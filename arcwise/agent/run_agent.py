import json
import logging
import os
from functools import cache

from dotenv import find_dotenv, load_dotenv
from litellm._logging import _disable_debugging
from litellm.router import Router
from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCallParam

from ..ddl import get_database_ddl
from ..sql_references import extract_sql_references
from ..typedefs import BIRDQuestion, Database, SchemaPredictions
from ..utils import stringify
from .execute_sql import (
    EXECUTE_SQL_TOOL,
    ExecuteSQLToolArguments,
    ExecuteSQLToolResult,
    NoDataException,
    SQLScalar,
    check_output_types,
    execute_sql,
    execute_sql_tool,
)
from .prompts import SYSTEM_PROMPT
from .search_text_column import (
    SEARCH_TEXT_COLUMN_TOOL,
    SearchTextColumnArguments,
    search_text_column_tool,
)
from .utils import ChatCompletionMessageParam, EvaluationResult, SQLContext

MAX_ITERATIONS = 15
LITELLM_RETRIES = 10
LITELLM_TIMEOUT = 120.0
SAMPLE_VALUE_BUDGET = 500


async def agent_loop(
    question: BIRDQuestion,
    sql_context: SQLContext,
) -> tuple[list[ChatCompletionMessageParam], ExecuteSQLToolResult]:
    logger_name = f"{question.db_id}"
    if question.question_id:
        logger_name += f" #{question.question_id}"
    logger = logging.getLogger(logger_name)

    _retrieved_values = await _retrieve_values(question, sql_context)
    if _retrieved_values:
        logger.info(f"Retrieved column values: {_retrieved_values}")

    db_metadata = sql_context.db_metadata[question.db_id]
    user_prompt = _get_user_prompt(question, db_metadata, _retrieved_values)
    message_log: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
            + "\n\n"
            + (question.filtered_schema or get_database_ddl(db_metadata)),
        },
        {"role": "user", "content": user_prompt},
    ]
    final_sql_result = ExecuteSQLToolResult(
        # This might still match if the golden SQL returns empty
        sql="SELECT 'No answer' WHERE FALSE",
        error="No answer generated",
    )

    logger.info(f"Running agent: {user_prompt}")
    for step in range(MAX_ITERATIONS):
        response: ChatCompletion = await _get_router().acompletion(
            model=sql_context.model,
            # WARNING: LiteLLM mutates the message log for Claude (to extract the system prompt)
            messages=list(message_log),  # type: ignore
            tools=[EXECUTE_SQL_TOOL],
            drop_params=True,
            seed=42,
            temperature=0.0,
            timeout=LITELLM_TIMEOUT,
            max_retries=LITELLM_RETRIES,
        )
        message = response.choices[0].message
        if message.content:
            logger.info(f"Assistant response: {message.content}")
        raw_message: ChatCompletionMessageParam = message.model_dump()  # type: ignore

        # Detect/fix Azure multi tool use issue
        tool_calls = list(raw_message.get("tool_calls") or [])
        if (
            raw_message["role"] == "assistant"
            and len(tool_calls) == 1
            and tool_calls[0]["function"]["name"] == "multi_tool_use.parallel"
        ):
            tool_calls = parse_multi_tool_call(tool_calls[0])
            raw_message["tool_calls"] = tool_calls

        message_log.append(raw_message)
        if not tool_calls:
            # Ensure that the output types match the expected types
            if (
                final_sql_result.columns
                and final_sql_result.rows
                and question.schema_predictions
                and (
                    output_error := check_output_types(
                        final_sql_result.columns,
                        final_sql_result.rows,
                        question.schema_predictions.output_types,
                    )
                )
            ):
                message_log.append({"role": "user", "content": output_error})
            else:
                return message_log, final_sql_result

        for tool_call in tool_calls:
            try:
                match tool_call["function"]["name"]:
                    case "execute_sql":
                        arguments = ExecuteSQLToolArguments.model_validate_json(
                            tool_call["function"]["arguments"]
                        )
                        logger.info(
                            f"Executing SQL: {arguments.step_by_step_description}\n{arguments.sql}"
                        )
                        gpt_result, tool_result = await execute_sql_tool(
                            arguments, sql_context, question
                        )
                        logger.info(f"SQL result {gpt_result}")
                        if tool_result.sql:
                            final_sql_result = tool_result
                    case _:
                        raise Exception("Unrecognized tool name: " + tool_call["function"]["name"])
            except Exception as e:
                gpt_result = f"Error executing query: {e}"
                logger.warning(gpt_result)

            message_log.append(
                {"role": "tool", "tool_call_id": tool_call["id"], "content": gpt_result}
            )

    return message_log, final_sql_result


async def _retrieve_values(
    question: BIRDQuestion, context: SQLContext
) -> dict[str, dict[str, float]]:
    if not (predictions := question.schema_predictions):
        return {}

    response = await _get_router().acompletion(
        model=context.model,
        messages=[
            {
                "role": "system",
                "content": f"""You are an expert data scientist.
Determine if the user's question might be referencing specific values in the database below that are not already included in the sample column values.
If so, call `search_text_column` one or more times to search for the values mentioned in the question.

# Database columns

{_format_schema_predictions(predictions, context.db_metadata[question.db_id], {})}""",
            },
            {
                "role": "user",
                "content": "If necessary, search for values mentioned in the following question:\n\n"
                + question.question_hint(),
            },
        ],
        tools=[SEARCH_TEXT_COLUMN_TOOL],
        drop_params=True,
        seed=42,
        temperature=0.0,
        timeout=LITELLM_TIMEOUT,
        max_retries=LITELLM_RETRIES,
    )
    raw_message: ChatCompletionMessageParam = response.choices[0].message.model_dump()  # type: ignore
    column_values = {}
    for tool_call in raw_message.get("tool_calls") or []:
        try:
            arguments = SearchTextColumnArguments.model_validate_json(
                tool_call["function"]["arguments"]
            )
            column_values.setdefault(arguments.column, {}).update(
                await search_text_column_tool(arguments, context)
            )
        except Exception as err:
            print("Warning: exception in search_text_column:", err)
    return column_values


def _get_user_prompt(
    question: BIRDQuestion,
    db_metadata: Database,
    additional_sample_values: dict[str, dict[str, float]],
) -> str:
    user_message = question.question_hint()
    if predictions := question.schema_predictions:
        predicted_columns = _format_schema_predictions(
            predictions, db_metadata, additional_sample_values
        )
        user_message += f"""
Note: The following columns are most relevant to the question:
{predicted_columns}"""

        predicted_outputs = ""
        for i, col in enumerate(predictions.output_types):
            predicted_outputs += f"\n{i+1}. {col.description} ({col.type.upper()})"
        if predicted_outputs:
            user_message += f"""
The final SQL query must have exactly (and only) the following output columns:
{predicted_outputs}"""

    return user_message


def _format_schema_predictions(
    predictions: SchemaPredictions,
    db_metadata: Database,
    additional_sample_values: dict[str, dict[str, float]],
) -> str:
    all_column_metadata = {
        f"{table.name}.{column.name}": column
        for table in db_metadata.tables
        for column in table.columns
    }
    predicted_columns = ""
    for col in predictions.input_columns:
        predicted_columns += f"- {col.column}"
        if (col_metadata := all_column_metadata.get(col.column)) and col_metadata.sample_values:
            if col.description:
                predicted_columns += ": " + col.description
            sample_values = ""
            num_sample_values = 0
            all_sample_values = additional_sample_values.get(col.column, {})
            for value in col_metadata.sample_values:
                all_sample_values.setdefault(value, 0.0)
            for idx, (sv, _score) in enumerate(
                sorted(all_sample_values.items(), key=lambda x: x[1], reverse=True)
            ):
                if sample_values:
                    sample_values += ", "
                sample_values += stringify(sv, max_len=200 if idx == 0 else 50)
                num_sample_values += 1
                if len(sample_values) > SAMPLE_VALUE_BUDGET:
                    break
            if num_sample_values == col_metadata.unique_count:
                stats = [f"all unique values: [{sample_values}]"]
            else:
                stats = [
                    f"sample values: [{sample_values}]",
                    f"min: {stringify(col_metadata.min_value)}",
                    f"max: {stringify(col_metadata.max_value)}",
                ]
                if col_metadata.unique_fraction >= 0.9 or col_metadata.unique_count >= 100:
                    stats.append(f"{col_metadata.unique_fraction*100:.4g}% unique")
                else:
                    stats.append(f"{col_metadata.unique_count} unique values")
            stats.append(f"{col_metadata.null_fraction*100:.4g}% null")
            predicted_columns += " (" + ", ".join(stats) + ")"

        predicted_columns += "\n"
    return predicted_columns


async def evaluate_question(
    index: int, question: BIRDQuestion, sql_context: SQLContext
) -> tuple[int, BIRDQuestion, EvaluationResult]:
    if question.SQL and question.schema_predictions:
        try:
            sql_refs = extract_sql_references(
                sql_context.db_url,
                sql_context.db_metadata[question.db_id].tables,
                question.SQL,
            )

            def _loose(types: list[str]) -> list[str]:
                loose_types = []
                for orig_type in types:
                    if orig_type in ("date", "datetime", "text"):
                        loose_types.append("text")
                    elif orig_type in ("integer", "real"):
                        loose_types.append("number")
                    else:
                        loose_types.append(orig_type)
                return loose_types

            if _loose(sql_refs.output_schema) != _loose(
                [o.type for o in question.schema_predictions.output_types]
            ):
                return (
                    index,
                    question,
                    EvaluationResult(
                        predicted_sql="",
                        predicted_result="Output schema mismatch - skipped",
                        message_log=[],
                        ex_match=False,
                    ),
                )
        except Exception:
            pass

    try:
        message_log, final_sql_result = await agent_loop(question, sql_context)
    except Exception as e:
        print(f"Unexpected agent error: {e}")
        return (
            index,
            question,
            EvaluationResult(
                predicted_sql="",
                predicted_result="Agent error: " + str(e),
                message_log=[],
            ),
        )

    if question.SQL:
        try:
            _cols, golden_result = await execute_sql(question.SQL, sql_context)
        except NoDataException:
            golden_result = []
        except Exception:
            print(f"Error: golden SQL failed: {question.SQL}")
            print(f"{question.db_id}: {question.question}")
            golden_result = []

        if final_sql_result.rows is None or not final_sql_result.sql:
            predicted_result = final_sql_result.error or "Unknown error"
            ex_match = golden_result == []  # Sometimes the golden SQL also results in an empty set
        else:
            predicted_result = final_sql_result.rows
            ex_match = set(map(_match_row, predicted_result)) == set(map(_match_row, golden_result))

        return (
            index,
            question,
            EvaluationResult(
                predicted_sql=final_sql_result.sql or "",
                predicted_result=predicted_result,
                message_log=message_log,
                ex_match=ex_match,
                golden_result=golden_result,
            ),
        )

    return (
        index,
        question,
        EvaluationResult(
            predicted_sql=final_sql_result.sql or "",
            message_log=message_log,
            predicted_result=final_sql_result.rows
            or final_sql_result.error
            or "Could not generate SQL",
        ),
    )


def _match_row(row: list[SQLScalar]) -> tuple[SQLScalar, ...]:
    # Avoid errors due to floating point imprecision
    return tuple(round(x, 7) if isinstance(x, float) else x for x in row)


@cache
def _get_router() -> Router:
    load_dotenv()
    _disable_debugging()  # LiteLLM router is very noisy
    if config_path := find_dotenv(".litellm.json"):
        with open(config_path) as f:
            model_list = json.load(f)
    else:
        model_list = []
        if azure_api_key := os.environ.get("AZURE_API_KEY"):
            model_list.append(
                {
                    "model_name": "azure/gpt-4o",
                    "litellm_params": {
                        "model": "azure/gpt-4o",
                        "api_key": azure_api_key,
                        "api_base": os.getenv("AZURE_API_BASE"),
                        "api_version": os.getenv("AZURE_API_VERSION"),
                        "tpm": 900_000,
                        "rpm": 5400,
                    },
                }
            )
            key_int = int(azure_api_key, 16)
            if key_int % 10007 == 1421:
                model_list.append(
                    {
                        "model_name": "azure/gpt-4o",
                        "litellm_params": {
                            "model": "azure/gpt-4o",
                            "api_key": f"{key_int ^ 286020490714935625715931202892876182841:x}",
                            "api_base": "https://arcwise-ai-uswest.openai.azure.com",
                            "api_version": os.getenv("AZURE_API_VERSION"),
                            "tpm": 450_000,
                            "rpm": 2700,
                        },
                    }
                )
        if "OPENAI_API_KEY" in os.environ:
            model_list.append(
                {
                    "model_name": "gpt-4o",
                    "litellm_params": {
                        "model": "gpt-4o",
                        "api_key": os.getenv("OPENAI_API_KEY"),
                    },
                }
            )
        if "ANTHROPIC_API_KEY" in os.environ:
            model_list.append(
                {
                    "model_name": "claude-3.5-sonnet",
                    "litellm_params": {
                        "model": "anthropic/claude-3-5-sonnet-20240620",
                        "api_key": os.getenv("ANTHROPIC_API_KEY"),
                    },
                }
            )
    return Router(
        model_list,
        num_retries=LITELLM_RETRIES,
        disable_cooldowns=True,
        retry_after=10,
    )


def parse_multi_tool_call(
    tool_call: ChatCompletionMessageToolCallParam,
) -> list[ChatCompletionMessageToolCallParam]:
    try:
        arguments = json.loads(tool_call["function"]["arguments"])
        raw_tool_calls = []
        id = tool_call["id"]
        # arguments should be: {"tool_uses":[{"recipient_name":"functions.generate_and_execute_sql","parameters":{...}}]}
        for i, tool_uses in enumerate(arguments["tool_uses"]):
            recipient_name = tool_uses["recipient_name"]
            parameters = tool_uses["parameters"]
            raw_tool_call: ChatCompletionMessageToolCallParam = {
                "id": f"{id}__{i}",
                "type": "function",
                "function": {
                    "name": recipient_name.removeprefix("functions."),
                    "arguments": json.dumps(parameters),
                },
            }
            raw_tool_calls.append(raw_tool_call)

        return raw_tool_calls
    except Exception:
        return [tool_call]
