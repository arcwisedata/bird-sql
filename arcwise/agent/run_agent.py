import asyncio
import json
import logging
import os
from functools import cache

from litellm._logging import _disable_debugging
from litellm.router import Router
from dotenv import load_dotenv, find_dotenv
from openai.types.chat import ChatCompletion

from ..ddl import get_database_ddl
from ..typedefs import BIRDQuestion, Database
from .execute_sql import (
    EXECUTE_SQL_TOOL,
    ExecuteSQLToolArguments,
    ExecuteSQLToolResult,
    NoDataException,
    SQLScalar,
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
from ..utils import stringify
from openai.types.chat import ChatCompletionMessageToolCallParam


MAX_ITERATIONS = 15
LITELLM_RETRIES = 10
LITELLM_TIMEOUT = 120.0
SAMPLE_VALUE_BUDGET = 200


async def agent_loop(
    question: BIRDQuestion,
    sql_context: SQLContext,
) -> tuple[list[ChatCompletionMessageParam], ExecuteSQLToolResult]:
    db_metadata = sql_context.db_metadata[question.db_id]
    user_prompt = _get_user_prompt(question, db_metadata)
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

    logger_name = f"{question.db_id}"
    if question.question_id:
        logger_name += f" #{question.question_id}"
    logger = logging.getLogger(logger_name)
    logger.info(f"Running agent: {user_prompt}")

    terminal_messages = 0
    sql_by_exec_result_id: dict[str, str] = {}
    for _ in range(MAX_ITERATIONS):
        response: ChatCompletion = await _get_router().acompletion(
            model=sql_context.model,
            # WARNING: LiteLLM mutates the message log for Claude (to extract the system prompt)
            messages=list(message_log),  # type: ignore
            tools=[EXECUTE_SQL_TOOL, SEARCH_TEXT_COLUMN_TOOL],
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
            if _has_final_answer(message_log):
                break
            elif terminal_messages >= 1:
                return message_log, final_sql_result
            else:
                message_log.append(
                    {"role": "user", "content": "Please call execute_sql with the final_answer."}
                )
                terminal_messages += 1

        for tool_call in tool_calls:
            try:
                match tool_call["function"]["name"]:
                    case "execute_sql":
                        arguments = ExecuteSQLToolArguments.model_validate_json(
                            tool_call["function"]["arguments"]
                        )
                        logger.info(
                            f"Executing SQL: {arguments.query_description}\n{arguments.sql}"
                        )
                        gpt_result, tool_result = await execute_sql_tool(
                            arguments,
                            sql_by_exec_result_id,
                            sql_context,
                            question,
                            (
                                question.schema_predictions.output_types
                                if question.schema_predictions
                                else None
                            ),
                        )
                        logger.info(f"SQL result {gpt_result}")
                        if tool_result.exec_result_id and tool_result.sql:
                            sql_by_exec_result_id[tool_result.exec_result_id] = tool_result.sql
                            final_sql_result = tool_result
                    case "search_text_column":
                        search_args = SearchTextColumnArguments.model_validate_json(
                            tool_call["function"]["arguments"]
                        )
                        logger.info(
                            f"search_text_column: '{search_args.search_value}' in {search_args.table}.{search_args.column}"
                        )
                        gpt_result = await asyncio.to_thread(
                            lambda: search_text_column_tool(search_args, sql_context)
                        )
                        logger.info("search_text_column result: " + gpt_result)
                    case _:
                        raise Exception("Unrecognized tool name: " + tool_call["function"]["name"])
            except Exception as e:
                gpt_result = f"Error executing query: {e}"
                logger.warning(gpt_result)

            message_log.append(
                {"role": "tool", "tool_call_id": tool_call["id"], "content": gpt_result}
            )

    return message_log, final_sql_result


def _get_user_prompt(question: BIRDQuestion, db_metadata: Database) -> str:
    user_message = question.question.strip()
    if question.evidence:
        # TODO: need to flag this as potentially inaccurate
        user_message += "\nContext: " + question.evidence.strip()

    all_column_metadata = {
        f"{table.name}.{column.name}": column
        for table in db_metadata.tables
        for column in table.columns
    }
    if predictions := question.schema_predictions:
        predicted_columns = ""
        for col in predictions.input_columns:
            predicted_columns += f"- {col.column}"
            if (col_metadata := all_column_metadata.get(col.column)) and col_metadata.sample_values:
                if col.description:
                    predicted_columns += ": " + col.description
                sample_values = ""
                num_sample_values = 0
                for sv in col_metadata.sample_values:
                    if sample_values:
                        sample_values += ", "
                    sample_values += stringify(sv)
                    num_sample_values += 1
                    if len(sample_values) > SAMPLE_VALUE_BUDGET:
                        break
                if num_sample_values == col_metadata.unique_count:
                    stats = [f"all unique values: [{sample_values}]"]
                else:
                    stats = [
                        f"min: {stringify(col_metadata.min_value)}",
                        f"max: {stringify(col_metadata.max_value)}",
                    ]
                    if col_metadata.type.lower() == "text":
                        stats.insert(0, f"sample values: [{sample_values}]")
                    if col_metadata.unique_fraction >= 0.9 or col_metadata.unique_count >= 100:
                        stats.append(f"{col_metadata.unique_fraction*100:.4g}% unique")
                    else:
                        stats.append(f"{col_metadata.unique_count} unique values")
                stats.append(f"{col_metadata.null_fraction*100:.4g}% null")
                predicted_columns += " (" + ", ".join(stats) + ")"

            predicted_columns += "\n"

        user_message += f"""
Hint: The following columns are most relevant to the question:
{predicted_columns}
When you have the final answer, you MUST conclude by running `execute_sql` with a `query_identifier` of "final_answer" with all information in one single query."""

        predicted_outputs = ""
        for i, col in enumerate(predictions.output_types):
            predicted_outputs += f"\n{i+1}. {col.type.upper()} ({col.description})"
        if predicted_outputs:
            user_message += f"""
I need final_answer to have exactly (and only) the following column types:
{predicted_outputs}"""

    return user_message


async def evaluate_question(
    index: int, question: BIRDQuestion, sql_context: SQLContext
) -> tuple[int, BIRDQuestion, EvaluationResult]:
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


def _has_final_answer(message_log: list[ChatCompletionMessageParam]) -> bool:
    for message in message_log[-1:0:-1]:
        if not (tcs := message.get("tool_calls")):
            continue

        for tc in tcs:
            try:
                if tc["function"]["name"] != "execute_sql":
                    continue
                args = ExecuteSQLToolArguments.model_validate_json(tc["function"]["arguments"])
                if args.query_identifier.startswith("final_answer"):
                    return True
            except Exception:
                pass

        # Only check the last set of tool calls
        break

    return False


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
