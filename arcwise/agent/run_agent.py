import logging

import litellm
from dotenv import load_dotenv
from openai.types.chat import ChatCompletion

from ..ddl import get_database_ddl
from ..typedefs import BIRDQuestion
from .execute_sql import (
    EXECUTE_SQL_TOOL,
    ExecuteSQLToolArguments,
    ExecuteSQLToolResult,
    execute_sql,
    execute_sql_tool,
)
from .prompts import SYSTEM_PROMPT
from .utils import ChatCompletionMessageParam, EvaluationResult, SQLContext
from ..utils import truncate_val
from openai.types.chat import ChatCompletionMessageToolCall, ChatCompletionUserMessageParam

load_dotenv()


MAX_ITERATIONS = 10
LITELLM_RETRIES = 3
LITELLM_TIMEOUT = 60.0


async def agent_loop(
    question: BIRDQuestion,
    sql_context: SQLContext,
) -> tuple[list[ChatCompletionMessageParam], ExecuteSQLToolResult]:
    user_message = question.question.strip()
    db_metadata = sql_context.db_metadata[question.db_id]
    if question.evidence:
        # TODO: need to flag this as potentially inaccurate
        user_message += "\nContext: " + question.evidence.strip()

    if predictions := question.schema_predictions:
        predicted_columns = ""
        for col in predictions.input_columns:         
            predicted_columns += f"- {col.column}"
            table_name, col_name = col.column.split(".")
            table_metadata = next((t for t in db_metadata.tables if t.name == table_name), None)
            if table_metadata is not None:
                col_metadata = next((c for c in table_metadata.columns if c.name == col_name), None)
                if col_metadata is not None:
                    samples = [f"'{truncate_val(sv)}'" if isinstance(sv, str) else str(sv) for sv in col_metadata.sample_values]
                    predicted_columns += f" (sample values: {', '.join(samples)}, ...)"
                    
            predicted_columns += "\n"

        user_message += f"""
Hint: The following columns are most relevant to the question:
{predicted_columns}
When you have the final answer, you MUST conclude by running `execute_sql` with a `query_identifier` of "final_answer" with all information in one single query."""

        predicted_outputs = ""
        for i, col in enumerate(predictions.output_types):
            predicted_outputs += f"\n{i+1}. {col.type.upper()}: {col.description}"
        if predicted_outputs:
            user_message += f"""
I need final_answer to have exactly the following column types:
{predicted_outputs}"""

    message_log: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
            + "\n\n"
            + (
                question.filtered_schema
                or get_database_ddl(db_metadata)
            ),
        },
        {"role": "user", "content": user_message},
    ]
    final_sql_result = ExecuteSQLToolResult(error="execute_sql was never called")

    logger_name = f"{question.db_id}"
    if question.question_id:
        logger_name += f" #{question.question_id}"
    logger = logging.getLogger(logger_name)

    logger.info(f"Running agent: {user_message}")

    sql_by_exec_result_id: dict[str, str] = {}
    for _ in range(MAX_ITERATIONS):
        response: ChatCompletion = await litellm.acompletion(
            model=sql_context.model,
            # WARNING: LiteLLM mutates the message log for Claude (to extract the system prompt)
            messages=list(message_log),  # type: ignore
            tools=[EXECUTE_SQL_TOOL],
            drop_params=True,
            temperature=0.0,
            timeout=60.0,
            max_retries=3,
        )
        message = response.choices[0].message
        if message.content:
            logger.info(f"Assistant response: {message.content}")
        message_log.append(message.model_dump())  # type: ignore

        if not (tool_calls := getattr(message, "tool_calls", None)):
            last_tool_call_json = next((tcs[0] for message in message_log[-1:0:-1] if (tcs := message.get("tool_calls"))), None)
            final_answer_tool_call_user_message: ChatCompletionMessageParam = {"role": "user", "content": "Please provide the requested final_answer as the last tool call."}
            if last_tool_call_json is None:
                message_log.append(final_answer_tool_call_user_message)
            else:
                last_tool_call = ChatCompletionMessageToolCall.model_validate(last_tool_call_json)
                last_execute_sql_tool_call = ExecuteSQLToolArguments.model_validate_json(last_tool_call.function.arguments)
                if last_execute_sql_tool_call.query_identifier.startswith("final_answer"):
                    # Agent is finished and the last tool call does indeed have a "final answer"
                    break
                else:
                    message_log.append(final_answer_tool_call_user_message)

        for tool_call in tool_calls or []:
            try:
                arguments = ExecuteSQLToolArguments.model_validate_json(
                    tool_call.function.arguments
                )
                logger.info(
                    f"Executing SQL: {arguments.query_description}\n{arguments.sql}"
                )
                gpt_result, tool_result = await execute_sql_tool(
                    arguments, sql_by_exec_result_id, sql_context, question, predictions.output_types if predictions else None
                )
                logger.info(f"SQL result {gpt_result}")
                if tool_result.exec_result_id and tool_result.sql:
                    sql_by_exec_result_id[tool_result.exec_result_id] = tool_result.sql
                    final_sql_result = tool_result
            except Exception as e:
                gpt_result = f"Error parsing execute_sql arguments: {e}"
                logger.warning(gpt_result)
                tool_result = ExecuteSQLToolResult(error=gpt_result)

            message_log.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": gpt_result}
            )

    return message_log, final_sql_result


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
        except Exception:
            print(f"Error: golden SQL failed: {question.SQL}")
            print(f"{question.db_id}: {question.question}")
            golden_result = []

        if final_sql_result.rows is None or not final_sql_result.sql:
            predicted_result = final_sql_result.error or "Unknown error"
            ex_match = False
            final_sql = ""
        else:
            predicted_result = final_sql_result.rows
            ex_match = set(map(tuple, predicted_result)) == set(
                map(tuple, golden_result)
            )
            final_sql = final_sql_result.sql

        return (
            index,
            question,
            EvaluationResult(
                predicted_sql=final_sql,
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
