import json
import re

import litellm
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from .typedefs import ColumnInfo, Database, Table
from .utils import stringify


def _normalize_for_json(name: str):
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _format_column(c: ColumnInfo):
    result = f"# {c.name}"
    if c.null_fraction == 1:
        return result + "\nNot useful"

    if c.original_name:
        result += f"\nOriginal name: {c.original_name}"
    if c.description:
        desc = c.description.replace("\n", "\t\n")
        result += f"\nOriginal description: {desc}"
    if c.value_description:
        desc = c.value_description.replace("\n", "\t\n")
        result += f"\nValue description: {desc}"
    if c.sample_values:
        result += "\n" + _format_sample_values(c)
    result += f"\nRange: {stringify(c.min_value)} to {stringify(c.max_value)}"
    return result


def _format_sample_values(c: ColumnInfo) -> str:
    sample_values = ""
    num_sample_values = 0
    for sv in c.sample_values:
        if sample_values:
            sample_values += ", "
        sample_values += stringify(sv)
        num_sample_values += 1
        if len(sample_values) > 200:
            break
    if num_sample_values < c.unique_count:
        sample_values += ", ..."
    return f"Sample values ({c.unique_count} unique): " + sample_values


async def generate_table_and_columns_ai_description(table: Table, model: str):
    system_prompt = """You are an expert, detail-oriented, data analyst.
Your task is to call `describe_table` with high-quality descriptions based on user-provided tables.

For each column, provide as concise and informative of a description as you can given the following constraints:
- Omit the description if the column name makes it completely obvious.
- If a column is the same as a previous column (and its sample values are of the same format), its description should be 'See [previous column]'.
- IMPORTANT: if "Value description" explains the meanings of certain values, or mentions that the values are not useful, this information MUST be preserved in the final description.
    - If "Value description" is inconsistent with the sample values, the sample values take priority and must be used instead.
- If the values appear to follow a consistent format or pattern, describe the format or pattern. e.g. if the values are '<html>...</html>', describe them simply as HTML values without providing sample values.
- If the values are numerical, describe the range of values.
- Otherwise, as long as the values are human-readable strings, provide a few sample values. If there are fewer than 5, list them all. NEVER include sample values that have been truncated and end with '…'.
- Put single quotes around ALL string values.

The ordering of the columns should match their original ordering. Only use the information provided by the user.
At the end, provide a table_description (but do not mention the exact row count.)"""
    tool = {
        "type": "function",
        "function": {
            "name": "describe_table",
            "description": "See instructions",
            "parameters": {
                "type": "object",
                "properties": {
                    **{
                        _normalize_for_json(c.name) + "-description": {
                            "type": "string",
                            "description": f"Description for column {c.name}",
                        }
                        for c in table.columns
                    },
                    "table_description": {
                        "type": "string",
                        "description": "Brief 1-line table description",
                    },
                },
                "additionalProperties": False,
                "required": [_normalize_for_json(c.name) + ".description" for c in table.columns]
                + ["table_description"],
            },
        },
    }
    column_data = "\n".join([_format_column(c) for c in table.columns])
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f'Describe this SQLite table named "{table.name}" with {table.row_count} rows and the following columns:\n\n{column_data}',
        },
    ]
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=5, max=30),
        ):
            with attempt:
                result = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    tools=[tool],
                    tool_choice={"type": "function", "function": {"name": "describe_table"}},  # type: ignore
                    temperature=0.0,
                    timeout=60.0,
                )
        descriptions = json.loads(result.choices[0].message.tool_calls[0].function.arguments)  # type: ignore
    except Exception:
        for column in table.columns:
            description = ""
            if column.description:
                description += f"\n{column.description[:200]}"
            if column.value_description:
                description += f"\nValue description: {column.value_description[:200]}"
            else:
                description += "\n" + _format_sample_values(column)
            column.ai_description = description.strip()
        return

    table.ai_description = descriptions.get("table_description", "")
    for column in table.columns:
        column.ai_description = (
            descriptions.get(_normalize_for_json(column.name) + "-description") or ""
        )


def use_pregenerated_descriptions(
    database: Database,
    column_descriptions: dict[str, str],
) -> None:
    for table in database.tables:
        # TODO: should we still generate a table description?
        for column in table.columns:
            description = column_descriptions.get(f"{database.name}|{table.name}|{column.name}")
            if description:
                # Strip all leading '# ' characters
                description = re.sub(r"^[# ]*", "", description)
            else:
                print(
                    f"Warning: no pre-generated description for {database.name}.{table.name}.{column.name}"
                )
            column.ai_description = description
