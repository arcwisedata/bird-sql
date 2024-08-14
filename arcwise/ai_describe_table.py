import asyncio
import re

import litellm
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from .typedefs import ColumnInfo, Database, Table
from .utils import stringify


def _format_column(c: ColumnInfo):
    result = f"# {c.name}"
    if c.null_fraction == 1:
        return result + "\nNot useful"
    result += f"\nType: {c.type}"
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
    system_prompt = """You are an expert, detail-oriented data analyst.
Your task is to provide high-quality but concise documentation for each column in a user-provided SQL table.
Repeat each column on its own line as `# column_name` followed by a short one-line description as a line comment, e.g. `-- Description`.
Try to keep descriptions to under 100 characters.
Information such as descriptions, sample values, and ranges will be provided for each column. Only use the information provided by the user.
Do not respond with any other information.

- If a column is the same as a previous column (and its sample values are of the same format), its description should be 'See [previous column]'.
- IMPORTANT: if "Value description" explains the meanings of certain values, or mentions that the values are not useful, this information MUST be preserved in the final description.
    - If "Value description" is inconsistent with the sample values, the sample values take priority and must be used instead.
- If the values appear to follow a consistent format or pattern, describe the format or pattern. e.g. if the values are '<html>...</html>', describe them simply as HTML values without providing sample values.
- If the values are numerical, describe the range of values.
- Otherwise, as long as the values are human-readable strings, provide up to 5 sample values that are most representative. If there are fewer than 5, list them all. NEVER include sample values that have been truncated and end with '…'.
- Put single quotes around ALL string values.
- The ordering of the columns must match their original ordering.

For example, given the columns:

<example>
# status
Original description: status of the order
Value description: 1 = Pending; 2 = Approved; 3 = Rejected; 4 = Complete
Sample values (4 unique): '1', '2', '3', '4'
# zip_code
Original description: zip code of the order address
Sample values (1234 unique): '12345', '54321', '90210', '00000'
# order_flag
Value description: not useful
Range: 0 to 1
</example>

<example_response>
# status
-- Order status. '1' = Pending, '2' = Approved, '3' = Rejected, '4' = Complete
# zip_code
-- 5-digit ZIP code for the order delivery address, e.g. '90210'
# order_flag
-- Not useful
</example_response>"""
    column_data = "\n".join([_format_column(c) for c in table.columns])
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Describe this SQLite table named `{table.name}` with {table.row_count} rows and the following columns:\n\n{column_data}",
        },
    ]
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=5, max=30),
        ):
            with attempt:
                result = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    timeout=60.0,
                )
        descriptions = result.choices[0].message.content.splitlines()  # type: ignore
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

    columns_by_name = {c.name: c for c in table.columns}
    last_column = None
    for line in descriptions:
        if not line:
            continue
        if line.startswith("# "):
            last_column = columns_by_name.get(line[2:].strip())
            if not last_column:
                print(f"Warning: {line[2:]} is not a valid column")
        elif line.startswith("-- ") and last_column:
            last_column.ai_description = line[3:].strip()


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


if __name__ == "__main__":
    table = Table(
        name="legalities",
        ai_description="Stores the legality status of cards across various game formats\n427907 rows, primary key: (id)",
        row_count=427907,
        primary_key=["id"],
        foreign_keys=[],
        columns=[
            ColumnInfo(
                name="id",
                original_name=None,
                type="integer",
                description="unique id identifying this legality",
                value_description=None,
                null_fraction=0.0,
                unique_count=427907,
                unique_fraction=1.0,
                sample_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                min_value=1,
                max_value=427907,
            ),
            ColumnInfo(
                name="format",
                original_name=None,
                type="text",
                description="format of play",
                value_description="each value refers to different rules to play",
                null_fraction=0.0,
                unique_count=15,
                unique_fraction=3.505434592095946e-05,
                sample_values=[
                    "vintage",
                    "legacy",
                    "commander",
                    "duel",
                    "modern",
                    "penny",
                    "pauper",
                    "pioneer",
                    "premodern",
                    "historic",
                ],
                min_value="brawl",
                max_value="vintage",
            ),
            ColumnInfo(
                name="status",
                original_name=None,
                type="text",
                description=None,
                value_description="• legal\n• banned\n• restricted",
                null_fraction=0.0,
                unique_count=3,
                unique_fraction=7.010869184191892e-06,
                sample_values=["Legal", "Banned", "Restricted"],
                min_value="Banned",
                max_value="Restricted",
            ),
            ColumnInfo(
                name="uuid",
                original_name=None,
                type="text",
                description=None,
                value_description=None,
                null_fraction=0.0,
                unique_count=55608,
                unique_fraction=0.1299534711981809,
                sample_values=[
                    "fed8bfd4-eea9-5466-acf5-0421c83789fa",
                    "fea502b2-f7ed-5947-8c11-2659905e539a",
                ],
                min_value="00010d56-fe38-5e35-8aed-518019aa36a5",
                max_value="fffdd333-3789-5104-a8be-37be199a2cb1",
            ),
        ],
    )
    asyncio.run(generate_table_and_columns_ai_description(table, "gpt-4o"))

    for c in table.columns:
        print("#", c.name)
        print(c.ai_description)
