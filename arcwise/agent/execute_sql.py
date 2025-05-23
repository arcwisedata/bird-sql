import asyncio
import csv
from io import StringIO

import sqlglot
import sqlglot.expressions as exp
from pydantic import BaseModel, Field

from .utils import SQLContext, execute_process_json
from ..typedefs import BIRDQuestion, SchemaPredictions
from ..utils import stringify

SQLScalar = str | int | float | bool | None


class ExecuteSQLToolArguments(BaseModel):
    step_by_step_description: str = Field(
        description="A step-by-step description of what this query does. Provide detailed reasoning for each step."
    )
    sql: str


class ExecuteSQLToolResult(BaseModel):
    error: str | None = None
    columns: list[str] | None = None
    rows: list[list[SQLScalar]] | None = None
    sql: str | None = None


class NoDataException(Exception):
    pass


EXECUTE_SQL_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_sql",
        "description": "Executes a SQL query on the database",
        "parameters": ExecuteSQLToolArguments.model_json_schema(),
    },
}


async def execute_sql_tool(
    arguments: ExecuteSQLToolArguments,
    sql_context: SQLContext,
    question: BIRDQuestion,
) -> tuple[str, ExecuteSQLToolResult]:
    sql = arguments.sql.strip().rstrip(";")
    sg_query = sqlglot.parse_one(sql, dialect=sql_context.dialect).transform(_lint_sql)
    assert isinstance(sg_query, exp.Query), "Only SELECT statements are supported"

    sql = sg_query.sql(dialect=sql_context.dialect)

    # Do some sanity checks before executing the final query
    _check_rounding(question, sql)
    # _check_mixed_division(sg_query)

    try:
        columns, rows = await execute_sql(sql, sql_context)
        tool_result = ExecuteSQLToolResult(rows=rows, columns=columns, sql=sql)
        gpt_result = _get_gpt_result(rows=rows, columns=columns)
    except Exception as exc:
        error_message = str(exc)
        tool_result = ExecuteSQLToolResult(sql=sql, error=error_message)
        gpt_result = "Error executing query: " + error_message
    return gpt_result, tool_result


EXECUTE_SQL_ROWS_BYTE_LIMIT = 512


def _get_gpt_result(
    rows: list[list[SQLScalar]],
    columns: list[str],
) -> str:
    tsv_preview = StringIO()
    writer = csv.writer(tsv_preview, delimiter="\t", lineterminator="\n")
    writer.writerow(columns)
    for i, row in enumerate(rows):
        if tsv_preview.tell() > EXECUTE_SQL_ROWS_BYTE_LIMIT:
            tsv_preview.write(f"```\n{len(rows) - i} more rows hidden")
            break
        writer.writerow([stringify(cell, quote_strings=False) for cell in row])

    return f"""row_count: {len(rows)}
```tsv
{tsv_preview.getvalue()}"""


def _get_unique_str(proposed_base_str: str, others: list[str]) -> str:
    final_str = proposed_base_str
    i = 1
    lowercase_others = set(s.lower() for s in others)
    while final_str.lower() in lowercase_others:
        final_str = f"{proposed_base_str}_{i}"
        i += 1
    return final_str


async def execute_sql(
    sql: str, sql_context: SQLContext, timeout: float = 30.0
) -> tuple[list[str], list[list[SQLScalar]]]:
    assert sql_context.dialect == "sqlite"
    rows = await asyncio.wait_for(_execute_sqlite(sql, sql_context), timeout)
    if not len(rows):
        # TODO: is there a way to force SQLite to always return the header?
        raise NoDataException("Query returned no results")
    return rows[0], list(rows[1:])  # type: ignore


async def _execute_sqlite(sql: str, sql_context: SQLContext) -> list[list[SQLScalar]]:
    response_json = await execute_process_json(
        ["sqlite3", "-json", "-readonly", sql_context.db_url, sql]
    )
    if not response_json:
        return []
    return [list(response_json[0].keys())] + [list(row.values()) for row in response_json]


def _lint_sql(node: exp.Expression) -> exp.Expression:
    # If there are any asc ORDER BYs, make sure that nulls_first is False
    if isinstance(node, exp.Ordered):
        if not node.args.get("desc"):
            node.args["nulls_first"] = False
    elif isinstance(node, exp.Div):
        # SQLite defaults to integer division, which is not desirable
        if node.find(exp.Cast) is None:
            node.args["this"] = sqlglot.cast(node.this, "real")

    return node


def _check_rounding(question: BIRDQuestion, sql: str):
    full_question = question.question_evidence().lower()
    round_in_query = "round(" in sql.lower()
    question_references_rounding = "decimal place" in full_question or "round(" in full_question

    if question_references_rounding and not round_in_query:
        raise RuntimeError(
            "Please ROUND your answer(s) to the appropriate number of decimal places."
        )
    elif round_in_query and not question_references_rounding:
        raise RuntimeError("Answers should not be rounded unless the question asks for it.")


# If a division mixes column references from two CTEs (or a CTE and the query itself),
# assert that the two scopes have the same set of tables.
# If they don't, it's a sign that the numerator & denominator operate on different quantities/units.
def _check_mixed_division(sg_query: exp.Query):
    # Find all tables used in each CTE
    ctes_to_tables: dict[str, set[str]] = {
        t.alias or t.name: {t.name for t in t.this.find_all(exp.Table)}
        for t in sg_query.find_all(exp.CTE)
    }
    if not ctes_to_tables:
        return

    # Find all tables used in the root query scope
    root_tables = {t.name for t in sg_query.find_all(exp.Table) if t.name not in ctes_to_tables}

    for node in sg_query.find_all(exp.Div):
        numerator_tables = _find_tables_used(node.left, ctes_to_tables, root_tables)
        denominator_tables = _find_tables_used(node.right, ctes_to_tables, root_tables)
        if numerator_tables and denominator_tables and numerator_tables != denominator_tables:
            raise RuntimeError(
                f"""
Division `{node.sql()}` is operating on different units.
The numerator and denominator must be derived from the same set of tables, e.g.

```sql
SELECT COUNT(DISTINCT CASE WHEN t2.is_valid THEN t1.id END) * 100.0 / COUNT(DISTINCT t1.id) percent
FROM table1 AS t1
INNER JOIN table2 AS t2 ON t1.id = t2.id
```

Please rework the query and try again."""
            )


def check_output_types(
    columns: list[str],
    rows: list[list[SQLScalar]],
    predicted_output_types: list[SchemaPredictions.OutputType],
) -> str | None:
    if len(rows) == 0:
        return None

    if len(columns) != len(predicted_output_types):
        return f"Expected {len(predicted_output_types)} columns, got {len(columns)}"

    # check whether the types match in the first row
    for val, pred in zip(rows[0], predicted_output_types):
        if val is None:
            # None values can be anything
            continue
        elif pred.type in {"text", "date", "datetime"} and isinstance(val, str):
            continue
        elif pred.type == "integer" and isinstance(val, int):
            continue
        elif pred.type == "real" and isinstance(val, float):
            continue
        else:
            return "Expected output column type(s): " + ", ".join(
                output_type.type for output_type in predicted_output_types
            )


# Find all table names used in column "scopes" for a given node
def _find_tables_used(
    node: exp.Expression, cte_table_refs: dict[str, set[str]], root_table_refs: set[str]
) -> set[str]:
    result = set()
    for column in node.find_all(exp.Column):
        result.update(cte_table_refs.get(column.table) or root_table_refs)
    return result
