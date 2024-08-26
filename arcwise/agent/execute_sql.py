import asyncio
import csv
import re
from io import StringIO
from typing import Literal

import sqlglot
import sqlglot.expressions as exp
from pydantic import BaseModel, Field

from .utils import SQLContext, execute_process_json
from ..typedefs import BIRDQuestion, SchemaPredictions
from ..utils import stringify

SQLScalar = str | int | float | bool | None


class Join(BaseModel):
    source: str = Field(description="The table or previous exec_result_id to join")
    description: str = Field(description="Explain why this join is needed")
    alias: str | None = Field(default=None, description="Optional alias for the join table")
    on: str = Field(description="Join condition SQL")
    missing_behavior: str = Field(
        description="Should rows from the source be included if no matching join is found? Provide a brief explanation"
    )
    join_type: Literal["inner", "left"] = Field(
        description="Use a left join if rows without joins should be included"
    )


class Column(BaseModel):
    description: str = Field(description="A brief explanation of the purpose of this column")
    alias: str = Field(description="A short lower_camel_case identifier for this column")
    deduplication_behavior: str = Field(
        description="Does this column reference joined tables? Is there a risk of duplicates affecting the result? Specify whether or not deduplication is required",
    )
    sql: str = Field(
        description="The SQL expression for this column. References should be fully qualified and escaped if needed"
    )


class ExecuteSQLToolArguments(BaseModel):
    query_description: str = Field(description="An explanation of the purpose of this query")
    query_identifier: str = Field(
        description="A short lower_camel_case identifier for the query result"
    )
    from_: str = Field(description="A table or previous exec_result_id to use as the source table")
    joins: list[Join] = Field(default=[], description="JOIN clauses")
    where: str | None = Field(default=None, description="WHERE clause")
    columns: list[Column] = Field(description="One or more columns to SELECT")
    having: str | None = Field(default=None, description="HAVING clause")
    group_by: list[str] = Field(
        default=[],
        description="GROUP BY clauses. Prefer to reference columns by number if possible.",
    )
    order_by: list[str] = Field(
        default=[],
        description="ORDER BY clauses. Prefer to reference columns by number if possible.",
    )
    limit: str | None = Field(
        default=None, description="LIMIT clause, with offset if needed as well"
    )


class ExecuteSQLToolResult(BaseModel):
    error: str | None = None
    columns: list[str] | None = None
    rows: list[list[SQLScalar]] | None = None
    exec_result_id: str | None = None
    sql: str | None = None


class NoDataException(Exception):
    pass


EXECUTE_SQL_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_sql",
        "description": "Executes a SQL query",
        "parameters": ExecuteSQLToolArguments.model_json_schema(),
    },
}


async def execute_sql_tool(
    arguments: ExecuteSQLToolArguments,
    previous_sql_queries: dict[str, str],
    sql_context: SQLContext,
    question: BIRDQuestion,
    predicted_output_types: list[SchemaPredictions.OutputType] | None,
) -> tuple[str, ExecuteSQLToolResult]:
    exec_result_id = _get_unique_str(
        re.sub("[^A-Za-z0-9_]", "_", arguments.query_identifier).lower(),
        list(previous_sql_queries.keys())
        # Do not shadow table names in the DB
        + [t.name for t in sql_context.db_metadata[question.db_id].tables],
    )

    sql = "SELECT\n  "
    sql += ", ".join(f"{col.sql} AS {col.alias}" for col in arguments.columns)
    sql += f"\nFROM {arguments.from_}"

    for join in arguments.joins:
        join_type = "LEFT JOIN" if join.join_type == "left" else "INNER JOIN"
        sql += f"\n{join_type} {join.source}"
        if join.alias:
            sql += f" AS {join.alias}"
        sql += f" ON {join.on}"

    if arguments.where:
        sql += f"\nWHERE {arguments.where}"

    if arguments.group_by:
        sql += f"\nGROUP BY {', '.join(arguments.group_by)}"

    if arguments.having:
        sql += f"\nHAVING {arguments.having}"

    if arguments.order_by:
        sql += f"\nORDER BY {', '.join(arguments.order_by)}"

    if arguments.limit:
        sql += f"\nLIMIT {arguments.limit}"

    sg_query = sqlglot.parse_one(sql, dialect=sql_context.dialect).transform(_lint_sql)
    assert isinstance(sg_query, exp.Query), "Only SELECT statements are supported"
    for table in [arguments.from_] + [j.source for j in arguments.joins]:
        if previous_sql := previous_sql_queries.get(table):
            if sg_query.ctes:
                # Needs to be prepended, as existing CTEs may reference `cte_name`
                new_cte = exp.CTE(
                    alias=table,
                    this=sqlglot.parse_one(previous_sql, dialect=sql_context.dialect),
                )
                sg_query.ctes.insert(0, new_cte)
            else:
                sg_query = sg_query.with_(
                    alias=table, as_=previous_sql, dialect=sql_context.dialect
                )

    sql = sg_query.sql(dialect=sql_context.dialect)
    if arguments.query_identifier.startswith("final_answer"):
        # Do some sanity checks before executing the final query
        _check_rounding(question, sql)
        _check_mixed_division(sg_query)

    try:
        columns, rows = await execute_sql(sql, sql_context)

        # Ensure that the output types match the expected types
        if arguments.query_identifier.startswith("final_answer") and predicted_output_types:
            _check_output_types(columns, rows, predicted_output_types)
        tool_result = ExecuteSQLToolResult(
            exec_result_id=exec_result_id,
            rows=rows,
            columns=columns,
            sql=sql,
        )
        gpt_result = _get_gpt_result(
            rows=rows,
            columns=columns,
            exec_result_id=exec_result_id,
        )
    except Exception as exc:
        error_message = str(exc)
        tool_result = ExecuteSQLToolResult(
            sql=sql,
            exec_result_id=exec_result_id,
            error=error_message,
        )
        gpt_result = "Error executing query: " + error_message
    return gpt_result, tool_result


EXECUTE_SQL_ROWS_BYTE_LIMIT = 512


def _get_gpt_result(
    rows: list[list[SQLScalar]],
    columns: list[str],
    exec_result_id: str,
) -> str:
    tsv_preview = StringIO()
    writer = csv.writer(tsv_preview, delimiter="\t", lineterminator="\n")
    writer.writerow(columns)
    for i, row in enumerate(rows):
        if tsv_preview.tell() > EXECUTE_SQL_ROWS_BYTE_LIMIT:
            tsv_preview.write(f"```\n{len(rows) - i} more rows hidden")
            break
        writer.writerow([stringify(cell, quote_strings=False) for cell in row])

    return f"""exec_result_id: {exec_result_id}
row_count: {len(rows)}
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


def _check_output_types(
    columns: list[str],
    rows: list[list[SQLScalar]],
    predicted_output_types: list[SchemaPredictions.OutputType],
):
    if len(rows) == 0:
        return

    if len(columns) != len(predicted_output_types):
        raise RuntimeError(f"Expected {len(predicted_output_types)} columns, got {len(columns)}")

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
            raise RuntimeError(
                f"Expected output column type(s): {', '.join(output_type.type for output_type in predicted_output_types)}"
            )


# Find all table names used in column "scopes" for a given node
def _find_tables_used(
    node: exp.Expression, cte_table_refs: dict[str, set[str]], root_table_refs: set[str]
) -> set[str]:
    result = set()
    for column in node.find_all(exp.Column):
        result.update(cte_table_refs.get(column.table) or root_table_refs)
    return result
