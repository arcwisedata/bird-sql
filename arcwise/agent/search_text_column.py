import asyncio
from pydantic import BaseModel, Field

from .utils import SQLContext, execute_process_json
from ..utils import stringify


class SearchTextColumnArguments(BaseModel):
    column: str = Field(description="Column name in table.column format. Do not add quoting")
    search_value: str


SEARCH_TEXT_COLUMN_TOOL = {
    "type": "function",
    "function": {
        "name": "search_text_column",
        "description": "Searches for similar values within a text column. Returns the closest values by similarity",
        "parameters": SearchTextColumnArguments.model_json_schema(),
    },
}


async def search_text_column_tool(
    args: SearchTextColumnArguments, context: SQLContext
) -> dict[str, float]:
    table, column = args.column.split(".", 1)
    search_value = args.search_value.replace("'", "''")
    try:
        response_json = await asyncio.wait_for(
            execute_process_json(
                [
                    "duckdb",
                    "-json",
                    "-readonly",
                    "-c",
                    f"SELECT value, jaro_winkler_similarity(value, '{search_value}') AS similarity "
                    f'FROM (SELECT DISTINCT CAST("{column}" AS STRING) AS value FROM "{table}") '
                    "ORDER BY 2 DESC LIMIT 5",
                    context.db_url,
                ]
            ),
            30.0,
        )
    except Exception:
        search_value = search_value.lower()
        length = len(search_value)
        response_json = await asyncio.wait_for(
            execute_process_json(
                [
                    "sqlite3",
                    "-json",
                    "-readonly",
                    context.db_url,
                    f"SELECT value, 1.0 * {length} / length(value) AS similarity "
                    f'FROM (SELECT DISTINCT CAST("{column}" AS TEXT) AS value FROM "{table}") '
                    f"WHERE LOWER(value) LIKE '%{search_value}%' LIMIT 10",
                ]
            ),
            30.0,
        )

    if not response_json:
        return {}

    return {
        stringify(row["value"], quote_strings=False): row["similarity"] or 0.0
        for row in response_json
    }
