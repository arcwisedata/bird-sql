from pydantic import BaseModel
import duckdb

from .utils import SQLContext
from ..utils import stringify


class SearchTextColumnArguments(BaseModel):
    table: str
    column: str
    search_value: str


SEARCH_TEXT_COLUMN_TOOL = {
    "type": "function",
    "function": {
        "name": "search_text_column",
        "description": "Searches for similar values within a text column. Returns the closest values by similarity",
        "parameters": SearchTextColumnArguments.model_json_schema(),
    },
}


def search_text_column_tool(args: SearchTextColumnArguments, context: SQLContext) -> str:
    with duckdb.connect(context.db_url, read_only=True) as conn:
        results = conn.execute(
            f'SELECT value, jaro_winkler_similarity(value, ?) FROM (SELECT DISTINCT CAST("{args.column}" AS STRING) AS value FROM "{args.table}") ORDER BY 2 DESC LIMIT 5',
            [args.search_value],
        ).fetchall()
        return "Closest values:\nvalue\tsimilarity\n" + "\n".join(
            f"{stringify(value, quote_strings=False)}\t{similarity:.4g}"
            for value, similarity in results
        )
