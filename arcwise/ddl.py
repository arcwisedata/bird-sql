from functools import cache
from .typedefs import Database, ForeignKey, Table, ColumnInfo
import sqlglot
import sqlglot.expressions as exp


SQLITE_DIALECT = sqlglot.Dialect.get_or_raise("sqlite")
MAX_DESC_LEN = 512


def get_database_ddl(db: Database) -> str:
    return "\n".join(get_table_ddl(table) for table in db.tables)


# Returns a CREATE TABLE statement.
def get_table_ddl(table: Table) -> str:
    schema = ""
    if table.ai_description:
        schema = "-- " + table.ai_description[:MAX_DESC_LEN].replace("\n", "\n-- ") + "\n"
    schema += f"CREATE TABLE {quote_identifier(table.name)} (\n"
    for idx, column in enumerate(table.columns):
        if idx:
            schema += ",\n"
        # Description will be added as a comment above the column line
        if column.ai_description:
            schema += "-- " + column.ai_description[:MAX_DESC_LEN].replace("\n", "\n-- ") + "\n"
        schema += quote_identifier(column.name) + " " + column.type.upper()

    if table.primary_key:
        schema += (
            ",\nPRIMARY KEY (" + ", ".join(quote_identifier(col) for col in table.primary_key) + ")"
        )

    for fkey in table.foreign_keys:
        schema += (
            f",\nFOREIGN KEY ({', '.join(quote_identifier(col) for col in fkey.columns)})"
            f" REFERENCES {quote_identifier(fkey.reference_table)}"
            f" ({', '.join(quote_identifier(col) for col in fkey.reference_columns)})"
        )
        if fkey.relationship:
            schema += " -- " + fkey.relationship

    return schema + "\n);"


@cache
def get_sqlite_keywords() -> set[str]:
    # SQLGlot treats "ORDER BY", "GROUP BY", etc as a single keyword
    return {
        part.upper() for keyword in SQLITE_DIALECT.tokenizer.KEYWORDS for part in keyword.split()
    }


def quote_identifier(name: str) -> str:
    if exp.SAFE_IDENTIFIER_RE.match(name) and name.upper() not in get_sqlite_keywords():
        return name
    return f"`{name}`"


if __name__ == "__main__":
    table = Table(
        name="test table",
        ai_description="test\ndescription",
        row_count=123,
        primary_key=["id"],
        foreign_keys=[
            ForeignKey(
                columns=["id2"],
                reference_table="other_table",
                reference_columns=["id"],
                relationship="one-to-one",
            )
        ],
        columns=[
            ColumnInfo(
                name="test_column",
                type="text",
                ai_description="col description",
                null_fraction=0.0,
                unique_count=123,
                unique_fraction=1.0,
                sample_values=[],
                min_value=0,
                max_value=1,
            )
        ],
    )
    print(get_table_ddl(table))
