from collections import defaultdict
from dataclasses import dataclass
import os
import sqlite3

from .typedefs import ForeignKey


@dataclass
class DatabaseColumn:
    name: str
    type: str


@dataclass
class DatabaseTable:
    db_id: str
    name: str
    columns: list[DatabaseColumn]
    primary_key: list[str]
    foreign_keys: list[ForeignKey]


COLUMN_TYPE_MAP = {
    "char": "text",
    "character": "text",
    "varchar": "text",
    "nchar": "text",
    "nvarchar": "text",
    "decimal": "real",
    "double precision": "real",
    "double": "real",
    "float": "real",
    "number": "real",
    "numeric": "real",
    "int": "integer",
    "bigint": "integer",
    "smallint": "integer",
    "mediumint": "integer",
    "tinyint": "integer",
}


def _map_column_type(col_type: str) -> str:
    col_type = col_type.lower()
    # Strip parens
    if "(" in col_type:
        col_type = col_type[: col_type.index("(")]
    return COLUMN_TYPE_MAP.get(col_type, col_type)


def index_db_tables(db_path: str) -> list[DatabaseTable]:
    db_tables = []
    for db_name in os.listdir(db_path):
        db_file = os.path.join(db_path, db_name, db_name + ".sqlite")
        if not os.path.exists(db_file):
            continue

        with sqlite3.connect(db_file) as conn:
            cursor = conn.cursor()

            # Get all table names
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            table_names = cursor.fetchall()

            for table_name in table_names:
                table_name = table_name[0]
                escaped_name = table_name.replace('"', '""')

                # Get foreign key information
                cursor.execute(f'PRAGMA foreign_key_list("{escaped_name}")')
                # id, seq, table, from, to, on_update, on_delete, match
                fk_info = sorted(cursor.fetchall())
                fk_by_id: dict[int, list[tuple[str, str, str]]] = defaultdict(list)
                for id, _seq, reference_table, from_col, to_col, _, _, _ in fk_info:
                    reference_table_name: str | None = next(
                        (
                            name
                            for row in table_names
                            if (name := row[0]) and name.lower() == reference_table.lower()
                        ),
                        None,
                    )
                    if reference_table_name is None:
                        continue
                    if from_col and to_col:
                        fk_by_id[id].append((reference_table_name, from_col, to_col))

                # Get column info
                cursor.execute(f'PRAGMA table_info("{escaped_name}")')
                # cid, name, type, notnull, dflt_value, pk
                columns_info = cursor.fetchall()
                columns = []
                primary_key = []
                for _cid, col_name, col_type, _notnull, _dflt_value, is_pk in columns_info:
                    columns.append(DatabaseColumn(name=col_name, type=_map_column_type(col_type)))
                    if is_pk:
                        primary_key.append(col_name)

                db_tables.append(
                    DatabaseTable(
                        db_id=db_name,
                        name=table_name,
                        columns=columns,
                        primary_key=primary_key,
                        foreign_keys=[
                            ForeignKey(
                                columns=[from_col for _, from_col, _ in fks],
                                reference_table=fks[0][0],
                                reference_columns=[to_col for _, _, to_col in fks],
                                relationship=None,
                            )
                            for fks in fk_by_id.values()
                        ],
                    )
                )

    assert db_tables, f"No database tables found in {db_path}"
    return db_tables


if __name__ == "__main__":
    db_tables = index_db_tables("mock_dataset/databases")
    print(f"Total tables: {len(db_tables)}")
    for table in db_tables:
        print(table)
