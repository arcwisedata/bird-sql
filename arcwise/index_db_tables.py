from collections import defaultdict
from dataclasses import dataclass
import os
import sqlite3

from .typedefs import ForeignKey


@dataclass
class DatabaseColumn:
    name: str
    type: str
    foreign_keys: list[ForeignKey]


@dataclass
class DatabaseTable:
    db_id: str
    name: str
    columns: list[DatabaseColumn]
    primary_key: list[str]


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
                fk_info = cursor.fetchall()
                fk_list = defaultdict(list)
                seen_fks = set()
                for _id, _seq, table, from_col, to_col, _, _, _ in fk_info:
                    if from_col and to_col and (table, from_col, to_col) not in seen_fks:
                        seen_fks.add((table, from_col, to_col))
                        fk_list[from_col].append(
                            ForeignKey(
                                reference_table=table,
                                reference_column=to_col,
                                relationship="",  # This will be determined later
                            )
                        )

                # Get column info
                cursor.execute(f'PRAGMA table_info("{escaped_name}")')
                # cid, name, type, notnull, dflt_value, pk
                columns_info = cursor.fetchall()
                columns = []
                primary_key = []
                for _cid, col_name, col_type, _notnull, _dflt_value, is_pk in columns_info:
                    columns.append(
                        DatabaseColumn(
                            name=col_name,
                            type=col_type,
                            foreign_keys=fk_list[col_name],
                        )
                    )
                    if is_pk:
                        primary_key.append(col_name)

                db_tables.append(
                    DatabaseTable(
                        db_id=db_name, name=table_name, columns=columns, primary_key=primary_key
                    )
                )

    assert db_tables, f"No database tables found in {db_path}"
    return db_tables


if __name__ == "__main__":
    db_tables = index_db_tables("mock_dataset/databases")
    print(f"Total tables: {len(db_tables)}")
    for table in db_tables:
        print(table)
