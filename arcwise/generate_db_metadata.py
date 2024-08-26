from dataclasses import dataclass
import re
import click
import json
import os
import pathlib
import sqlite3
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import aclosing
from functools import partial
from typing import Any

import pandas as pd
from fastapi.encoders import jsonable_encoder
from tqdm import tqdm

from .ai_describe_table import (
    generate_table_and_columns_ai_description,
    use_pregenerated_descriptions,
)
from .index_db_tables import index_db_tables, DatabaseTable
from .typedefs import ColumnInfo, Database, Table
from .utils import coro, run_with_concurrency

RELATIONSHIPS = {
    (True, True): "one-to-one",
    (True, False): "one-to-many",
    (False, True): "many-to-one",
    (False, False): "many-to-many",
}
TABLE_DESCRIPTION_CSV_COLUMNS = [
    "original_column_name",
    "column_name",
    "column_description",
    "data_format",
    "value_description",
]


@dataclass
class ColumnStatistics:
    row_count: int
    null_fraction: float
    distinct_count: int
    distinct_percent: float
    most_common_vals: list[Any] | None = None
    histogram: list[Any] | None = None


def get_column_stats(
    db_path: str,
    table: DatabaseTable,
) -> list[ColumnStatistics]:
    db_id = table.db_id
    sqlite_path = pathlib.Path(db_path) / f"{db_id}/{db_id}.sqlite"
    try:
        with sqlite3.connect(sqlite_path) as conn:
            cursor = conn.cursor()

            # Construct a single query to get all stats for all columns
            escaped_table = table.name.replace('"', '""')
            escaped_cols = [col.name.replace('"', '""') for col in table.columns]
            query = f"""
            SELECT
                COUNT(*) AS row_count,
                {', '.join([f'''
                SUM(CASE WHEN "{col}" IS NULL THEN 1 ELSE 0 END) AS null_count_{col_i},
                COUNT(DISTINCT "{col}") AS distinct_count_{col_i},
                (
                    SELECT JSON_GROUP_ARRAY(val)
                    FROM (
                        SELECT "{col}" AS val, COUNT(*)
                        FROM "{escaped_table}"
                        WHERE "{col}" IS NOT NULL
                        GROUP BY 1
                        ORDER BY 2 DESC
                        LIMIT 10
                    )
                ) AS most_common_{col_i},
                JSON_ARRAY(MIN("{col}"), MAX("{col}")) AS histogram_{col_i}
                ''' for col_i, col in enumerate(escaped_cols)])}
            FROM "{escaped_table}"
            """

            cursor.execute(query)
            result_row = cursor.fetchone()

        row_count = int(result_row[0])
        table_stats = []

        for i, _column in enumerate(table.columns):
            null_count, distinct_count, most_common, histogram = result_row[i * 4 + 1 : i * 4 + 5]
            null_count = int(null_count or 0)
            distinct_count = int(distinct_count or 0)
            null_fraction = null_count / row_count if row_count else 1.0
            distinct_percent = distinct_count / row_count if row_count else 0.0

            stats = ColumnStatistics(
                row_count=row_count,
                null_fraction=null_fraction,
                distinct_count=distinct_count,
                distinct_percent=distinct_percent,
                most_common_vals=json.loads(most_common),
                histogram=json.loads(histogram) if histogram != "[null,null]" else None,
            )
            table_stats.append(stats)

        return table_stats
    except Exception as err:
        print(f"Error reading {db_id}.{table.name}: {err}")
        raise


def get_cleaned_metadata(db_path: str) -> list[Database]:
    print("Listing tables...")
    tables = index_db_tables(db_path)
    print(f"Found {len(tables)} tables.")

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(get_column_stats, db_path, table) for table in tables]
        table_column_stats = [f.result() for f in tqdm(futures, desc="Getting column stats")]

    all_columns: dict[tuple[str, str, str], tuple[ColumnInfo, ColumnStatistics, str]] = {}
    tables_by_db_id: dict[str, list[Table]] = defaultdict(list)
    for table, column_stats in zip(tables, table_column_stats):
        db_id = table.db_id

        if (descriptions := read_table_description(db_path, db_id, table.name)) is not None:
            description_rows = list(descriptions.itertuples(index=False))
            description_by_name = {row[0].strip().lower(): row for row in description_rows}
        else:
            description_by_name = {}

        output_columns = []
        for column, stats in zip(table.columns, column_stats):
            col_description = description_by_name.get(column.name.lower())
            orig_col_name = None
            description = None
            value_description = None
            if col_description is None:
                print(f"Warning: missing column description for {db_id}.{table.name}.{column.name}")
            else:
                if len(col_description) < 5:
                    print(
                        f"Warning: malformed column description for {db_id}.{table.name}.{column.name}"
                    )
                else:
                    expected_cols, additional_cols = (
                        col_description[:5],
                        col_description[5:],
                    )
                    (
                        _,
                        orig_col_name,
                        description,
                        _type,
                        value_description,
                    ) = expected_cols
                    additional_info = [
                        str_i
                        for i in additional_cols
                        if (str_i := str(i)) and not isinstance(i, int)
                    ]
                    additional_info = (
                        (
                            "Some included information which may or may not be relevant:\n"
                            + "\n".join(additional_info)
                        )
                        if additional_info
                        else ""
                    )
                    if value_description and additional_info:
                        value_description += "\n" + additional_info
                    elif additional_info:
                        value_description = additional_info

            merged_values = (stats.histogram or []) + (stats.most_common_vals or [])
            try:
                min_value = min(merged_values, default=None)
                max_value = max(merged_values, default=None)
            except Exception:
                min_value = min([str(x) for x in merged_values], default=None)
                max_value = max([str(x) for x in merged_values], default=None)

            column_info = ColumnInfo(
                name=column.name,
                original_name=orig_col_name.strip() if orig_col_name else None,
                type=column.type,
                description=description.strip() if description else None,
                value_description=value_description.strip() if value_description else None,
                null_fraction=stats.null_fraction,
                unique_count=int(stats.distinct_count),
                unique_fraction=stats.distinct_percent,
                sample_values=(stats.most_common_vals or stats.histogram or [])[:10],
                min_value=min_value,
                max_value=max_value,
            )
            output_columns.append(column_info)
            all_columns[(db_id, table.name.lower(), column.name.lower())] = (
                column_info,
                stats,
                table.name,
            )

        tables_by_db_id[table.db_id].append(
            Table(
                name=table.name,
                row_count=column_stats[0].row_count,
                primary_key=table.primary_key,
                foreign_keys=table.foreign_keys,
                columns=output_columns,
            )
        )

    # Determine foreign key relationships
    databases_list = []
    for db_id, tables in tables_by_db_id.items():
        databases_list.append(Database(name=db_id, tables=tables))
        for table in tables:
            for fkey in table.foreign_keys:
                if len(fkey.columns) > 1 or fkey.relationship:
                    continue

                column = fkey.columns[0]
                ref_column = fkey.reference_columns[0]
                try:
                    _, from_col_stats, _ = all_columns[(db_id, table.name.lower(), column.lower())]
                    to_col, to_col_stats, to_col_table = all_columns[
                        (db_id, fkey.reference_table.lower(), ref_column.lower())
                    ]
                except Exception:
                    print(f"Warning: invalid foreign key {fkey}")
                    continue

                from_unique = approx_eq(
                    from_col_stats.distinct_percent + from_col_stats.null_fraction, 1
                )
                to_unique = approx_eq(to_col_stats.distinct_percent + to_col_stats.null_fraction, 1)
                # names may have been lowercased
                fkey.reference_table = to_col_table
                fkey.reference_columns[0] = to_col.name
                fkey.relationship = RELATIONSHIPS[(from_unique, to_unique)]

    return databases_list


def approx_eq(a, b, tol=1e-6):
    return abs(a - b) < tol


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "_", name.lower())


def read_table_description(db_path: str, db_id: str, table_name: str) -> pd.DataFrame | None:
    dir = pathlib.Path(db_path) / f"{db_id}/database_description"
    if not dir.exists():
        print(f"Warning: {db_id} has no descriptions")
        return None

    full_path = None
    normalized_table_name = _normalize_name(table_name)
    for csv_path in dir.glob("*.csv"):
        if _normalize_name(csv_path.stem) == normalized_table_name:
            full_path = csv_path
            break
    if full_path is None:
        print(f"Warning: No matching description file for {db_id}.{table_name}")
        return None

    descriptions = None
    for encoding in ["utf-8", "latin1", "ISO-8859-1", "cp1252"]:
        try:
            descriptions = pd.read_csv(full_path, encoding=encoding)
            break
        except Exception:
            pass
    if descriptions is None:
        print(f"Warning: could not read description file {full_path}")
        return None

    extra_columns = list(set(descriptions.columns) - set(TABLE_DESCRIPTION_CSV_COLUMNS))
    descriptions = descriptions[TABLE_DESCRIPTION_CSV_COLUMNS + extra_columns]
    descriptions.fillna("", inplace=True)
    return descriptions


# CLI params
@click.command()
@click.option("--db-path", help="Path to input directory with sqlite dbs", required=True)
@click.option("--description-file", help="Path to JSON file with column descriptions")
@click.option(
    "--output-file",
    help="Filepath where output metadata JSON file will be saved",
    required=True,
)
@click.option("--model", default="gpt-4o", help="LLM to use for AI descriptions")
@click.option("--concurrency", default=3, help="Number of tables to evaluate concurrently")
@click.option(
    "--ai-only",
    is_flag=True,
    default=False,
    help="Re-generates AI descriptions from a previous metadata file",
)
@coro
async def main(
    db_path: str,
    description_file: str | None,
    output_file: str,
    model: str,
    concurrency: int,
    ai_only: bool,
):
    if ai_only:
        try:
            with open(output_file, "r") as f:
                output_databases = [Database.model_validate(item) for item in json.load(f)]
        except Exception:
            print(f"--ai-only requires an existing output file ({output_file})")
    else:
        output_databases = get_cleaned_metadata(db_path)
        # Save initial metadata (without AI descriptions)
        if dirname := os.path.dirname(output_file):
            os.makedirs(dirname, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(json.dumps(jsonable_encoder(output_databases), indent=2))

    if description_file:
        print(f"Using pre-generated descriptions: {description_file}")
        with open(description_file, "r") as f:
            column_descriptions = json.load(f)
            assert isinstance(column_descriptions, dict)
        for database in output_databases:
            use_pregenerated_descriptions(database, column_descriptions)
    else:
        # Generate AI descriptions
        callables = [
            partial(generate_table_and_columns_ai_description, table, model)
            for database in output_databases
            for table in database.tables
            if ai_only or not table.ai_description  # --ai-only force-regenerates
        ]
        async with aclosing(run_with_concurrency(callables, concurrency)) as results:
            with tqdm(total=len(callables), desc="Generating AI descriptions") as pbar:
                async for _ in results:
                    pbar.update(1)

    with open(output_file, "w") as f:
        f.write(json.dumps(jsonable_encoder(output_databases), indent=2))


if __name__ == "__main__":
    main()
