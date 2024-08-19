import click
from tqdm import tqdm

from arcwise.utils import coro, load_database_metadata, load_questions
from arcwise.sql_references import extract_sql_references


@click.command()
@click.option("--predictions-file", help="Path to preditions", required=True)
@click.option("--metadata-file", help="Path to db metadata", required=True)
@click.option("--database-path", help="Path to databases", required=True)
@coro
async def main(
    predictions_file: str,
    metadata_file: str,
    database_path: str,
) -> None:
    questions = load_questions(predictions_file)
    metadata = load_database_metadata(metadata_file)

    stats = []
    for question in tqdm(questions):
        if (
            not question.SQL
            or not question.schema_predictions
            or not question.schema_predictions.raw_prediction
        ):
            continue

        db_id = question.db_id
        try:
            sql_refs = extract_sql_references(
                f"{database_path}/{db_id}/{db_id}.sqlite",
                metadata[db_id].tables,
                question.SQL,
            )
        except Exception:
            try:
                # Spider dataset queries have double-quoted strings
                sql_refs = extract_sql_references(
                    f"{database_path}/{db_id}/{db_id}.sqlite",
                    metadata[db_id].tables,
                    question.SQL.replace('"', "'"),
                )
            except Exception:
                print(f"Failed to extract schema for {question.db_id}: {question.SQL}")
                continue

        golden_column = set(sql_refs.columns)
        golden_table = set(sql_refs.tables)
        predicted_output_schema = [c.type for c in question.schema_predictions.output_types]
        if not predicted_output_schema:
            print(f"Warning: no output schema for {question.question}")

        predicted_column = set(c.column for c in question.schema_predictions.input_columns)
        predicted_table = set(
            c.column.split(".")[0] for c in question.schema_predictions.input_columns
        )
        # for table in metadata[db_id].tables:
        #     ref_tables = set()
        #     if table.name in predicted_table:
        #         for column in table.primary_key:
        #             predicted_column.add(f"{table.name}.{column}")
        # for table in metadata[db_id].tables:
        #     if table.name in ref_tables:
        #         for column in table.primary_key:
        #             predicted_column.add(f"{table.name}.{column}")
        # predicted_table.update(ref_tables)
        output_match = sql_refs.output_schema == predicted_output_schema
        # if not output_match:
        #     print(question.question)
        #     print("```")
        #     print(question.SQL)
        #     print("```")
        #     print(question.schema_predictions.raw_prediction)
        #     print("-----------")
        table_intersection = len(predicted_table & golden_table)
        column_intersection = len(predicted_column & golden_column)

        # Calculate table precision and recall
        stats.append(
            {
                "db_id": db_id,
                "all_correct": int(
                    output_match
                    and golden_table <= predicted_table
                    and golden_column <= predicted_column
                ),
                "output_correct": int(output_match),
                "table_precision": (
                    table_intersection / len(predicted_table) if predicted_table else 0.0
                ),
                "table_recall": table_intersection / len(golden_table),
                "table_correct": int(golden_table <= predicted_table),
                "column_precision": (
                    column_intersection / len(predicted_column) if predicted_column else 0.0
                ),
                "column_recall": column_intersection / len(golden_column)
                if len(golden_column)
                else 1.0,
                "column_correct": int(golden_column <= predicted_column),
                "column_ratio": len(predicted_column) / len(golden_column)
                if len(golden_column)
                else 1.0,
                "table_ratio": len(predicted_table) / len(golden_table),
            }
        )

    for db_id in [None, *metadata.keys()]:
        filtered_stats = (
            stats if db_id is None else [stat for stat in stats if stat["db_id"] == db_id]
        )
        total = len(filtered_stats)

        if total == 0:
            continue

        correct = sum(stat["all_correct"] for stat in filtered_stats)
        output_match_sum = sum(stat["output_correct"] for stat in filtered_stats)
        table_match_sum = sum(stat["table_correct"] for stat in filtered_stats)
        column_match_sum = sum(stat["column_correct"] for stat in filtered_stats)
        table_recall_sum = sum(stat["table_recall"] for stat in filtered_stats)
        column_recall_sum = sum(stat["column_recall"] for stat in filtered_stats)
        table_precision_sum = sum(stat["table_precision"] for stat in filtered_stats)
        column_precision_sum = sum(stat["column_precision"] for stat in filtered_stats)
        column_ratio = sum(stat["column_ratio"] for stat in filtered_stats)
        table_ratio = sum(stat["table_ratio"] for stat in filtered_stats)

        print(f"Database: {db_id or 'all databases'}:")
        print("Correct\tOutput OK\tTables OK\tColumns OK")
        print(
            f"{correct/total:.2%}\t{output_match_sum/total:.2%}\t{table_match_sum/total:.2%}\t{column_match_sum/total:.2%}"
        )

        print("Table recall\tColumn recall\tTable precision\tColumn precision")
        print(
            f"{table_recall_sum/total:.2%}\t{column_recall_sum/total:.2%}\t{table_precision_sum/total:.2%}\t{column_precision_sum/total:.2%}"
        )

        print("Column selection ratio\tTable selection ratio")
        print(f"{column_ratio/total:.2f}x\t{table_ratio/total:.2f}x")
        print("--------")


if __name__ == "__main__":
    main()
