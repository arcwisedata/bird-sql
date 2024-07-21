import click

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

    total = 0
    correct = 0
    hallucinations = 0
    output_match_sum = 0
    table_precision_sum = 0
    table_recall_sum = 0
    table_match_sum = 0
    column_precision_sum = 0
    column_recall_sum = 0
    column_match_sum = 0
    column_ratio = 0
    table_ratio = 0

    for question in questions:
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
            print(f"Failed to extract schema for {question.question_id}")
            continue

        total += 1
        golden_column = set(sql_refs.columns)
        golden_table = set(sql_refs.tables)
        all_columns = {
            f"{table.name}.{column.name}"
            for table in metadata[db_id].tables
            for column in table.columns
        }

        predicted_output_schema = [
            c.type for c in question.schema_predictions.output_types
        ]
        predicted_column = set(
            c.column for c in question.schema_predictions.input_columns
        )
        predicted_table = set(
            c.column.split(".")[0] for c in question.schema_predictions.input_columns
        )
        # lines = question.schema_predictions.raw_prediction.splitlines()
        # output_index = lines.index("Input Columns")
        # for line in lines[:output_index]:
        #     if not line.startswith("--"):
        #         predicted_output_schema.append(line)
        # for line in lines[output_index + 1 :]:
        #     if not line.startswith("--"):
        #         if line not in all_columns:
        #             print(f"Hallucinated column in {question.question_id}: {line}")
        #             hallucinations += 1
        #         else:
        #             predicted_table.add(line.split(".")[0])
        #             predicted_column.add(line)

        output_match = sql_refs.output_schema == predicted_output_schema
        output_match_sum += int(output_match)

        # Calculate table precision and recall
        correct += int(
            output_match
            and golden_table <= predicted_table
            and golden_column <= predicted_column
        )

        # Calculate common intersections
        table_intersection = len(predicted_table & golden_table)
        column_intersection = len(predicted_column & golden_column)

        # Calculate table precision and recall
        table_precision = (
            table_intersection / len(predicted_table) if predicted_table else 0.0
        )
        table_recall = table_intersection / len(golden_table)
        table_match_sum += golden_table <= predicted_table
        table_precision_sum += table_precision
        table_recall_sum += table_recall

        # # Calculate column precision and recall
        column_precision = (
            column_intersection / len(predicted_column) if predicted_column else 0.0
        )
        column_recall = column_intersection / len(golden_column)
        column_match_sum += golden_column <= predicted_column
        column_precision_sum += column_precision
        column_recall_sum += column_recall

        column_ratio += len(predicted_column) / len(golden_column)
        table_ratio += len(predicted_table) / len(golden_table)

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

    print(f"Hallucinations\t{hallucinations}")


if __name__ == "__main__":
    main()
