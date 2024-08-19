import json
import os
import random
import sys

from transformers import AutoTokenizer

from arcwise.prompts import COLUMN_PREDICTION_PROMPT
from arcwise.utils import load_database_metadata

TOKEN_BUDGET = 20000

with open(sys.argv[1], "r") as f:
    annotated = json.load(f)

databases = load_database_metadata(sys.argv[2])

tokenizer = AutoTokenizer.from_pretrained("arcwise/bird-mistral-nemo")


def token_count(text: str) -> int:
    return len(tokenizer(text)["input_ids"])  # type: ignore


random.shuffle(annotated)

for x in annotated:
    if not (sql_refs := x.get("sql_refs")) or not (
        sql_refs_annotated_v2 := x.get("sql_refs_annotated_v2")
    ):
        continue

    schema_tables = databases[x["db_id"]].tables

    if len(sql_refs["output_schema"]) != len(sql_refs_annotated_v2["output_columns"]):
        print("Output schema mismatch:", x["question"], file=sys.stderr)
        continue

    output = "Output Types\n"
    for col, desc in zip(sql_refs["output_schema"], sql_refs_annotated_v2["output_columns"]):
        output += f"-- {desc}\n{col}\n"

    input_cols = {}
    for col, desc in sql_refs_annotated_v2["input_columns"].items():
        input_cols[col.replace("'", "").replace('"', "").replace("`", "")] = desc

    output += "Input Columns\n"
    for col in sql_refs["columns"]:
        if not (desc := input_cols.get(col)):
            raise Exception("Missing input column annotation " + col)
        output += f"-- {desc}\n{col}\n"
    output = output.strip()

    if "ERROR" in output:
        continue

    tokens = token_count(output)
    tokens += token_count(f"{x['question']} {x.get('evidence', '')}".strip())

    schema = ""
    if not os.environ.get("EVAL"):
        tables = set(sql_refs["tables"])
        chosen, others = [], []
        for table in schema_tables:
            random.shuffle(table.columns)
            if table.name in tables:
                ddl = table.format_for_column_prediction()
                tokens += token_count(ddl)
                chosen.append(ddl)
            else:
                others.append(table)

        if len(chosen) != len(tables):
            raise ValueError(
                f"{x['question']}: could not find all tables {len(chosen)} vs {len(tables)}"
            )

        random.shuffle(others)

        for table in others:
            ddl = table.format_for_column_prediction()
            count = token_count(ddl)
            if tokens + count > TOKEN_BUDGET - 512:
                continue
            tokens += count
            chosen.append(ddl)

        if len(chosen) < len(schema_tables):
            print(
                f"Chose {len(chosen)} / {len(schema_tables)} tables from {x['db_id']}",
                file=sys.stderr,
            )

        random.shuffle(chosen)
        schema = "\n".join(chosen)
    else:
        schema = "\n".join(table.format_for_column_prediction() for table in schema_tables)
        tokens += token_count(schema)

    output_json: dict = {
        "messages": COLUMN_PREDICTION_PROMPT
        + [
            {
                "role": "user",
                "content": f"""Given a database named {x['db_id']}:
<schema>
{schema}
</schema>
{x['question'].strip()}
{"Context: " + x['evidence'].strip() if x.get('evidence') else ""}""".strip(),
            },
            {"role": "assistant", "content": output},
        ]
    }

    output_json["tokens"] = tokens
    output_json["db_id"] = x["db_id"]
    if "question_id" in x:
        output_json["question_id"] = x["question_id"]

    print(json.dumps(output_json))
