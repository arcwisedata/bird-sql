import asyncio
import json
from collections import defaultdict
from contextlib import aclosing
from dataclasses import dataclass
from functools import partial
from typing import Any

import click
import litellm
import numpy as np
from openai.types.chat import ChatCompletionMessageParam
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from .ddl import Table, get_table_ddl
from .typedefs import BIRDQuestion, Database, SchemaPredictions
from .utils import coro, load_database_metadata, load_questions
from vllm import LLM, SamplingParams

CONTEXT_TOKEN_LIMIT = 8000
RELEVANCY_THRESHOLD = 0.50
EMBED_BATCH_SIZE = 128


@dataclass
class DatabaseInfo:
    db: Database
    # Tokens/embeddings of each db.tables
    table_tokens: list[list[int]]
    table_embeddings: np.ndarray | None


@click.command()
@click.option("--questions-file", help="Path to questions JSON", required=True)
@click.option("--output-file", help="Path to output file", required=True)
@click.option("--metadata-file", help="Path to JSON metadata", required=True)
@click.option("--model", help="Model identifier", required=True)
@click.option("--embedding-model", help="Model identifier", required=True)
@coro
async def main(
    questions_file: str,
    output_file: str,
    metadata_file: str,
    model: str,
    embedding_model: str,
) -> None:
    questions = load_questions(questions_file)
    metadata = load_database_metadata(metadata_file)
    llm = LLM(
        model=model,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.95,
        max_num_batched_tokens=10240,
        max_model_len=10240,
    )
    database_info = {}
    tokenizer = AutoTokenizer.from_pretrained(model)
    for db_name, db in tqdm(metadata.items(), desc="Embedding schemas"):
        table_schemas = [_format_table(table) for table in db.tables]
        table_tokens: list[list[int]] = tokenizer(table_schemas, add_special_tokens=False)["input_ids"]  # type: ignore
        total_tokens = sum(len(tokens) for tokens in table_tokens)
        if total_tokens > CONTEXT_TOKEN_LIMIT:
            print(f"Note: {db_name} has long context length ({total_tokens})")
        database_info[db_name] = DatabaseInfo(
            db=db,
            table_tokens=table_tokens,
            table_embeddings=(
                await batch_embed(embedding_model, table_schemas)
                if total_tokens > CONTEXT_TOKEN_LIMIT
                else None
            ),
        )

    question_embeddings = await batch_embed(
        embedding_model, text=[q.question_evidence() for q in questions]
    )
    question_prompts = [
        _create_prompt(
            question,
            embedding,
            database_info[question.db_id],
            tokenizer,
        )
        for question, embedding in zip(questions, question_embeddings)
        if not question.schema_predictions
    ]
    prompt_token_ids: list[list[int]] = [
        # IMPORTANT: Strip Mistral's EOS token for assistant responses
        input_ids[:-1]  # type: ignore
        for input_ids in tokenizer.apply_chat_template(question_prompts)  # type: ignore
    ]
    outputs = llm.generate(
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(temperature=0, max_tokens=1000),
    )

    for question, output in zip(questions, outputs):
        completion = output.outputs[0].text
        question.schema_predictions = _process_prediction(
            question, database_info[question.db_id], completion
        )

    with open(output_file, "w") as f:
        json.dump([q.model_dump() for q in questions], f, indent=2)


def _format_table(table: Table) -> str:
    assert table.ai_description, "AI descriptions are required"
    schema = (
        "-- " + table.ai_description.replace("\n", "\n-- ") + "\n# Table: {table.name}"
    )

    for col in table.columns:
        assert col.ai_description, "AI descriptions are required"
        schema += "\n-- " + col.ai_description.replace("\n", "\n-- ")
        schema += f"\n{table.name}.{col.name}\t{col.type.upper()}"

    return schema


def _create_prompt(
    question: BIRDQuestion,
    question_embedding: np.ndarray,
    db_info: DatabaseInfo,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> list[ChatCompletionMessageParam]:
    db = db_info.db
    if (embeddings := db_info.table_embeddings) is not None:
        selected_tables = []
        selected_schemas = []
        token_usage = 0
        relevancy = np.dot(embeddings, question_embedding)
        for index, score in sorted(enumerate(relevancy), key=lambda x: -x[1]):
            if score < RELEVANCY_THRESHOLD and len(selected_tables) > 0:
                break
            selected_tables.append(db.tables[index])
            table_tokens: list[int] = db_info.table_tokens[index]
            if token_usage + len(table_tokens) >= CONTEXT_TOKEN_LIMIT:
                budget = max(CONTEXT_TOKEN_LIMIT - token_usage, 128)
                selected_schemas.append(tokenizer.decode(table_tokens[:budget]) + "...")
                break
            token_usage += len(table_tokens)
            selected_schemas.append(tokenizer.decode(table_tokens))
        schema = "\n".join(selected_schemas)
    else:
        selected_tables = db.tables
        schema = "\n".join(tokenizer.batch_decode(db_info.table_tokens))

    question.filtered_schema = "\n".join(
        get_table_ddl(table) for table in selected_tables
    )

    return SCHEMA_PREDICTION_PROMPT + [
        {
            "role": "user",
            "content": f"""Given the database:
<schema>
{schema}
</schema>
{question.question.strip()}
{'Context: ' + question.evidence if question.evidence else ''}""".strip(),
        },
        # Prefill response
        {"role": "assistant", "content": "Output Types\n"},
    ]


def _process_prediction(
    question: BIRDQuestion, db_info: DatabaseInfo, completion: str
) -> SchemaPredictions | None:
    valid_columns = set(
        f"{table.name}.{col.name}"
        for table in db_info.db.tables
        for col in table.columns
    )
    input_column_descriptions = defaultdict(list)
    lines = completion.splitlines()
    if "Input Columns" not in lines:
        print(f"Warning: malformed prediction for question: {question.question}")
        return

    if lines[0] == "Output Types":
        lines = lines[1:]

    input_columns_line = lines.index("Input Columns")
    output_types = _parse_output_types(lines[:input_columns_line])

    last_desc = None
    for line in lines[input_columns_line + 1 :]:
        if line.startswith("--"):
            last_desc = line[2:].strip()
        elif (col_name := line.strip()) and col_name in valid_columns:
            input_column_descriptions[col_name].append(last_desc)

    final_input_columns = [
        SchemaPredictions.InputColumn(
            column=col,
            description=descs[0],
        )
        for col, descs in input_column_descriptions.items()
    ]
    return SchemaPredictions(
        output_types=output_types,
        input_columns=final_input_columns,
        raw_prediction=completion,
    )


def _parse_output_types(lines: list[str]) -> list[SchemaPredictions.OutputType]:
    output_types = []
    last_desc = None
    for line in lines:
        if line.startswith("--"):
            last_desc = line[2:].strip()
        elif type_ := line.strip():
            if not last_desc or type_ not in {
                "real",
                "integer",
                "text",
                "date",
                "datetime",
            }:
                return []
            output_types.append(
                SchemaPredictions.OutputType(type=type_, description=last_desc)
            )
            last_desc = None
    return output_types


SCHEMA_PREDICTION_PROMPT: list[ChatCompletionMessageParam] = [
    {
        "role": "system",
        "content": """Given a SQL database and question, please determine a list of "Output Types" and "Input Columns" required to answer the question.
Before each list item, write a `--` line comment explaining why it is needed, citing the user's question when possible.
Output types should be one of: real, integer, text, date, datetime
Input columns should be formatted without quotes as: table_name.column_name
Ensure that the output types provide exactly the information needed to answer the question and nothing more or less.""",
    },
    {
        "role": "user",
        "content": """Given the database:
<schema>
-- Table containing sales info
# Table: sales
sales.sale_date\tDATE
sales.product_id\tINTEGER
sales.quantity\tREAL
-- Daily prices for each product
# Table: prices
prices.price_date\tDATE
prices.product_id\tINTEGER
prices.price\tREAL
-- Product-level information
# Table: products
products.product_id\tINTEGER
products.product_name\tTEXT
</schema>
What was the average price by product name in January 2024?""",
    },
    {
        "role": "assistant",
        "content": """Output Types
-- The 'product name' requested by the question
text
-- The average price 'in January 2024' for each product name
real
Input Columns
-- The question asks for the 'average price'. We should aggregate the price column in prices
prices.price
-- To filter for prices 'in January 2024', we should use the price_date column in prices
prices.price_date
-- We need to join prices with products by product_id to obtain the product name
prices.product_id
-- Join key for product_id in prices
products.product_id
-- Finally, we can group by the product_name column in products
products.product_name""",
    },
]


async def batch_embed(model: str, text: list[str]) -> np.ndarray:
    if len(text) > EMBED_BATCH_SIZE:
        return np.concatenate(
            await asyncio.gather(
                *[
                    batch_embed(model, text[i : i + EMBED_BATCH_SIZE])
                    for i in range(0, len(text), EMBED_BATCH_SIZE)
                ]
            )
        )

    embeddings = await litellm.aembedding(model=model, input=text)
    assert embeddings.data and len(embeddings.data) == len(
        text
    ), "Error getting embeddings"
    data = sorted(embeddings.data, key=lambda x: x["index"])
    return np.array([d["embedding"] for d in data])


if __name__ == "__main__":
    main()
