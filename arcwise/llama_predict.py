import asyncio
import json
from collections import defaultdict
from contextlib import aclosing
from dataclasses import dataclass
from functools import partial

import click
import litellm
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from .ddl import get_database_ddl, get_table_ddl
from .typedefs import BIRDQuestion, Database, LlamaPredictions
from .utils import coro, load_database_metadata, load_questions, run_with_concurrency

CONTEXT_TOKEN_LIMIT = 8000
RELEVANCY_THRESHOLD = 0.50
EMBED_BATCH_SIZE = 100


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
@click.option("--embedding-model", help="Model identifier", default="openai/voyage-code-2")
@click.option("--concurrency", default=2, help="Number of questions to evaluate concurrently")
@coro
async def main(
    questions_file: str,
    output_file: str,
    metadata_file: str,
    model: str,
    embedding_model: str,
    concurrency: int,
) -> None:
    questions = load_questions(questions_file)
    metadata = load_database_metadata(metadata_file)

    database_info = {}
    tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-bnb-4bit")
    for db_name, db in tqdm(metadata.items(), desc="Embedding schemas"):
        table_schemas = [get_table_ddl(table) for table in db.tables]
        table_tokens: list[list[int]] = tokenizer(table_schemas)["input_ids"]  # type: ignore
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
    callables = [
        partial(
            process_question, question, embedding, database_info[question.db_id], model, tokenizer
        )
        for question, embedding in zip(questions, question_embeddings)
        if not question.llama_predictions
    ]

    def _write_output() -> None:
        with open(output_file, "w") as f:
            json.dump([q.model_dump(exclude_none=True) for q in questions], f, indent=2)

    async with aclosing(run_with_concurrency(callables, concurrency)) as results:
        with tqdm(total=len(callables)) as pbar:
            async for _ in results:
                pbar.update(1)
                if pbar.n % 10 == 0:
                    _write_output()

    _write_output()


async def process_question(
    question: BIRDQuestion,
    question_embedding: np.ndarray,
    db_info: DatabaseInfo,
    model: str,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> None:
    db = db_info.db
    schema = get_database_ddl(db)
    tables = db.tables
    if (embeddings := db_info.table_embeddings) is not None:
        selected_tables = []
        selected_schemas = []
        token_usage = 0
        relevancy = np.dot(embeddings, question_embedding)
        for index, score in sorted(enumerate(relevancy), key=lambda x: -x[1]):
            if score < RELEVANCY_THRESHOLD:
                break
            selected_tables.append(tables[index])
            table_tokens: list[int] = db_info.table_tokens[index]
            if token_usage + len(table_tokens) >= CONTEXT_TOKEN_LIMIT:
                budget = max(CONTEXT_TOKEN_LIMIT - token_usage, 128)
                selected_schemas.append(
                    tokenizer.decode(table_tokens[:budget], skip_special_tokens=True) + "..."
                )
                break
            token_usage += len(table_tokens)
            selected_schemas.append(tokenizer.decode(table_tokens, skip_special_tokens=True))
        tables = selected_tables
        schema = "\n".join(selected_schemas)

    question.filtered_schema = schema
    valid_columns = set(f"{col.name}::{table.name}" for table in tables for col in table.columns)

    try:
        response = await litellm.acompletion(
            model=model,
            messages=SCHEMA_PREDICTION_PROMPT
            + [
                {
                    "role": "user",
                    "content": f"""Given the database:
<schema>
{question.filtered_schema}
</schema>
{question.question.strip()}
{'Context: ' + question.evidence if question.evidence else ''}""".strip(),
                },
            ],
            n=1,
            temperature=0.0,
            custom_llm_provider="openai",
            timeout=600.0,  # Allow for cold start time
            num_retries=5,
        )
        input_column_descriptions = defaultdict(list)
        output_types = []
        for choice in response.choices:  # type: ignore
            lines = choice.message.content.strip().splitlines()  # type: ignore
            if "Input Columns" not in lines:
                continue

            input_columns_line = lines.index("Input Columns")
            output_types = _parse_output_types(lines[1:input_columns_line])

            last_desc = None
            for line in lines[input_columns_line + 1 :]:
                if line.startswith("--"):
                    last_desc = line[2:].strip()
                elif (col_name := line.strip()) and col_name in valid_columns:
                    col, table = line.split("::")
                    input_column_descriptions[table + "." + col].append(last_desc)

        # (Uncomment if using multiple samples)
        # majority_output = None
        # for outputs in output_types_by_shape.values():
        #     if len(outputs) > N_SAMPLES / 2:
        #         majority_output = random.choice(outputs)

        final_input_columns = [
            LlamaPredictions.InputColumn(
                column=col,
                description=descs[0],
            )
            for col, descs in input_column_descriptions.items()
        ]

        question.llama_predictions = LlamaPredictions(
            output_types=output_types,
            input_columns=final_input_columns,
        )
    except Exception as err:
        print(
            "Error getting prediction for question:",
            question.model_dump_json(include={"db_id", "question"}),
        )
        print(f"Exception message: {err}")


def _parse_output_types(lines: list[str]) -> list[LlamaPredictions.OutputType]:
    output_types = []
    last_desc = None
    for line in lines:
        if line.startswith("--"):
            last_desc = line[2:].strip()
        elif type_ := line.strip():
            if not last_desc or type_ not in {"real", "integer", "text", "date", "datetime"}:
                return []
            output_types.append(LlamaPredictions.OutputType(type=type_, description=last_desc))
            last_desc = None
    return output_types


SCHEMA_PREDICTION_PROMPT = [
    {
        "role": "system",
        "content": """Given a SQL database and question, please determine a list of "Output Types" and "Input Columns" required to answer the question.
Before each list item, write a `--` line comment explaining why it is needed, citing the user's question when possible.
Output types should be one of: real, integer, text, date
Input columns should be formatted without quotes as: column_name::table_name
Ensure that the output types provide exactly the information needed to answer the question and nothing more or less.""",
    },
    {
        "role": "user",
        "content": """Given the database:
<schema>
CREATE TABLE sales (
sale_date date,
product_id integer,
quantity real
);
CREATE TABLE prices (
price_date date,
product_id integer,
price real
);
CREATE TABLE products (
product_id integer,
product_name text
);
</schema>
What was the average price by product name in January 2024?""",
    },
    {
        "role": "assistant",
        "content": """Output Types
-- The product name for the average price
text
-- 'average price of carrots in January 2024' for the product name
real
Input Columns
-- The question asks for the 'average price'. We should aggregate the price column in prices
price::prices
-- To filter for prices 'in January 2024', we should use the price_date column in prices
price_date::prices
-- To filter for prices 'of carrots', we need to use the product_id column to join against products
product_id::prices
-- Join key for product_id in prices
product_id::products
-- Finally, we can group by the product_name column in products
product_name::products""",
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
    assert embeddings.data and len(embeddings.data) == len(text), "Error getting embeddings"
    data = sorted(embeddings.data, key=lambda x: x["index"])
    return np.array([d["embedding"] for d in data])


if __name__ == "__main__":
    main()
