from collections import defaultdict
import json
import random

import click
import numpy as np
from openai.types.chat import ChatCompletionMessageParam
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from .ddl import get_table_ddl
from .embedding import batch_embed
from .prompts import COLUMN_PREDICTION_TYPES, COLUMN_PREDICTION_PROMPT
from .typedefs import BIRDQuestion, Database, SchemaPredictions
from .utils import coro, load_database_metadata, load_questions
from vllm import LLM, RequestOutput, SamplingParams

NUM_VOTES = 7
MIN_VOTES = 1
OUTPUT_TOKENS = 1500


@click.command()
@click.option("--questions-file", help="Path to questions JSON", required=True)
@click.option("--output-file", help="Path to output file", required=True)
@click.option("--metadata-file", help="Path to JSON metadata", required=True)
@click.option("--model", help="Model identifier", required=True)
@click.option("--max-model-len", help="Model context length", default=20_000)
@click.option("--embedding-model", help="Model identifier", required=True)
@coro
async def main(
    questions_file: str,
    output_file: str,
    metadata_file: str,
    model: str,
    max_model_len: int,
    embedding_model: str,
) -> None:
    questions = load_questions(questions_file)
    metadata = load_database_metadata(metadata_file)
    llm = LLM(
        model=model,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.95,
        max_num_batched_tokens=max_model_len,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model)
    context_token_limit = max_model_len - OUTPUT_TOKENS
    for db_name, db in metadata.items():
        db_questions = [q for q in questions if q.db_id == db_name]
        if not db_questions:
            continue
        print(f"Processing database {db_name} ({len(db_questions)} questions)")

        table_schemas = [table.format_for_column_prediction() for table in db.tables]
        table_tokens: list[list[int]] = tokenizer(table_schemas, add_special_tokens=False)[
            "input_ids"
        ]  # type: ignore
        total_tokens = sum(len(tokens) for tokens in table_tokens)

        table_embeddings = None
        question_embeddings = [None] * len(db_questions)
        if total_tokens > context_token_limit:
            print(f"Note: {db_name} has long context length ({total_tokens})")
            table_embeddings = await batch_embed(embedding_model, table_schemas)
            question_embeddings = await batch_embed(
                embedding_model, text=[q.question_evidence() for q in db_questions]
            )

        question_prompts = [
            _create_prompt(
                question,
                question_embedding,
                db,
                table_tokens,
                table_embeddings,
                tokenizer,
                context_token_limit,
            )
            for question, question_embedding in zip(db_questions, question_embeddings)
            if not question.schema_predictions
        ]
        prompt_token_ids: list[list[int]] = tokenizer.apply_chat_template(  # type: ignore
            question_prompts,  # type: ignore
            add_generation_prompt=True,
        )

        outputs = llm.generate(
            prompt_token_ids=prompt_token_ids,
            sampling_params=SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=OUTPUT_TOKENS,
                n=NUM_VOTES,
            ),
        )
        for question, output in zip(db_questions, outputs):
            question.schema_predictions = _process_prediction(question, db, output)

        with open(output_file, "w") as f:
            json.dump([q.model_dump() for q in questions], f, indent=2)


def _create_prompt(
    question: BIRDQuestion,
    question_embedding: np.ndarray | None,
    db: Database,
    table_tokens: list[list[int]],
    table_embeddings: np.ndarray | None,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    token_limit: int,
) -> list[ChatCompletionMessageParam]:
    if question_embedding is not None and table_embeddings is not None:
        selected_tables = []
        selected_schemas = []
        token_usage = 0
        relevancy = np.dot(table_embeddings, question_embedding)
        for index, score in sorted(enumerate(relevancy), key=lambda x: -x[1]):
            selected_tables.append(db.tables[index])
            tokens_to_add: list[int] = table_tokens[index]
            if token_usage + len(tokens_to_add) > token_limit:
                budget = max(token_limit - token_usage, 128)
                selected_schemas.append(tokenizer.decode(tokens_to_add[:budget]) + "...")
                break
            token_usage += len(tokens_to_add)
            selected_schemas.append(tokenizer.decode(tokens_to_add))
        schema = "\n".join(selected_schemas)
    else:
        selected_tables = db.tables
        schema = "\n".join(tokenizer.batch_decode(table_tokens))

    question.filtered_schema = "\n".join(get_table_ddl(table) for table in selected_tables)

    return COLUMN_PREDICTION_PROMPT + [
        {
            "role": "user",
            "content": f"""Given a database named {db.name}:
<schema>
{schema}
</schema>
{question.question_evidence()}""".strip(),
        },
    ]


def _process_prediction(
    question: BIRDQuestion, db: Database, output: RequestOutput
) -> SchemaPredictions | None:
    input_column_votes = defaultdict(list)
    output_type_votes = defaultdict(list)
    for completion in output.outputs:
        lines = completion.text.splitlines()
        if "Input Columns" not in lines:
            print(f"Warning: malformed prediction for question: {question.question}")
            return

        if lines[0].strip() == "Output Types":
            lines = lines[1:]

        input_columns_line = lines.index("Input Columns")
        output_types = _parse_output_types(lines[:input_columns_line])
        if output_types:
            output_type_votes[tuple(x.type for x in output_types)].append(output_types)

        last_desc = None
        all_columns = {
            f"{table.name}.{column.name}" for table in db.tables for column in table.columns
        }
        for line in lines[input_columns_line + 1 :]:
            if line.startswith("--"):
                last_desc = line[2:].strip()
            elif (col_name := line.strip()) and col_name in all_columns:
                input_column_votes[col_name].append(last_desc)

    final_output_votes = max(output_type_votes.values(), key=len, default=[])
    final_input_columns = [
        SchemaPredictions.InputColumn(
            column=col, description=random.choice(votes), votes=len(votes)
        )
        for col, votes in input_column_votes.items()
        if len(votes) >= MIN_VOTES
    ]
    final_input_columns.sort(key=lambda x: -(x.votes or 0))
    return SchemaPredictions(
        output_types=(
            # If we fail to reach agreement, let the agent decide
            final_output_votes[0] if len(final_output_votes) >= MIN_VOTES else []
        ),
        input_columns=final_input_columns,
        raw_prediction=output.outputs[0].text,
    )


# (This turned out to be slow and not that helpful.)
# def _guided_output_regex(db: Database) -> str:
#     # Create a union of all table/column combinations
#     # Example: (table1\.(col1|col2)|table2\.(col1|col2))...
#     table_patterns = []
#     for table in db.tables:
#         table_pattern = re.escape(table.name + ".")
#         column_pattern = "|".join(re.escape(col.name) for col in table.columns)
#         table_patterns.append(f"{table_pattern}({column_pattern})")
#     column_name_regex = "|".join(table_patterns)
#     column_type_regex = "|".join(COLUMN_TYPES)
#     return (
#         f"Output Types(\n-- [^\n]+\n({column_type_regex}))+\n"
#         f"Input Columns(\n-- [^\n]+\n({column_name_regex}))+"
#     )


def _parse_output_types(lines: list[str]) -> list[SchemaPredictions.OutputType]:
    output_types = []
    last_desc = None
    for line in lines:
        if line.startswith("--"):
            last_desc = line[2:].strip()
        elif type_ := line.strip():
            if not last_desc or type_ not in COLUMN_PREDICTION_TYPES:
                return []  # Void out bad guesses
            output_types.append(SchemaPredictions.OutputType(type=type_, description=last_desc))
    return output_types


if __name__ == "__main__":
    main()
