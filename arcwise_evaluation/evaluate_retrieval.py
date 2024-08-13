import json
import math
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import shared_memory

import click
import numpy as np
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer

from arcwise.ddl import get_table_ddl
from arcwise.embedding import batch_embed
from arcwise.sql_references import extract_sql_references
from arcwise.typedefs import Database, Table
from arcwise.utils import (
    BIRDQuestion,
    coro,
    load_database_metadata,
    load_questions,
)

load_dotenv()

MODEL_NAMES = {
    "gemini": "",
    "bedrock": "",
    "openai_large": "text-embedding-3-large",
    "openai_small": "text-embedding-3-small",
    "voyage": "voyage/voyage-code-2",
}

MODEL_LIMITS = {
    "gemini": 8000,
    "bedrock": 8000,
    "openai_large": 20000,
    "openai_small": 20000,
    "voyage": 40000,
}


def _get_document(table: Table, method: str | None) -> str:
    if method == "description":
        assert table.ai_description
        return table.ai_description
    return get_table_ddl(table)


@click.command()
@click.option("--questions-file", help="Path to questions JSON", required=True)
@click.option("--metadata-file", help="Path to JSON metadata", required=True)
@click.option("--persist-dir", help="Persistence path", default="retrieval_storage")
@click.option("--model", help="Model identifier", required=True)
@click.option("--method", help="Embedding method")
@click.option("--context-limit", help="Simulated context limit", default=8000)
@coro
async def main(
    questions_file: str,
    metadata_file: str,
    persist_dir: str,
    model: str,
    method: str | None,
    context_limit: int,
) -> None:
    metadata = load_database_metadata(metadata_file)
    questions = load_questions(questions_file)
    documents = {
        f"{db.name}.{table.name}": _get_document(table, method)
        for db in metadata.values()
        for table in db.tables
    }
    queries = {_question_key(model, q): q.question_evidence() for q in questions}

    docs_cache_path = persist_dir + "/" + f"{model}_docs.pkl"
    query_cache_path = persist_dir + "/" + f"{model}_queries.pkl"
    print("Getting document embeddings", file=sys.stderr)
    doc_embeddings = await _get_embeddings(documents, docs_cache_path, model)
    print("Getting question embeddings", file=sys.stderr)
    question_embeddings = await _get_embeddings(queries, query_cache_path, model)
    print("Calculating token counts", file=sys.stderr)
    table_token_counts = _calculate_token_counts(persist_dir, metadata)

    shm_name = "shared_memory"
    embedding_size = len(next(iter(doc_embeddings.values())))
    d_size = np.dtype(np.float64).itemsize * len(doc_embeddings) * embedding_size
    shm = shared_memory.SharedMemory(create=True, size=d_size, name=shm_name)
    doc_embeddings_np = np.ndarray(
        shape=(len(doc_embeddings), embedding_size), dtype=np.float64, buffer=shm.buf
    )
    for i, embedding in enumerate(doc_embeddings.values()):
        doc_embeddings_np[i] = embedding

    with ProcessPoolExecutor() as executor:
        print("Running evaluation", file=sys.stderr)
        futures = [
            executor.submit(
                _process_question,
                question,
                metadata[question.db_id].tables,
                question_embeddings[_question_key(model, question)],
                list(doc_embeddings.keys()),
                doc_embeddings_np.shape,
                shm_name,
                table_token_counts,
                context_limit,
            )
            for question in questions
            if question.db_id in ["works_cycles", "hockey"]
        ]
        print("db_id\tquestion_id\tquestion\tsql\trecall\tcorrectness\tndcg\tnotes")
        for future in tqdm(futures):
            if line := future.result():
                print(line)


@dataclass
class EvalResult:
    recall: float
    correctness: float
    ndcg: float
    notes: str


def _process_question(
    question: BIRDQuestion,
    tables: list[Table],
    question_embedding: list[float],
    document_ids: list[str],
    documents_shape: tuple[int, ...],
    documents_shm: str,
    table_token_counts: dict[str, int],
    context_limit: int,
) -> str:
    q = question.question_evidence()
    q_clean = q.replace("\n", " ")
    question_fields = f"{question.db_id}\t{question.question_id}\t{q_clean}\t{question.SQL}"
    shm = shared_memory.SharedMemory(name=documents_shm)
    doc_embeddings = np.ndarray(documents_shape, dtype=np.float64, buffer=shm.buf)

    try:
        assert question.SQL, "can't evaluate without golden SQL"
        sql_refs = extract_sql_references("", tables, question.SQL or "", query_runtime_types=False)

        retrieved_ids = []
        token_count = 0
        relevancy = np.dot(doc_embeddings, question_embedding)
        for index, _score in sorted(enumerate(relevancy), key=lambda x: -x[1]):
            id = document_ids[index]
            if not id.startswith(question.db_id + "."):
                continue
            size = table_token_counts[id]
            if token_count + size > context_limit:
                # TODO: need to give this a partial score
                break
            token_count += size
            retrieved_ids.append(id)
        relevant_ids = set(f"{question.db_id}.{table}" for table in sql_refs.tables)
        retrieved_id_set = set(retrieved_ids)
        recall = len(relevant_ids & retrieved_id_set) / len(relevant_ids)
        correctness = float(relevant_ids <= retrieved_id_set)

        # Compute NDCG
        def dcg_01(relevance):
            return sum(x / math.log2(i + 2) for i, x in enumerate(relevance))

        ideal_ranking = [1] * len(relevant_ids)
        actual_ranking = [int(id in relevant_ids) for id in retrieved_ids]
        idcg = dcg_01(ideal_ranking)
        dcg = dcg_01(actual_ranking)
        ndcg = dcg / idcg if idcg > 0 else 0

        missing = relevant_ids - retrieved_id_set
        notes = f"retrieved={len(retrieved_ids)} relevant={len(relevant_ids)}"
        if missing:
            notes += " missing=" + ",".join([x.split(".", 1)[1] for x in missing])
        return f"{question_fields}\t{recall}\t{correctness}\t{ndcg}\t{notes}"
    except Exception as _e:
        # Usually this is a SQL error - ignore it
        return ""
        # return f"{question_fields}\t0\t0\t0\tException: {e}"


def _question_key(model: str, question: BIRDQuestion) -> str:
    return f"{model}_text:{question.question_evidence()}"


async def _get_embeddings(
    documents: dict[str, str],
    cache_path: str,
    model: str,
) -> dict[str, list[float]]:
    cached_embeddings = {}
    try:
        with open(cache_path, "rb") as f:
            cached_embeddings = pickle.load(f)
    except Exception:
        pass

    uncached_docs: list[str] = []
    uncached_keys = []
    embeddings: dict[str, list[float]] = {}
    for key, document in documents.items():
        if embedding := cached_embeddings.get(key):
            embeddings[key] = embedding
        else:
            uncached_docs.append(document)
            uncached_keys.append(key)

    if uncached_docs:
        result = await batch_embed(MODEL_NAMES[model], uncached_docs)
        for key, embedding in zip(uncached_keys, result.tolist()):
            embeddings[key] = cached_embeddings[key] = embedding

        with open(cache_path, "wb") as f:
            pickle.dump(cached_embeddings, f)

    return embeddings


def _calculate_token_counts(persist_dir: str, metadata: dict[str, Database]) -> dict[str, int]:
    token_count_cache = persist_dir + "/token_counts.json"
    if os.path.exists(token_count_cache):
        with open(persist_dir + "/token_counts.json") as f:
            token_counts = json.load(f)
    else:
        tokenizer = AutoTokenizer.from_pretrained("arcwise/bird-mistral-nemo")
        ids = []
        tables = []
        for db in metadata.values():
            for table in db.tables:
                ids.append(f"{db.name}.{table.name}")
                tables.append(get_table_ddl(table))
        table_token_counts = tokenizer(tables, return_length=True)["length"]
        token_counts: dict[str, int] = dict(zip(ids, table_token_counts))  # type: ignore
        with open(persist_dir + "/token_counts.json", "w") as f:
            json.dump(token_counts, f)
    return token_counts


if __name__ == "__main__":
    main()
