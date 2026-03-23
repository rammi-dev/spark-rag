"""StackOverflow ingestion — fetch answered Spark questions via API.

Can be run standalone or called from an Airflow DAG.

Usage:
    uv run python -m spark_rag.ingestion.stackoverflow
    uv run python -m spark_rag.ingestion.stackoverflow --since 2025-01-01
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime

import requests

from pymilvus import MilvusClient

from spark_rag.chunking.so_chunker import chunk_question
from spark_rag.config import load_config
from spark_rag.embedding.client import EmbeddingClient
from spark_rag.milvus.collections import create_collection
from spark_rag.milvus.ingest import ingest_incremental_so

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SE_API_BASE = "https://api.stackexchange.com/2.3"


def fetch_questions(
    tags: list[str],
    max_questions: int = 5000,
    since: datetime | None = None,
    page_size: int = 100,
) -> list[dict]:
    """Fetch answered Spark questions from StackExchange API.

    Args:
        tags: Tags to filter by (e.g. ["apache-spark"]).
        max_questions: Max questions to fetch.
        since: Only fetch questions modified after this date (for incremental).
        page_size: API page size.

    Returns:
        List of question dicts with answers included.
    """
    questions: list[dict] = []
    page = 1

    params = {
        "order": "desc",
        "sort": "activity",
        "tagged": ";".join(tags),
        "site": "stackoverflow",
        "filter": "withbody",  # include body + answers
        "pagesize": min(page_size, 100),
    }
    if since:
        params["fromdate"] = int(since.timestamp())

    while len(questions) < max_questions:
        params["page"] = page
        try:
            resp = requests.get(f"{SE_API_BASE}/questions", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("SE API error on page %d: %s", page, e)
            break

        items = data.get("items", [])
        if not items:
            break

        # Filter to answered questions
        for item in items:
            if item.get("answer_count", 0) > 0:
                questions.append(item)

        if not data.get("has_more", False):
            break

        page += 1
        # Respect API rate limit
        remaining = data.get("quota_remaining", 100)
        if remaining < 10:
            logger.warning("SE API quota low: %d remaining", remaining)
            break
        time.sleep(0.5)

    logger.info("Fetched %d answered questions", len(questions))
    return questions[:max_questions]


def main():
    parser = argparse.ArgumentParser(description="Ingest StackOverflow Spark Q&A into Milvus")
    parser.add_argument("--since", help="Fetch questions modified after this date (YYYY-MM-DD)")
    parser.add_argument("--max", type=int, default=None, help="Override max_questions from config")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    so_cfg = cfg.ingestion.stackoverflow
    since = datetime.fromisoformat(args.since) if args.since else None
    max_q = args.max or so_cfg.max_questions

    # Fetch
    questions = fetch_questions(so_cfg.tags, max_questions=max_q, since=since)

    # Chunk
    data_by_question: dict[int, list[dict]] = {}
    total_chunks = 0
    for q in questions:
        chunks = chunk_question(q)
        total_chunks += len(chunks)
        data_by_question[q["question_id"]] = []
        for chunk in chunks:
            data_by_question[q["question_id"]].append(
                # Store without embedding for now — will embed below
                {"_chunk": chunk}
            )

    logger.info("Chunked %d questions → %d chunks", len(questions), total_chunks)

    if args.dry_run:
        logger.info("DRY RUN — not embedding or inserting")
        return

    # Embed all chunks
    embedding_client = EmbeddingClient(cfg.ollama)
    all_chunks = []
    for qid, chunk_wrappers in data_by_question.items():
        for cw in chunk_wrappers:
            all_chunks.append(cw["_chunk"])

    texts = [c.content for c in all_chunks]
    embed_result = embedding_client.embed_batch(texts, batch_size=args.batch_size)

    # Rebuild data_by_question with embeddings
    idx = 0
    embedded_by_question: dict[int, list[dict]] = {}
    for qid, chunk_wrappers in data_by_question.items():
        embedded_by_question[qid] = []
        for cw in chunk_wrappers:
            chunk = cw["_chunk"]
            embedded_by_question[qid].append(
                chunk.to_milvus_data(embed_result.vectors[idx])
            )
            idx += 1

    # Insert
    milvus_client = MilvusClient(uri=cfg.milvus.url)
    create_collection(milvus_client, "spark_stackoverflow", drop_existing=False)

    count = ingest_incremental_so(milvus_client, embedded_by_question, batch_size=args.batch_size)
    logger.info("Ingested %d SO records", count)
    milvus_client.close()


if __name__ == "__main__":
    main()
