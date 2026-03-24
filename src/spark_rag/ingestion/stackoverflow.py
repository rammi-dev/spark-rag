"""StackOverflow ingestion — fetch answered Spark questions via API → Lance.

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

from spark_rag.chunking.so_chunker import chunk_question
from spark_rag.config import load_config
from spark_rag.lance.schemas import so_chunks_to_table
from spark_rag.lance.store import LanceStore

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
    parser = argparse.ArgumentParser(description="Ingest StackOverflow Spark Q&A into Lance")
    parser.add_argument("--since", help="Fetch questions modified after this date (YYYY-MM-DD)")
    parser.add_argument("--max", type=int, default=None, help="Override max_questions from config")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    so_cfg = cfg.ingestion.stackoverflow
    since = datetime.fromisoformat(args.since) if args.since else None
    max_q = args.max or so_cfg.max_questions

    # Fetch
    questions = fetch_questions(so_cfg.tags, max_questions=max_q, since=since)

    # Chunk
    all_chunks = []
    chunks_by_question: dict[int, list] = {}
    for q in questions:
        chunks = chunk_question(q)
        all_chunks.extend(chunks)
        chunks_by_question[q["question_id"]] = chunks

    logger.info("Chunked %d questions → %d chunks", len(questions), len(all_chunks))

    if args.dry_run:
        logger.info("DRY RUN — not writing to Lance")
        return

    # Write to Lance (incremental by question_id)
    store = LanceStore(cfg.lance, cfg.polaris)
    total = 0
    for qid, chunks in chunks_by_question.items():
        table = so_chunks_to_table(chunks)
        total += store.replace_by_id("spark_so", "question_id", qid, table)

    # Register table in Polaris (once after all inserts)
    if total > 0:
        from spark_rag.lance.polaris_helpers import register_table
        register_table(
            cfg.polaris.endpoint, store.token, cfg.polaris.catalog,
            cfg.polaris.namespace, "spark_so", store._uri("spark_so"),
        )

    logger.info("Wrote %d SO chunks to Lance", total)


if __name__ == "__main__":
    main()
