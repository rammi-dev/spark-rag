"""Batch ingest data into Milvus collections.

Handles:
- Batch insertion with configurable size
- Version-based replacement (delete old → insert new) for code/docs
- Source-ID-based dedup for SO/issues incremental sync
"""

from __future__ import annotations

import logging
import time

from pymilvus import MilvusClient

from spark_rag.milvus.collections import (
    COLLECTION_NAMES,
    delete_by_source_id,
    delete_by_version,
)

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 500
FLUSH_WAIT_SECONDS = 2


def batch_insert(
    client: MilvusClient,
    collection_name: str,
    data: list[dict],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Insert data in batches.

    Args:
        client: MilvusClient instance.
        collection_name: Target collection.
        data: List of dicts, each matching the collection schema.
        batch_size: Max records per insert call.

    Returns:
        Total number of records inserted.
    """
    if not data:
        return 0

    if collection_name not in COLLECTION_NAMES:
        raise ValueError(f"Unknown collection: {collection_name}")

    total = 0
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        result = client.insert(collection_name=collection_name, data=batch)
        count = result.get("insert_count", len(batch))
        total += count

        if (i // batch_size) % 10 == 0 and i > 0:
            logger.info(
                "Inserted %d / %d into %s", total, len(data), collection_name
            )

    logger.info(
        "Inserted %d records into %s", total, collection_name
    )
    return total


def ingest_version(
    client: MilvusClient,
    collection_name: str,
    data: list[dict],
    spark_version: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Ingest data for a specific Spark version (code or docs).

    Deletes existing data for this version first, then inserts fresh.

    Args:
        client: MilvusClient instance.
        collection_name: Must be "spark_code" or "spark_docs".
        data: List of dicts with embeddings and metadata.
        spark_version: Version string (e.g. "4.1.0").
        batch_size: Max records per insert call.

    Returns:
        Number of records inserted.
    """
    if collection_name not in ("spark_code", "spark_docs"):
        raise ValueError(f"Version-based ingestion only for spark_code/spark_docs, got {collection_name}")

    logger.info(
        "Ingesting %d records for %s version %s (replacing existing)",
        len(data), collection_name, spark_version,
    )

    # Delete existing data for this version
    delete_by_version(client, collection_name, spark_version)
    time.sleep(FLUSH_WAIT_SECONDS)

    # Insert new data
    return batch_insert(client, collection_name, data, batch_size)


def ingest_incremental_so(
    client: MilvusClient,
    data_by_question: dict[int, list[dict]],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Incremental ingest for StackOverflow: upsert by question_id.

    For each question_id, deletes existing vectors then inserts new ones.

    Args:
        client: MilvusClient instance.
        data_by_question: {question_id: [chunk_dicts]}. Each chunk dict
            must have 'question_id' field.
        batch_size: Max records per insert call.

    Returns:
        Total records inserted.
    """
    total = 0
    all_data: list[dict] = []

    for question_id, chunks in data_by_question.items():
        delete_by_source_id(client, "spark_stackoverflow", "question_id", question_id)
        all_data.extend(chunks)

    if all_data:
        time.sleep(FLUSH_WAIT_SECONDS)
        total = batch_insert(client, "spark_stackoverflow", all_data, batch_size)

    logger.info(
        "Incremental SO ingest: %d questions, %d total records",
        len(data_by_question), total,
    )
    return total


def ingest_incremental_issues(
    client: MilvusClient,
    data_by_issue: dict[int, list[dict]],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Incremental ingest for GitHub Issues: upsert by issue_number.

    For each issue_number, deletes existing vectors (body + comments)
    then inserts new ones.

    Args:
        client: MilvusClient instance.
        data_by_issue: {issue_number: [chunk_dicts]}. Each chunk dict
            must have 'issue_number' field.
        batch_size: Max records per insert call.

    Returns:
        Total records inserted.
    """
    total = 0
    all_data: list[dict] = []

    for issue_number, chunks in data_by_issue.items():
        delete_by_source_id(client, "spark_issues", "issue_number", issue_number)
        all_data.extend(chunks)

    if all_data:
        time.sleep(FLUSH_WAIT_SECONDS)
        total = batch_insert(client, "spark_issues", all_data, batch_size)

    logger.info(
        "Incremental issues ingest: %d issues, %d total records",
        len(data_by_issue), total,
    )
    return total
