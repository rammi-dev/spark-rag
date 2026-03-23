"""Milvus collection schema definitions and lifecycle management.

4 collections:
- spark_code: source code chunks (per version, partition key: language)
- spark_docs: documentation chunks (per version)
- spark_stackoverflow: SO Q&A (cross-version, incremental)
- spark_issues: GitHub issues + comments (cross-version, incremental)

All use HNSW index with COSINE metric, 768-dim vectors (nomic-embed-text).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

logger = logging.getLogger(__name__)

DIMENSION = 768
HNSW_PARAMS = {"M": 16, "efConstruction": 256}
SEARCH_PARAMS = {"ef": 64}

COLLECTION_NAMES = ["spark_code", "spark_docs", "spark_stackoverflow", "spark_issues"]


def _base_fields() -> list[FieldSchema]:
    """Fields common to all collections."""
    return [
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema("content", DataType.VARCHAR, max_length=65535),
    ]


def spark_code_schema() -> CollectionSchema:
    fields = _base_fields() + [
        FieldSchema("spark_version", DataType.VARCHAR, max_length=32),
        FieldSchema("file_path", DataType.VARCHAR, max_length=512),
        FieldSchema("language", DataType.VARCHAR, max_length=32, is_partition_key=True),
        FieldSchema("chunk_type", DataType.VARCHAR, max_length=32),
        FieldSchema("qualified_name", DataType.VARCHAR, max_length=512),
        FieldSchema("signature", DataType.VARCHAR, max_length=1024),
        FieldSchema("spark_apis", DataType.JSON),
        FieldSchema("problem_indicators", DataType.JSON),
    ]
    return CollectionSchema(fields, description="Spark source code chunks, per version")


def spark_docs_schema() -> CollectionSchema:
    fields = _base_fields() + [
        FieldSchema("spark_version", DataType.VARCHAR, max_length=32),
        FieldSchema("doc_url", DataType.VARCHAR, max_length=512),
        FieldSchema("doc_section", DataType.VARCHAR, max_length=256),
        FieldSchema("heading_hierarchy", DataType.JSON),
        FieldSchema("content_type", DataType.VARCHAR, max_length=32),
        FieldSchema("related_configs", DataType.JSON),
    ]
    return CollectionSchema(fields, description="Spark documentation chunks, per version")


def spark_stackoverflow_schema() -> CollectionSchema:
    fields = _base_fields() + [
        FieldSchema("question_id", DataType.INT64),
        FieldSchema("is_question", DataType.BOOL),
        FieldSchema("score", DataType.INT64),
        FieldSchema("is_accepted", DataType.BOOL),
        FieldSchema("tags", DataType.JSON),
        FieldSchema("error_type", DataType.VARCHAR, max_length=256),
        FieldSchema("spark_apis_mentioned", DataType.JSON),
        FieldSchema("spark_versions_mentioned", DataType.JSON),
    ]
    return CollectionSchema(fields, description="StackOverflow Q&A, cross-version")


def spark_issues_schema() -> CollectionSchema:
    fields = _base_fields() + [
        FieldSchema("issue_number", DataType.INT64),
        FieldSchema("state", DataType.VARCHAR, max_length=16),
        FieldSchema("is_comment", DataType.BOOL),
        FieldSchema("parent_issue_number", DataType.INT64),
        FieldSchema("author", DataType.VARCHAR, max_length=128),
        FieldSchema("labels", DataType.JSON),
        FieldSchema("created_at", DataType.VARCHAR, max_length=32),
        FieldSchema("closed_at", DataType.VARCHAR, max_length=32),
        FieldSchema("spark_versions_mentioned", DataType.JSON),
        FieldSchema("linked_prs", DataType.JSON),
    ]
    return CollectionSchema(fields, description="GitHub issues + comments, cross-version")


SCHEMAS = {
    "spark_code": spark_code_schema,
    "spark_docs": spark_docs_schema,
    "spark_stackoverflow": spark_stackoverflow_schema,
    "spark_issues": spark_issues_schema,
}


@dataclass
class CollectionInfo:
    name: str
    exists: bool
    num_entities: int


def create_collection(
    client: MilvusClient,
    name: str,
    drop_existing: bool = False,
) -> None:
    """Create a single collection with HNSW index.

    Args:
        client: MilvusClient instance.
        name: Collection name (must be in SCHEMAS).
        drop_existing: If True, drop and recreate. If False, skip if exists.
    """
    if name not in SCHEMAS:
        raise ValueError(f"Unknown collection: {name}. Must be one of {COLLECTION_NAMES}")

    if client.has_collection(name):
        if drop_existing:
            logger.info("Dropping existing collection: %s", name)
            client.drop_collection(name)
        else:
            logger.info("Collection %s already exists, skipping", name)
            return

    schema = SCHEMAS[name]()

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="HNSW",
        metric_type="COSINE",
        params=HNSW_PARAMS,
    )

    logger.info("Creating collection: %s", name)
    client.create_collection(
        collection_name=name,
        schema=schema,
        index_params=index_params,
    )
    logger.info("Created collection: %s", name)


def create_all_collections(
    client: MilvusClient,
    drop_existing: bool = False,
) -> None:
    """Create all 4 collections."""
    for name in COLLECTION_NAMES:
        create_collection(client, name, drop_existing=drop_existing)


def get_collection_info(client: MilvusClient, name: str) -> CollectionInfo:
    """Get info about a collection."""
    exists = client.has_collection(name)
    num_entities = 0
    if exists:
        stats = client.get_collection_stats(name)
        num_entities = stats.get("row_count", 0)
    return CollectionInfo(name=name, exists=exists, num_entities=num_entities)


def get_all_collection_info(client: MilvusClient) -> list[CollectionInfo]:
    """Get info about all 4 collections."""
    return [get_collection_info(client, name) for name in COLLECTION_NAMES]


def delete_by_version(
    client: MilvusClient,
    collection_name: str,
    spark_version: str,
) -> None:
    """Delete all chunks for a specific Spark version (for re-ingestion).

    Only applicable to spark_code and spark_docs (version-tagged collections).
    """
    if collection_name not in ("spark_code", "spark_docs"):
        raise ValueError(f"Version-based deletion only for spark_code/spark_docs, got {collection_name}")

    logger.info("Deleting chunks for version %s from %s", spark_version, collection_name)
    client.delete(
        collection_name=collection_name,
        filter=f'spark_version == "{spark_version}"',
    )


def delete_by_source_id(
    client: MilvusClient,
    collection_name: str,
    field: str,
    value: int,
) -> None:
    """Delete chunks by source ID (for incremental sync dedup).

    Examples:
        delete_by_source_id(client, "spark_stackoverflow", "question_id", 12345)
        delete_by_source_id(client, "spark_issues", "issue_number", 678)
    """
    logger.info("Deleting from %s where %s == %d", collection_name, field, value)
    client.delete(
        collection_name=collection_name,
        filter=f"{field} == {value}",
    )
