"""Lance → Milvus loader — read embeddings + metadata from Lance, insert into Milvus."""

from __future__ import annotations

import logging

from pymilvus import MilvusClient

from spark_rag.lance.store import LanceStore
from spark_rag.milvus.collections import create_collection
from spark_rag.milvus.ingest import batch_insert

logger = logging.getLogger(__name__)

# Mapping from Lance table name to Milvus collection name
_TABLE_TO_COLLECTION = {
    "spark_code": "spark_code",
    "spark_docs": "spark_docs",
    "spark_so": "spark_stackoverflow",
    "spark_issues": "spark_issues",
}

# Fields to drop when converting Lance → Milvus (not in Milvus schema)
_DROP_FIELDS = {"chunk_id", "ingested_at"}

# Arrow list columns that need re-wrapping as JSON dicts for Milvus
_LIST_TO_JSON = {
    "spark_code": {
        "spark_apis": "apis",
        "problem_indicators": "patterns",
    },
    "spark_docs": {
        "heading_hierarchy": None,  # stays as-is (JSON list in Milvus)
        "related_configs": "configs",
    },
    "spark_so": {
        "tags": "tags",
        "spark_apis_mentioned": "apis",
        "spark_versions_mentioned": "versions",
    },
    "spark_issues": {
        "labels": "labels",
        "spark_versions_mentioned": "versions",
        "linked_prs": "prs",
    },
}


class LanceMilvusLoader:
    """Load embeddings + metadata from a Lance table into Milvus."""

    def __init__(self, store: LanceStore, milvus_client: MilvusClient):
        self.store = store
        self.milvus = milvus_client

    def load(
        self,
        table_name: str,
        embedding_column: str,
        spark_version: str | None = None,
        batch_size: int = 500,
        drop_existing: bool = False,
    ) -> int:
        """Read Lance dataset, convert to Milvus format, insert.

        Args:
            table_name: Lance table name (e.g. 'spark_code').
            embedding_column: Which embedding column to use (e.g. 'emb_nomic').
            spark_version: Filter by version (for code/docs). None = all versions.
            batch_size: Milvus insert batch size.
            drop_existing: If True, recreate the Milvus collection first.

        Returns:
            Number of rows loaded into Milvus.
        """
        collection = _TABLE_TO_COLLECTION.get(table_name, table_name)

        # Read from Lance
        ds = self.store.open_dataset(table_name)
        filter_expr = f"spark_version = '{spark_version}'" if spark_version else None
        arrow_table = ds.to_table(filter=filter_expr)

        if len(arrow_table) == 0:
            logger.warning("No rows in '%s'%s", table_name,
                           f" for version {spark_version}" if spark_version else "")
            return 0

        # Check embedding column exists
        if embedding_column not in [f.name for f in arrow_table.schema]:
            raise ValueError(
                f"Embedding column '{embedding_column}' not found in '{table_name}'. "
                f"Run embedding first: python -m spark_rag.lance.cli_embed --table {table_name} --column {embedding_column}"
            )

        # Convert to list of dicts for Milvus
        milvus_data = self._to_milvus_format(arrow_table, table_name, embedding_column)

        # Create collection and insert
        create_collection(self.milvus, collection, drop_existing=drop_existing)
        count = batch_insert(self.milvus, collection, milvus_data, batch_size=batch_size)
        logger.info("Loaded %d rows from '%s' → Milvus '%s'", count, table_name, collection)
        return count

    def _to_milvus_format(
        self, arrow_table, table_name: str, embedding_column: str,
    ) -> list[dict]:
        """Convert Arrow table to list of Milvus-compatible dicts."""
        json_mappings = _LIST_TO_JSON.get(table_name, {})
        rows = arrow_table.to_pydict()
        n = len(arrow_table)
        result = []

        for i in range(n):
            row = {}
            for col_name in rows:
                if col_name in _DROP_FIELDS:
                    continue
                if col_name == embedding_column:
                    row["embedding"] = rows[col_name][i]
                    continue

                # Skip other embedding columns
                if col_name.startswith("emb_"):
                    continue

                value = rows[col_name][i]

                # Re-wrap list columns as JSON dicts for Milvus
                if col_name in json_mappings:
                    wrapper_key = json_mappings[col_name]
                    if wrapper_key is not None:
                        value = {wrapper_key: value if value is not None else []}
                    # If wrapper_key is None, keep as-is (already a list)

                # Truncate strings to Milvus VARCHAR limits
                if isinstance(value, str):
                    value = _truncate(value, col_name)

                row[col_name] = value

            result.append(row)

        return result


# Milvus VARCHAR limits by field
_FIELD_LIMITS = {
    "content": 65535,
    "qualified_name": 512,
    "signature": 1024,
    "doc_url": 512,
    "doc_section": 256,
    "error_type": 256,
    "author": 128,
    "state": 16,
    "created_at": 32,
    "closed_at": 32,
}


def _truncate(value: str, field: str) -> str:
    limit = _FIELD_LIMITS.get(field)
    if limit and len(value) > limit:
        return value[:limit]
    return value
