"""LanceStore — read/write Lance datasets on S3, registered in Polaris."""

from __future__ import annotations

import logging

import lance
import pyarrow as pa

from spark_rag.config import LanceConfig, PolarisConfig
from spark_rag.lance.polaris_helpers import (
    ensure_catalog,
    ensure_namespace,
    get_token,
    register_table,
)

logger = logging.getLogger(__name__)


class LanceStore:
    """Manages Lance datasets on Ceph S3 with Polaris catalog registration."""

    def __init__(self, lance_cfg: LanceConfig, polaris_cfg: PolarisConfig):
        self.storage_options = lance_cfg.storage_options
        self.s3_base = f"s3://{lance_cfg.s3_bucket}/{lance_cfg.s3_prefix}"
        self.polaris = polaris_cfg
        self.token = get_token(
            polaris_cfg.endpoint, polaris_cfg.client_id, polaris_cfg.client_secret,
        )
        # Ensure catalog and namespace exist (idempotent)
        ensure_catalog(
            polaris_cfg.endpoint, self.token, polaris_cfg.catalog, lance_cfg.s3_bucket,
        )
        ensure_namespace(
            polaris_cfg.endpoint, self.token, polaris_cfg.catalog, polaris_cfg.namespace,
        )

    def _uri(self, table_name: str) -> str:
        return f"{self.s3_base}/{table_name}"

    def write_table(self, table_name: str, data: pa.Table, mode: str = "overwrite") -> int:
        """Write Arrow table to Lance on S3 and register in Polaris.

        Args:
            table_name: Lance table name (e.g. 'spark_code').
            data: PyArrow Table to write.
            mode: 'overwrite' replaces entire dataset, 'append' adds rows.

        Returns:
            Number of rows written.
        """
        uri = self._uri(table_name)
        lance.write_dataset(data, uri, mode=mode, storage_options=self.storage_options)
        register_table(
            self.polaris.endpoint, self.token, self.polaris.catalog,
            self.polaris.namespace, table_name, uri,
        )
        logger.info("Wrote %d rows to Lance table '%s' (mode=%s)", len(data), table_name, mode)
        return len(data)

    def open_dataset(self, table_name: str) -> lance.LanceDataset:
        """Open a Lance dataset for reading."""
        uri = self._uri(table_name)
        return lance.dataset(uri, storage_options=self.storage_options)

    def replace_version(self, table_name: str, spark_version: str, data: pa.Table) -> int:
        """Replace all rows for a Spark version. For code/docs tables.

        Deletes existing rows with matching spark_version, then appends new data.
        If the dataset doesn't exist yet, creates it.
        """
        uri = self._uri(table_name)
        try:
            ds = lance.dataset(uri, storage_options=self.storage_options)
            ds.delete(f"spark_version = '{spark_version}'")
            lance.write_dataset(data, uri, mode="append", storage_options=self.storage_options)
        except Exception:
            # Dataset doesn't exist yet — create it
            lance.write_dataset(data, uri, mode="create", storage_options=self.storage_options)

        register_table(
            self.polaris.endpoint, self.token, self.polaris.catalog,
            self.polaris.namespace, table_name, uri,
        )
        logger.info(
            "Replaced version %s in '%s' (%d rows)", spark_version, table_name, len(data),
        )
        return len(data)

    def replace_by_id(
        self, table_name: str, id_field: str, id_value: int, data: pa.Table,
    ) -> int:
        """Replace rows by source ID (e.g. question_id, issue_number).

        For incremental SO/issues ingestion.
        """
        uri = self._uri(table_name)
        try:
            ds = lance.dataset(uri, storage_options=self.storage_options)
            ds.delete(f"{id_field} = {id_value}")
            lance.write_dataset(data, uri, mode="append", storage_options=self.storage_options)
        except Exception:
            lance.write_dataset(data, uri, mode="create", storage_options=self.storage_options)

        logger.debug("Replaced %s=%d in '%s' (%d rows)", id_field, id_value, table_name, len(data))
        return len(data)

    def add_embeddings(
        self,
        table_name: str,
        chunk_ids: list[str],
        column_name: str,
        vectors: list[list[float]],
        dim: int,
    ) -> int:
        """Merge embedding vectors into existing rows by chunk_id.

        Uses Lance dataset merge (schema evolution) to add or update
        an embedding column.
        """
        flat_values = [v for vec in vectors for v in vec]
        emb_array = pa.FixedSizeListArray.from_arrays(
            pa.array(flat_values, type=pa.float32()), dim,
        )
        merge_table = pa.table({"chunk_id": chunk_ids, column_name: emb_array})

        ds = self.open_dataset(table_name)
        ds.merge(merge_table, on="chunk_id")
        logger.info(
            "Merged %d embeddings into '%s' column '%s'",
            len(chunk_ids), table_name, column_name,
        )
        return len(chunk_ids)

    def read_unembedded(self, table_name: str, emb_column: str) -> pa.Table:
        """Read rows where the embedding column is null (not yet embedded)."""
        ds = self.open_dataset(table_name)
        return ds.to_table(filter=f"{emb_column} IS NULL")

    def dataset_exists(self, table_name: str) -> bool:
        """Check if a Lance dataset exists at the expected URI."""
        try:
            lance.dataset(self._uri(table_name), storage_options=self.storage_options)
            return True
        except Exception:
            return False

    def row_count(self, table_name: str) -> int:
        """Return total row count for a dataset."""
        ds = self.open_dataset(table_name)
        return ds.count_rows()
