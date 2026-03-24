"""Embedding experiment runner — add vector columns to Lance datasets."""

from __future__ import annotations

import logging

from spark_rag.embedding.client import EmbeddingClient
from spark_rag.lance.store import LanceStore

logger = logging.getLogger(__name__)


class EmbeddingExperiment:
    """Add an embedding column to a Lance dataset.

    Each experiment adds a fixed_size_list<float32>(dim) column.
    Incremental: only embeds rows where the column doesn't exist yet.
    """

    def __init__(
        self,
        store: LanceStore,
        embedding_client: EmbeddingClient,
        column_name: str,
        dimensions: int,
    ):
        self.store = store
        self.client = embedding_client
        self.column_name = column_name
        self.dimensions = dimensions

    def run(self, table_name: str, batch_size: int = 200) -> int:
        """Embed all unembedded rows and merge vectors back.

        Returns:
            Number of rows newly embedded.
        """
        # Check if column exists — if not, embed all rows
        ds = self.store.open_dataset(table_name)
        schema_names = [f.name for f in ds.schema]

        if self.column_name in schema_names:
            # Column exists — only embed rows with NULL
            unembedded = self.store.read_unembedded(table_name, self.column_name)
        else:
            # Column doesn't exist — embed all rows
            unembedded = ds.to_table(columns=["chunk_id", "content"])

        if len(unembedded) == 0:
            logger.info("All rows in '%s' already have '%s'", table_name, self.column_name)
            return 0

        logger.info(
            "Embedding %d rows in '%s' for column '%s'...",
            len(unembedded), table_name, self.column_name,
        )

        texts = unembedded.column("content").to_pylist()
        chunk_ids = unembedded.column("chunk_id").to_pylist()

        result = self.client.embed_batch(texts, batch_size=batch_size)

        self.store.add_embeddings(
            table_name, chunk_ids, self.column_name, result.vectors, self.dimensions,
        )

        logger.info(
            "Embedded %d rows in '%s' → column '%s' (dim=%d)",
            len(texts), table_name, self.column_name, self.dimensions,
        )
        return len(texts)
