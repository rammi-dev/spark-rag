"""Ollama embedding client with batching support."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

from spark_rag.config import OllamaConfig

logger = logging.getLogger(__name__)

# Ollama /api/embed accepts multiple inputs in one call
DEFAULT_BATCH_SIZE = 32


@dataclass
class EmbeddingResult:
    vectors: list[list[float]]
    model: str
    dimension: int


class EmbeddingClient:
    """Client for Ollama embedding API."""

    def __init__(self, config: OllamaConfig | None = None):
        self.config = config or OllamaConfig()
        self.base_url = self.config.url.rstrip("/")
        self.model = self.config.embedding_model
        self._dimension: int | None = None

    @property
    def embed_url(self) -> str:
        return f"{self.base_url}/api/embed"

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text string. Returns a single vector."""
        result = self.embed_batch([text])
        return result.vectors[0]

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> EmbeddingResult:
        """Embed multiple texts, splitting into batches if needed.

        Args:
            texts: List of strings to embed.
            batch_size: Max texts per Ollama API call.

        Returns:
            EmbeddingResult with all vectors in input order.
        """
        if not texts:
            return EmbeddingResult(vectors=[], model=self.model, dimension=0)

        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors = self._call_embed(batch)
            all_vectors.extend(vectors)

            if i > 0 and i % (batch_size * 10) == 0:
                logger.info("Embedded %d / %d texts", i, len(texts))

        dim = len(all_vectors[0]) if all_vectors else 0
        self._dimension = dim

        return EmbeddingResult(
            vectors=all_vectors,
            model=self.model,
            dimension=dim,
        )

    def _call_embed(self, texts: list[str]) -> list[list[float]]:
        """Single call to Ollama /api/embed."""
        resp = requests.post(
            self.embed_url,
            json={"model": self.model, "input": texts},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        embeddings = data.get("embeddings", [])
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Expected {len(texts)} embeddings, got {len(embeddings)}"
            )
        return embeddings

    @property
    def dimension(self) -> int | None:
        """Return embedding dimension (available after first embed call)."""
        return self._dimension

    def check_health(self) -> bool:
        """Check if Ollama is reachable and model is available."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            # Model names in Ollama can have :latest suffix
            return any(
                self.model in name for name in models
            )
        except Exception:
            return False
