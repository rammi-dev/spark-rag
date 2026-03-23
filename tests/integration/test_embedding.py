"""Integration test: embedding client against live Ollama.

Requires: kubectl -n ollama port-forward svc/ollama 11434:11434
"""

import numpy as np
import pytest
import requests

from spark_rag.config import OllamaConfig
from spark_rag.embedding.client import EmbeddingClient

OLLAMA_URL = "http://localhost:11434"


@pytest.fixture(scope="module")
def client():
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
    except Exception:
        pytest.skip("Ollama not reachable at localhost:11434")
    return EmbeddingClient(OllamaConfig(url=OLLAMA_URL))


class TestLiveEmbedding:
    def test_health_check(self, client):
        assert client.check_health() is True

    def test_embed_returns_768_dims(self, client):
        vec = client.embed_one("Apache Spark DataFrame collect")
        assert len(vec) == 768
        assert all(isinstance(v, float) for v in vec)

    def test_different_texts_produce_different_vectors(self, client):
        result = client.embed_batch([
            "How to fix OutOfMemoryError in Spark",
            "Python list comprehension tutorial",
        ])
        v1 = np.array(result.vectors[0])
        v2 = np.array(result.vectors[1])
        cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # Different topics should have lower similarity
        assert cosine_sim < 0.95

    def test_similar_texts_have_high_similarity(self, client):
        result = client.embed_batch([
            "Spark OOM error on groupByKey with large dataset",
            "OutOfMemoryError when using groupByKey in Apache Spark",
        ])
        v1 = np.array(result.vectors[0])
        v2 = np.array(result.vectors[1])
        cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        assert cosine_sim > 0.5

    def test_batch_of_ten(self, client):
        texts = [f"Spark API example {i}: {op}" for i, op in enumerate([
            "select", "filter", "groupBy", "agg", "join",
            "union", "cache", "persist", "collect", "show",
        ])]
        result = client.embed_batch(texts)
        assert len(result.vectors) == 10
        assert result.dimension == 768
