"""Unit tests for embedding client (mocked Ollama)."""

from unittest.mock import MagicMock, patch

import pytest

from spark_rag.config import OllamaConfig
from spark_rag.embedding.client import EmbeddingClient


@pytest.fixture
def client():
    return EmbeddingClient(OllamaConfig(url="http://fake:11434"))


def _mock_response(embeddings):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"embeddings": embeddings}
    resp.raise_for_status = MagicMock()
    return resp


class TestEmbedOne:
    @patch("spark_rag.embedding.client.requests.post")
    def test_returns_single_vector(self, mock_post, client):
        vec = [0.1] * 768
        mock_post.return_value = _mock_response([vec])

        result = client.embed_one("test text")

        assert result == vec
        mock_post.assert_called_once()
        call_json = mock_post.call_args.kwargs["json"]
        assert call_json["input"] == ["test text"]
        assert call_json["model"] == "nomic-embed-text"


class TestEmbedBatch:
    @patch("spark_rag.embedding.client.requests.post")
    def test_batch_preserves_order(self, mock_post, client):
        vecs = [[float(i)] * 768 for i in range(5)]
        mock_post.return_value = _mock_response(vecs)

        result = client.embed_batch(["a", "b", "c", "d", "e"])

        assert len(result.vectors) == 5
        assert result.vectors[0][0] == 0.0
        assert result.vectors[4][0] == 4.0
        assert result.dimension == 768

    @patch("spark_rag.embedding.client.requests.post")
    def test_splits_into_batches(self, mock_post, client):
        vec = [0.1] * 768
        # Each batch call returns its batch size worth of vectors
        mock_post.side_effect = [
            _mock_response([vec] * 3),
            _mock_response([vec] * 2),
        ]

        result = client.embed_batch(["a", "b", "c", "d", "e"], batch_size=3)

        assert len(result.vectors) == 5
        assert mock_post.call_count == 2
        # First call: 3 texts, second: 2
        assert len(mock_post.call_args_list[0].kwargs["json"]["input"]) == 3
        assert len(mock_post.call_args_list[1].kwargs["json"]["input"]) == 2

    @patch("spark_rag.embedding.client.requests.post")
    def test_empty_input(self, mock_post, client):
        result = client.embed_batch([])

        assert result.vectors == []
        assert result.dimension == 0
        mock_post.assert_not_called()

    @patch("spark_rag.embedding.client.requests.post")
    def test_dimension_set_after_embed(self, mock_post, client):
        assert client.dimension is None
        mock_post.return_value = _mock_response([[0.1] * 768])
        client.embed_one("test")
        assert client.dimension == 768


class TestHealthCheck:
    @patch("spark_rag.embedding.client.requests.get")
    def test_healthy_with_model(self, mock_get, client):
        resp = MagicMock()
        resp.json.return_value = {
            "models": [{"name": "nomic-embed-text:latest"}]
        }
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        assert client.check_health() is True

    @patch("spark_rag.embedding.client.requests.get")
    def test_unhealthy_missing_model(self, mock_get, client):
        resp = MagicMock()
        resp.json.return_value = {
            "models": [{"name": "tinyllama:latest"}]
        }
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        assert client.check_health() is False

    @patch("spark_rag.embedding.client.requests.get")
    def test_unhealthy_connection_error(self, mock_get, client):
        mock_get.side_effect = ConnectionError("refused")
        assert client.check_health() is False
