"""Integration test: create collections, insert, search, delete against live Milvus.

Requires: kubectl -n milvus port-forward svc/milvus 19530:19530
"""

import time

import numpy as np
import pytest
from pymilvus import MilvusClient

from spark_rag.milvus.collections import (
    COLLECTION_NAMES,
    create_all_collections,
    create_collection,
    delete_by_source_id,
    delete_by_version,
    get_all_collection_info,
    get_collection_info,
)

MILVUS_URI = "http://localhost:19530"
TEST_PREFIX = "test_crud_"


def _test_name(name: str) -> str:
    return f"{TEST_PREFIX}{name}"


def _random_vectors(n, dim=768):
    vecs = np.random.random((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / norms).tolist()


@pytest.fixture(scope="module")
def client():
    try:
        c = MilvusClient(uri=MILVUS_URI)
        c.get_server_version()
    except Exception:
        pytest.skip("Milvus not reachable at localhost:19530")
    yield c
    # Cleanup
    for name in c.list_collections():
        if name.startswith(TEST_PREFIX):
            c.drop_collection(name)
    c.close()


class TestCreateCollections:
    def test_create_spark_code(self, client):
        name = _test_name("spark_code")
        # Monkey-patch the name for testing
        create_collection(client, "spark_code", drop_existing=True)
        info = get_collection_info(client, "spark_code")
        assert info.exists is True

    def test_create_all(self, client):
        create_all_collections(client, drop_existing=True)
        for name in COLLECTION_NAMES:
            assert client.has_collection(name)

    def test_skip_existing(self, client):
        """Creating without drop_existing should not error."""
        create_collection(client, "spark_code", drop_existing=False)
        assert client.has_collection("spark_code")

    def test_unknown_collection_raises(self, client):
        with pytest.raises(ValueError, match="Unknown collection"):
            create_collection(client, "nonexistent")


class TestInsertAndSearch:
    def test_insert_code_chunks(self, client):
        create_collection(client, "spark_code", drop_existing=True)

        vectors = _random_vectors(10)
        data = [
            {
                "embedding": vec,
                "content": f"def method_{i}(): pass",
                "spark_version": "4.1.0",
                "file_path": f"sql/core/src/main/scala/File{i}.scala",
                "language": "scala",
                "chunk_type": "method",
                "qualified_name": f"o.a.s.sql.File{i}.method_{i}",
                "signature": f"def method_{i}(): Unit",
                "spark_apis": {"apis": ["DataFrame.select"]},
                "problem_indicators": {"patterns": []},
            }
            for i, vec in enumerate(vectors)
        ]
        res = client.insert(collection_name="spark_code", data=data)
        assert res["insert_count"] == 10

        time.sleep(2)

        results = client.search(
            collection_name="spark_code",
            data=_random_vectors(1),
            limit=5,
            output_fields=["content", "spark_version", "language", "qualified_name"],
        )
        assert len(results[0]) == 5
        assert results[0][0]["entity"]["spark_version"] == "4.1.0"

    def test_insert_issues(self, client):
        create_collection(client, "spark_issues", drop_existing=True)

        vectors = _random_vectors(5)
        data = [
            {
                "embedding": vec,
                "content": f"Issue body {i}: NullPointerException in optimizer",
                "issue_number": 48000 + i,
                "state": "closed" if i < 3 else "open",
                "is_comment": False,
                "parent_issue_number": 48000 + i,
                "author": f"user{i}",
                "labels": {"labels": ["bug", "SQL"]},
                "created_at": "2025-01-15T10:00:00Z",
                "closed_at": "2025-02-01T12:00:00Z" if i < 3 else "",
                "spark_versions_mentioned": {"versions": ["4.1.0"]},
                "linked_prs": {"prs": [12345 + i]},
            }
            for i, vec in enumerate(vectors)
        ]
        res = client.insert(collection_name="spark_issues", data=data)
        assert res["insert_count"] == 5

        time.sleep(2)

        # Search with filter
        results = client.search(
            collection_name="spark_issues",
            data=_random_vectors(1),
            filter='state == "closed" and is_comment == false',
            limit=10,
            output_fields=["issue_number", "state", "labels"],
        )
        for hit in results[0]:
            assert hit["entity"]["state"] == "closed"


class TestVersionDeletion:
    def test_delete_by_version(self, client):
        create_collection(client, "spark_code", drop_existing=True)

        vectors = _random_vectors(10)
        data = [
            {
                "embedding": vec,
                "content": f"code chunk {i}",
                "spark_version": "4.1.0" if i < 6 else "3.5.4",
                "file_path": f"file{i}.scala",
                "language": "scala",
                "chunk_type": "method",
                "qualified_name": f"method_{i}",
                "signature": f"def method_{i}()",
                "spark_apis": {"apis": []},
                "problem_indicators": {"patterns": []},
            }
            for i, vec in enumerate(vectors)
        ]
        client.insert(collection_name="spark_code", data=data)
        time.sleep(2)

        # Delete 4.1.0 chunks
        delete_by_version(client, "spark_code", "4.1.0")
        time.sleep(2)

        # Search should only find 3.5.4
        results = client.search(
            collection_name="spark_code",
            data=_random_vectors(1),
            limit=10,
            output_fields=["spark_version"],
        )
        for hit in results[0]:
            assert hit["entity"]["spark_version"] == "3.5.4"

    def test_delete_by_version_rejects_wrong_collection(self, client):
        with pytest.raises(ValueError, match="Version-based deletion only"):
            delete_by_version(client, "spark_stackoverflow", "4.1.0")


class TestSourceIdDeletion:
    def test_delete_so_by_question_id(self, client):
        create_collection(client, "spark_stackoverflow", drop_existing=True)

        vectors = _random_vectors(5)
        data = [
            {
                "embedding": vec,
                "content": f"SO answer {i}",
                "question_id": 99900 + i,
                "is_question": i == 0,
                "score": 10 * i,
                "is_accepted": i == 1,
                "tags": {"tags": ["apache-spark"]},
                "error_type": "",
                "spark_apis_mentioned": {"apis": []},
                "spark_versions_mentioned": {"versions": []},
            }
            for i, vec in enumerate(vectors)
        ]
        client.insert(collection_name="spark_stackoverflow", data=data)
        time.sleep(2)

        # Delete question 99900
        delete_by_source_id(client, "spark_stackoverflow", "question_id", 99900)
        time.sleep(2)

        # Verify it's gone
        results = client.query(
            collection_name="spark_stackoverflow",
            filter="question_id == 99900",
            output_fields=["question_id"],
        )
        assert len(results) == 0


class TestCollectionInfo:
    def test_get_info(self, client):
        create_collection(client, "spark_docs", drop_existing=True)
        info = get_collection_info(client, "spark_docs")
        assert info.name == "spark_docs"
        assert info.exists is True

    def test_get_all_info(self, client):
        create_all_collections(client, drop_existing=True)
        infos = get_all_collection_info(client)
        assert len(infos) == 4
        for info in infos:
            assert info.exists is True
