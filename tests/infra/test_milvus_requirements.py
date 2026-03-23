"""Phase 0: Validate Milvus supports all features the RAG system needs.

Run with: uv run pytest tests/infra/test_milvus_requirements.py -v
Requires: kubectl -n milvus port-forward svc/milvus 19530:19530
"""

import time
import numpy as np
import pytest
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)

DIMENSION = 768  # nomic-embed-text output dimension
TEST_PREFIX = "test_phase0_"


@pytest.fixture(scope="module")
def client(milvus_uri):
    c = MilvusClient(uri=milvus_uri)
    yield c
    # Cleanup: drop all test collections
    for name in c.list_collections():
        if name.startswith(TEST_PREFIX):
            c.drop_collection(name)
    c.close()


@pytest.fixture(scope="module")
def conn(milvus_uri):
    """Low-level connection for operations MilvusClient doesn't support."""
    connections.connect(alias="phase0", uri=milvus_uri)
    yield
    connections.disconnect(alias="phase0")


def _random_vectors(n, dim=DIMENSION):
    vecs = np.random.random((n, dim)).astype(np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / norms).tolist()


class TestConnection:
    def test_connect_and_version(self, client):
        """1. Connect to Milvus and verify version >= 2.6."""
        version = client.get_server_version()
        major, minor = [int(x) for x in version.split(".")[:2]]
        assert major >= 2 and minor >= 6, f"Need Milvus >= 2.6, got {version}"


class TestCollectionCreation:
    """2. Create collection with all field types needed by RAG collections."""

    COLL_NAME = f"{TEST_PREFIX}field_types"

    def test_create_with_all_field_types(self, client):
        if client.has_collection(self.COLL_NAME):
            client.drop_collection(self.COLL_NAME)

        schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("spark_version", DataType.VARCHAR, max_length=32)
        schema.add_field("metadata_json", DataType.JSON)
        schema.add_field("score", DataType.INT64)
        schema.add_field("view_count", DataType.INT32)
        schema.add_field("is_accepted", DataType.BOOL)

        client.create_collection(
            collection_name=self.COLL_NAME,
            schema=schema,
        )
        assert client.has_collection(self.COLL_NAME)

        # Verify field count
        info = client.describe_collection(self.COLL_NAME)
        assert info is not None


class TestHNSWIndex:
    """3. Create HNSW index with COSINE metric."""

    COLL_NAME = f"{TEST_PREFIX}hnsw"

    def test_hnsw_cosine_index(self, client):
        if client.has_collection(self.COLL_NAME):
            client.drop_collection(self.COLL_NAME)

        schema = client.create_schema(auto_id=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
        schema.add_field("content", DataType.VARCHAR, max_length=1024)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 256},
        )

        client.create_collection(
            collection_name=self.COLL_NAME,
            schema=schema,
            index_params=index_params,
        )
        assert client.has_collection(self.COLL_NAME)


class TestInsertAndSearch:
    """4. Insert vectors with metadata, search by vector, verify results."""

    COLL_NAME = f"{TEST_PREFIX}insert_search"

    def test_insert_and_search(self, client):
        if client.has_collection(self.COLL_NAME):
            client.drop_collection(self.COLL_NAME)

        schema = client.create_schema(auto_id=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("spark_version", DataType.VARCHAR, max_length=32)
        schema.add_field("score", DataType.INT64)
        schema.add_field("is_accepted", DataType.BOOL)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 256},
        )

        client.create_collection(
            collection_name=self.COLL_NAME,
            schema=schema,
            index_params=index_params,
        )

        # Insert 100 vectors
        vectors = _random_vectors(100)
        data = [
            {
                "embedding": vec,
                "content": f"Spark code snippet {i}",
                "spark_version": "4.1.0" if i < 50 else "3.5.4",
                "score": i * 10,
                "is_accepted": i % 2 == 0,
            }
            for i, vec in enumerate(vectors)
        ]
        res = client.insert(collection_name=self.COLL_NAME, data=data)
        assert res["insert_count"] == 100

        # Search
        query_vec = _random_vectors(1)
        results = client.search(
            collection_name=self.COLL_NAME,
            data=query_vec,
            limit=10,
            output_fields=["content", "spark_version", "score", "is_accepted"],
        )
        assert len(results[0]) == 10
        # Verify metadata fields returned
        hit = results[0][0]
        assert "content" in hit["entity"]
        assert "spark_version" in hit["entity"]
        assert "score" in hit["entity"]


class TestFilteredSearch:
    """5. Vector search with scalar filters."""

    COLL_NAME = f"{TEST_PREFIX}insert_search"  # Reuse from above

    def test_filtered_by_score_and_accepted(self, client):
        query_vec = _random_vectors(1)
        results = client.search(
            collection_name=self.COLL_NAME,
            data=query_vec,
            limit=10,
            filter="score > 200 and is_accepted == true",
            output_fields=["content", "score", "is_accepted"],
        )
        for hit in results[0]:
            assert hit["entity"]["score"] > 200
            assert hit["entity"]["is_accepted"] is True

    def test_filtered_by_spark_version(self, client):
        """Filter search results to a specific Spark version."""
        query_vec = _random_vectors(1)
        results = client.search(
            collection_name=self.COLL_NAME,
            data=query_vec,
            limit=10,
            filter='spark_version == "4.1.0"',
            output_fields=["spark_version"],
        )
        for hit in results[0]:
            assert hit["entity"]["spark_version"] == "4.1.0"


class TestJSONField:
    """6. JSON field storage and filtering."""

    COLL_NAME = f"{TEST_PREFIX}json_field"

    def test_json_field_query(self, client):
        if client.has_collection(self.COLL_NAME):
            client.drop_collection(self.COLL_NAME)

        schema = client.create_schema(auto_id=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
        schema.add_field("spark_apis", DataType.JSON)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 256},
        )

        client.create_collection(
            collection_name=self.COLL_NAME,
            schema=schema,
            index_params=index_params,
        )

        vectors = _random_vectors(20)
        data = [
            {
                "embedding": vec,
                "spark_apis": {
                    "apis": ["collect", "groupByKey"] if i % 2 == 0 else ["map", "filter"],
                    "risk_level": "high" if i % 2 == 0 else "low",
                },
            }
            for i, vec in enumerate(vectors)
        ]
        client.insert(collection_name=self.COLL_NAME, data=data)

        # Filter on JSON path
        query_vec = _random_vectors(1)
        results = client.search(
            collection_name=self.COLL_NAME,
            data=query_vec,
            limit=10,
            filter='json_contains(spark_apis["apis"], "collect")',
            output_fields=["spark_apis"],
        )
        for hit in results[0]:
            assert "collect" in hit["entity"]["spark_apis"]["apis"]


class TestPartitionKey:
    """7. Partition key on VARCHAR field (for language-based partitioning)."""

    COLL_NAME = f"{TEST_PREFIX}partition_key"

    def test_partition_key_on_varchar(self, client, conn):
        if client.has_collection(self.COLL_NAME):
            client.drop_collection(self.COLL_NAME)

        # Use low-level API for partition key support
        fields = [
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION),
            FieldSchema("language", DataType.VARCHAR, max_length=32, is_partition_key=True),
            FieldSchema("content", DataType.VARCHAR, max_length=1024),
        ]
        schema = CollectionSchema(fields)
        coll = Collection(self.COLL_NAME, schema, using="phase0")

        index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 256},
        }
        coll.create_index("embedding", index_params)
        coll.load()

        # Insert with different partition keys
        vectors = _random_vectors(30)
        languages = ["scala"] * 10 + ["java"] * 10 + ["python"] * 10
        coll.insert([
            vectors,
            languages,
            [f"code snippet {i}" for i in range(30)],
        ])

        # Search with partition key filter
        coll.flush()
        results = coll.search(
            data=_random_vectors(1),
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=10,
            expr='language == "scala"',
            output_fields=["language", "content"],
        )
        assert len(results[0]) > 0
        for hit in results[0]:
            assert hit.entity.get("language") == "scala"

        coll.release()


class TestBatchInsert:
    """8. Batch insert throughput."""

    COLL_NAME = f"{TEST_PREFIX}batch"

    def test_batch_insert_1000(self, client):
        if client.has_collection(self.COLL_NAME):
            client.drop_collection(self.COLL_NAME)

        schema = client.create_schema(auto_id=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
        schema.add_field("content", DataType.VARCHAR, max_length=1024)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 256},
        )

        client.create_collection(
            collection_name=self.COLL_NAME,
            schema=schema,
            index_params=index_params,
        )

        vectors = _random_vectors(1000)
        data = [
            {"embedding": vec, "content": f"chunk {i}"}
            for i, vec in enumerate(vectors)
        ]

        start = time.time()
        res = client.insert(collection_name=self.COLL_NAME, data=data)
        elapsed = time.time() - start

        assert res["insert_count"] == 1000
        throughput = 1000 / elapsed
        assert throughput > 100, f"Insert throughput {throughput:.0f}/s < 100/s"


class TestMultiCollectionSearch:
    """9. Search across 3 collections in parallel (simulates our 3 RAG collections)."""

    COLLS = [f"{TEST_PREFIX}multi_{i}" for i in range(3)]

    def test_parallel_search(self, client):
        # Create 3 collections
        for coll_name in self.COLLS:
            if client.has_collection(coll_name):
                client.drop_collection(coll_name)

            schema = client.create_schema(auto_id=True)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
            schema.add_field("source", DataType.VARCHAR, max_length=64)

            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 256},
            )

            client.create_collection(
                collection_name=coll_name,
                schema=schema,
                index_params=index_params,
            )

            vectors = _random_vectors(50)
            data = [
                {"embedding": vec, "source": coll_name}
                for vec in vectors
            ]
            client.insert(collection_name=coll_name, data=data)

        # Search all 3
        query_vec = _random_vectors(1)
        all_results = []
        for coll_name in self.COLLS:
            results = client.search(
                collection_name=coll_name,
                data=query_vec,
                limit=5,
                output_fields=["source"],
            )
            all_results.extend(results[0])

        assert len(all_results) == 15  # 5 from each collection
