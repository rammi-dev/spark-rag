"""Unit tests for Milvus ingest module."""

import pytest

from spark_rag.milvus.ingest import batch_insert, ingest_version


class TestBatchInsert:
    def test_rejects_unknown_collection(self):
        with pytest.raises(ValueError, match="Unknown collection"):
            batch_insert(None, "nonexistent", [{"x": 1}])

    def test_empty_data_returns_zero(self):
        # Should not call client at all
        assert batch_insert(None, "spark_code", []) == 0


class TestIngestVersion:
    def test_rejects_non_versioned_collection(self):
        with pytest.raises(ValueError, match="Version-based ingestion only"):
            ingest_version(None, "spark_stackoverflow", [], "4.1.0")

    def test_rejects_issues_collection(self):
        with pytest.raises(ValueError, match="Version-based ingestion only"):
            ingest_version(None, "spark_issues", [], "4.1.0")
