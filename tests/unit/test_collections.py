"""Unit tests for Milvus collection schema definitions."""

import pytest
from pymilvus import DataType

from spark_rag.milvus.collections import (
    COLLECTION_NAMES,
    DIMENSION,
    SCHEMAS,
    spark_code_schema,
    spark_docs_schema,
    spark_issues_schema,
    spark_stackoverflow_schema,
)


def _field_map(schema):
    """Return {field_name: FieldSchema} from a CollectionSchema."""
    return {f.name: f for f in schema.fields}


class TestSparkCodeSchema:
    def test_has_required_fields(self):
        fields = _field_map(spark_code_schema())
        assert "id" in fields
        assert "embedding" in fields
        assert "content" in fields
        assert "spark_version" in fields
        assert "file_path" in fields
        assert "language" in fields
        assert "chunk_type" in fields
        assert "qualified_name" in fields
        assert "signature" in fields
        assert "spark_apis" in fields
        assert "problem_indicators" in fields

    def test_embedding_dimension(self):
        fields = _field_map(spark_code_schema())
        assert fields["embedding"].params["dim"] == DIMENSION

    def test_language_is_partition_key(self):
        fields = _field_map(spark_code_schema())
        assert fields["language"].is_partition_key is True

    def test_id_is_auto_pk(self):
        fields = _field_map(spark_code_schema())
        assert fields["id"].is_primary is True
        assert fields["id"].auto_id is True

    def test_json_fields(self):
        fields = _field_map(spark_code_schema())
        assert fields["spark_apis"].dtype == DataType.JSON
        assert fields["problem_indicators"].dtype == DataType.JSON


class TestSparkDocsSchema:
    def test_has_required_fields(self):
        fields = _field_map(spark_docs_schema())
        assert "spark_version" in fields
        assert "doc_url" in fields
        assert "doc_section" in fields
        assert "heading_hierarchy" in fields
        assert "content_type" in fields
        assert "related_configs" in fields

    def test_no_partition_key(self):
        fields = _field_map(spark_docs_schema())
        for f in fields.values():
            assert not f.is_partition_key


class TestSparkStackoverflowSchema:
    def test_has_required_fields(self):
        fields = _field_map(spark_stackoverflow_schema())
        assert "question_id" in fields
        assert "is_question" in fields
        assert "score" in fields
        assert "is_accepted" in fields
        assert "tags" in fields
        assert "error_type" in fields
        assert "spark_apis_mentioned" in fields
        assert "spark_versions_mentioned" in fields

    def test_score_is_int64(self):
        fields = _field_map(spark_stackoverflow_schema())
        assert fields["score"].dtype == DataType.INT64

    def test_bool_fields(self):
        fields = _field_map(spark_stackoverflow_schema())
        assert fields["is_question"].dtype == DataType.BOOL
        assert fields["is_accepted"].dtype == DataType.BOOL


class TestSparkIssuesSchema:
    def test_has_required_fields(self):
        fields = _field_map(spark_issues_schema())
        assert "issue_number" in fields
        assert "state" in fields
        assert "is_comment" in fields
        assert "parent_issue_number" in fields
        assert "author" in fields
        assert "labels" in fields
        assert "created_at" in fields
        assert "closed_at" in fields
        assert "spark_versions_mentioned" in fields
        assert "linked_prs" in fields

    def test_issue_number_is_int64(self):
        fields = _field_map(spark_issues_schema())
        assert fields["issue_number"].dtype == DataType.INT64
        assert fields["parent_issue_number"].dtype == DataType.INT64


class TestSchemaRegistry:
    def test_all_four_collections_registered(self):
        assert len(SCHEMAS) == 4
        assert set(SCHEMAS.keys()) == set(COLLECTION_NAMES)

    def test_all_schemas_have_base_fields(self):
        for name, schema_fn in SCHEMAS.items():
            fields = _field_map(schema_fn())
            assert "id" in fields, f"{name} missing id"
            assert "embedding" in fields, f"{name} missing embedding"
            assert "content" in fields, f"{name} missing content"
            assert fields["embedding"].params["dim"] == DIMENSION, f"{name} wrong dimension"
