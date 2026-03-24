"""Tests for Lance schema converters and chunk_id stability."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from spark_rag.lance.schemas import (
    _chunk_id,
    _unwrap_list,
    code_chunks_to_table,
    doc_chunks_to_table,
    issue_chunks_to_table,
    so_chunks_to_table,
)


# --- Minimal chunk dataclass stubs (mirror the real ones) ---

@dataclass
class FakeCodeChunk:
    content: str = "def foo(): pass"
    chunk_type: str = "method"
    language: str = "python"
    file_path: str = "core/src/main.py"
    qualified_name: str = "Main.foo"
    signature: str = "def foo()"
    spark_apis: dict = None
    problem_indicators: dict = None
    start_line: int = 1
    end_line: int = 5

    def __post_init__(self):
        if self.spark_apis is None:
            self.spark_apis = {"apis": ["SparkSession.builder"]}
        if self.problem_indicators is None:
            self.problem_indicators = {"patterns": [{"name": "collect_on_large", "risk": "HIGH"}]}


@dataclass
class FakeDocChunk:
    content: str = "# Configuration\nSet spark.sql.shuffle.partitions"
    doc_url: str = "docs/configuration.md"
    doc_section: str = "Configuration"
    heading_hierarchy: list = None
    content_type: str = "prose"
    related_configs: dict = None

    def __post_init__(self):
        if self.heading_hierarchy is None:
            self.heading_hierarchy = ["Configuration", "SQL"]
        if self.related_configs is None:
            self.related_configs = {"configs": ["spark.sql.shuffle.partitions"]}


@dataclass
class FakeSOChunk:
    content: str = "How to fix OOM in Spark?"
    question_id: int = 12345
    is_question: bool = True
    score: int = 15
    is_accepted: bool = False
    tags: dict = None
    error_type: str = "OutOfMemoryError"
    spark_apis_mentioned: dict = None
    spark_versions_mentioned: dict = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {"tags": ["apache-spark", "pyspark"]}
        if self.spark_apis_mentioned is None:
            self.spark_apis_mentioned = {"apis": ["DataFrame.collect"]}
        if self.spark_versions_mentioned is None:
            self.spark_versions_mentioned = {"versions": ["3.5.4"]}


@dataclass
class FakeIssueChunk:
    content: str = "NullPointerException in shuffle"
    issue_number: int = 42000
    state: str = "closed"
    is_comment: bool = False
    parent_issue_number: int = 42000
    author: str = "user123"
    labels: dict = None
    created_at: str = "2025-01-15T10:00:00Z"
    closed_at: str = "2025-02-01T12:00:00Z"
    spark_versions_mentioned: dict = None
    linked_prs: dict = None

    def __post_init__(self):
        if self.labels is None:
            self.labels = {"labels": ["bug", "SQL"]}
        if self.spark_versions_mentioned is None:
            self.spark_versions_mentioned = {"versions": ["4.1.0"]}
        if self.linked_prs is None:
            self.linked_prs = {"prs": [42001]}


# --- Tests ---


class TestChunkId:
    def test_deterministic(self):
        """Same input always produces same chunk_id."""
        id1 = _chunk_id("test:key:1")
        id2 = _chunk_id("test:key:1")
        assert id1 == id2

    def test_different_inputs(self):
        """Different inputs produce different chunk_ids."""
        id1 = _chunk_id("test:key:1")
        id2 = _chunk_id("test:key:2")
        assert id1 != id2

    def test_uuid_format(self):
        """chunk_id is a valid UUID string."""
        import uuid
        cid = _chunk_id("test")
        uuid.UUID(cid)  # raises if invalid


class TestUnwrapList:
    def test_unwrap_apis(self):
        assert _unwrap_list({"apis": ["a", "b"]}, "apis") == ["a", "b"]

    def test_unwrap_empty_dict(self):
        assert _unwrap_list({}, "any") == []

    def test_unwrap_none(self):
        assert _unwrap_list(None, "any") == []

    def test_unwrap_tags(self):
        assert _unwrap_list({"tags": ["spark"]}, "tags") == ["spark"]


class TestCodeChunksToTable:
    def test_basic_conversion(self):
        chunks = [FakeCodeChunk(), FakeCodeChunk(qualified_name="Main.bar", start_line=10)]
        table = code_chunks_to_table(chunks, "4.1.0")
        assert isinstance(table, pa.Table)
        assert len(table) == 2
        assert "chunk_id" in table.column_names
        assert "content" in table.column_names
        assert "spark_version" in table.column_names
        assert "spark_apis" in table.column_names

    def test_chunk_id_stability(self):
        chunks = [FakeCodeChunk()]
        t1 = code_chunks_to_table(chunks, "4.1.0")
        t2 = code_chunks_to_table(chunks, "4.1.0")
        assert t1.column("chunk_id")[0].as_py() == t2.column("chunk_id")[0].as_py()

    def test_spark_apis_as_list(self):
        chunks = [FakeCodeChunk()]
        table = code_chunks_to_table(chunks, "4.1.0")
        apis = table.column("spark_apis")[0].as_py()
        assert isinstance(apis, list)
        assert "SparkSession.builder" in apis

    def test_empty_chunks(self):
        table = code_chunks_to_table([], "4.1.0")
        assert len(table) == 0


class TestDocChunksToTable:
    def test_basic_conversion(self):
        chunks = [FakeDocChunk()]
        table = doc_chunks_to_table(chunks, "4.1.0")
        assert len(table) == 1
        assert "heading_hierarchy" in table.column_names
        assert "related_configs" in table.column_names

    def test_heading_hierarchy_as_list(self):
        chunks = [FakeDocChunk()]
        table = doc_chunks_to_table(chunks, "4.1.0")
        h = table.column("heading_hierarchy")[0].as_py()
        assert h == ["Configuration", "SQL"]


class TestSOChunksToTable:
    def test_basic_conversion(self):
        chunks = [FakeSOChunk(), FakeSOChunk(is_question=False, is_accepted=True)]
        table = so_chunks_to_table(chunks)
        assert len(table) == 2
        assert "question_id" in table.column_names
        assert "tags" in table.column_names

    def test_tags_as_list(self):
        chunks = [FakeSOChunk()]
        table = so_chunks_to_table(chunks)
        tags = table.column("tags")[0].as_py()
        assert isinstance(tags, list)
        assert "apache-spark" in tags

    def test_different_chunk_ids_for_q_and_a(self):
        q = FakeSOChunk(is_question=True)
        a = FakeSOChunk(is_question=False, content="Use repartition instead")
        table = so_chunks_to_table([q, a])
        ids = table.column("chunk_id").to_pylist()
        assert ids[0] != ids[1]


class TestIssueChunksToTable:
    def test_basic_conversion(self):
        chunks = [FakeIssueChunk()]
        table = issue_chunks_to_table(chunks)
        assert len(table) == 1
        assert "issue_number" in table.column_names
        assert "linked_prs" in table.column_names

    def test_linked_prs_as_int_list(self):
        chunks = [FakeIssueChunk()]
        table = issue_chunks_to_table(chunks)
        prs = table.column("linked_prs")[0].as_py()
        assert prs == [42001]

    def test_labels_as_list(self):
        chunks = [FakeIssueChunk()]
        table = issue_chunks_to_table(chunks)
        labels = table.column("labels")[0].as_py()
        assert "bug" in labels
