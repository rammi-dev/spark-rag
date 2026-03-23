"""Unit tests for synthesis module."""

import asyncio

import pytest

from spark_rag.config import SynthesisConfig
from spark_rag.milvus.search import SearchHit, SearchResult
from spark_rag.synthesis import create_provider
from spark_rag.synthesis.base import SynthesisInput
from spark_rag.synthesis.claude import _build_prompt
from spark_rag.synthesis.noop import NoopSynthesis


def _make_search_result(hits=None):
    return SearchResult(
        hits=hits or [],
        query_text="test query",
        input_type="question",
        version=None,
    )


def _make_input(user_input="Why does groupByKey cause OOM?", **kwargs):
    return SynthesisInput(
        user_input=user_input,
        input_type=kwargs.get("input_type", "question"),
        detected_patterns=kwargs.get("patterns", []),
        search_result=kwargs.get("search_result", _make_search_result()),
    )


class TestNoopSynthesis:
    def test_returns_none(self):
        provider = NoopSynthesis()
        result = asyncio.run(provider.analyze(_make_input()))
        assert result is None

    def test_name(self):
        assert NoopSynthesis().name == "noop"


class TestFactory:
    def test_disabled_returns_noop(self):
        config = SynthesisConfig(enabled=False)
        provider = create_provider(config)
        assert isinstance(provider, NoopSynthesis)

    def test_enabled_claude(self):
        config = SynthesisConfig(enabled=True, provider="claude", model="claude-sonnet-4-20250514")
        provider = create_provider(config)
        assert provider.name.startswith("claude")

    def test_unknown_provider_raises(self):
        config = SynthesisConfig(enabled=True, provider="gpt-5")
        with pytest.raises(ValueError, match="Unknown synthesis provider"):
            create_provider(config)


class TestPromptBuilding:
    def test_includes_user_input(self):
        input = _make_input("df.collect() causes OOM")
        prompt = _build_prompt(input)
        assert "df.collect() causes OOM" in prompt
        assert "question" in prompt  # input_type

    def test_includes_patterns(self):
        input = _make_input(
            patterns=[
                {"name": "collect_on_large", "risk": "HIGH", "description": "Pulls all data to driver"}
            ]
        )
        prompt = _build_prompt(input)
        assert "collect_on_large" in prompt
        assert "HIGH" in prompt

    def test_includes_context_chunks(self):
        hits = [
            SearchHit(
                source="spark_code",
                content="def collect(): Array[Row]",
                score=0.9,
                vector_similarity=0.9,
                metadata={"file_path": "sql/Dataset.scala", "spark_version": "4.1.0"},
            ),
            SearchHit(
                source="spark_stackoverflow",
                content="Use take() instead of collect()",
                score=0.85,
                vector_similarity=0.85,
                metadata={"question_id": 12345},
            ),
        ]
        input = _make_input(search_result=_make_search_result(hits))
        prompt = _build_prompt(input)
        assert "Dataset.scala" in prompt
        assert "SO #12345" in prompt
        assert "def collect()" in prompt

    def test_limits_context_chunks(self):
        hits = [
            SearchHit(
                source="spark_code",
                content=f"chunk {i}",
                score=0.5,
                vector_similarity=0.5,
            )
            for i in range(30)
        ]
        input = _make_input(search_result=_make_search_result(hits))
        input.max_context_chunks = 5
        prompt = _build_prompt(input)
        # Should only include 5 chunks
        assert "chunk 0" in prompt
        assert "chunk 4" in prompt
        assert "chunk 5" not in prompt

    def test_empty_patterns_and_hits(self):
        input = _make_input(patterns=[], search_result=_make_search_result([]))
        prompt = _build_prompt(input)
        # Should still have user input and task sections
        assert "User Input" in prompt
        assert "Task" in prompt

    def test_includes_issue_metadata(self):
        hits = [
            SearchHit(
                source="spark_issues",
                content="NPE in optimizer",
                score=0.8,
                vector_similarity=0.8,
                metadata={"issue_number": 48123},
            ),
        ]
        input = _make_input(search_result=_make_search_result(hits))
        prompt = _build_prompt(input)
        assert "issue #48123" in prompt

    def test_includes_doc_metadata(self):
        hits = [
            SearchHit(
                source="spark_docs",
                content="Configure shuffle partitions",
                score=0.8,
                vector_similarity=0.8,
                metadata={"doc_url": "docs/configuration.md", "spark_version": "4.1.0"},
            ),
        ]
        input = _make_input(search_result=_make_search_result(hits))
        prompt = _build_prompt(input)
        assert "docs/configuration.md" in prompt
        assert "v4.1.0" in prompt
