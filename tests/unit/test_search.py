"""Unit tests for Milvus search and re-ranking logic."""

from spark_rag.milvus.search import (
    COLLECTION_WEIGHTS,
    SearchHit,
    SearchResult,
    _build_version_filter,
    _rerank_hit,
)


class TestVersionFilter:
    def test_code_exact_match(self):
        f = _build_version_filter("spark_code", "4.1.0")
        assert f == 'spark_version == "4.1.0"'

    def test_docs_exact_match(self):
        f = _build_version_filter("spark_docs", "3.5.4")
        assert f == 'spark_version == "3.5.4"'

    def test_so_no_filter(self):
        f = _build_version_filter("spark_stackoverflow", "4.1.0")
        assert f == ""

    def test_issues_no_filter(self):
        f = _build_version_filter("spark_issues", "4.1.0")
        assert f == ""

    def test_all_version_no_filter(self):
        f = _build_version_filter("spark_code", "all")
        assert f == ""

    def test_none_version_no_filter(self):
        f = _build_version_filter("spark_code", None)
        assert f == ""


class TestReranking:
    def test_base_score_uses_weight(self):
        hit = {"distance": 0.8, "entity": {}}
        score = _rerank_hit(hit, "spark_code", "code")
        expected = 0.8 * COLLECTION_WEIGHTS["code"]["spark_code"]
        assert abs(score - expected) < 0.001

    def test_api_overlap_boost(self):
        hit = {
            "distance": 0.8,
            "entity": {"spark_apis": {"apis": ["DataFrame.collect", "DataFrame.select"]}},
        }
        score_no_overlap = _rerank_hit(hit, "spark_code", "code", api_overlap=set())
        score_with_overlap = _rerank_hit(
            hit, "spark_code", "code",
            api_overlap={"DataFrame.collect", "DataFrame.join"},
        )
        assert score_with_overlap > score_no_overlap

    def test_so_accepted_answer_boost(self):
        base_hit = {"distance": 0.8, "entity": {"score": 5, "is_accepted": False}}
        accepted_hit = {"distance": 0.8, "entity": {"score": 5, "is_accepted": True}}
        base_score = _rerank_hit(base_hit, "spark_stackoverflow", "logs")
        accepted_score = _rerank_hit(accepted_hit, "spark_stackoverflow", "logs")
        assert accepted_score > base_score

    def test_so_high_score_boost(self):
        low_hit = {"distance": 0.8, "entity": {"score": 2, "is_accepted": False}}
        high_hit = {"distance": 0.8, "entity": {"score": 50, "is_accepted": False}}
        low_score = _rerank_hit(low_hit, "spark_stackoverflow", "logs")
        high_score = _rerank_hit(high_hit, "spark_stackoverflow", "logs")
        assert high_score > low_score

    def test_closed_issue_boost(self):
        open_hit = {"distance": 0.8, "entity": {"state": "open"}}
        closed_hit = {"distance": 0.8, "entity": {"state": "closed"}}
        open_score = _rerank_hit(open_hit, "spark_issues", "logs")
        closed_score = _rerank_hit(closed_hit, "spark_issues", "logs")
        assert closed_score > open_score

    def test_version_mention_boost(self):
        no_ver = {"distance": 0.8, "entity": {"spark_versions_mentioned": {"versions": []}}}
        has_ver = {"distance": 0.8, "entity": {"spark_versions_mentioned": {"versions": ["4.1.0"]}}}
        no_score = _rerank_hit(no_ver, "spark_stackoverflow", "question", target_version="4.1.0")
        ver_score = _rerank_hit(has_ver, "spark_stackoverflow", "question", target_version="4.1.0")
        assert ver_score > no_score

    def test_input_type_changes_weights(self):
        hit = {"distance": 0.8, "entity": {}}
        code_score = _rerank_hit(hit, "spark_docs", "code")
        question_score = _rerank_hit(hit, "spark_docs", "question")
        # Docs weighted higher for questions than code
        assert question_score > code_score


class TestSearchResult:
    def test_by_source(self):
        result = SearchResult(
            hits=[
                SearchHit(source="spark_code", content="a", score=0.9, vector_similarity=0.9),
                SearchHit(source="spark_docs", content="b", score=0.8, vector_similarity=0.8),
                SearchHit(source="spark_code", content="c", score=0.7, vector_similarity=0.7),
            ],
            query_text="test",
            input_type="question",
            version=None,
        )
        code_hits = result.by_source("spark_code")
        assert len(code_hits) == 2
        docs_hits = result.by_source("spark_docs")
        assert len(docs_hits) == 1
