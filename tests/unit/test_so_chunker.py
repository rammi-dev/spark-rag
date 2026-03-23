"""Unit tests for StackOverflow chunker."""

from spark_rag.chunking.so_chunker import chunk_question, _extract_versions


SAMPLE_SO_ITEM = {
    "question_id": 12345,
    "title": "OutOfMemoryError when using groupByKey in Spark 3.5",
    "body": "<p>I'm getting <code>java.lang.OutOfMemoryError</code> when calling "
            "<code>rdd.groupByKey()</code> on a large dataset in Spark 3.5.4.</p>",
    "score": 15,
    "tags": ["apache-spark", "pyspark", "out-of-memory"],
    "answers": [
        {
            "answer_id": 67890,
            "body": "<p>Use <code>reduceByKey</code> instead of <code>groupByKey</code>. "
                    "See the Spark 4.0 migration guide for details.</p>",
            "score": 25,
            "is_accepted": True,
        },
        {
            "answer_id": 67891,
            "body": "<p>You can also try increasing <code>spark.executor.memory</code>.</p>",
            "score": 3,
            "is_accepted": False,
        },
    ],
}


class TestSOChunker:
    def test_produces_question_and_answers(self):
        chunks = chunk_question(SAMPLE_SO_ITEM)
        assert len(chunks) == 3  # 1 question + 2 answers
        questions = [c for c in chunks if c.is_question]
        answers = [c for c in chunks if not c.is_question]
        assert len(questions) == 1
        assert len(answers) == 2

    def test_question_has_title_and_body(self):
        chunks = chunk_question(SAMPLE_SO_ITEM)
        q = [c for c in chunks if c.is_question][0]
        assert "OutOfMemoryError" in q.content
        assert "groupByKey" in q.content

    def test_question_metadata(self):
        chunks = chunk_question(SAMPLE_SO_ITEM)
        q = [c for c in chunks if c.is_question][0]
        assert q.question_id == 12345
        assert q.score == 15
        assert "apache-spark" in q.tags["tags"]

    def test_extracts_error_type(self):
        chunks = chunk_question(SAMPLE_SO_ITEM)
        q = [c for c in chunks if c.is_question][0]
        assert q.error_type == "OutOfMemoryError"

    def test_extracts_spark_apis(self):
        chunks = chunk_question(SAMPLE_SO_ITEM)
        q = [c for c in chunks if c.is_question][0]
        assert "RDD.groupByKey" in q.spark_apis_mentioned["apis"]

    def test_extracts_versions(self):
        chunks = chunk_question(SAMPLE_SO_ITEM)
        q = [c for c in chunks if c.is_question][0]
        assert "3.5.4" in q.spark_versions_mentioned["versions"] or "3.5" in q.spark_versions_mentioned["versions"]

    def test_accepted_answer_flagged(self):
        chunks = chunk_question(SAMPLE_SO_ITEM)
        accepted = [c for c in chunks if c.is_accepted]
        assert len(accepted) == 1
        assert accepted[0].score == 25

    def test_answer_extracts_version(self):
        chunks = chunk_question(SAMPLE_SO_ITEM)
        accepted = [c for c in chunks if c.is_accepted][0]
        # Answer mentions "Spark 4.0"
        assert "4.0" in accepted.spark_versions_mentioned["versions"]

    def test_to_milvus_data(self):
        chunks = chunk_question(SAMPLE_SO_ITEM)
        data = chunks[0].to_milvus_data([0.1] * 768)
        assert data["question_id"] == 12345
        assert data["is_question"] is True
        assert len(data["embedding"]) == 768


class TestVersionExtraction:
    def test_spark_version_formats(self):
        assert "3.5" in _extract_versions("using Spark 3.5")
        assert "3.5.4" in _extract_versions("Spark v3.5.4")
        assert "4.1.0" in _extract_versions("spark 4.1.0")

    def test_no_false_versions(self):
        versions = _extract_versions("I have 100.200.300 items")
        # 100.x is not a plausible Spark version
        assert len(versions) == 0

    def test_multiple_versions(self):
        versions = _extract_versions("migrating from Spark 3.5 to 4.1.0")
        assert "3.5" in versions
        assert "4.1.0" in versions
