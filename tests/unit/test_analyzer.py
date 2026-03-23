"""Unit tests for the query pipeline analyzer."""

from spark_rag.api.analyzer import (
    _build_queries,
    detect_input_type,
)
from spark_rag.chunking.spark_patterns import PatternResult, SparkAPIUsage


class TestInputTypeDetection:
    def test_stacktrace_is_logs(self):
        text = """
java.lang.OutOfMemoryError: Java heap space
    at org.apache.spark.sql.execution.SparkPlan.execute(SparkPlan.scala:120)
    at org.apache.spark.sql.Dataset.collect(Dataset.scala:1234)
"""
        assert detect_input_type(text) == "logs"

    def test_python_traceback_is_logs(self):
        text = """
Traceback (most recent call last):
  File "job.py", line 10, in <module>
    df.collect()
pyspark.errors.PySparkException: Task failed
"""
        assert detect_input_type(text) == "logs"

    def test_spark_code_detected(self):
        text = """
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("test").getOrCreate()
df = spark.read.parquet("data.parquet")
result = df.select("col1").filter("col1 > 0")
result.show()
"""
        assert detect_input_type(text) == "code"

    def test_scala_code_detected(self):
        text = """
val spark = SparkSession.builder.appName("test").getOrCreate()
val df = spark.read.parquet("data")
df.select("col1").filter($"col1" > 0).show()
"""
        assert detect_input_type(text) == "code"

    def test_question_detected(self):
        text = "How do I fix OutOfMemoryError when using groupByKey on a large dataset?"
        assert detect_input_type(text) == "question"

    def test_short_question(self):
        text = "What is the default value for spark.sql.shuffle.partitions?"
        assert detect_input_type(text) == "question"


class TestQueryBuilding:
    def test_question_single_query(self):
        result = PatternResult()
        queries = _build_queries("How does Catalyst work?", "question", result)
        assert len(queries) == 1
        assert queries[0] == "How does Catalyst work?"

    def test_code_adds_api_query(self):
        result = PatternResult(
            apis=[
                SparkAPIUsage(api="DataFrame.collect", line=1),
                SparkAPIUsage(api="DataFrame.groupBy", line=2),
            ]
        )
        queries = _build_queries("df.groupBy('key').collect()", "code", result)
        assert len(queries) == 2
        assert "DataFrame.collect" in queries[1]
        assert "DataFrame.groupBy" in queries[1]

    def test_logs_adds_error_query(self):
        text = "java.lang.OutOfMemoryError: GC overhead limit"
        result = PatternResult()
        queries = _build_queries(text, "logs", result)
        assert len(queries) == 2
        assert "OutOfMemoryError" in queries[1]

    def test_logs_no_error_single_query(self):
        text = "Task failed with unknown error"
        result = PatternResult()
        queries = _build_queries(text, "logs", result)
        assert len(queries) == 1
