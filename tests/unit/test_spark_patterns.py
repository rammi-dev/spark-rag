"""Unit tests for Spark pattern detection."""

from spark_rag.chunking.spark_patterns import (
    RiskLevel,
    analyze,
    detect_apis,
    detect_patterns,
    extract_error_types,
    is_stacktrace,
)


class TestAPIDetection:
    def test_detects_spark_session(self):
        code = 'val spark = SparkSession.builder.appName("test").getOrCreate()'
        apis = detect_apis(code)
        names = [a.api for a in apis]
        assert "SparkSession.builder" in names

    def test_detects_dataframe_ops(self):
        code = """
df = spark.read.parquet("data.parquet")
result = df.select("col1").filter("col1 > 0").groupBy("col1").agg(count("*"))
result.show()
"""
        apis = detect_apis(code)
        names = [a.api for a in apis]
        assert "DataFrame.read" in names
        assert "DataFrame.select" in names
        assert "DataFrame.filter" in names
        assert "DataFrame.groupBy" in names
        assert "DataFrame.agg" in names
        assert "DataFrame.show" in names

    def test_detects_rdd_ops(self):
        code = "rdd.map(lambda x: x*2).reduceByKey(lambda a, b: a+b)"
        apis = detect_apis(code)
        names = [a.api for a in apis]
        assert "RDD.map" in names
        assert "RDD.reduceByKey" in names

    def test_line_numbers(self):
        code = "line1\ndf.collect()\nline3"
        apis = detect_apis(code)
        collect = [a for a in apis if "collect" in a.api]
        assert collect[0].line == 2

    def test_no_false_positives_on_plain_text(self):
        code = "This is just a comment about data processing"
        apis = detect_apis(code)
        assert len(apis) == 0

    def test_deduplicates_same_api(self):
        code = "df.select('a')\ndf2.select('b')"
        apis = detect_apis(code)
        select_apis = [a for a in apis if a.api == "DataFrame.select"]
        # Same API called twice → only one entry
        assert len(select_apis) == 1


class TestProblemPatterns:
    def test_collect_detected(self):
        code = "result = df.collect()"
        patterns = detect_patterns(code)
        assert any(p.name == "collect_on_large" for p in patterns)

    def test_groupByKey_detected(self):
        code = "rdd.groupByKey().mapValues(list)"
        patterns = detect_patterns(code)
        assert any(p.name == "groupByKey" for p in patterns)
        p = [p for p in patterns if p.name == "groupByKey"][0]
        assert p.risk == RiskLevel.MEDIUM

    def test_coalesce_one_detected(self):
        code = "df.coalesce(1).write.csv('out')"
        patterns = detect_patterns(code)
        assert any(p.name == "coalesce_one" for p in patterns)

    def test_coalesce_other_not_detected(self):
        code = "df.coalesce(8).write.csv('out')"
        patterns = detect_patterns(code)
        assert not any(p.name == "coalesce_one" for p in patterns)

    def test_toPandas_detected(self):
        code = "pdf = df.toPandas()"
        patterns = detect_patterns(code)
        assert any(p.name == "toPandas_large" for p in patterns)

    def test_cache_without_unpersist(self):
        code = "df.cache()\ndf.count()"
        patterns = detect_patterns(code)
        assert any(p.name == "cache_no_unpersist" for p in patterns)

    def test_cache_with_unpersist_ok(self):
        code = "df.cache()\ndf.count()\ndf.unpersist()"
        patterns = detect_patterns(code)
        assert not any(p.name == "cache_no_unpersist" for p in patterns)

    def test_crossjoin_detected(self):
        code = "df1.crossJoin(df2)"
        patterns = detect_patterns(code)
        assert any(p.name == "crossjoin" for p in patterns)

    def test_repartition_one(self):
        code = "df.repartition(1)"
        patterns = detect_patterns(code)
        assert any(p.name == "repartition_to_one" for p in patterns)

    def test_collect_in_loop(self):
        code = "for item in items:\n    data = df.filter(col('id') == item).collect()"
        patterns = detect_patterns(code)
        assert any(p.name == "collect_in_loop" for p in patterns)

    def test_clean_code_no_patterns(self):
        code = """
df = spark.read.parquet("input")
result = df.select("a", "b").filter(col("a") > 0)
result.write.parquet("output")
"""
        patterns = detect_patterns(code)
        assert len(patterns) == 0

    def test_pattern_has_line_number(self):
        code = "line1\ndf.collect()\nline3"
        patterns = detect_patterns(code)
        p = [p for p in patterns if p.name == "collect_on_large"][0]
        assert p.line == 2


class TestAnalyze:
    def test_full_analysis(self):
        code = """
spark = SparkSession.builder.appName("test").getOrCreate()
df = spark.read.parquet("data")
result = df.groupBy("key").agg(collect_list("val"))
result.collect()
"""
        result = analyze(code)
        assert len(result.apis) > 0
        assert "SparkSession.builder" in result.api_names
        assert result.has_problems
        assert any(p.name == "collect_on_large" for p in result.patterns)

    def test_to_metadata(self):
        code = "df.collect()"
        result = analyze(code)
        meta = result.to_metadata()
        assert "apis" in meta
        assert "patterns" in meta
        assert isinstance(meta["apis"], list)
        assert meta["patterns"][0]["risk"] == "HIGH"


class TestErrorExtraction:
    def test_oom(self):
        log = "Exception: java.lang.OutOfMemoryError: Java heap space"
        errors = extract_error_types(log)
        assert "OutOfMemoryError" in errors

    def test_spark_exception(self):
        log = "org.apache.spark.SparkException: Task failed"
        errors = extract_error_types(log)
        assert "SparkException" in errors

    def test_multiple_errors(self):
        log = """
java.lang.OutOfMemoryError: GC overhead limit
Caused by: org.apache.spark.SparkException: Task not serializable
"""
        errors = extract_error_types(log)
        assert "OutOfMemoryError" in errors
        assert "SparkException" in errors

    def test_no_errors_in_clean_text(self):
        assert extract_error_types("Everything is fine") == []

    def test_fetch_failed(self):
        log = "FetchFailedException: Failed to connect to host"
        errors = extract_error_types(log)
        assert "FetchFailedException" in errors


class TestStacktraceDetection:
    def test_java_stacktrace(self):
        text = """
java.lang.OutOfMemoryError: Java heap space
    at org.apache.spark.sql.execution.SparkPlan.execute(SparkPlan.scala:120)
    at org.apache.spark.sql.Dataset.collect(Dataset.scala:1234)
"""
        assert is_stacktrace(text) is True

    def test_python_traceback(self):
        text = """
Traceback (most recent call last):
  File "main.py", line 10, in <module>
    df.collect()
"""
        assert is_stacktrace(text) is True

    def test_caused_by(self):
        text = "Something failed\nCaused by: java.lang.NullPointerException"
        assert is_stacktrace(text) is True

    def test_plain_text_not_stacktrace(self):
        text = "How do I fix a Spark OOM error when using groupByKey?"
        assert is_stacktrace(text) is False
