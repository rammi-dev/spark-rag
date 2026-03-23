"""Unit tests for AST-aware code chunking."""

import pytest

from spark_rag.chunking.code_chunker import (
    CodeChunk,
    chunk_file,
    detect_language,
)


SAMPLE_PYTHON = '''
import os
from pyspark.sql import SparkSession

def create_session():
    return SparkSession.builder.appName("test").getOrCreate()

class DataProcessor:
    def __init__(self, spark):
        self.spark = spark

    def process(self, path):
        df = self.spark.read.parquet(path)
        return df.select("col1", "col2").filter("col1 > 0")

    def bad_collect(self):
        return self.spark.read.parquet("big").collect()
'''

SAMPLE_SCALA = '''
package org.apache.spark.sql

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext

class DataFrame {
  def select(cols: String*): DataFrame = {
    // implementation
    this
  }

  def collect(): Array[Row] = {
    // pulls everything to driver
    Array.empty
  }
}

object DataFrameUtils {
  def create(): DataFrame = new DataFrame()
}
'''

SAMPLE_JAVA = '''
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;

public class SparkJob {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder().appName("job").getOrCreate();
        Dataset df = spark.read().parquet("data");
    }

    public void process() {
        // do work
    }
}
'''


class TestPythonChunking:
    def test_extracts_imports(self):
        chunks = chunk_file(SAMPLE_PYTHON, "processor.py", "python")
        imports = [c for c in chunks if c.chunk_type == "imports"]
        assert len(imports) == 1
        assert "import os" in imports[0].content
        assert "from pyspark.sql import SparkSession" in imports[0].content

    def test_extracts_top_level_function(self):
        chunks = chunk_file(SAMPLE_PYTHON, "processor.py", "python")
        methods = [c for c in chunks if c.qualified_name == "create_session"]
        assert len(methods) == 1
        assert methods[0].chunk_type == "method"
        assert "SparkSession.builder" in methods[0].content

    def test_extracts_class_methods(self):
        chunks = chunk_file(SAMPLE_PYTHON, "processor.py", "python")
        methods = [c for c in chunks if c.chunk_type == "method" and "DataProcessor" in c.qualified_name]
        names = [c.qualified_name for c in methods]
        assert "DataProcessor.__init__" in names
        assert "DataProcessor.process" in names
        assert "DataProcessor.bad_collect" in names

    def test_class_summary(self):
        chunks = chunk_file(SAMPLE_PYTHON, "processor.py", "python")
        summaries = [c for c in chunks if c.chunk_type == "class_summary"]
        assert len(summaries) == 1
        assert summaries[0].qualified_name == "DataProcessor"
        assert "__init__" in summaries[0].content
        assert "process" in summaries[0].content

    def test_detects_spark_apis(self):
        chunks = chunk_file(SAMPLE_PYTHON, "processor.py", "python")
        process = [c for c in chunks if c.qualified_name == "DataProcessor.process"][0]
        assert "DataFrame.read" in process.spark_apis["apis"] or "DataFrame.select" in process.spark_apis["apis"]

    def test_detects_problem_patterns(self):
        chunks = chunk_file(SAMPLE_PYTHON, "processor.py", "python")
        bad = [c for c in chunks if c.qualified_name == "DataProcessor.bad_collect"][0]
        pattern_names = [p["name"] for p in bad.problem_indicators["patterns"]]
        assert "collect_on_large" in pattern_names

    def test_has_line_numbers(self):
        chunks = chunk_file(SAMPLE_PYTHON, "processor.py", "python")
        methods = [c for c in chunks if c.chunk_type == "method"]
        for m in methods:
            assert m.start_line > 0


class TestScalaChunking:
    def test_extracts_imports_and_package(self):
        chunks = chunk_file(SAMPLE_SCALA, "DataFrame.scala", "scala")
        imports = [c for c in chunks if c.chunk_type == "imports"]
        assert len(imports) == 1
        assert "package org.apache.spark.sql" in imports[0].content
        assert "import org.apache.spark.rdd.RDD" in imports[0].content

    def test_extracts_class_methods(self):
        chunks = chunk_file(SAMPLE_SCALA, "DataFrame.scala", "scala")
        methods = [c for c in chunks if c.chunk_type == "method" and "DataFrame" in c.qualified_name]
        names = [c.qualified_name for c in methods]
        assert "DataFrame.select" in names
        assert "DataFrame.collect" in names

    def test_class_and_object_summaries(self):
        chunks = chunk_file(SAMPLE_SCALA, "DataFrame.scala", "scala")
        summaries = [c for c in chunks if c.chunk_type == "class_summary"]
        names = [s.qualified_name for s in summaries]
        assert "DataFrame" in names
        assert "DataFrameUtils" in names


class TestJavaChunking:
    def test_extracts_imports(self):
        chunks = chunk_file(SAMPLE_JAVA, "SparkJob.java", "java")
        imports = [c for c in chunks if c.chunk_type == "imports"]
        assert len(imports) == 1
        assert "import org.apache.spark.sql.SparkSession" in imports[0].content

    def test_extracts_methods(self):
        chunks = chunk_file(SAMPLE_JAVA, "SparkJob.java", "java")
        methods = [c for c in chunks if c.chunk_type == "method"]
        names = [c.qualified_name for c in methods]
        assert "SparkJob.main" in names
        assert "SparkJob.process" in names

    def test_class_summary(self):
        chunks = chunk_file(SAMPLE_JAVA, "SparkJob.java", "java")
        summaries = [c for c in chunks if c.chunk_type == "class_summary"]
        assert len(summaries) == 1
        assert summaries[0].qualified_name == "SparkJob"


class TestLongChunkSplitting:
    def test_splits_long_method(self):
        # Generate a long Python function
        lines = ["def long_func():"]
        for i in range(200):
            lines.append(f"    x_{i} = {i} * 2  # padding line to make it long enough")
        code = "\n".join(lines)

        chunks = chunk_file(code, "long.py", "python")
        methods = [c for c in chunks if c.chunk_type == "method"]
        # Should be split into multiple parts
        assert len(methods) > 1
        # First part keeps the signature
        assert methods[0].signature != ""


class TestDetectLanguage:
    def test_scala(self):
        assert detect_language("src/main/scala/Foo.scala") == "scala"

    def test_java(self):
        assert detect_language("src/main/java/Foo.java") == "java"

    def test_python(self):
        assert detect_language("python/pyspark/sql/functions.py") == "python"

    def test_unknown(self):
        assert detect_language("README.md") is None
        assert detect_language("build.sbt") is None


class TestToMilvusData:
    def test_converts_to_dict(self):
        chunk = CodeChunk(
            content="def foo(): pass",
            chunk_type="method",
            language="python",
            file_path="test.py",
            qualified_name="foo",
            signature="def foo()",
        )
        data = chunk.to_milvus_data("4.1.0", [0.1] * 768)
        assert data["spark_version"] == "4.1.0"
        assert data["language"] == "python"
        assert data["chunk_type"] == "method"
        assert len(data["embedding"]) == 768
