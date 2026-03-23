"""Unit tests for Markdown documentation chunker."""

from spark_rag.chunking.doc_chunker import chunk_markdown


SAMPLE_MD = """\
# Spark SQL Guide

Spark SQL is a module for structured data processing.

## Data Sources

Spark SQL supports reading data from various sources.
Set spark.sql.sources.default to change the default format.

### Parquet

Parquet is a columnar format supported by many data processing systems.

```python
df = spark.read.parquet("data.parquet")
df.write.parquet("output")
```

## Configuration

<table>
<tr><th>Property</th><th>Default</th><th>Description</th></tr>
<tr><td>spark.sql.shuffle.partitions</td><td>200</td><td>Number of partitions for shuffles</td></tr>
<tr><td>spark.sql.adaptive.enabled</td><td>true</td><td>Enable adaptive query execution</td></tr>
</table>
"""

SAMPLE_WITH_INCLUDE = """\
# Quick Start

Here is an example:

{% include_example quick_start python/examples/quick_start.py %}

And some more text after the example.
"""


class TestMarkdownChunker:
    def test_splits_on_headings(self):
        chunks = chunk_markdown(SAMPLE_MD, "docs/sql-guide.md")
        assert len(chunks) > 0
        sections = set(c.doc_section for c in chunks)
        assert "Spark SQL Guide" in sections

    def test_heading_hierarchy(self):
        chunks = chunk_markdown(SAMPLE_MD, "docs/sql-guide.md")
        parquet = [c for c in chunks if "Parquet" in str(c.heading_hierarchy)]
        assert len(parquet) > 0
        h = parquet[0].heading_hierarchy
        assert "Spark SQL Guide" in h
        assert "Data Sources" in h
        assert "Parquet" in h

    def test_code_example_separate_chunk(self):
        chunks = chunk_markdown(SAMPLE_MD, "docs/sql-guide.md")
        code_chunks = [c for c in chunks if c.content_type == "code_example"]
        assert len(code_chunks) >= 1
        assert "spark.read.parquet" in code_chunks[0].content

    def test_config_table_from_html(self):
        chunks = chunk_markdown(SAMPLE_MD, "docs/sql-guide.md")
        table_chunks = [c for c in chunks if c.content_type == "config_table"]
        assert len(table_chunks) >= 1
        assert "spark.sql.shuffle.partitions" in table_chunks[0].content

    def test_extracts_config_keys(self):
        chunks = chunk_markdown(SAMPLE_MD, "docs/sql-guide.md")
        all_configs = []
        for c in chunks:
            all_configs.extend(c.related_configs.get("configs", []))
        assert "spark.sql.shuffle.partitions" in all_configs
        assert "spark.sql.sources.default" in all_configs

    def test_doc_url_is_file_path(self):
        chunks = chunk_markdown(SAMPLE_MD, "docs/sql-guide.md")
        for c in chunks:
            assert c.doc_url == "docs/sql-guide.md"

    def test_to_milvus_data(self):
        chunks = chunk_markdown(SAMPLE_MD, "docs/sql-guide.md")
        data = chunks[0].to_milvus_data("4.1.0", [0.1] * 768)
        assert data["spark_version"] == "4.1.0"
        assert data["content_type"] in ("prose", "code_example", "config_table")
        assert len(data["embedding"]) == 768

    def test_empty_content(self):
        assert chunk_markdown("", "docs/empty.md") == []
        assert chunk_markdown("   ", "docs/empty.md") == []

    def test_no_trivial_chunks(self):
        chunks = chunk_markdown(SAMPLE_MD, "docs/sql-guide.md")
        for c in chunks:
            assert len(c.content) > 10


class TestIncludeExample:
    def test_resolves_include_example(self):
        examples = {
            "quick_start": "spark = SparkSession.builder.getOrCreate()\ndf = spark.read.csv('data.csv')"
        }
        chunks = chunk_markdown(
            SAMPLE_WITH_INCLUDE, "docs/quick-start.md",
            examples_lookup=examples,
        )
        code_chunks = [c for c in chunks if c.content_type == "code_example"]
        assert len(code_chunks) >= 1
        assert "SparkSession.builder" in code_chunks[0].content

    def test_unresolved_example_placeholder(self):
        chunks = chunk_markdown(SAMPLE_WITH_INCLUDE, "docs/quick-start.md")
        # Without lookup, should produce a placeholder text in prose
        all_content = " ".join(c.content for c in chunks)
        assert "quick_start" in all_content


class TestLongContent:
    def test_splits_long_prose(self):
        # Generate long markdown section
        lines = ["# Big Section", ""]
        for i in range(100):
            lines.append(f"This is paragraph {i} with enough text to contribute to a long chunk for testing purposes.")
            lines.append("")
        md = "\n".join(lines)

        chunks = chunk_markdown(md, "docs/big.md")
        prose = [c for c in chunks if c.content_type == "prose"]
        # Should have been split
        assert len(prose) >= 1
        for c in prose:
            assert len(c.content) <= 2200  # some tolerance over MAX_CHUNK_CHARS
