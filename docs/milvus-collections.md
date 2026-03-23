# Milvus Collections Reference

## Common Config

All 4 collections:
- **Index**: HNSW (M=16, efConstruction=256)
- **Metric**: COSINE
- **Embedding**: 768 dims (nomic-embed-text)
- **Search params**: ef=64

Total: ~500MB across all collections.

---

## 1. spark_code (~50K vectors per version)

Spark source, tree-sitter AST parsed. **Ingested per version.**

| Field | Type | Notes |
|---|---|---|
| `id` | INT64 (PK, auto) | |
| `embedding` | FLOAT_VECTOR(768) | |
| `content` | VARCHAR(65535) | Source code text |
| `spark_version` | VARCHAR(32) | `"4.1.0"`, `"3.5.4"` |
| `file_path` | VARCHAR(512) | Relative path in repo |
| `language` | VARCHAR(32) | `scala`/`java`/`python` — **partition key** |
| `chunk_type` | VARCHAR(32) | `method`/`class_summary`/`imports` |
| `qualified_name` | VARCHAR(512) | `o.a.s.sql.DataFrame.select` |
| `signature` | VARCHAR(1024) | Method signature |
| `spark_apis` | JSON | `{"apis": ["SparkSession.builder"]}` |
| `problem_indicators` | JSON | `{"patterns": [{"name": "collect_on_large", "risk": "HIGH"}]}` |

---

## 2. spark_docs (~1K vectors per version)

Official docs from spark.apache.org. **Ingested per version.**

| Field | Type | Notes |
|---|---|---|
| `id` | INT64 (PK, auto) | |
| `embedding` | FLOAT_VECTOR(768) | |
| `content` | VARCHAR(65535) | Doc text |
| `spark_version` | VARCHAR(32) | |
| `doc_url` | VARCHAR(512) | Source URL |
| `doc_section` | VARCHAR(256) | Top-level section |
| `heading_hierarchy` | JSON | `["Spark SQL Guide", "Data Sources", "Parquet"]` |
| `content_type` | VARCHAR(32) | `prose`/`code_example`/`config_table` |
| `related_configs` | JSON | `{"configs": ["spark.sql.shuffle.partitions"]}` |

---

## 3. spark_stackoverflow (~20K vectors)

SO questions + answers, `apache-spark` tag, answered only. **Cross-version, Airflow incremental sync.**

| Field | Type | Notes |
|---|---|---|
| `id` | INT64 (PK, auto) | |
| `embedding` | FLOAT_VECTOR(768) | |
| `content` | VARCHAR(65535) | Question or answer text |
| `question_id` | INT64 | SO question ID |
| `is_question` | BOOL | true=question, false=answer |
| `score` | INT64 | Vote score |
| `is_accepted` | BOOL | Accepted answer |
| `tags` | JSON | `["apache-spark", "pyspark"]` |
| `error_type` | VARCHAR(256) | Exception class if present |
| `spark_apis_mentioned` | JSON | `{"apis": ["groupByKey"]}` |
| `spark_versions_mentioned` | JSON | `["3.5", "4.0"]` |

**Dedup**: upsert by `question_id` + chunk index. Updated posts replaced on sync.

---

## 4. spark_issues (~50K vectors)

GitHub Issues, apache/spark, all states + comments. **Cross-version, Airflow incremental sync.**

| Field | Type | Notes |
|---|---|---|
| `id` | INT64 (PK, auto) | |
| `embedding` | FLOAT_VECTOR(768) | |
| `content` | VARCHAR(65535) | Issue body or comment |
| `issue_number` | INT64 | GitHub issue number |
| `state` | VARCHAR(16) | `open`/`closed` |
| `is_comment` | BOOL | true=comment, false=issue body |
| `parent_issue_number` | INT64 | Links comment to issue |
| `author` | VARCHAR(128) | GitHub username |
| `labels` | JSON | `["bug", "SQL"]` |
| `created_at` | VARCHAR(32) | ISO 8601 |
| `closed_at` | VARCHAR(32) | ISO 8601 or `""` |
| `spark_versions_mentioned` | JSON | `["4.0.0"]` |
| `linked_prs` | JSON | `[12345]` |

**Dedup**: upsert by `issue_number` + `is_comment` + comment index. Edited issues replaced, new comments added.

---

## Incremental Sync

SO and issues use a high-water mark (last sync timestamp) stored in Ceph S3 at `s3://spark-rag-raw/sync-state/{source}.json`.

Each Airflow run:
1. Read high-water mark
2. Fetch items modified after that timestamp
3. Delete existing vectors with matching source IDs
4. Insert new/updated chunks
5. Update high-water mark

## Version Lifecycle

```bash
# Add a version — no schema change needed
uv run python -m spark_rag.ingestion.code --version 3.5.4
uv run python -m spark_rag.ingestion.docs --version 3.5.4

# Re-ingest a version — deletes old, inserts fresh
uv run python -m spark_rag.ingestion.code --version 4.1.0 --replace
```
