# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

RAG-based Spark expert system. Indexes 4 knowledge sources to build a comprehensive Spark knowledge base:
- **Spark source code** — multi-version (4.1, 3.5, etc.), AST-parsed
- **Spark documentation** — multi-version, structured by sections
- **StackOverflow** — general Spark questions, answered posts only
- **GitHub Issues** — apache/spark repo, all issues (open + closed) with comments

Use cases:
- Answer questions about Spark internals, APIs, and configuration
- Troubleshoot Spark problems from logs and error messages
- Analyze user Spark code for issues and anti-patterns
- Compare behavior across Spark versions (e.g. 4.1 vs 3.5)

Two modes: retrieval-only (default, no API key needed) and retrieval + Claude synthesis (requires ANTHROPIC_API_KEY).

## Infrastructure Dependencies

Runs on the K8s cluster managed by `/mnt/c/Work/playground`.

### Required components
- **Milvus** (`milvus` ns) — vector store (4 collections: spark_code, spark_docs, spark_stackoverflow, spark_issues)
- **Ollama** (`ollama` ns) — embedding model (nomic-embed-text, 768 dims)
- **Airflow** (`airflow` ns) — orchestrates ingestion DAGs (4 DAGs)
- **Ceph** (`rook-ceph` ns) — S3 for raw data bucket + Milvus storage backend

To activate missing components:
```bash
/mnt/c/Work/playground/components/<name>/scripts/build.sh
```

## Build and Run

```bash
uv venv && uv sync

# Run API server (retrieval only)
uv run uvicorn spark_rag.api.main:app --reload

# With Claude synthesis
ANTHROPIC_API_KEY=sk-... uv run uvicorn spark_rag.api.main:app --reload

# Run tests
uv run pytest                              # all tests
uv run pytest tests/unit/                  # unit only (no infra needed)
uv run pytest tests/infra/                 # Milvus + Airflow validation
uv run pytest tests/integration/           # needs Ollama + Milvus port-forwarded
```

### Port-forwarding (for local dev)
```bash
kubectl -n milvus port-forward svc/milvus 19530:19530 &
kubectl -n milvus port-forward svc/milvus-attu 3001:3000 &
kubectl -n ollama port-forward svc/ollama 11434:11434 &
kubectl -n airflow port-forward svc/airflow-api-server 8080:8080 &
```

## Project Structure

```
src/spark_rag/
  config.py          — load config.yaml + env vars
  chunking/          — tree-sitter AST (code), beautifulsoup4 (docs/SO/issues)
  embedding/         — Ollama nomic-embed-text client
  milvus/            — collection schemas, ingest, search + re-rank
  synthesis/         — swappable LLM provider (claude/noop)
  ingestion/         — GitHub clone, docs scrape, SO API, GitHub Issues API
  api/               — FastAPI service + query pipeline
airflow/dags/        — 4 ingestion DAGs (KubernetesPodOperator)
deployments/         — K8s manifests for API service
config.yaml          — project config (endpoints, synthesis toggle, versions, sources)
```

## 4 Milvus Collections

| Collection | Source | Version-tagged | Key metadata |
|---|---|---|---|
| spark_code | GitHub apache/spark source | Yes (per ingested version) | file_path, language, chunk_type, qualified_name, spark_apis |
| spark_docs | spark.apache.org/docs | Yes (per ingested version) | doc_url, doc_section, heading_hierarchy, content_type |
| spark_stackoverflow | StackOverflow API | Cross-version (extracted when detectable) | question_id, score, is_accepted, tags, error_type |
| spark_issues | GitHub Issues API | Cross-version (labels/milestone) | issue_number, state, labels, author, is_comment, created_at |

## Key Design Decisions
- Code + docs have `spark_version` field — ingested per version, queryable by version
- SO + Issues are cross-version — `spark_version` extracted from content/labels when detectable
- Queries accept optional `version` filter; default is baseline (4.1.0)
- Ingestion DAGs accept version parameter for code/docs; SO/issues ingest everything
- tree-sitter for code chunking (Scala/Java/Python AST), beautifulsoup4 for docs/SO/issues
- GitHub Issues include comments (each comment = separate chunk linked to parent issue)
- SO filtered to answered posts only (has accepted answer or answer score > 0)

## Airflow Notes
- DAGs PVC is mounted on the **dag-processor** pod, not the scheduler
- Deploy DAGs via: `kubectl cp` to the dag-processor pod's `/opt/airflow/dags/`
- Airflow 3.x API uses JWT auth: `POST /auth/token` with admin:admin
