# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

RAG-based Spark expert system. Indexes Apache Spark source code, documentation, and StackOverflow to:
- Answer questions about Spark internals, APIs, and configuration
- Troubleshoot Spark problems from logs and error messages
- Analyze user Spark code for issues and anti-patterns
- Compare behavior across Spark versions (e.g. 4.1 vs 3.5)

Multi-version: all indexed data is tagged with `spark_version`. Baseline is Spark 4.1, but any version can be ingested and queried.

Two modes: retrieval-only (default, no API key needed) and retrieval + Claude synthesis (requires ANTHROPIC_API_KEY).

## Infrastructure Dependencies

Runs on the K8s cluster managed by `/mnt/c/Work/playground`.

### Required components
- **Milvus** (`milvus` ns) — vector store (3 collections: spark_code, spark_docs, spark_stackoverflow)
- **Ollama** (`ollama` ns) — embedding model (nomic-embed-text, 768 dims)
- **Airflow** (`airflow` ns) — orchestrates ingestion DAGs
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
uv run pytest tests/infra/                 # Milvus capability validation
uv run pytest tests/integration/           # needs Ollama + Milvus port-forwarded
```

### Port-forwarding (for local dev)
```bash
kubectl -n milvus port-forward svc/milvus 19530:19530 &
kubectl -n milvus port-forward svc/milvus-attu 3001:3000 &
kubectl -n ollama port-forward svc/ollama 11434:11434 &
```

## Project Structure

```
src/spark_rag/
  config.py          — load config.yaml + env vars
  chunking/          — tree-sitter AST (code), beautifulsoup4 (docs/SO)
  embedding/         — Ollama nomic-embed-text client
  milvus/            — collection schemas, ingest, search + re-rank
  synthesis/         — swappable LLM provider (claude/noop)
  ingestion/         — GitHub clone, docs scrape, SO API
  api/               — FastAPI service + query pipeline
airflow/dags/        — 3 ingestion DAGs (KubernetesPodOperator)
deployments/         — K8s manifests for API service
config.yaml          — project config (endpoints, synthesis toggle, versions, sources)
```

## Key Design Decisions
- All collections have `spark_version` field — every chunk is version-tagged
- Queries accept optional `version` filter; default is baseline (4.1.0)
- Ingestion DAGs accept version parameter — can ingest any Spark release
- tree-sitter for code chunking (Scala/Java/Python AST), beautifulsoup4 for docs/SO
