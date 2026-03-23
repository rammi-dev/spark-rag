# spark-rag

RAG-based Spark expert system. Ask questions about Apache Spark — get answers grounded in source code, official docs, StackOverflow, and GitHub issues.

## What It Does

- **Answer Spark questions** — "How does Catalyst optimize joins?" → relevant source code + doc sections + community answers
- **Troubleshoot from logs** — paste a Spark stacktrace → matching GitHub issues, SO solutions, relevant docs
- **Analyze Spark code** — submit code → detect anti-patterns (collect on large, groupByKey, etc.) + related references
- **Compare across versions** — "How did shuffle behavior change between 3.5 and 4.1?" → version-filtered results

## Knowledge Sources

| Source | What's Indexed | Update Strategy |
|---|---|---|
| **Spark source code** | `sql/`, `core/`, `mllib/`, `python/pyspark/` — AST-parsed into methods, classes, imports | One-time CLI per version |
| **Spark documentation** | spark.apache.org/docs — prose, code examples, config tables | One-time CLI per version |
| **StackOverflow** | `apache-spark` tag, answered posts only — questions + answers | Airflow weekly incremental |
| **GitHub Issues** | apache/spark, all states (open + closed) + comments | Airflow daily incremental |

Multi-version: code and docs are ingested per Spark version (4.1, 3.5, etc.). Queries can target a specific version or search across all.

## How It Works

```
User input (question / code / logs)
  → detect input type
  → parse + extract patterns (tree-sitter AST for code, error extraction for logs)
  → embed via Ollama (nomic-embed-text, 768d)
  → search 4 Milvus collections in parallel
  → re-rank (similarity + API overlap + metadata signals)
  → return patterns + ranked references
  → optionally: send top results to Claude API for written analysis
```

Two modes:
- **Retrieval only** (default) — no API key needed, returns detected patterns + ranked references
- **Retrieval + Claude synthesis** — same retrieval, plus a written analysis with root cause and recommendations

## Quick Start

```bash
# Setup
uv venv && uv sync

# Port-forward infrastructure (from K8s cluster)
kubectl -n milvus port-forward svc/milvus 19530:19530 &
kubectl -n ollama port-forward svc/ollama 11434:11434 &

# One-time ingestion (start with docs — fastest)
uv run python -m spark_rag.ingestion.docs --version 4.1.0
uv run python -m spark_rag.ingestion.code --version 4.1.0

# Run API
uv run uvicorn spark_rag.api.main:app --reload

# Query
curl localhost:8000/analyze -d '{"input": "Why does my Spark job OOM on groupByKey?"}'
```

## Infrastructure

Runs on a local K8s cluster (minikube, 3 Hyper-V nodes) managed by the [playground](file:///mnt/c/Work/playground) repo.

| Component | Namespace | Role |
|---|---|---|
| Milvus | `milvus` | Vector store — 4 collections, HNSW index, COSINE metric |
| Ollama | `ollama` | Embedding model — nomic-embed-text (768 dims) |
| Airflow | `airflow` | Periodic ingestion — SO + GitHub Issues incremental sync |
| Ceph | `rook-ceph` | S3 storage — raw scraped data + Milvus backend |

## Project Structure

```
src/spark_rag/
  config.py            — config.yaml + env vars
  chunking/            — tree-sitter AST (code), beautifulsoup4 (docs/SO/issues)
  embedding/           — Ollama nomic-embed-text client
  milvus/              — collection schemas, ingest, search + re-rank
  synthesis/           — swappable LLM provider (claude / noop)
  ingestion/           — CLI scripts (code, docs) + modules for SO, issues
  api/                 — FastAPI service + query pipeline
airflow/dags/          — 2 Airflow DAGs (SO + issues incremental sync)
deployments/           — K8s manifests
docs/
  architecture.md      — full architecture with Mermaid diagrams
  milvus-collections.md — collection schemas + example queries
```

## Docs

- [Architecture](docs/architecture.md) — system design, ingestion flows, query pipeline, deployment topology
- [Milvus Collections](docs/milvus-collections.md) — schemas, indexes, example queries, lifecycle

## Phase 2: LlamaIndex Migration

Phase 1 (current) uses custom code with direct pymilvus, tree-sitter, and beautifulsoup4 — built for learning and full control over every piece of the RAG pipeline.

Phase 2 will replace the custom plumbing with [LlamaIndex](https://docs.llamaindex.ai/), which is purpose-built for RAG:

| Component | Phase 1 (Custom) | Phase 2 (LlamaIndex) |
|---|---|---|
| Code chunking | Custom tree-sitter parser | `CodeSplitter` (tree-sitter under the hood) |
| Doc/SO/Issue chunking | Custom beautifulsoup4 | Custom `NodeParser` subclasses |
| Embedding client | Direct Ollama HTTP | `OllamaEmbedding` |
| Milvus operations | Direct pymilvus | `MilvusVectorStore` |
| Multi-collection search | Custom parallel search + merge | `QueryFusionRetriever` (reciprocal rank fusion) |
| Re-ranking | Custom scoring logic | `NodePostprocessor` subclass |
| Incremental sync | Custom high-water mark | `IngestionPipeline` (built-in hash dedup) |
| Claude synthesis | Direct anthropic SDK | `Anthropic` LLM + `ResponseSynthesizer` |

Migration reuses the same Milvus collections, config, and API layer — only the pipeline internals change. The test suite validates both phases against the same expected behavior.
