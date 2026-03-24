# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Source Layout

`src/` layout (hatchling build backend). Application code lives in `src/spark_rag/`, not top-level. No linting or formatting tools configured (no ruff, mypy, black, or pre-commit hooks).

## Project Overview

RAG-based Spark expert system. Indexes 4 knowledge sources to build a comprehensive Spark knowledge base:
- **Spark source code** — multi-version (4.1, 3.5, etc.), AST-parsed via tree-sitter
- **Spark documentation** — multi-version, Markdown from repo `docs/*.md`
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

### Required components (all deployed and validated)
- **Milvus** (`milvus` ns) — v2.6.8 cluster, 4 collections, HNSW/COSINE, 768d vectors
- **Ollama** (`ollama` ns) — nomic-embed-text (768 dims), CPU-only
- **Airflow** (`airflow` ns) — 3.1.8, CeleryExecutor, 2 workers, KubernetesPodOperator RBAC
- **Ceph** (`rook-ceph` ns) — S3 gateway + block storage + CephFS

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
uv run pytest                              # all tests (162 unit + 21 integration + 20 infra)
uv run pytest tests/unit/                  # unit only (no infra needed)
uv run pytest tests/infra/                 # Milvus + Airflow + Ollama validation
uv run pytest tests/integration/           # needs Ollama + Milvus port-forwarded

# Single test file or function
uv run pytest tests/unit/test_config.py              # single file
uv run pytest tests/unit/test_config.py::test_name   # single test
uv run pytest -k "pattern"                            # by name pattern

# Custom Milvus URI for integration/infra tests
uv run pytest tests/integration/ --milvus-uri http://custom:19530
```

### Port-forwarding (for local dev)
```bash
kubectl -n milvus port-forward svc/milvus 19530:19530 &
kubectl -n milvus port-forward svc/milvus-attu 3001:3000 &
kubectl -n ollama port-forward svc/ollama 11434:11434 &
kubectl -n airflow port-forward svc/airflow-api-server 8080:8080 &
```

### Ingestion
```bash
# One-time: code + docs per version
uv run python -m spark_rag.ingestion.code --version 4.1.0
uv run python -m spark_rag.ingestion.docs --version 4.1.0

# Incremental: SO + issues (or let Airflow handle)
uv run python -m spark_rag.ingestion.stackoverflow
uv run python -m spark_rag.ingestion.issues --token $GITHUB_TOKEN

# Dry run (chunk + count, no embedding)
uv run python -m spark_rag.ingestion.code --version 4.1.0 --dry-run
```

## Module Architecture

```
config.py → loads config.yaml + env overrides

chunking/
  spark_patterns.py → API detection (30+ patterns), problem rules (11), error extraction
  code_chunker.py   → tree-sitter AST → method/class_summary/imports chunks
  doc_chunker.py    → Markdown headings/code blocks/HTML tables → prose/code_example/config_table
  so_chunker.py     → SO API JSON → question + answer chunks
  issue_chunker.py  → GitHub API → issue body + comment chunks

embedding/client.py → Ollama HTTP client, batching, health check

milvus/
  collections.py → 4 schemas (spark_code, spark_docs, spark_stackoverflow, spark_issues)
  ingest.py      → batch insert, version replace, incremental dedup
  search.py      → multi-collection parallel search, re-ranking (similarity × weight + boosts)

synthesis/
  base.py   → SynthesisProvider ABC
  noop.py   → returns None (retrieval-only)
  claude.py → builds structured prompt → Claude API → written analysis

ingestion/
  github.py         → git clone/checkout (sparse, blob-filter)
  code.py           → CLI: repo → tree-sitter chunk → embed → Milvus
  docs.py           → CLI: repo docs/*.md → Markdown chunk → embed → Milvus
  stackoverflow.py  → CLI: StackExchange API → chunk → embed → incremental Milvus
  issues.py         → CLI: GitHub API → chunk → embed → incremental Milvus

api/
  main.py     → FastAPI (POST /analyze, GET /health), lifespan startup
  analyzer.py → query pipeline: detect type → parse → embed → search → rerank → synthesize
```

## 4 Milvus Collections

| Collection | Source | Version-tagged | Key metadata |
|---|---|---|---|
| spark_code | GitHub apache/spark source | Yes (per version) | file_path, language (partition key), chunk_type, qualified_name, spark_apis |
| spark_docs | `docs/*.md` from repo | Yes (per version) | doc_url, doc_section, heading_hierarchy, content_type, related_configs |
| spark_stackoverflow | StackExchange API | Cross-version | question_id, score, is_accepted, tags, error_type, spark_versions_mentioned |
| spark_issues | GitHub Issues API | Cross-version | issue_number, state, labels, author, is_comment, spark_versions_mentioned, linked_prs |

## Key Design Decisions
- Code + docs ingested per version via CLI scripts (not Airflow — overkill for one-time)
- SO + Issues synced incrementally via Airflow DAGs (or CLI with `--since`)
- Docs sourced from repo Markdown (not website scrape) — includes inline HTML config tables + Jekyll include_example resolution
- tree-sitter for code chunking (Scala/Java/Python), Markdown parsing for docs, beautifulsoup4 for SO/issues
- Re-ranking: vector similarity × collection weight (varies by input type) + API overlap + SO score + accepted answer + closed issue + version mention boosts
- Synthesis prompt includes: user input + detected patterns + top 15 context chunks with source metadata

## Airflow Notes
- DAGs PVC mounted on **dag-processor** pod, not the scheduler
- Deploy DAGs via dag-processor: `kubectl exec -n airflow <dag-processor-pod> -c dag-processor -- ...`
- Airflow 3.x API uses JWT auth: `POST /auth/token` with admin:admin

## Design Docs

- `docs/architecture.md` — system architecture and data flow
- `docs/milvus-collections.md` — collection schemas and indexing details
