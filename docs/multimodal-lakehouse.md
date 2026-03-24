# Multimodal Lakehouse Architecture

A general-purpose multimodal lakehouse on Kubernetes that unifies structured data, unstructured text, and vector embeddings in a single open platform. All data lives in Lance format on Ceph S3, cataloged by Apache Polaris, and accessed by multiple compute engines for different workloads.

## Layer Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           APPLICATION LAYER                                 │
│                                                                             │
│   spark-rag        ML apps          Analytics         Ad-hoc queries        │
│   (RAG pipeline)   (Kubeflow)       (dashboards)      (notebooks)           │
└──────┬──────────────┬────────────────┬─────────────────┬────────────────────┘
       │              │                │                 │
┌──────▼──────────────▼────────────────▼─────────────────▼────────────────────┐
│                          SERVING LAYER                                      │
│                                                                             │
│   Milvus                 Trino                   API servers                │
│   (vector search)        (federated SQL)         (FastAPI, etc)             │
│                                                                             │
│   Load selected          Query Lance tables      Read from Milvus           │
│   embeddings from        directly via Polaris    or Trino                   │
│   Lance → serve          catalog connector                                  │
└──────┬──────────────┬────────────────────────────┬──────────────────────────┘
       │              │                            │
┌──────▼──────────────▼────────────────────────────▼──────────────────────────┐
│                         PROCESSING LAYER                                    │
│                                                                             │
│   Spark                          Airflow                  Kubeflow          │
│   (batch processing)             (orchestration)          (ML experiments)  │
│                                                                             │
│   - Read/write Lance tables      - Schedule ingestion     - Track embedding │
│     via Polaris catalog          - Trigger embed jobs       model versions  │
│   - Large-scale transforms       - Lance → Milvus load    - Compare metrics │
│   - Cross-dataset joins          - Monitor pipelines        across runs     │
│   - Chunking at scale                                    - Hyperparameter   │
│                                                            search on        │
│                                                            chunking +       │
│                                                            embedding        │
└──────┬──────────────┬────────────────────────────┬──────────────────────────┘
       │              │                            │
┌──────▼──────────────▼────────────────────────────▼──────────────────────────┐
│                          CATALOG LAYER                                      │
│                                                                             │
│                        Apache Polaris                                       │
│                    (unified metadata catalog)                               │
│                                                                             │
│   - Registers Lance tables as generic-tables (format=lance)                 │
│   - Namespaces organize by project / domain                                 │
│   - Single source of truth for table locations + properties                 │
│   - REST API: all engines discover tables the same way                      │
│   - OAuth2 authentication (client credentials flow)                         │
│   - Generic Table API (no Iceberg dependency for Lance tables)              │
│                                                                             │
│   Namespace layout:                                                         │
│     spark_rag/                                                              │
│       ├── spark_code        (code chunks, multi-version)                    │
│       ├── spark_docs        (doc chunks, multi-version)                     │
│       ├── spark_so          (StackOverflow Q&A)                             │
│       └── spark_issues      (GitHub issues + comments)                      │
│     <future_project>/                                                       │
│       └── ...                                                               │
│                                                                             │
│   Each table entry:                                                         │
│     { name, format: "lance", base-location: "s3://...",                     │
│       properties: { embedding_columns, dimensions, ... } }                  │
└──────┬──────────────────────────────────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────────────────────┐
│                          STORAGE LAYER                                      │
│                                                                             │
│   Lance format on Ceph S3                                                   │
│   (rook-ceph namespace, S3 gateway)                                         │
│                                                                             │
│   s3://<bucket>/lance/                                                      │
│     ├── <table>/             Lance dataset (Arrow columnar + versioned)     │
│     ├── <table>/                                                            │
│     └── ...                                                                 │
│                                                                             │
│   Why Lance:                                                                │
│   - Columnar (Arrow-native) — efficient selective column reads              │
│   - Schema evolution — add embedding columns without rewriting data         │
│   - Versioned — time-travel, rollback                                       │
│   - Vector index support (IVF_PQ) — optional ANN search without Milvus     │
│   - Zero-copy integration with PyArrow, Pandas, Polars                      │
│   - S3-native via Rust object-store crate                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Engine Access Patterns

| Engine | Access Pattern | Use Case |
|---|---|---|
| **pylance** | Direct S3 read/write. Location resolved from Polaris catalog. | Ingestion, embedding, Lance-to-Milvus loading |
| **Spark** | Lance Namespace + Polaris REST catalog connector. `spark.sql.catalog.lakehouse...` | Batch processing, large-scale transforms, cross-dataset joins |
| **Trino** | Polaris catalog connector (Iceberg REST + generic tables) | Ad-hoc SQL queries, analytics, dashboards |
| **Milvus** | Loaded from Lance via loader CLI. Selected embedding column only. | Low-latency vector search at serving time |
| **Kubeflow** | pylance in pipeline steps. Metrics logged per embedding column. | Experiment tracking, embedding model comparison |
| **Airflow** | Orchestrates pylance + CLI tools via KubernetesPodOperator | Scheduled ingestion, embed, load pipelines |

## Data Flow: Three Decoupled Stages

```
STAGE 1: Ingest                   STAGE 2: Embed                   STAGE 3: Serve
─────────────────                 ──────────────────               ─────────────────

Source ──► Chunker ──► Lance      Lance ──► Model ──► Lance        Lance ──► Milvus
           (Spark or              (read content)  (merge col)      (selected col)
            Python)

Produces:                         Produces:                        Produces:
 - Typed Arrow columns            - New embedding column           - Milvus collection
 - Stable chunk_id (uuid5)          per model experiment             with chosen vectors
 - Registered in Polaris          - Incremental (skips done)       - Ready for search
 - No embedding yet               - Tracked in Polaris props

Orchestrated by:                  Orchestrated by:                 Orchestrated by:
 Airflow DAGs                     Airflow or Kubeflow              Airflow DAGs
 (or CLI for one-time)            (experiment tracking)            (or CLI)
```

Stages are fully independent — re-embed without re-ingesting, swap serving model without re-embedding.

## Embedding Experiment Model

```
┌─────────────────────────────────────────────────────────────────────┐
│ Lance Dataset                                                       │
│                                                                     │
│ ┌──────────┬─────────┬─────────────┬─────────────┬────────────┐    │
│ │ chunk_id │ content │ metadata... │ emb_nomic   │ emb_bge_m3 │    │
│ │ (stable) │ (text)  │ (typed)     │ (768d)      │ (1024d)    │    │
│ ├──────────┼─────────┼─────────────┼─────────────┼────────────┤    │
│ │ a1b2...  │ ...     │ ...         │ [0.12, ...] │ [0.45, ...]│    │
│ │ c3d4...  │ ...     │ ...         │ [0.34, ...] │ NULL       │    │
│ └──────────┴─────────┴─────────────┴─────────────┴────────────┘    │
│                                      ▲              ▲               │
│                                  experiment 1   experiment 2        │
│                                  (complete)     (in progress)       │
│                                                                     │
│ Schema evolution: each new model = new fixed_size_list column       │
│ Incremental: only rows with NULL in that column get embedded        │
└─────────────────────────────────────────────────────────────────────┘
```

- Each embedding model adds a `fixed_size_list<float32>(dim)` column (e.g. `emb_nomic`, `emb_bge_m3`)
- Columns are added via Lance schema evolution (`merge`) — no data rewrite
- Embedding is incremental: only rows with NULL in the target column get processed
- Polaris table properties track column metadata:
  ```json
  {
    "emb_nomic": {"model": "nomic-embed-text", "dim": 768, "status": "complete"},
    "emb_bge_m3": {"model": "bge-m3", "dim": 1024, "status": "in_progress"}
  }
  ```
- Load any column into Milvus for serving — switch models without re-ingesting

## Polaris Catalog Interaction

```
Any Engine                          Polaris REST API
    │                                      │
    ├── Auth ─────────────────────────────►│ POST /api/catalog/v1/oauth/tokens
    │   Basic(client_id:secret)            │   ──► { access_token }
    │                                      │
    ├── Discover table ───────────────────►│ GET  /api/catalog/v1/{catalog}
    │                                      │   /namespaces/{ns}/generic-tables/{name}
    │   ◄── { format: "lance",             │
    │         base-location: "s3://...",    │
    │         properties: {...} }           │
    │                                      │
    ├── Register new table ───────────────►│ POST /api/catalog/v1/{catalog}
    │   body: { name, format: "lance",     │   /namespaces/{ns}/generic-tables
    │           base-location: "s3://..." } │
    │                                      │
    ├── List all tables ──────────────────►│ GET  .../generic-tables/
    │                                      │
    └── Drop table ───────────────────────►│ DELETE .../generic-tables/{name}
        (metadata only, data stays on S3)  │
```

Polaris is metadata only — it records where data lives but does not manage storage. Data is written directly to Ceph S3 via pylance; Polaris provides the discovery and governance layer.

## Airflow Orchestration

```
DAG: ingest_and_embed (scheduled or triggered)

  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ ingest_code  │────►│ embed_code   │────►│ load_code    │
  │ (chunk→Lance)│     │ (Lance→Lance)│     │ (Lance→Milvus)
  └──────────────┘     └──────────────┘     └──────────────┘
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ ingest_docs  │────►│ embed_docs   │────►│ load_docs    │
  └──────────────┘     └──────────────┘     └──────────────┘
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ ingest_so    │────►│ embed_so     │────►│ load_so      │
  └──────────────┘     └──────────────┘     └──────────────┘
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ ingest_issues│────►│ embed_issues │────►│ load_issues  │
  └──────────────┘     └──────────────┘     └──────────────┘

  Each task = KubernetesPodOperator running the CLI command.
  Stages are independent: re-embed without re-ingesting.
```

## Kubeflow Experiment Tracking

```
Kubeflow Pipeline: embedding_experiment

  Parameters:
    - dataset: spark_code | spark_docs | ...
    - model: nomic-embed-text | bge-m3 | ...
    - column: emb_nomic | emb_bge_m3 | ...

  Steps:
    1. Embed (pylance + Ollama)
       └── log: rows_embedded, duration, model_version
    2. Load to Milvus (temporary test collection)
    3. Evaluate (recall@k on held-out queries)
       └── log: recall@10, recall@50, mrr, latency_p99
    4. Compare across runs in Kubeflow UI

  Output: metrics per (model × dataset) combination
  Decision: which embedding column to promote to production Milvus
```

## Spark Processing

```python
# Configure Spark to use Polaris catalog for Lance tables
spark.sql.catalog.lakehouse = org.apache.spark.sql.lance.catalog.LanceCatalog
spark.sql.catalog.lakehouse.type = rest
spark.sql.catalog.lakehouse.uri = http://polaris:8181
spark.sql.catalog.lakehouse.credential = client_id:client_secret

# Query Lance tables as Spark DataFrames
df = spark.table("lakehouse.spark_rag.spark_code")
df.filter("spark_version = '4.1.0'").select("content", "emb_nomic").show()

# Cross-dataset joins
code = spark.table("lakehouse.spark_rag.spark_code")
docs = spark.table("lakehouse.spark_rag.spark_docs")
joined = code.join(docs, code.spark_version == docs.spark_version)

# Write results back to Lance via catalog
enriched.writeTo("lakehouse.spark_rag.spark_code_enriched").create()
```

## Trino SQL Access

```sql
-- Query Lance tables via federated SQL
SELECT content, spark_version, spark_apis
FROM lakehouse.spark_rag.spark_code
WHERE spark_version = '4.1.0'
  AND cardinality(spark_apis) > 0;

-- Cross-collection analytics
SELECT
  s.question_id, s.score, s.tags,
  c.qualified_name, c.file_path
FROM lakehouse.spark_rag.spark_so s
JOIN lakehouse.spark_rag.spark_code c
  ON contains(s.spark_apis_mentioned, c.qualified_name);
```

## Infrastructure Components

```
┌─────────────────────────────────────────────────────────────┐
│                    minikube (3 Hyper-V nodes)                │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ polaris ns   │  │ milvus ns   │  │ rook-ceph ns        │ │
│  │              │  │             │  │                     │ │
│  │ Polaris      │  │ Milvus      │  │ Ceph S3 gateway     │ │
│  │ (catalog)    │  │ (vector     │  │ (object storage)    │ │
│  │              │  │  search)    │  │                     │ │
│  └──────────────┘  └─────────────┘  └─────────────────────┘ │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ ollama ns   │  │ airflow ns  │  │ kubeflow ns         │ │
│  │             │  │             │  │                     │ │
│  │ Ollama      │  │ Airflow     │  │ Kubeflow Pipelines  │ │
│  │ (embedding  │  │ (orchestr.) │  │ (ML experiments)    │ │
│  │  models)    │  │             │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│  ┌─────────────┐  ┌─────────────┐                           │
│  │ spark ns    │  │ trino ns    │                           │
│  │             │  │             │                           │
│  │ Spark       │  │ Trino       │                           │
│  │ (batch      │  │ (federated  │                           │
│  │  processing)│  │  SQL)       │                           │
│  └─────────────┘  └─────────────┘                           │
└─────────────────────────────────────────────────────────────┘

All components access Lance data via:
  1. Polaris catalog (discover table location)
  2. Ceph S3 (read/write Lance files)
```

## Data Model

All Lance tables share base fields:

| Field | Type | Purpose |
|---|---|---|
| `chunk_id` | `utf8` | Stable UUID5 primary key from identity fields |
| `content` | `large_utf8` | Chunk text |
| `ingested_at` | `timestamp[us, UTC]` | Ingestion time |

Embedding columns added dynamically via schema evolution:

| Column | Type | Example |
|---|---|---|
| `emb_nomic` | `fixed_size_list<float32>(768)` | nomic-embed-text vectors |
| `emb_bge_m3` | `fixed_size_list<float32>(1024)` | bge-m3 vectors |

Table-specific metadata columns are typed Arrow columns (not JSON blobs) — lists, structs, ints, bools — for efficient columnar reads and native filtering.

## Key Design Principles

1. **Open formats only** — Lance (Arrow-native) on S3. No vendor lock-in. Any engine that reads Arrow can read the data.
2. **Catalog-first** — all tables discovered through Polaris. No hardcoded S3 paths in application code.
3. **Decouple ingest from embed from serve** — each stage runs independently. Change one without touching the others.
4. **Column-per-experiment** — embedding models are columns, not separate tables. Cheap to add, trivial to compare.
5. **Incremental by default** — embedding fills NULLs, ingestion deduplicates by stable IDs.
6. **Multi-engine** — same data accessible from Python, Spark, Trino, and Milvus via the shared Polaris catalog.
