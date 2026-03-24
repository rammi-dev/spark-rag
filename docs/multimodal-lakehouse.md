# Multimodal Lakehouse Architecture

A general-purpose multimodal lakehouse on Kubernetes that unifies structured data, unstructured text, and vector embeddings in a single open platform. All data lives in Lance format on Ceph S3, cataloged by Apache Polaris, and accessed by multiple compute engines for different workloads.

## Component Roles

| Component | Role | What it is NOT |
|---|---|---|
| **Lance** | Open columnar file format on S3 (like Parquet for AI) | Not a database, not a search engine |
| **Polaris** | Metadata catalog — registers table locations + properties | Not storage, does not manage data files |
| **Spark** | Primary engine for reading/writing Lance tables at scale | Not a storage layer |
| **Milvus** | Vector search serving layer — low-latency ANN queries | Not the source of truth for data |
| **Ceph S3** | Object storage — where Lance files physically live | Not a compute engine |
| **Ollama** | Embedding model server (nomic-embed-text, etc.) | Not storage |
| **Airflow** | Orchestration — schedules ingest/embed/load pipelines | Not a compute engine |
| **Kubeflow** | ML experiment tracking — compares embedding models | Not orchestration |

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
│   Loaded from Lance      Query Lance tables      Read from Milvus           │
│   (embeddings + meta     directly via Polaris    or Trino                   │
│    only, not raw data)   catalog connector                                  │
└──────┬──────────────┬────────────────────────────┬──────────────────────────┘
       │              │                            │
┌──────▼──────────────▼────────────────────────────▼──────────────────────────┐
│                         PROCESSING LAYER                                    │
│                                                                             │
│   Spark                          Airflow                  Kubeflow          │
│   (batch read/write)             (orchestration)          (ML experiments)  │
│                                                                             │
│   - Read/write Lance tables      - Schedule ingestion     - Track embedding │
│     via Polaris catalog          - Trigger embed jobs       model versions  │
│   - Chunking at scale            - Lance → Milvus load    - Compare metrics │
│   - Add embedding columns        - Monitor pipelines        across runs     │
│   - Cross-dataset joins                                                     │
│   - Primary compute engine                                                  │
│     for Lance I/O                                                           │
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
│     spark_rag_catalog / spark_rag /                                         │
│       ├── spark_code        (code chunks, multi-version)                    │
│       ├── spark_docs        (doc chunks, multi-version)                     │
│       ├── spark_so          (StackOverflow Q&A)                             │
│       └── spark_issues      (GitHub issues + comments)                      │
│     <future_catalog> / <namespace> /                                        │
│       └── ...                                                               │
│                                                                             │
│   Each table entry:                                                         │
│     { name, format: "lance", base-location: "s3://...",                     │
│       properties: { embedding_columns, dimensions, ... } }                  │
│                                                                             │
│   API paths (verified on Polaris 1.3.0):                                    │
│     OAuth:          POST /api/catalog/v1/oauth/tokens                       │
│     Management:     /api/management/v1/catalogs                             │
│     Namespaces:     /api/catalog/v1/{catalog}/namespaces                    │
│     Generic tables: /api/catalog/polaris/v1/{catalog}/namespaces/{ns}/      │
│                     generic-tables  (realm prefix 'polaris' required)       │
└──────┬──────────────────────────────────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────────────────────┐
│                          STORAGE LAYER                                      │
│                                                                             │
│   Lance format on Ceph S3                                                   │
│   (rook-ceph namespace, S3 gateway)                                         │
│                                                                             │
│   s3://<bucket>/lance/<table>/                                              │
│     ├── data/*.lance          (Arrow columnar fragments)                    │
│     └── _versions/            (version manifest, time-travel)               │
│                                                                             │
│   Why Lance:                                                                │
│   - Columnar (Arrow-native) — efficient selective column reads              │
│   - Schema evolution — add embedding columns without rewriting data         │
│   - Versioned — time-travel, rollback                                       │
│   - Zero-copy integration with PyArrow, Pandas, Polars, Spark              │
│   - S3-native via Rust object-store crate                                   │
│   - NOT a database — just a file format (like Parquet, but for AI)          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Engine Access Patterns

| Engine | Access Pattern | Use Case |
|---|---|---|
| **Spark** | Lance catalog connector via Polaris REST. `spark.sql.catalog.lance.impl = polaris`. Primary engine for batch I/O. | Chunking, embedding, transforms, cross-dataset joins, Lance→Milvus load |
| **pylance** | Direct S3 read/write via `lance.write_dataset()` / `lance.dataset()`. S3 URI from Polaris. | Lightweight Python scripts, CLI tools, embedding pipelines |
| **Trino** | Lance connector via Polaris. `connector.name=lance, lance.impl=polaris` | Ad-hoc SQL queries, analytics, dashboards |
| **Milvus** | Loaded from Lance (embeddings + metadata only). Not connected to Polaris. | Low-latency vector search at serving time |
| **Kubeflow** | Spark or pylance in pipeline steps. Metrics logged per embedding column. | Experiment tracking, embedding model comparison |
| **Airflow** | Orchestrates Spark jobs + CLI tools via KubernetesPodOperator. | Scheduled ingestion, embed, load pipelines |

## Data Flow: Three Decoupled Stages

```
STAGE 1: Ingest                   STAGE 2: Embed                   STAGE 3: Serve
─────────────────                 ──────────────────               ─────────────────

Source ──► Chunker ──► Lance      Lance ──► Model ──► Lance        Lance ──► Milvus
           (Spark or              (read content)  (add column)     (embeddings +
            pylance)                                                metadata only)

Produces:                         Produces:                        Produces:
 - Typed Arrow columns            - New embedding column           - Milvus collection
 - Stable chunk_id (uuid5)          per model experiment             with chosen vectors
 - Registered in Polaris          - Schema evolution (merge)       - Ready for ANN search
 - No embedding yet               - Tracked in Polaris props

Orchestrated by:                  Orchestrated by:                 Orchestrated by:
 Airflow DAGs                     Airflow or Kubeflow              Airflow DAGs
 (or CLI for one-time)            (experiment tracking)            (or CLI)
```

Stages are fully independent — re-embed without re-ingesting, swap serving model without re-embedding.

## How pylance Interacts with Polaris

pylance writes Lance files directly to S3. Polaris only stores metadata (location + properties). The interaction is:

```
Python (pylance + requests)                  Polaris REST API
    │                                              │
    ├── Get token ────────────────────────────────►│ POST /oauth/tokens
    │   ◄── access_token                           │
    │                                              │
    ├── Write Lance data to S3 ───────────────────►│ (no Polaris call — direct S3)
    │   lance.write_dataset(table, "s3://...")      │
    │                                              │
    ├── Register table in catalog ────────────────►│ POST /generic-tables
    │   { name, format: "lance",                   │   { base-location: "s3://..." }
    │     base-location: "s3://..." }              │
    │                                              │
    ├── Discover table location ──────────────────►│ GET /generic-tables/{name}
    │   ◄── { base-location: "s3://..." }          │
    │                                              │
    ├── Read Lance data from S3                    │ (no Polaris call — direct S3)
    │   ds = lance.dataset("s3://...")             │
    │   table = ds.to_table(columns=[...])         │
    │                                              │
    └── Update properties ────────────────────────►│ DELETE + POST (no update API)
        { emb_nomic: {dim: 768, status: done} }   │
```

## Spark Processing

```python
# Configure Spark to use Polaris catalog for Lance tables
spark = SparkSession.builder \
    .config("spark.jars.packages", "org.lance:lance-spark-bundle-3.5_2.12:0.0.7") \
    .config("spark.sql.catalog.lance", "org.lance.spark.LanceNamespaceSparkCatalog") \
    .config("spark.sql.catalog.lance.impl", "polaris") \
    .config("spark.sql.catalog.lance.endpoint", "http://polaris:8181") \
    .config("spark.sql.catalog.lance.auth_token", token) \
    .getOrCreate()

# Read Lance tables as DataFrames
df = spark.table("lance.spark_rag_catalog.spark_rag.spark_code")
df.filter("spark_version = '4.1.0'").select("content", "emb_nomic").show()

# Cross-dataset joins
code = spark.table("lance.spark_rag_catalog.spark_rag.spark_code")
docs = spark.table("lance.spark_rag_catalog.spark_rag.spark_docs")
joined = code.join(docs, code.spark_version == docs.spark_version)

# Write results back to Lance via catalog
enriched.writeTo("lance.spark_rag_catalog.spark_rag.spark_code_enriched").create()
```

## Trino SQL Access

```sql
-- Trino catalog properties: connector.name=lance, lance.impl=polaris, lance.endpoint=http://polaris:8181

SELECT content, spark_version, spark_apis
FROM lance.spark_rag_catalog.spark_rag.spark_code
WHERE spark_version = '4.1.0';

-- Cross-collection analytics
SELECT s.question_id, s.score, c.qualified_name
FROM lance.spark_rag_catalog.spark_rag.spark_so s
JOIN lance.spark_rag_catalog.spark_rag.spark_code c
  ON contains(s.spark_apis_mentioned, c.qualified_name);
```

## Embedding Experiment Model

```
┌─────────────────────────────────────────────────────────────────────┐
│ Lance Dataset: spark_code                                           │
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
│ Schema evolution via lance.dataset.merge() — no data rewrite        │
│ Incremental: only rows with NULL in target column get embedded      │
│ Polaris table properties track column metadata                      │
│ Spark reads any column: df.select("content", "emb_nomic")          │
└─────────────────────────────────────────────────────────────────────┘

Polaris table properties:
  { "emb_nomic": {"model": "nomic-embed-text", "dim": 768, "status": "complete"},
    "emb_bge_m3": {"model": "bge-m3", "dim": 1024, "status": "in_progress"} }
```

## Airflow Orchestration

```
DAG: ingest_and_embed (scheduled or triggered)

  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ ingest_code  │────►│ embed_code   │────►│ load_code    │
  │ (chunk→Lance)│     │ (add column) │     │ (Lance→Milvus)
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

  Each task = KubernetesPodOperator (Spark submit or Python CLI).
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
    1. Embed (Spark or pylance + Ollama) — add column to Lance table
       └── log: rows_embedded, duration, model_version
    2. Load to Milvus (temporary test collection)
    3. Evaluate (recall@k on held-out queries)
       └── log: recall@10, recall@50, mrr, latency_p99
    4. Compare across runs in Kubeflow UI

  Output: metrics per (model × dataset) combination
  Decision: which embedding column to promote to production Milvus
```

## Infrastructure Components

```
┌─────────────────────────────────────────────────────────────┐
│                    minikube (3 Hyper-V nodes)                │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ polaris ns   │  │ milvus ns   │  │ rook-ceph ns        │ │
│  │              │  │             │  │                     │ │
│  │ Polaris 1.3  │  │ Milvus 2.6  │  │ Ceph S3 gateway     │ │
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
│  │ (batch I/O  │  │ (federated  │                           │
│  │  for Lance) │  │  SQL)       │                           │
│  └─────────────┘  └─────────────┘                           │
└─────────────────────────────────────────────────────────────┘

Data flows:
  1. Polaris catalog → discover table S3 location
  2. Ceph S3 → read/write Lance files (all engines)
  3. Lance → Milvus (embeddings + metadata loaded for serving)
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
6. **Spark as primary engine** — batch reads/writes to Lance via Polaris catalog. pylance for lightweight Python-only pipelines.
7. **Milvus for serving only** — loaded from Lance with selected embedding column. Not the source of truth.
8. **No extra databases** — Lance is a file format, not a DB. Polaris is a catalog, not a DB. Milvus serves vectors. Each has one job.
