# Multimodal Lakehouse — Gap Analysis

Gap between the current spark-rag implementation and the target multimodal lakehouse architecture.

## Current vs Target Architecture

```mermaid
graph TB
    subgraph CURRENT["Current State"]
        direction TB
        S1[Source] --> C1[Chunker]
        C1 --> E1[Embed<br/>Ollama]
        E1 --> M1[Milvus<br/>direct insert]
        M1 --> API1[FastAPI<br/>search + synthesis]

        style S1 fill:#4ade80
        style C1 fill:#4ade80
        style E1 fill:#4ade80
        style M1 fill:#4ade80
        style API1 fill:#4ade80
    end

    subgraph TARGET["Target State"]
        direction TB
        S2[Source] --> C2[Chunker]
        C2 --> L2[Lance on S3<br/>via Polaris]
        L2 --> E2[Embed<br/>column-per-model]
        E2 --> L2
        L2 --> M2[Milvus<br/>selected column]
        M2 --> API2[FastAPI]
        L2 -.-> SP2[Spark]
        L2 -.-> TR2[Trino]
        L2 -.-> KF2[Kubeflow<br/>experiments]

        style S2 fill:#4ade80
        style C2 fill:#4ade80
        style L2 fill:#f87171
        style E2 fill:#f87171
        style M2 fill:#4ade80
        style API2 fill:#4ade80
        style SP2 fill:#fbbf24
        style TR2 fill:#fbbf24
        style KF2 fill:#f87171
    end
```

```
Legend:  🟢 exists   🟡 infra exists, no integration   🔴 not started
```

## Gap by Layer

```mermaid
block-beta
    columns 4
    block:APP["APPLICATION LAYER"]:4
        A1["spark-rag API\n✅ complete"]
        A2["ML apps\n(Kubeflow)\n❌ missing"]
        A3["Analytics\n(dashboards)\n❌ missing"]
        A4["Notebooks\n⚠️ no Lance"]
    end
    space:4
    block:SERVE["SERVING LAYER"]:4
        SV1["Milvus\n✅ deployed\n⚠️ no Lance loader"]
        SV2["Trino\n❌ not deployed\n❌ no connector"]
        SV3["FastAPI\n✅ complete"]
        SV4[" "]
    end
    space:4
    block:PROC["PROCESSING LAYER"]:4
        P1["Spark\n⚠️ deployed\n❌ no Lance catalog"]
        P2["Airflow\n✅ deployed\n❌ no DAGs"]
        P3["Kubeflow\n❌ not deployed"]
        P4[" "]
    end
    space:4
    block:CAT["CATALOG LAYER"]:4
        CT1["Polaris\n❌ not deployed\n❌ no client code"]:4
    end
    space:4
    block:STOR["STORAGE LAYER"]:4
        ST1["Ceph S3\n✅ deployed"]:2
        ST2["Lance format\n❌ no datasets\n❌ no store module"]:2
    end

    style A1 fill:#4ade80
    style A2 fill:#f87171
    style A3 fill:#f87171
    style A4 fill:#fbbf24
    style SV1 fill:#fbbf24
    style SV2 fill:#f87171
    style SV3 fill:#4ade80
    style P1 fill:#fbbf24
    style P2 fill:#fbbf24
    style P3 fill:#f87171
    style CT1 fill:#f87171
    style ST1 fill:#4ade80
    style ST2 fill:#f87171
```

## Infrastructure Gap

```mermaid
graph LR
    subgraph K8S["minikube (3 Hyper-V nodes)"]
        direction TB
        subgraph DEPLOYED["✅ Deployed & Validated"]
            MILVUS["Milvus 2.6\nmilvus ns"]
            OLLAMA["Ollama\nollama ns"]
            AIRFLOW["Airflow 3.1\nairflow ns"]
            CEPH["Ceph S3\nrook-ceph ns"]
            MON["Monitoring\nmonitoring ns"]
        end
        subgraph PENDING["❌ Not Deployed"]
            POLARIS["Polaris\npolaris ns\n(Helm ready)"]
            KUBEFLOW["Kubeflow\nkubeflow ns"]
            TRINO["Trino\ntrino ns"]
        end
        subgraph PARTIAL["⚠️ Deployed, No Integration"]
            SPARK["Spark\nspark ns\n(no Lance catalog)"]
        end
    end

    style MILVUS fill:#4ade80
    style OLLAMA fill:#4ade80
    style AIRFLOW fill:#4ade80
    style CEPH fill:#4ade80
    style MON fill:#4ade80
    style POLARIS fill:#f87171
    style KUBEFLOW fill:#f87171
    style TRINO fill:#f87171
    style SPARK fill:#fbbf24
```

## Data Pipeline Gap

### Current: Tightly Coupled

```mermaid
flowchart LR
    subgraph INGEST["Ingestion (4 pipelines)"]
        GIT[Git clone] --> CHUNK[Chunk\ntree-sitter / markdown]
        SO_API[SO API] --> CHUNK_SO[Chunk\nQ&A parser]
        GH_API[GitHub API] --> CHUNK_ISS[Chunk\nissue parser]
    end

    CHUNK --> EMBED[Embed\nOllama\nnomic-embed-text]
    CHUNK_SO --> EMBED
    CHUNK_ISS --> EMBED

    EMBED --> MILVUS[(Milvus\n4 collections)]
    MILVUS --> SEARCH[Search + Rerank]
    SEARCH --> SYNTH[Claude Synthesis]

    style EMBED fill:#f87171,color:#fff
    style MILVUS fill:#4ade80

    linkStyle 3,4,5 stroke:#f87171,stroke-width:3
```

**Problem**: Changing the embedding model requires re-running the entire pipeline (chunk + embed + insert). ~5-6 hours on CPU for full dataset.

### Target: Three Decoupled Stages

```mermaid
flowchart LR
    subgraph STAGE1["Stage 1: Ingest"]
        GIT2[Git clone] --> CHUNK2[Chunk]
        SO2[SO API] --> CHUNK2_SO[Chunk]
        GH2[GitHub API] --> CHUNK2_ISS[Chunk]
    end

    CHUNK2 --> LANCE[(Lance on S3\nvia Polaris)]
    CHUNK2_SO --> LANCE
    CHUNK2_ISS --> LANCE

    subgraph STAGE2["Stage 2: Embed"]
        LANCE --> |read content| EMB_N[Embed\nnomic-embed-text]
        LANCE --> |read content| EMB_B[Embed\nbge-m3]
        EMB_N --> |merge col| LANCE
        EMB_B --> |merge col| LANCE
    end

    subgraph STAGE3["Stage 3: Serve"]
        LANCE --> |selected col| MILVUS2[(Milvus)]
    end

    MILVUS2 --> SEARCH2[Search + Rerank]

    style LANCE fill:#f87171,color:#fff
    style EMB_N fill:#f87171,color:#fff
    style EMB_B fill:#f87171,color:#fff
```

**All red boxes = not implemented yet.**

## Embedding Experiment Gap

```mermaid
flowchart TB
    subgraph CURRENT_EMB["Current: Single Model, No Experiments"]
        TEXT1[content] --> OLL1[Ollama\nnomic-embed-text] --> VEC1["768d vector"]
        VEC1 --> MIL1[(Milvus)]
    end

    subgraph TARGET_EMB["Target: Column-per-Model in Lance"]
        TEXT2[content] --> OLL2[nomic-embed-text] --> COL1["emb_nomic\n768d column"]
        TEXT2 --> BGE2[bge-m3] --> COL2["emb_bge_m3\n1024d column"]
        TEXT2 --> FUT2[future model] --> COL3["emb_xxx\nNd column"]

        COL1 --> LANCE2[(Lance dataset)]
        COL2 --> LANCE2
        COL3 --> LANCE2

        LANCE2 --> |select best| MIL2[(Milvus)]

        LANCE2 -.-> KF[Kubeflow\nrecall@k evaluation]
        KF -.-> |promote winner| MIL2
    end

    style OLL1 fill:#4ade80
    style MIL1 fill:#4ade80
    style LANCE2 fill:#f87171,color:#fff
    style KF fill:#f87171,color:#fff
    style COL1 fill:#f87171,color:#fff
    style COL2 fill:#f87171,color:#fff
    style COL3 fill:#f87171,color:#fff
```

## Polaris Catalog Integration Gap

```mermaid
sequenceDiagram
    participant App as spark-rag
    participant Pol as Polaris REST API
    participant S3 as Ceph S3
    participant Mil as Milvus

    Note over App,Pol: ❌ None of this exists yet

    App->>Pol: POST /oauth/tokens (client credentials)
    Pol-->>App: access_token

    App->>Pol: POST /generic-tables {name: spark_code, format: lance, base-location: s3://...}
    Pol-->>App: 201 Created

    App->>S3: lance.write_dataset(chunks, s3://spark-rag/lance/spark_code/)
    S3-->>App: OK

    App->>Pol: GET /generic-tables/spark_code
    Pol-->>App: {base-location: s3://..., properties: {emb_nomic: complete}}

    App->>S3: lance.dataset(s3://...).to_table(columns=[emb_nomic, content, metadata])
    S3-->>App: Arrow Table

    App->>Mil: batch_insert(table → Milvus format)
    Mil-->>App: insert_count: N
```

## Airflow DAG Gap

```mermaid
flowchart TB
    subgraph CURRENT_AF["Current: No DAGs"]
        AF_EMPTY["airflow/dags/\n(empty directory)"]
    end

    subgraph TARGET_AF["Target: 3-Stage DAG"]
        direction LR
        subgraph ING["Ingest Tasks"]
            I_CODE[ingest_code]
            I_DOCS[ingest_docs]
            I_SO[ingest_so]
            I_ISS[ingest_issues]
        end
        subgraph EMB["Embed Tasks"]
            E_CODE[embed_code]
            E_DOCS[embed_docs]
            E_SO[embed_so]
            E_ISS[embed_issues]
        end
        subgraph LOAD["Load Tasks"]
            L_CODE[load_code]
            L_DOCS[load_docs]
            L_SO[load_so]
            L_ISS[load_issues]
        end
        I_CODE --> E_CODE --> L_CODE
        I_DOCS --> E_DOCS --> L_DOCS
        I_SO --> E_SO --> L_SO
        I_ISS --> E_ISS --> L_ISS
    end

    style AF_EMPTY fill:#f87171,color:#fff
    style I_CODE fill:#f87171,color:#fff
    style I_DOCS fill:#f87171,color:#fff
    style I_SO fill:#f87171,color:#fff
    style I_ISS fill:#f87171,color:#fff
    style E_CODE fill:#f87171,color:#fff
    style E_DOCS fill:#f87171,color:#fff
    style E_SO fill:#f87171,color:#fff
    style E_ISS fill:#f87171,color:#fff
    style L_CODE fill:#f87171,color:#fff
    style L_DOCS fill:#f87171,color:#fff
    style L_SO fill:#f87171,color:#fff
    style L_ISS fill:#f87171,color:#fff
```

## Code Module Gap

```mermaid
graph TB
    subgraph EXISTS["✅ Implemented"]
        CONFIG[config.py]
        CHUNK_MOD[chunking/\n5 files]
        EMB_MOD[embedding/\nclient.py]
        MIL_MOD[milvus/\n3 files]
        SYNTH_MOD[synthesis/\n3 files]
        ING_MOD[ingestion/\n5 files]
        API_MOD[api/\n2 files]
    end

    subgraph PARTIAL["⚠️ Config Added, No Module"]
        CONFIG_LANCE[config.py\nLanceConfig ✅\nPolarisConfig ❌]
        PYPROJECT[pyproject.toml\npylance ✅]
    end

    subgraph MISSING["❌ Not Created"]
        LANCE_MOD["lance/\nschemas.py\nstore.py\nembeddings.py\nloader.py\ncli_embed.py\ncli_load.py"]
        CATALOG_MOD["catalog.py\nPolaris REST client"]
        KF_MOD["kubeflow/\npipeline definition"]
        DAG_MOD["airflow/dags/\nDAG definitions"]
    end

    CONFIG --> CONFIG_LANCE
    ING_MOD -.-> |needs refactor| LANCE_MOD
    LANCE_MOD --> CATALOG_MOD
    LANCE_MOD --> MIL_MOD
    EMB_MOD --> LANCE_MOD

    style EXISTS fill:#4ade80
    style PARTIAL fill:#fbbf24
    style MISSING fill:#f87171,color:#fff
```

## Gap Summary

| Component | Layer | Status | What Exists | What's Missing |
|---|---|---|---|---|
| **Lance store** | Storage | ❌ | config added, pylance dep added | `lance/` module (6 files): schemas, store, embeddings, loader, 2 CLIs |
| **Polaris catalog** | Catalog | ❌ | nothing | config, REST client, OAuth, table registration |
| **Polaris deploy** | Infra | ❌ | Helm chart ready | K8s deployment (user handles) |
| **Ingestion refactor** | Processing | ⚠️ | 4 CLIs work (chunk→embed→Milvus) | Decouple: chunk→Lance, embed→Lance, load→Milvus |
| **Airflow DAGs** | Processing | ❌ | empty `dags/` dir, Airflow deployed | DAG definitions for 3-stage pipeline |
| **Kubeflow** | Processing | ❌ | nothing | deploy + pipeline definition + metrics |
| **Spark catalog** | Processing | ⚠️ | Spark deployed | Lance catalog connector config |
| **Trino connector** | Serving | ❌ | nothing | deploy + Polaris catalog connector |
| **Milvus loader** | Serving | ⚠️ | Milvus works, ingest functions exist | `lance/loader.py` to read Lance → Milvus |
| **FastAPI** | Application | ✅ | complete | none |
| **Ceph S3** | Storage | ✅ | deployed, validated | none |

## Implementation Phases

```mermaid
gantt
    title Multimodal Lakehouse Implementation
    dateFormat YYYY-MM-DD
    axisFormat %b

    section Phase 1 — Foundation
    Deploy Polaris (user)           :p1a, 2026-03-24, 3d
    Polaris config + client         :p1b, after p1a, 2d
    Lance schemas.py                :p1c, 2026-03-24, 2d
    Lance store.py                  :p1d, after p1c, 3d

    section Phase 2 — Decouple Pipeline
    Lance embeddings.py + cli       :p2a, after p1d, 2d
    Lance loader.py + cli           :p2b, after p1d, 2d
    Refactor 4 ingestion CLIs       :p2c, after p2a, 3d
    Airflow DAGs                    :p2d, after p2c, 2d

    section Phase 3 — Experiments
    Deploy Kubeflow (user)          :p3a, after p2d, 3d
    Kubeflow pipeline               :p3b, after p3a, 3d
    Evaluation metrics              :p3c, after p3b, 2d

    section Phase 4 — Multi-Engine
    Spark Lance catalog config      :p4a, after p2d, 1d
    Deploy Trino (user)             :p4b, after p2d, 2d
    Trino Polaris connector         :p4c, after p4b, 1d
```

## Critical Path

```mermaid
flowchart LR
    POLARIS_DEPLOY["Deploy Polaris\n(user)"] --> POLARIS_CLIENT["Polaris client\ncatalog.py"]
    POLARIS_CLIENT --> STORE["Lance store\nstore.py"]

    SCHEMAS["Lance schemas\nschemas.py"] --> STORE

    STORE --> EMBED_MOD["Embedding module\nembeddings.py"]
    STORE --> LOADER["Lance loader\nloader.py"]

    EMBED_MOD --> REFACTOR["Refactor ingestion\n4 CLIs"]
    LOADER --> REFACTOR

    REFACTOR --> DAGS["Airflow DAGs"]
    REFACTOR --> KUBEFLOW["Kubeflow pipeline"]

    style POLARIS_DEPLOY fill:#f87171,color:#fff
    style POLARIS_CLIENT fill:#f87171,color:#fff
    style SCHEMAS fill:#f87171,color:#fff
    style STORE fill:#f87171,color:#fff
    style EMBED_MOD fill:#f87171,color:#fff
    style LOADER fill:#f87171,color:#fff
    style REFACTOR fill:#fbbf24
    style DAGS fill:#f87171,color:#fff
    style KUBEFLOW fill:#f87171,color:#fff
```

**Blocking dependency**: Polaris deployment must happen first — everything else discovers tables through the catalog.

**Parallel tracks after Polaris**: Lance schemas + Polaris client can start together, then store, then embedding + loader in parallel, then ingestion refactor.
