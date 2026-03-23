# Remaining Work

## What's Done

All core modules implemented and tested (162 unit + 21 integration + 20 infra validation tests):
- Config, embedding client, Milvus collections/ingest/search
- All 4 chunkers (code, docs, SO, issues) with pattern detection
- Synthesis module (noop + Claude)
- FastAPI API (POST /analyze, GET /health)
- 4 ingestion CLI scripts (code, docs, SO, issues)
- E2E validated: docs ingested → API queried → correct results returned

## What's Left

### 1. Airflow DAGs (2 DAGs)

Create `airflow/dags/ingest_stackoverflow.py` and `airflow/dags/ingest_github_issues.py`.

Both DAGs follow the same pattern:

```python
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from datetime import datetime, timedelta

with DAG(
    dag_id="ingest_stackoverflow",
    schedule="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["spark-rag", "ingestion"],
) as dag:

    ingest = KubernetesPodOperator(
        task_id="ingest_so",
        namespace="airflow",
        image="spark-rag:latest",  # needs Dockerfile
        cmds=["uv", "run", "python", "-m", "spark_rag.ingestion.stackoverflow"],
        arguments=["--since", "{{ prev_ds }}"],  # incremental from last run
        name="ingest-so",
        service_account_name="airflow-worker",
        get_logs=True,
        is_delete_operator_pod=True,
    )
```

Similarly for `ingest_github_issues` with `schedule="@daily"`.

**Key details:**
- Both use `KubernetesPodOperator` — runs in ephemeral pods, not Celery workers
- `--since {{ prev_ds }}` for incremental (Airflow provides previous execution date)
- `service_account_name="airflow-worker"` — already validated RBAC (can create pods + read logs)
- `is_delete_operator_pod=True` — clean up after run
- Needs a Docker image with the project installed (see Dockerfile below)

**Deploy DAGs to Airflow:**
```bash
# Find dag-processor pod (it mounts the CephFS DAGs PVC)
DAGPROC=$(kubectl -n airflow get pod -l component=dag-processor -o jsonpath='{.items[0].metadata.name}')

# Copy DAG files
kubectl cp airflow/dags/ingest_stackoverflow.py airflow/${DAGPROC}:/opt/airflow/dags/ -c dag-processor
kubectl cp airflow/dags/ingest_github_issues.py airflow/${DAGPROC}:/opt/airflow/dags/ -c dag-processor
```

### 2. Dockerfile

Create `Dockerfile` at project root for K8s deployment (API service + ingestion pods):

```dockerfile
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install git (needed for ingestion clone)
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ src/
COPY config.yaml .

# API server
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "spark_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and push to a registry accessible by minikube:
```bash
# Using minikube's Docker daemon
eval $(minikube docker-env)
docker build -t spark-rag:latest .
```

### 3. K8s Deployment Manifests

Create `deployments/api-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spark-rag-api
  namespace: spark-rag
spec:
  replicas: 1
  selector:
    matchLabels:
      app: spark-rag-api
  template:
    metadata:
      labels:
        app: spark-rag-api
    spec:
      containers:
        - name: api
          image: spark-rag:latest
          imagePullPolicy: Never  # local image
          ports:
            - containerPort: 8000
          env:
            - name: MILVUS_URL
              value: "http://milvus.milvus.svc:19530"
            - name: OLLAMA_URL
              value: "http://ollama.ollama.svc:11434"
            # Uncomment for synthesis:
            # - name: SYNTHESIS_ENABLED
            #   value: "true"
            # - name: ANTHROPIC_API_KEY
            #   valueFrom:
            #     secretKeyRef:
            #       name: spark-rag-secrets
            #       key: anthropic-api-key
          resources:
            requests:
              cpu: 250m
              memory: 512Mi
            limits:
              cpu: 500m
              memory: 1Gi
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: spark-rag-api
  namespace: spark-rag
spec:
  selector:
    app: spark-rag-api
  ports:
    - port: 8000
      targetPort: 8000
  type: ClusterIP
```

Create namespace and deploy:
```bash
kubectl create namespace spark-rag
kubectl apply -f deployments/api-deployment.yaml
kubectl -n spark-rag port-forward svc/spark-rag-api 8000:8000
```

### 4. Ceph S3 Bucket for Raw Data

Create the `spark-rag-raw` bucket for storing raw scraped data (backup/audit):

```bash
# Create S3 user
kubectl -n rook-ceph exec deploy/rook-ceph-tools -- \
  radosgw-admin user create --uid=spark-rag --display-name="spark-rag" --caps="buckets=*;users=read"

# Create bucket
kubectl -n rook-ceph exec deploy/rook-ceph-tools -- \
  radosgw-admin bucket create --bucket=spark-rag-raw

# Or create via CephObjectStoreUser manifest (deployments/s3-user.yaml)
```

This is optional — ingestion works without raw storage. Add it when you want audit/backup of scraped data.

### 5. E2E Test Suite

Create `tests/e2e/test_query_pipeline.py` — requires all services running + at least docs ingested:

```python
"""E2E: full query pipeline against live services."""

import pytest
import requests

API_BASE = "http://localhost:8000"

@pytest.fixture(scope="module")
def api():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        r.raise_for_status()
        health = r.json()
        if not health["ollama"] or not health["milvus"]:
            pytest.skip("Services not healthy")
    except Exception:
        pytest.skip("API not reachable")
    return API_BASE

class TestQuestionQuery:
    def test_returns_references(self, api):
        r = requests.post(f"{api}/analyze", json={"input": "How does AQE work in Spark?"})
        assert r.status_code == 200
        data = r.json()
        assert data["input_type"] == "question"
        assert len(data["references"]) > 0

class TestCodeQuery:
    def test_detects_patterns(self, api):
        r = requests.post(f"{api}/analyze", json={
            "input": "df = spark.read.parquet('big')\ndf.collect()"
        })
        data = r.json()
        assert data["input_type"] == "code"
        assert any(p["name"] == "collect_on_large" for p in data["detected_patterns"])

class TestLogQuery:
    def test_extracts_errors(self, api):
        r = requests.post(f"{api}/analyze", json={
            "input": "java.lang.OutOfMemoryError: Java heap space\n    at org.apache.spark.sql.Dataset.collect"
        })
        data = r.json()
        assert data["input_type"] == "logs"
        assert "OutOfMemoryError" in data["error_types"]

class TestVersionFilter:
    def test_version_in_response(self, api):
        r = requests.post(f"{api}/analyze", json={
            "input": "What is AQE?",
            "version": "4.1.0"
        })
        data = r.json()
        assert data["version"] == "4.1.0"
```

Run: `uv run pytest tests/e2e/ -v` (after ingesting docs and starting the API)

### 6. Full Ingestion Run

After deploying everything, run the full initial ingestion:

```bash
# Port-forward services
kubectl -n milvus port-forward svc/milvus 19530:19530 &
kubectl -n ollama port-forward svc/ollama 11434:11434 &

# Phase 1: Docs (fastest, ~18 min per version)
uv run python -m spark_rag.ingestion.docs --version 4.1.0

# Phase 2: Code (longest, ~2-4h per version on CPU)
uv run python -m spark_rag.ingestion.code --version 4.1.0

# Phase 3: SO + Issues (moderate, ~30min + ~1-2h)
uv run python -m spark_rag.ingestion.stackoverflow
uv run python -m spark_rag.ingestion.issues --token $GITHUB_TOKEN

# Add another version
uv run python -m spark_rag.ingestion.docs --version 3.5.4
uv run python -m spark_rag.ingestion.code --version 3.5.4
```

Estimated total for first full ingestion (4.1 only): ~5-6 hours (CPU embedding bottleneck).

### 7. Access Script

Create `scripts/access.sh` for convenient port-forwarding:

```bash
#!/bin/bash
# Port-forward all services for local development
kubectl -n milvus port-forward svc/milvus 19530:19530 &
kubectl -n milvus port-forward svc/milvus-attu 3001:3000 &
kubectl -n ollama port-forward svc/ollama 11434:11434 &
kubectl -n airflow port-forward svc/airflow-api-server 8080:8080 &
echo "Milvus:  localhost:19530"
echo "Attu UI: localhost:3001"
echo "Ollama:  localhost:11434"
echo "Airflow: localhost:8080"
echo "Press Ctrl+C to stop all"
wait
```

### 8. Lance Cold Storage Layer (optimization)

Add `lance` as a cold storage layer to avoid re-embedding on Milvus rebuilds. Embedding is the bottleneck (~5-6h for full ingestion on CPU). Lance persists embeddings so Milvus can be rebuilt in minutes.

**Dependency:**
```bash
uv add lance
```

**New module:** `src/spark_rag/lance/`

```
src/spark_rag/lance/
  __init__.py
  store.py     — write embedded chunks to Lance datasets on Ceph S3
  loader.py    — read from Lance → batch insert into Milvus
```

**`store.py`** — write after embedding, before Milvus insert:

```python
import lance
import pyarrow as pa

LANCE_BASE = "s3://spark-rag-lance"  # or local path for dev

def write_collection(collection_name: str, data: list[dict], version: str | None = None):
    """Write embedded chunks to a Lance dataset.

    Each collection maps to a Lance dataset at {LANCE_BASE}/{collection_name}/
    Appends data. Use version param to tag versioned data (code/docs).
    """
    # Convert to PyArrow table
    table = pa.Table.from_pylist(data)

    uri = f"{LANCE_BASE}/{collection_name}"
    ds = lance.write_dataset(table, uri, mode="append")
    return ds.count_rows()
```

**`loader.py`** — rebuild Milvus from Lance (no re-embedding needed):

```python
import lance

def load_to_milvus(
    milvus_client,
    collection_name: str,
    lance_uri: str,
    filter_expr: str | None = None,
    batch_size: int = 500,
):
    """Load vectors from Lance dataset into Milvus.

    Args:
        filter_expr: Optional Lance SQL filter, e.g. 'spark_version = "4.1.0"'
    """
    ds = lance.dataset(lance_uri)
    scanner = ds.scanner(filter=filter_expr) if filter_expr else ds.scanner()

    total = 0
    for batch in scanner.to_batches():
        rows = batch.to_pylist()
        if rows:
            milvus_client.insert(collection_name=collection_name, data=rows)
            total += len(rows)
    return total
```

**Integration into ingestion pipeline:**

Change the flow from:
```
chunk → embed → Milvus
```
To:
```
chunk → embed → Lance (persist) → Milvus (serve)
```

Each ingestion script gets a `--lance` flag:
```bash
# Ingest to both Lance + Milvus
uv run python -m spark_rag.ingestion.docs --version 4.1.0 --lance

# Rebuild Milvus from Lance (no re-embedding, minutes instead of hours)
uv run python -m spark_rag.lance.loader --collection spark_docs --version 4.1.0

# Rebuild all from Lance
uv run python -m spark_rag.lance.loader --all
```

**Lance storage locations:**
- Local dev: `/tmp/spark-rag-lance/{collection}/`
- K8s/production: `s3://spark-rag-lance/{collection}/` (Ceph S3, same RGW as Milvus)

**Config addition:**
```yaml
lance:
  enabled: false                              # opt-in
  uri: "/tmp/spark-rag-lance"                 # local path or s3://spark-rag-lance
```

**When to implement:** After the first full ingestion run. If re-embedding time (~5-6h) becomes a pain point, Lance is the fix. The ingestion scripts already produce the data in the right shape — adding Lance is wiring, not redesign.

## Summary of Files to Create

| File | What |
|---|---|
| `airflow/dags/ingest_stackoverflow.py` | Weekly SO sync DAG |
| `airflow/dags/ingest_github_issues.py` | Daily issues sync DAG |
| `Dockerfile` | Container image for API + ingestion |
| `deployments/api-deployment.yaml` | K8s Deployment + Service for API |
| `deployments/s3-user.yaml` | CephObjectStoreUser (optional) |
| `tests/e2e/test_query_pipeline.py` | E2E tests against live services |
| `scripts/access.sh` | Port-forward convenience script |
| `src/spark_rag/lance/store.py` | Write to Lance (optimization, deferred) |
| `src/spark_rag/lance/loader.py` | Load Lance → Milvus (optimization, deferred) |
