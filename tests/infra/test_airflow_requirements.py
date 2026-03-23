"""Phase 0: Validate Airflow is ready for spark-rag ingestion DAGs.

Run with: uv run pytest tests/infra/test_airflow_requirements.py -v
Requires: kubectl -n airflow port-forward svc/airflow-api-server 8080:8080
"""

import subprocess
import time

import pytest
import requests

AIRFLOW_API = "http://localhost:8080"
NAMESPACE = "airflow"


def _kubectl(*args, check=True):
    result = subprocess.run(
        ["kubectl", "-n", NAMESPACE, *args],
        capture_output=True, text=True, timeout=30,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"kubectl failed: {result.stderr}")
    return result


def _get_jwt_token(base_url):
    """Get JWT token from Airflow 3.x auth endpoint."""
    r = requests.post(
        f"{base_url}/auth/token",
        json={"username": "admin", "password": "admin"},
        timeout=5,
    )
    r.raise_for_status()
    return r.json()["access_token"]


@pytest.fixture(scope="module")
def api_session():
    """Verify API server is reachable, return (base_url, session_with_auth)."""
    try:
        r = requests.get(f"{AIRFLOW_API}/api/v2/version", timeout=5)
        r.raise_for_status()
    except Exception as e:
        pytest.skip(f"Airflow API not reachable at {AIRFLOW_API}: {e}")

    token = _get_jwt_token(AIRFLOW_API)
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    return AIRFLOW_API, session


class TestAirflowHealth:
    """Verify Airflow cluster is healthy and API responsive."""

    def test_api_version(self, api_session):
        """1. API reachable and Airflow >= 3.0."""
        api_base, _ = api_session
        r = requests.get(f"{api_base}/api/v2/version")
        data = r.json()
        version = data["version"]
        major = int(version.split(".")[0])
        assert major >= 3, f"Need Airflow >= 3.0, got {version}"

    def test_scheduler_running(self):
        """2. Scheduler pod is Running."""
        result = _kubectl(
            "get", "pod", "-l", "component=scheduler",
            "-o", "jsonpath={.items[0].status.phase}",
        )
        assert result.stdout.strip() == "Running"

    def test_workers_ready(self):
        """3. At least 1 Celery worker is Running."""
        result = _kubectl(
            "get", "pod", "-l", "component=worker",
            "-o", "jsonpath={range .items[*]}{.status.phase}{\"\\n\"}{end}",
        )
        phases = [p for p in result.stdout.strip().split("\n") if p]
        running = [p for p in phases if p == "Running"]
        assert len(running) >= 1, f"No running workers, phases: {phases}"

    def test_dag_processor_running(self):
        """4. DAG processor pod is Running."""
        result = _kubectl(
            "get", "pod", "-l", "component=dag-processor",
            "-o", "jsonpath={.items[0].status.phase}",
        )
        assert result.stdout.strip() == "Running"


class TestDAGPVC:
    """Verify DAGs PVC is mounted and writable."""

    def test_dags_pvc_exists(self):
        """5. airflow-dags PVC exists and is Bound."""
        result = _kubectl(
            "get", "pvc", "airflow-dags",
            "-o", "jsonpath={.status.phase}",
        )
        assert result.stdout.strip() == "Bound"

    def test_dags_directory_writable(self):
        """6. Can write to DAGs directory via dag-processor pod (has PVC mount)."""
        dag_proc = _kubectl(
            "get", "pod", "-l", "component=dag-processor",
            "-o", "jsonpath={.items[0].metadata.name}",
        ).stdout.strip()

        # Write a test file via dag-processor (it mounts the CephFS DAGs PVC)
        _kubectl(
            "exec", dag_proc, "-c", "dag-processor",
            "--", "bash", "-c",
            "echo 'test' > /opt/airflow/dags/_test_write && rm /opt/airflow/dags/_test_write",
        )
        # If we get here without error, write succeeded


class TestKubernetesPodOperator:
    """Verify RBAC allows launching pods from Airflow."""

    def test_worker_service_account_exists(self):
        """7. Airflow worker service account exists."""
        result = _kubectl("get", "sa", "airflow-worker", "-o", "name")
        assert "serviceaccount/airflow-worker" in result.stdout.strip()

    def test_worker_sa_can_create_pods(self):
        """8. Worker service account has permission to create pods."""
        result = _kubectl(
            "auth", "can-i", "create", "pods",
            "--as=system:serviceaccount:airflow:airflow-worker",
            check=False,
        )
        assert result.stdout.strip() == "yes", (
            "airflow-worker SA cannot create pods — KubernetesPodOperator won't work. "
            "Check RBAC/ClusterRoleBinding in values.yaml"
        )

    def test_worker_sa_can_read_pod_logs(self):
        """9. Worker SA can read pod logs (needed to stream KPO output)."""
        result = _kubectl(
            "auth", "can-i", "get", "pods/log",
            "--as=system:serviceaccount:airflow:airflow-worker",
            check=False,
        )
        assert result.stdout.strip() == "yes", (
            "airflow-worker SA cannot read pod logs — KPO log streaming won't work"
        )


class TestDAGDeployment:
    """Verify we can deploy a DAG and Airflow picks it up."""

    DAG_CONTENT = '''\
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="spark_rag_test_probe",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["test", "spark-rag"],
) as dag:
    BashOperator(task_id="hello", bash_command="echo spark-rag-probe-ok")
'''

    def test_deploy_and_detect_dag(self, api_session):
        """10. Copy a test DAG to PVC via dag-processor, verify Airflow detects it, then clean up."""
        api_base, session = api_session

        dag_proc = _kubectl(
            "get", "pod", "-l", "component=dag-processor",
            "-o", "jsonpath={.items[0].metadata.name}",
        ).stdout.strip()

        dag_path = "/opt/airflow/dags/spark_rag_test_probe.py"

        try:
            # Write DAG file via dag-processor (it mounts the CephFS DAGs PVC)
            _kubectl(
                "exec", dag_proc, "-c", "dag-processor",
                "--", "bash", "-c",
                f"cat > {dag_path} << 'DAGEOF'\n{self.DAG_CONTENT}DAGEOF",
            )

            # Wait for DAG processor to pick it up (up to 60s)
            detected = False
            for _ in range(30):
                time.sleep(2)
                r = session.get(
                    f"{api_base}/api/v2/dags/spark_rag_test_probe",
                    timeout=5,
                )
                if r.status_code == 200:
                    detected = True
                    break

            assert detected, "Airflow did not detect test DAG within 60s"
        finally:
            # Clean up
            _kubectl(
                "exec", dag_proc, "-c", "dag-processor",
                "--", "rm", "-f", dag_path,
                check=False,
            )
