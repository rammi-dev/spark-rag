"""Load project configuration from config.yaml with env var overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


@dataclass
class OllamaConfig:
    url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"


@dataclass
class MilvusConfig:
    url: str = "http://localhost:19530"


@dataclass
class SynthesisConfig:
    enabled: bool = False
    provider: str = "claude"
    model: str = "claude-sonnet-4-20250514"


@dataclass
class SparkVersion:
    tag: str  # git tag, e.g. "v4.1.0"
    version: str  # semver, e.g. "4.1.0"


@dataclass
class SparkVersionsConfig:
    baseline: str = "4.1.0"
    available: list[SparkVersion] = field(default_factory=list)

    def get_tag(self, version: str) -> str | None:
        """Get git tag for a version string."""
        for v in self.available:
            if v.version == version:
                return v.tag
        return None


@dataclass
class SparkCodeIngestion:
    repo_url: str = "https://github.com/apache/spark.git"
    paths: list[str] = field(default_factory=lambda: ["sql/", "core/", "mllib/", "python/pyspark/"])


@dataclass
class SparkDocsIngestion:
    paths: list[str] = field(default_factory=lambda: ["docs/"])
    examples_path: str = "examples/src/main/"
    glob: str = "*.md"


@dataclass
class StackOverflowIngestion:
    tags: list[str] = field(default_factory=lambda: ["apache-spark"])
    filter: str = "answered"
    max_questions: int = 5000


@dataclass
class GitHubIssuesIngestion:
    repo: str = "apache/spark"
    state: str = "all"
    include_comments: bool = True
    max_issues: int = 10000


@dataclass
class PolarisConfig:
    endpoint: str = "http://localhost:8181"
    catalog: str = "spark_rag_catalog"
    namespace: str = "spark_rag"
    client_id: str = "root"
    client_secret_env: str = "POLARIS_CLIENT_SECRET"

    @property
    def client_secret(self) -> str:
        return os.environ.get(self.client_secret_env, "")


@dataclass
class LanceConfig:
    s3_endpoint: str = "http://localhost:8080"
    s3_bucket: str = "spark-rag"
    s3_prefix: str = "lance"
    s3_access_key_env: str = "CEPH_ACCESS_KEY"
    s3_secret_key_env: str = "CEPH_SECRET_KEY"
    s3_region: str = "us-east-1"

    @property
    def s3_access_key(self) -> str:
        return os.environ.get(self.s3_access_key_env, "")

    @property
    def s3_secret_key(self) -> str:
        return os.environ.get(self.s3_secret_key_env, "")

    @property
    def storage_options(self) -> dict[str, str]:
        return {
            "aws_endpoint": self.s3_endpoint,
            "aws_access_key_id": self.s3_access_key,
            "aws_secret_access_key": self.s3_secret_key,
            "aws_region": self.s3_region,
            "allow_http": "true",
            "aws_virtual_hosted_style_request": "false",
        }


@dataclass
class EmbeddingModelConfig:
    name: str = "nomic-embed-text"
    column: str = "emb_nomic"
    dimensions: int = 768


@dataclass
class EmbeddingExperimentsConfig:
    models: list[EmbeddingModelConfig] = field(default_factory=lambda: [EmbeddingModelConfig()])
    active: str = "emb_nomic"

    def get_model(self, column: str) -> EmbeddingModelConfig | None:
        for m in self.models:
            if m.column == column:
                return m
        return None


@dataclass
class IngestionConfig:
    spark_code: SparkCodeIngestion = field(default_factory=SparkCodeIngestion)
    spark_docs: SparkDocsIngestion = field(default_factory=SparkDocsIngestion)
    stackoverflow: StackOverflowIngestion = field(default_factory=StackOverflowIngestion)
    github_issues: GitHubIssuesIngestion = field(default_factory=GitHubIssuesIngestion)


@dataclass
class Config:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    spark_versions: SparkVersionsConfig = field(default_factory=SparkVersionsConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    polaris: PolarisConfig = field(default_factory=PolarisConfig)
    lance: LanceConfig = field(default_factory=LanceConfig)
    embedding_experiments: EmbeddingExperimentsConfig = field(default_factory=EmbeddingExperimentsConfig)


def _apply_env_overrides(cfg: Config) -> None:
    """Override config values from environment variables."""
    if url := os.environ.get("OLLAMA_URL"):
        cfg.ollama.url = url
    if model := os.environ.get("OLLAMA_EMBEDDING_MODEL"):
        cfg.ollama.embedding_model = model
    if url := os.environ.get("MILVUS_URL"):
        cfg.milvus.url = url
    if os.environ.get("SYNTHESIS_ENABLED", "").lower() in ("true", "1"):
        cfg.synthesis.enabled = True
    if model := os.environ.get("SYNTHESIS_MODEL"):
        cfg.synthesis.model = model
    if url := os.environ.get("POLARIS_ENDPOINT"):
        cfg.polaris.endpoint = url
    if url := os.environ.get("LANCE_S3_ENDPOINT"):
        cfg.lance.s3_endpoint = url
    if bucket := os.environ.get("LANCE_S3_BUCKET"):
        cfg.lance.s3_bucket = bucket


def load_config(path: Path | str | None = None) -> Config:
    """Load config from YAML file with env var overrides.

    Args:
        path: Path to config.yaml. Defaults to project root config.yaml.

    Returns:
        Populated Config dataclass.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        cfg = Config()
        _apply_env_overrides(cfg)
        return cfg

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    # Build config from YAML
    ollama_raw = raw.get("ollama", {})
    milvus_raw = raw.get("milvus", {})
    synthesis_raw = raw.get("synthesis", {})
    versions_raw = raw.get("spark_versions", {})
    ingestion_raw = raw.get("ingestion", {})
    polaris_raw = raw.get("polaris", {})
    lance_raw = raw.get("lance", {})
    emb_exp_raw = raw.get("embedding_experiments", {})

    cfg = Config(
        ollama=OllamaConfig(
            url=ollama_raw.get("url", OllamaConfig.url),
            embedding_model=ollama_raw.get("embedding_model", OllamaConfig.embedding_model),
        ),
        milvus=MilvusConfig(
            url=milvus_raw.get("url", MilvusConfig.url),
        ),
        synthesis=SynthesisConfig(
            enabled=synthesis_raw.get("enabled", False),
            provider=synthesis_raw.get("provider", "claude"),
            model=synthesis_raw.get("model", SynthesisConfig.model),
        ),
        spark_versions=SparkVersionsConfig(
            baseline=versions_raw.get("baseline", "4.1.0"),
            available=[
                SparkVersion(tag=v["tag"], version=v["version"])
                for v in versions_raw.get("available", [])
            ],
        ),
        ingestion=IngestionConfig(
            spark_code=SparkCodeIngestion(
                repo_url=ingestion_raw.get("spark_code", {}).get("repo_url", SparkCodeIngestion.repo_url),
                paths=ingestion_raw.get("spark_code", {}).get("paths", ["sql/", "core/", "mllib/", "python/pyspark/"]),
            ) if "spark_code" in ingestion_raw else SparkCodeIngestion(),
            spark_docs=SparkDocsIngestion(
                paths=ingestion_raw.get("spark_docs", {}).get("paths", ["docs/"]),
                examples_path=ingestion_raw.get("spark_docs", {}).get("examples_path", "examples/src/main/"),
                glob=ingestion_raw.get("spark_docs", {}).get("glob", "*.md"),
            ) if "spark_docs" in ingestion_raw else SparkDocsIngestion(),
            stackoverflow=StackOverflowIngestion(
                tags=ingestion_raw.get("stackoverflow", {}).get("tags", ["apache-spark"]),
                filter=ingestion_raw.get("stackoverflow", {}).get("filter", "answered"),
                max_questions=ingestion_raw.get("stackoverflow", {}).get("max_questions", 5000),
            ) if "stackoverflow" in ingestion_raw else StackOverflowIngestion(),
            github_issues=GitHubIssuesIngestion(
                repo=ingestion_raw.get("github_issues", {}).get("repo", "apache/spark"),
                state=ingestion_raw.get("github_issues", {}).get("state", "all"),
                include_comments=ingestion_raw.get("github_issues", {}).get("include_comments", True),
                max_issues=ingestion_raw.get("github_issues", {}).get("max_issues", 10000),
            ) if "github_issues" in ingestion_raw else GitHubIssuesIngestion(),
        ),
        polaris=PolarisConfig(
            endpoint=polaris_raw.get("endpoint", PolarisConfig.endpoint),
            catalog=polaris_raw.get("catalog", PolarisConfig.catalog),
            namespace=polaris_raw.get("namespace", PolarisConfig.namespace),
            client_id=polaris_raw.get("client_id", PolarisConfig.client_id),
            client_secret_env=polaris_raw.get("client_secret_env", PolarisConfig.client_secret_env),
        ) if polaris_raw else PolarisConfig(),
        lance=LanceConfig(
            s3_endpoint=lance_raw.get("s3_endpoint", LanceConfig.s3_endpoint),
            s3_bucket=lance_raw.get("s3_bucket", LanceConfig.s3_bucket),
            s3_prefix=lance_raw.get("s3_prefix", LanceConfig.s3_prefix),
            s3_access_key_env=lance_raw.get("s3_access_key_env", LanceConfig.s3_access_key_env),
            s3_secret_key_env=lance_raw.get("s3_secret_key_env", LanceConfig.s3_secret_key_env),
            s3_region=lance_raw.get("s3_region", LanceConfig.s3_region),
        ) if lance_raw else LanceConfig(),
        embedding_experiments=EmbeddingExperimentsConfig(
            models=[
                EmbeddingModelConfig(
                    name=m["name"], column=m["column"], dimensions=m["dimensions"],
                )
                for m in emb_exp_raw.get("models", [])
            ] if emb_exp_raw.get("models") else [EmbeddingModelConfig()],
            active=emb_exp_raw.get("active", "emb_nomic"),
        ) if emb_exp_raw else EmbeddingExperimentsConfig(),
    )

    _apply_env_overrides(cfg)
    return cfg
