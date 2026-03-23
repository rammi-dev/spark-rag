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
    )

    _apply_env_overrides(cfg)
    return cfg
