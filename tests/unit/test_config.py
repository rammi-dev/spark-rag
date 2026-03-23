"""Unit tests for config loading."""

import os
import textwrap
from pathlib import Path

import pytest

from spark_rag.config import Config, load_config


@pytest.fixture
def config_file(tmp_path):
    """Write a minimal config.yaml and return its path."""
    content = textwrap.dedent("""\
        ollama:
          url: "http://ollama.test:11434"
          embedding_model: "test-embed"
        milvus:
          url: "http://milvus.test:19530"
        synthesis:
          enabled: false
          provider: "claude"
          model: "claude-sonnet-4-20250514"
        spark_versions:
          baseline: "4.1.0"
          available:
            - tag: "v4.1.0"
              version: "4.1.0"
            - tag: "v3.5.4"
              version: "3.5.4"
        ingestion:
          spark_code:
            repo_url: "https://github.com/apache/spark.git"
            paths: ["sql/", "core/"]
          spark_docs:
            base_url: "https://spark.apache.org/docs/"
          stackoverflow:
            tags: ["apache-spark"]
            filter: "answered"
            max_questions: 100
          github_issues:
            repo: "apache/spark"
            state: "all"
            include_comments: true
            max_issues: 500
    """)
    p = tmp_path / "config.yaml"
    p.write_text(content)
    return p


class TestLoadConfig:
    def test_loads_from_yaml(self, config_file):
        cfg = load_config(config_file)
        assert cfg.ollama.url == "http://ollama.test:11434"
        assert cfg.ollama.embedding_model == "test-embed"
        assert cfg.milvus.url == "http://milvus.test:19530"

    def test_synthesis_defaults_off(self, config_file):
        cfg = load_config(config_file)
        assert cfg.synthesis.enabled is False
        assert cfg.synthesis.provider == "claude"

    def test_spark_versions(self, config_file):
        cfg = load_config(config_file)
        assert cfg.spark_versions.baseline == "4.1.0"
        assert len(cfg.spark_versions.available) == 2
        assert cfg.spark_versions.available[0].version == "4.1.0"
        assert cfg.spark_versions.available[0].tag == "v4.1.0"

    def test_get_tag(self, config_file):
        cfg = load_config(config_file)
        assert cfg.spark_versions.get_tag("4.1.0") == "v4.1.0"
        assert cfg.spark_versions.get_tag("3.5.4") == "v3.5.4"
        assert cfg.spark_versions.get_tag("9.9.9") is None

    def test_ingestion_config(self, config_file):
        cfg = load_config(config_file)
        assert cfg.ingestion.spark_code.paths == ["sql/", "core/"]
        assert cfg.ingestion.stackoverflow.max_questions == 100
        assert cfg.ingestion.github_issues.state == "all"
        assert cfg.ingestion.github_issues.include_comments is True
        assert cfg.ingestion.github_issues.max_issues == 500


class TestEnvOverrides:
    def test_ollama_url_override(self, config_file, monkeypatch):
        monkeypatch.setenv("OLLAMA_URL", "http://override:11434")
        cfg = load_config(config_file)
        assert cfg.ollama.url == "http://override:11434"

    def test_milvus_url_override(self, config_file, monkeypatch):
        monkeypatch.setenv("MILVUS_URL", "http://override:19530")
        cfg = load_config(config_file)
        assert cfg.milvus.url == "http://override:19530"

    def test_synthesis_enabled_override(self, config_file, monkeypatch):
        monkeypatch.setenv("SYNTHESIS_ENABLED", "true")
        cfg = load_config(config_file)
        assert cfg.synthesis.enabled is True

    def test_synthesis_model_override(self, config_file, monkeypatch):
        monkeypatch.setenv("SYNTHESIS_MODEL", "claude-opus-4-20250514")
        cfg = load_config(config_file)
        assert cfg.synthesis.model == "claude-opus-4-20250514"


class TestMissingConfig:
    def test_missing_file_returns_defaults(self, tmp_path):
        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert cfg.ollama.url == "http://localhost:11434"
        assert cfg.milvus.url == "http://localhost:19530"
        assert cfg.synthesis.enabled is False

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        cfg = load_config(p)
        assert isinstance(cfg, Config)


class TestProjectConfig:
    def test_loads_real_config(self):
        """Load the actual project config.yaml."""
        cfg = load_config()
        assert cfg.spark_versions.baseline == "4.1.0"
        assert len(cfg.spark_versions.available) >= 1
        assert cfg.ingestion.github_issues.repo == "apache/spark"
