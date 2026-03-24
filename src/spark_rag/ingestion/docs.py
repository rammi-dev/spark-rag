"""One-time ingestion: Spark documentation (Markdown) → Lance.

Reads docs/*.md from the same git clone used for code ingestion.
Resolves {% include_example %} tags from examples/src/main/.

Usage:
    uv run python -m spark_rag.ingestion.docs --version 4.1.0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from spark_rag.chunking.doc_chunker import chunk_markdown
from spark_rag.config import load_config
from spark_rag.ingestion.github import checkout_version, ensure_repo
from spark_rag.lance.schemas import doc_chunks_to_table
from spark_rag.lance.store import LanceStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_examples_lookup(repo_dir: Path, examples_path: str) -> dict[str, str]:
    """Build a lookup dict for {% include_example %} resolution.

    Maps example name → file content. The Jekyll tag format is:
        {% include_example name path/to/file %}
    We index by the relative path from examples_path.
    """
    examples_dir = repo_dir / examples_path
    if not examples_dir.exists():
        logger.warning("Examples dir not found: %s", examples_dir)
        return {}

    lookup: dict[str, str] = {}
    for f in examples_dir.rglob("*"):
        if f.is_file() and f.suffix in (".py", ".scala", ".java", ".r", ".R"):
            rel = str(f.relative_to(repo_dir))
            lookup[rel] = f.read_text(encoding="utf-8", errors="replace")
    logger.info("Built examples lookup: %d files", len(lookup))
    return lookup


def main():
    parser = argparse.ArgumentParser(description="Ingest Spark documentation into Lance")
    parser.add_argument("--version", required=True, help="Spark version (e.g. 4.1.0)")
    parser.add_argument("--dry-run", action="store_true", help="Chunk and count without writing to Lance")
    args = parser.parse_args()

    cfg = load_config()
    spark_version = args.version

    # Resolve git tag
    git_tag = cfg.spark_versions.get_tag(spark_version)
    if not git_tag:
        logger.error("Version %s not in config.yaml spark_versions.available", spark_version)
        sys.exit(1)

    # Clone and checkout
    repo_dir = ensure_repo(cfg.ingestion.spark_code.repo_url)
    checkout_version(repo_dir, git_tag)

    # Build examples lookup for include_example resolution
    examples_lookup = _build_examples_lookup(repo_dir, cfg.ingestion.spark_docs.examples_path)

    # Find markdown doc files
    all_chunks = []
    doc_paths = cfg.ingestion.spark_docs.paths
    glob_pattern = cfg.ingestion.spark_docs.glob

    for rel_path in doc_paths:
        docs_dir = repo_dir / rel_path
        if not docs_dir.exists():
            logger.warning("Docs path %s not found in repo", rel_path)
            continue

        md_files = sorted(docs_dir.glob(glob_pattern))
        logger.info("Found %d markdown files in %s", len(md_files), rel_path)

        for md_file in md_files:
            try:
                content = md_file.read_text(encoding="utf-8", errors="replace")
                file_rel = str(md_file.relative_to(repo_dir))
                chunks = chunk_markdown(content, file_rel, examples_lookup=examples_lookup)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning("Failed to chunk %s: %s", md_file, e)

    logger.info("Chunked → %d doc chunks", len(all_chunks))

    if args.dry_run:
        logger.info("DRY RUN — not embedding or inserting")
        _print_stats(all_chunks)
        return

    # Embed
    logger.info("Embedding %d chunks via Ollama (%s)...", len(all_chunks), cfg.ollama.embedding_model)
    embedding_client = EmbeddingClient(cfg.ollama)
    texts = [c.content for c in all_chunks]
    embed_result = embedding_client.embed_batch(texts, batch_size=args.batch_size)
    logger.info("Embedded %d chunks (dim=%d)", len(embed_result.vectors), embed_result.dimension)

    # Build Milvus data
    data = [
        chunk.to_milvus_data(spark_version, embedding)
        for chunk, embedding in zip(all_chunks, embed_result.vectors)
    ]

    # Insert into Milvus
    milvus_client = MilvusClient(uri=cfg.milvus.url)
    create_collection(milvus_client, "spark_docs", drop_existing=False)

    count = ingest_version(milvus_client, "spark_docs", data, spark_version, batch_size=args.batch_size)
    logger.info("Ingested %d records into spark_docs for version %s", count, spark_version)

    milvus_client.close()


def _print_stats(chunks):
    from collections import Counter
    by_type = Counter(c.content_type for c in chunks)
    by_section = Counter(c.doc_section for c in chunks)
    logger.info("By content type: %s", dict(by_type))
    logger.info("Top sections: %s", dict(by_section.most_common(10)))
    total_chars = sum(len(c.content) for c in chunks)
    logger.info("Total content: %d chars (~%d tokens)", total_chars, total_chars // 4)


if __name__ == "__main__":
    main()
