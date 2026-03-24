"""One-time ingestion: Spark source code → Lance.

Usage:
    uv run python -m spark_rag.ingestion.code --version 4.1.0
    uv run python -m spark_rag.ingestion.code --version 3.5.4 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys

from spark_rag.chunking.code_chunker import chunk_file, detect_language
from spark_rag.config import load_config
from spark_rag.ingestion.github import checkout_version, ensure_repo, iter_files
from spark_rag.lance.schemas import code_chunks_to_table
from spark_rag.lance.store import LanceStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest Spark source code into Lance")
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

    # Find source files
    code_extensions = {".scala", ".java", ".py"}
    files = iter_files(repo_dir, cfg.ingestion.spark_code.paths, code_extensions)
    logger.info("Found %d source files for version %s", len(files), spark_version)

    # Chunk all files
    all_chunks = []
    skipped = 0
    for filepath in files:
        lang = detect_language(str(filepath))
        if not lang:
            skipped += 1
            continue
        try:
            source = filepath.read_text(encoding="utf-8", errors="replace")
            chunks = chunk_file(source, str(filepath.relative_to(repo_dir)), lang)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.warning("Failed to chunk %s: %s", filepath, e)
            skipped += 1

    logger.info("Chunked %d files → %d chunks (skipped %d)", len(files), len(all_chunks), skipped)

    if args.dry_run:
        logger.info("DRY RUN — not embedding or inserting")
        _print_stats(all_chunks)
        return

    # Write to Lance
    logger.info("Writing %d chunks to Lance table 'spark_code'...", len(all_chunks))
    table = code_chunks_to_table(all_chunks, spark_version)
    store = LanceStore(cfg.lance, cfg.polaris)
    count = store.replace_version("spark_code", spark_version, table)
    logger.info("Wrote %d rows to Lance for version %s", count, spark_version)


def _print_stats(chunks):
    """Print chunk statistics for dry run."""
    from collections import Counter
    by_type = Counter(c.chunk_type for c in chunks)
    by_lang = Counter(c.language for c in chunks)
    logger.info("By chunk type: %s", dict(by_type))
    logger.info("By language: %s", dict(by_lang))
    total_chars = sum(len(c.content) for c in chunks)
    logger.info("Total content: %d chars (~%d tokens)", total_chars, total_chars // 4)


if __name__ == "__main__":
    main()
