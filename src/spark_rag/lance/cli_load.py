"""CLI: Load Lance dataset → Milvus.

Usage:
    uv run python -m spark_rag.lance.cli_load --table spark_code --embedding-column emb_nomic --version 4.1.0
    uv run python -m spark_rag.lance.cli_load --table spark_docs --embedding-column emb_nomic --version 4.1.0
    uv run python -m spark_rag.lance.cli_load --table spark_so --embedding-column emb_nomic
"""

from __future__ import annotations

import argparse
import logging
import sys

from pymilvus import MilvusClient

from spark_rag.config import load_config
from spark_rag.lance.loader import LanceMilvusLoader
from spark_rag.lance.store import LanceStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Load Lance dataset into Milvus")
    parser.add_argument("--table", required=True, help="Lance table name (e.g. spark_code)")
    parser.add_argument("--embedding-column", required=True, help="Embedding column (e.g. emb_nomic)")
    parser.add_argument("--version", default=None, help="Spark version filter (for code/docs)")
    parser.add_argument("--batch-size", type=int, default=500, help="Milvus insert batch size")
    parser.add_argument("--drop-existing", action="store_true", help="Recreate Milvus collection")
    args = parser.parse_args()

    cfg = load_config()

    store = LanceStore(cfg.lance, cfg.polaris)

    if not store.dataset_exists(args.table):
        logger.error("Lance table '%s' does not exist. Run ingestion first.", args.table)
        sys.exit(1)

    milvus_client = MilvusClient(uri=cfg.milvus.url)
    loader = LanceMilvusLoader(store, milvus_client)

    count = loader.load(
        args.table,
        args.embedding_column,
        spark_version=args.version,
        batch_size=args.batch_size,
        drop_existing=args.drop_existing,
    )
    logger.info("Done. Loaded %d rows into Milvus.", count)
    milvus_client.close()


if __name__ == "__main__":
    main()
