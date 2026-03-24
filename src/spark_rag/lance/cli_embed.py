"""CLI: Run embedding experiment on a Lance dataset.

Usage:
    uv run python -m spark_rag.lance.cli_embed --table spark_code --model nomic-embed-text --column emb_nomic
    uv run python -m spark_rag.lance.cli_embed --table spark_docs --model nomic-embed-text --column emb_nomic --batch-size 100
"""

from __future__ import annotations

import argparse
import logging
import sys

from spark_rag.config import load_config
from spark_rag.embedding.client import EmbeddingClient
from spark_rag.lance.embeddings import EmbeddingExperiment
from spark_rag.lance.store import LanceStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Add embedding column to a Lance dataset")
    parser.add_argument("--table", required=True, help="Lance table name (e.g. spark_code)")
    parser.add_argument("--model", required=True, help="Ollama embedding model name")
    parser.add_argument("--column", required=True, help="Embedding column name (e.g. emb_nomic)")
    parser.add_argument("--batch-size", type=int, default=200, help="Embedding batch size")
    args = parser.parse_args()

    cfg = load_config()

    # Resolve dimensions from embedding_experiments config
    model_cfg = cfg.embedding_experiments.get_model(args.column)
    if model_cfg:
        dimensions = model_cfg.dimensions
    else:
        logger.warning("Column '%s' not in embedding_experiments config, assuming 768 dims", args.column)
        dimensions = 768

    store = LanceStore(cfg.lance, cfg.polaris)

    if not store.dataset_exists(args.table):
        logger.error("Lance table '%s' does not exist. Run ingestion first.", args.table)
        sys.exit(1)

    embedding_client = EmbeddingClient(cfg.ollama)
    experiment = EmbeddingExperiment(store, embedding_client, args.column, dimensions)
    count = experiment.run(args.table, batch_size=args.batch_size)
    logger.info("Done. Embedded %d rows.", count)


if __name__ == "__main__":
    main()
