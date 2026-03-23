"""Multi-collection search with version filtering and re-ranking.

Query pipeline:
1. Search all 4 collections in parallel
2. Apply collection-specific weights based on input type
3. Re-rank combining: vector similarity, API overlap, metadata signals
4. Return merged, ranked results
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from pymilvus import MilvusClient

from spark_rag.milvus.collections import COLLECTION_NAMES, DIMENSION

logger = logging.getLogger(__name__)

InputType = Literal["code", "logs", "question"]

# Per-collection weight by input type
COLLECTION_WEIGHTS: dict[InputType, dict[str, float]] = {
    "code": {"spark_code": 0.4, "spark_docs": 0.3, "spark_stackoverflow": 0.2, "spark_issues": 0.1},
    "logs": {"spark_code": 0.15, "spark_docs": 0.2, "spark_stackoverflow": 0.35, "spark_issues": 0.3},
    "question": {"spark_code": 0.2, "spark_docs": 0.35, "spark_stackoverflow": 0.3, "spark_issues": 0.15},
}

SEARCH_PARAMS = {"metric_type": "COSINE", "params": {"ef": 64}}

# Output fields per collection
_OUTPUT_FIELDS = {
    "spark_code": ["content", "spark_version", "file_path", "language", "chunk_type",
                    "qualified_name", "signature", "spark_apis", "problem_indicators"],
    "spark_docs": ["content", "spark_version", "doc_url", "doc_section",
                    "heading_hierarchy", "content_type", "related_configs"],
    "spark_stackoverflow": ["content", "question_id", "is_question", "score",
                             "is_accepted", "tags", "error_type",
                             "spark_apis_mentioned", "spark_versions_mentioned"],
    "spark_issues": ["content", "issue_number", "state", "is_comment",
                      "parent_issue_number", "author", "labels",
                      "created_at", "closed_at", "spark_versions_mentioned", "linked_prs"],
}


@dataclass
class SearchHit:
    """A single search result from one collection."""
    source: str  # collection name
    content: str
    score: float  # final weighted score
    vector_similarity: float  # raw cosine similarity
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """Merged search results across all collections."""
    hits: list[SearchHit]
    query_text: str
    input_type: InputType
    version: str | None

    @property
    def top(self) -> list[SearchHit]:
        return self.hits

    def by_source(self, source: str) -> list[SearchHit]:
        return [h for h in self.hits if h.source == source]


def _build_version_filter(
    collection_name: str,
    version: str | None,
) -> str:
    """Build Milvus filter expression for version filtering."""
    if not version or version == "all":
        return ""

    # Code and docs have spark_version field — exact match
    if collection_name in ("spark_code", "spark_docs"):
        return f'spark_version == "{version}"'

    # SO and issues: boost but don't filter (version may be in content)
    # No filter applied — re-ranking handles version relevance
    return ""


def _rerank_hit(
    hit: dict,
    collection_name: str,
    input_type: InputType,
    api_overlap: set[str] | None = None,
    target_version: str | None = None,
) -> float:
    """Compute re-ranked score for a single hit.

    Combines vector similarity with metadata signals.
    """
    entity = hit.get("entity", {})
    vector_sim = hit.get("distance", 0.0)

    # Base score: vector similarity × collection weight
    weight = COLLECTION_WEIGHTS[input_type].get(collection_name, 0.1)
    score = vector_sim * weight

    # Boost: API overlap (code and SO)
    if api_overlap and collection_name in ("spark_code", "spark_stackoverflow"):
        hit_apis = set()
        for key in ("spark_apis", "spark_apis_mentioned"):
            apis_data = entity.get(key, {})
            if isinstance(apis_data, dict):
                hit_apis.update(apis_data.get("apis", []))
        overlap = len(api_overlap & hit_apis)
        if overlap:
            score += 0.05 * overlap  # small boost per overlapping API

    # Boost: SO score and accepted answer
    if collection_name == "spark_stackoverflow":
        so_score = entity.get("score", 0)
        if so_score > 10:
            score += 0.03
        if entity.get("is_accepted"):
            score += 0.05

    # Boost: closed issues (resolved problems are more useful)
    if collection_name == "spark_issues":
        if entity.get("state") == "closed":
            score += 0.02

    # Boost: version mention in SO/issues
    if target_version and collection_name in ("spark_stackoverflow", "spark_issues"):
        versions_data = entity.get("spark_versions_mentioned", {})
        if isinstance(versions_data, dict):
            mentioned = versions_data.get("versions", [])
            if any(target_version in v for v in mentioned):
                score += 0.04

    return score


def search(
    client: MilvusClient,
    query_vectors: list[list[float]],
    input_type: InputType = "question",
    version: str | None = None,
    limit_per_collection: int = 10,
    total_limit: int = 20,
    api_overlap: set[str] | None = None,
    collections: list[str] | None = None,
) -> SearchResult:
    """Search across multiple collections with re-ranking.

    Args:
        client: MilvusClient instance.
        query_vectors: List of query embedding vectors. Uses first vector
            for search; additional vectors for future multi-query support.
        input_type: Detected input type (code/logs/question).
        version: Optional Spark version filter.
        limit_per_collection: Max hits per collection.
        total_limit: Max total hits returned after re-ranking.
        api_overlap: Set of Spark API names detected in input (for boosting).
        collections: Which collections to search (defaults to all 4).

    Returns:
        SearchResult with merged, re-ranked hits.
    """
    target_collections = collections or COLLECTION_NAMES
    query_vector = query_vectors[0]  # primary query

    all_hits: list[SearchHit] = []

    for coll_name in target_collections:
        if not client.has_collection(coll_name):
            logger.warning("Collection %s not found, skipping", coll_name)
            continue

        version_filter = _build_version_filter(coll_name, version)
        output_fields = _OUTPUT_FIELDS.get(coll_name, ["content"])

        try:
            results = client.search(
                collection_name=coll_name,
                data=[query_vector],
                limit=limit_per_collection,
                filter=version_filter if version_filter else None,
                output_fields=output_fields,
                search_params=SEARCH_PARAMS,
            )
        except Exception as e:
            logger.error("Search failed on %s: %s", coll_name, e)
            continue

        for hit in results[0]:
            entity = hit.get("entity", {})
            reranked_score = _rerank_hit(
                hit, coll_name, input_type,
                api_overlap=api_overlap,
                target_version=version,
            )
            all_hits.append(SearchHit(
                source=coll_name,
                content=entity.get("content", ""),
                score=reranked_score,
                vector_similarity=hit.get("distance", 0.0),
                metadata={k: v for k, v in entity.items() if k != "content"},
            ))

    # Sort by score descending, take top N
    all_hits.sort(key=lambda h: h.score, reverse=True)

    return SearchResult(
        hits=all_hits[:total_limit],
        query_text="",  # set by caller
        input_type=input_type,
        version=version,
    )
