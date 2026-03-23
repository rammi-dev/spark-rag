"""Query pipeline: detect → parse → embed → search → rerank → synthesize.

This is the core logic behind the /analyze endpoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from spark_rag.chunking.spark_patterns import (
    PatternResult,
    analyze as analyze_patterns,
    extract_error_types,
    is_stacktrace,
)
from spark_rag.embedding.client import EmbeddingClient
from spark_rag.milvus.search import InputType, SearchResult, search
from spark_rag.synthesis.base import SynthesisInput, SynthesisProvider

from pymilvus import MilvusClient

logger = logging.getLogger(__name__)


@dataclass
class AnalyzeRequest:
    input: str
    version: str | None = None  # optional Spark version filter


@dataclass
class Reference:
    source: str
    content: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class AnalyzeResponse:
    input_type: str
    detected_patterns: list[dict]
    error_types: list[str]
    references: list[Reference]
    analysis: str | None  # None when synthesis disabled
    version: str | None


def detect_input_type(text: str) -> InputType:
    """Classify user input as code, logs, or question."""
    if is_stacktrace(text):
        return "logs"

    # Heuristic: code has more special chars and indentation
    lines = text.strip().split("\n")
    code_indicators = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("def ", "val ", "var ", "class ", "import ", "from ", "public ", "private ")):
            code_indicators += 1
        if stripped.endswith(("{", "}", ":", ")", ";")) or "=" in stripped:
            code_indicators += 1
        if stripped.startswith(("//", "#", "/*", "*")):
            code_indicators += 1

    # If >30% of lines look like code, classify as code
    if len(lines) > 0 and code_indicators / len(lines) > 0.3:
        return "code"

    return "question"


def _build_queries(text: str, input_type: InputType, pattern_result: PatternResult) -> list[str]:
    """Build 1-3 query strings for embedding.

    - Always embed the input text itself
    - For code: add a synthetic query from detected APIs/patterns
    - For logs: add the extracted error message
    """
    queries = [text]

    if input_type == "code" and pattern_result.api_names:
        # Synthetic query: "Spark [api1] [api2] usage patterns"
        top_apis = pattern_result.api_names[:5]
        queries.append(f"Spark {' '.join(top_apis)} usage patterns and best practices")

    if input_type == "logs":
        errors = extract_error_types(text)
        if errors:
            queries.append(f"Spark {errors[0]} cause and fix")

    return queries


async def analyze(
    request: AnalyzeRequest,
    embedding_client: EmbeddingClient,
    milvus_client: MilvusClient,
    synthesis_provider: SynthesisProvider,
    default_version: str | None = None,
) -> AnalyzeResponse:
    """Run the full query pipeline.

    1. Detect input type (code / logs / question)
    2. Parse input (pattern detection for code, error extraction for logs)
    3. Embed queries via Ollama
    4. Search Milvus (4 collections, parallel, with version filter)
    5. Re-rank results
    6. Optionally synthesize with Claude
    7. Return response
    """
    text = request.input
    version = request.version or default_version

    # 1. Detect input type
    input_type = detect_input_type(text)
    logger.info("Input type detected: %s", input_type)

    # 2. Parse
    pattern_result = analyze_patterns(text)
    error_types = extract_error_types(text)

    # 3. Build and embed queries
    queries = _build_queries(text, input_type, pattern_result)
    embed_result = embedding_client.embed_batch(queries)
    query_vectors = embed_result.vectors

    # 4+5. Search + re-rank
    api_overlap = set(pattern_result.api_names) if pattern_result.api_names else None
    search_result = search(
        client=milvus_client,
        query_vectors=query_vectors,
        input_type=input_type,
        version=version,
        api_overlap=api_overlap,
    )
    search_result.query_text = text

    # 6. Synthesize (if enabled)
    synthesis_input = SynthesisInput(
        user_input=text,
        input_type=input_type,
        detected_patterns=[
            {"name": p.name, "risk": p.risk.value, "description": p.description}
            for p in pattern_result.patterns
        ],
        search_result=search_result,
    )
    analysis_text = await synthesis_provider.analyze(synthesis_input)

    # 7. Build response
    references = [
        Reference(
            source=hit.source,
            content=hit.content[:2000],  # truncate for response
            score=round(hit.score, 4),
            metadata=hit.metadata,
        )
        for hit in search_result.hits
    ]

    return AnalyzeResponse(
        input_type=input_type,
        detected_patterns=[
            {"name": p.name, "risk": p.risk.value, "description": p.description, "line": p.line}
            for p in pattern_result.patterns
        ],
        error_types=error_types,
        references=references,
        analysis=analysis_text,
        version=version,
    )
