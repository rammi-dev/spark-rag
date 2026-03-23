"""FastAPI application — spark-rag query service."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pymilvus import MilvusClient

from spark_rag.config import load_config
from spark_rag.embedding.client import EmbeddingClient
from spark_rag.milvus.collections import get_all_collection_info
from spark_rag.synthesis import create_provider
from spark_rag.api.analyzer import AnalyzeRequest, AnalyzeResponse, analyze

logger = logging.getLogger(__name__)

# Global state (set during lifespan)
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load config, connect to Milvus + Ollama, create synthesis provider."""
    cfg = load_config()

    embedding_client = EmbeddingClient(cfg.ollama)
    milvus_client = MilvusClient(uri=cfg.milvus.url)
    synthesis_provider = create_provider(cfg.synthesis)

    # Health checks
    if not embedding_client.check_health():
        logger.warning("Ollama not reachable at %s — embedding will fail", cfg.ollama.url)

    try:
        milvus_client.get_server_version()
    except Exception as e:
        logger.warning("Milvus not reachable at %s: %s", cfg.milvus.url, e)

    _state["config"] = cfg
    _state["embedding"] = embedding_client
    _state["milvus"] = milvus_client
    _state["synthesis"] = synthesis_provider

    logger.info(
        "spark-rag started — synthesis: %s, baseline version: %s",
        synthesis_provider.name, cfg.spark_versions.baseline,
    )

    yield

    milvus_client.close()
    _state.clear()


app = FastAPI(
    title="spark-rag",
    description="Spark expert — ask questions, troubleshoot from logs, analyze code",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Request/Response models ─────────────────────────────────────────

class AnalyzeRequestModel(BaseModel):
    input: str = Field(..., description="Spark question, code snippet, or error logs")
    version: str | None = Field(None, description="Spark version filter (e.g. '4.1.0', 'all')")


class PatternModel(BaseModel):
    name: str
    risk: str
    description: str
    line: int | None = None


class ReferenceModel(BaseModel):
    source: str
    content: str
    score: float
    metadata: dict = {}


class AnalyzeResponseModel(BaseModel):
    input_type: str
    detected_patterns: list[PatternModel]
    error_types: list[str]
    references: list[ReferenceModel]
    analysis: str | None
    version: str | None


class HealthModel(BaseModel):
    status: str
    ollama: bool
    milvus: bool
    milvus_version: str | None
    synthesis: str
    collections: list[dict]


# ── Endpoints ───────────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponseModel)
async def analyze_endpoint(request: AnalyzeRequestModel):
    """Analyze Spark code, logs, or question."""
    if not request.input.strip():
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    req = AnalyzeRequest(
        input=request.input,
        version=request.version,
    )

    result = await analyze(
        request=req,
        embedding_client=_state["embedding"],
        milvus_client=_state["milvus"],
        synthesis_provider=_state["synthesis"],
        default_version=_state["config"].spark_versions.baseline,
    )

    return AnalyzeResponseModel(
        input_type=result.input_type,
        detected_patterns=[
            PatternModel(**p) for p in result.detected_patterns
        ],
        error_types=result.error_types,
        references=[
            ReferenceModel(
                source=r.source,
                content=r.content,
                score=r.score,
                metadata=r.metadata,
            )
            for r in result.references
        ],
        analysis=result.analysis,
        version=result.version,
    )


@app.get("/health", response_model=HealthModel)
async def health():
    """Health check — reports status of all dependencies."""
    embedding: EmbeddingClient = _state.get("embedding")
    milvus: MilvusClient = _state.get("milvus")
    synthesis = _state.get("synthesis")

    ollama_ok = embedding.check_health() if embedding else False

    milvus_ok = False
    milvus_version = None
    collections = []
    if milvus:
        try:
            milvus_version = milvus.get_server_version()
            milvus_ok = True
            for info in get_all_collection_info(milvus):
                collections.append({
                    "name": info.name,
                    "exists": info.exists,
                    "num_entities": info.num_entities,
                })
        except Exception:
            pass

    status = "healthy" if (ollama_ok and milvus_ok) else "degraded"

    return HealthModel(
        status=status,
        ollama=ollama_ok,
        milvus=milvus_ok,
        milvus_version=milvus_version,
        synthesis=synthesis.name if synthesis else "none",
        collections=collections,
    )
