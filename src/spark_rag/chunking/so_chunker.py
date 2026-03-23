"""Chunk StackOverflow Q&A data for Milvus ingestion.

Each question becomes one chunk, each answer becomes a separate chunk.
Extracts metadata: error types, Spark APIs mentioned, version references.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

from spark_rag.chunking.spark_patterns import detect_apis, extract_error_types

# Regex to find Spark version mentions like "Spark 3.5", "spark 4.1.0", "v3.5.4"
_VERSION_PATTERN = re.compile(r"(?:spark\s*)?v?(\d+\.\d+(?:\.\d+)?)", re.IGNORECASE)


@dataclass
class SOChunk:
    content: str
    question_id: int
    is_question: bool
    score: int
    is_accepted: bool
    tags: dict  # {"tags": ["apache-spark", ...]}
    error_type: str
    spark_apis_mentioned: dict  # {"apis": [...]}
    spark_versions_mentioned: dict  # {"versions": [...]}

    def to_milvus_data(self, embedding: list[float]) -> dict:
        return {
            "embedding": embedding,
            "content": self.content[:65535],
            "question_id": self.question_id,
            "is_question": self.is_question,
            "score": self.score,
            "is_accepted": self.is_accepted,
            "tags": self.tags,
            "error_type": self.error_type[:256],
            "spark_apis_mentioned": self.spark_apis_mentioned,
            "spark_versions_mentioned": self.spark_versions_mentioned,
        }


def _html_to_text(html: str) -> str:
    """Strip HTML tags, return plain text."""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def _extract_versions(text: str) -> list[str]:
    """Extract Spark version references from text."""
    matches = _VERSION_PATTERN.findall(text)
    # Filter to plausible Spark versions (1.x - 9.x)
    versions = []
    seen = set()
    for v in matches:
        major = int(v.split(".")[0])
        if 1 <= major <= 9 and v not in seen:
            versions.append(v)
            seen.add(v)
    return versions


def chunk_question(item: dict) -> list[SOChunk]:
    """Chunk a StackOverflow API response item into question + answer chunks.

    Args:
        item: Dict from StackExchange API /questions endpoint with
              filter=withbody. Expected keys:
              - question_id, title, body, score, tags
              - answers: list of {answer_id, body, score, is_accepted}

    Returns:
        List of SOChunks (one for question, one per answer).
    """
    chunks: list[SOChunk] = []
    tags = item.get("tags", [])
    question_id = item["question_id"]

    # Question chunk
    title = item.get("title", "")
    body_html = item.get("body", "")
    body_text = _html_to_text(body_html)
    question_text = f"{title}\n\n{body_text}"

    errors = extract_error_types(question_text)
    apis = detect_apis(question_text)
    versions = _extract_versions(question_text)

    chunks.append(SOChunk(
        content=question_text,
        question_id=question_id,
        is_question=True,
        score=item.get("score", 0),
        is_accepted=False,
        tags={"tags": tags},
        error_type=errors[0] if errors else "",
        spark_apis_mentioned={"apis": [a.api for a in apis]},
        spark_versions_mentioned={"versions": versions},
    ))

    # Answer chunks
    for answer in item.get("answers", []):
        answer_html = answer.get("body", "")
        answer_text = _html_to_text(answer_html)

        a_errors = extract_error_types(answer_text)
        a_apis = detect_apis(answer_text)
        a_versions = _extract_versions(answer_text)

        chunks.append(SOChunk(
            content=answer_text,
            question_id=question_id,
            is_question=False,
            score=answer.get("score", 0),
            is_accepted=answer.get("is_accepted", False),
            tags={"tags": tags},
            error_type=a_errors[0] if a_errors else "",
            spark_apis_mentioned={"apis": [a.api for a in a_apis]},
            spark_versions_mentioned={"versions": a_versions},
        ))

    return chunks
