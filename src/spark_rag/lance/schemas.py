"""PyArrow schemas and chunk-to-table converters for Lance datasets."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone

import pyarrow as pa

# Fixed namespace UUID for deterministic chunk_id generation
_NAMESPACE_UUID = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def _chunk_id(key: str) -> str:
    """Generate stable UUID5 from a deterministic key string."""
    return str(uuid.uuid5(_NAMESPACE_UUID, key))


def _content_hash(content: str) -> str:
    """Short hash of content for dedup keys."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _unwrap_list(d: dict, key: str) -> list:
    """Extract list from JSON dict wrapper like {"apis": [...]}."""
    if not d:
        return []
    # The dict has one key wrapping the list, e.g. {"apis": [...]}
    for v in d.values():
        if isinstance(v, list):
            return v
    return []


# --- Code chunks ---

def code_chunks_to_table(chunks, spark_version: str) -> pa.Table:
    """Convert CodeChunk list to PyArrow Table."""
    now = _now()
    rows = {
        "chunk_id": [],
        "content": [],
        "ingested_at": [],
        "spark_version": [],
        "file_path": [],
        "language": [],
        "chunk_type": [],
        "qualified_name": [],
        "signature": [],
        "spark_apis": [],
        "problem_indicators": [],
        "start_line": [],
        "end_line": [],
    }
    for c in chunks:
        rows["chunk_id"].append(
            _chunk_id(f"{spark_version}:{c.file_path}:{c.qualified_name}:{c.start_line}")
        )
        rows["content"].append(c.content)
        rows["ingested_at"].append(now)
        rows["spark_version"].append(spark_version)
        rows["file_path"].append(c.file_path)
        rows["language"].append(c.language)
        rows["chunk_type"].append(c.chunk_type)
        rows["qualified_name"].append(c.qualified_name)
        rows["signature"].append(c.signature)
        rows["spark_apis"].append(_unwrap_list(c.spark_apis, "apis"))
        indicators = _unwrap_list(c.problem_indicators, "patterns")
        rows["problem_indicators"].append([p.get("name", "") for p in indicators] if indicators else [])
        rows["start_line"].append(c.start_line)
        rows["end_line"].append(c.end_line)

    return pa.table(
        rows,
        schema=pa.schema([
            pa.field("chunk_id", pa.utf8(), nullable=False),
            pa.field("content", pa.large_utf8(), nullable=False),
            pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("spark_version", pa.utf8()),
            pa.field("file_path", pa.utf8()),
            pa.field("language", pa.utf8()),
            pa.field("chunk_type", pa.utf8()),
            pa.field("qualified_name", pa.utf8()),
            pa.field("signature", pa.utf8()),
            pa.field("spark_apis", pa.list_(pa.utf8())),
            pa.field("problem_indicators", pa.list_(pa.utf8())),
            pa.field("start_line", pa.int32()),
            pa.field("end_line", pa.int32()),
        ]),
    )


# --- Doc chunks ---

def doc_chunks_to_table(chunks, spark_version: str) -> pa.Table:
    """Convert DocChunk list to PyArrow Table."""
    now = _now()
    rows = {
        "chunk_id": [],
        "content": [],
        "ingested_at": [],
        "spark_version": [],
        "doc_url": [],
        "doc_section": [],
        "heading_hierarchy": [],
        "content_type": [],
        "related_configs": [],
    }
    for c in chunks:
        rows["chunk_id"].append(
            _chunk_id(f"{spark_version}:{c.doc_url}:{c.doc_section}:{_content_hash(c.content)}")
        )
        rows["content"].append(c.content)
        rows["ingested_at"].append(now)
        rows["spark_version"].append(spark_version)
        rows["doc_url"].append(c.doc_url)
        rows["doc_section"].append(c.doc_section)
        rows["heading_hierarchy"].append(c.heading_hierarchy if c.heading_hierarchy else [])
        rows["content_type"].append(c.content_type)
        rows["related_configs"].append(_unwrap_list(c.related_configs, "configs"))

    return pa.table(
        rows,
        schema=pa.schema([
            pa.field("chunk_id", pa.utf8(), nullable=False),
            pa.field("content", pa.large_utf8(), nullable=False),
            pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("spark_version", pa.utf8()),
            pa.field("doc_url", pa.utf8()),
            pa.field("doc_section", pa.utf8()),
            pa.field("heading_hierarchy", pa.list_(pa.utf8())),
            pa.field("content_type", pa.utf8()),
            pa.field("related_configs", pa.list_(pa.utf8())),
        ]),
    )


# --- StackOverflow chunks ---

def so_chunks_to_table(chunks) -> pa.Table:
    """Convert SOChunk list to PyArrow Table."""
    now = _now()
    rows = {
        "chunk_id": [],
        "content": [],
        "ingested_at": [],
        "question_id": [],
        "is_question": [],
        "score": [],
        "is_accepted": [],
        "tags": [],
        "error_type": [],
        "spark_apis_mentioned": [],
        "spark_versions_mentioned": [],
    }
    for c in chunks:
        q_or_a = "q" if c.is_question else "a"
        rows["chunk_id"].append(
            _chunk_id(f"so:{c.question_id}:{q_or_a}:{_content_hash(c.content)}")
        )
        rows["content"].append(c.content)
        rows["ingested_at"].append(now)
        rows["question_id"].append(c.question_id)
        rows["is_question"].append(c.is_question)
        rows["score"].append(c.score)
        rows["is_accepted"].append(c.is_accepted)
        rows["tags"].append(_unwrap_list(c.tags, "tags"))
        rows["error_type"].append(c.error_type)
        rows["spark_apis_mentioned"].append(_unwrap_list(c.spark_apis_mentioned, "apis"))
        rows["spark_versions_mentioned"].append(_unwrap_list(c.spark_versions_mentioned, "versions"))

    return pa.table(
        rows,
        schema=pa.schema([
            pa.field("chunk_id", pa.utf8(), nullable=False),
            pa.field("content", pa.large_utf8(), nullable=False),
            pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("question_id", pa.int64()),
            pa.field("is_question", pa.bool_()),
            pa.field("score", pa.int64()),
            pa.field("is_accepted", pa.bool_()),
            pa.field("tags", pa.list_(pa.utf8())),
            pa.field("error_type", pa.utf8()),
            pa.field("spark_apis_mentioned", pa.list_(pa.utf8())),
            pa.field("spark_versions_mentioned", pa.list_(pa.utf8())),
        ]),
    )


# --- Issue chunks ---

def issue_chunks_to_table(chunks) -> pa.Table:
    """Convert IssueChunk list to PyArrow Table."""
    now = _now()
    rows = {
        "chunk_id": [],
        "content": [],
        "ingested_at": [],
        "issue_number": [],
        "state": [],
        "is_comment": [],
        "parent_issue_number": [],
        "author": [],
        "labels": [],
        "created_at": [],
        "closed_at": [],
        "spark_versions_mentioned": [],
        "linked_prs": [],
    }
    for c in chunks:
        role = "comment" if c.is_comment else "body"
        rows["chunk_id"].append(
            _chunk_id(f"issue:{c.issue_number}:{role}:{c.author}:{_content_hash(c.content)}")
        )
        rows["content"].append(c.content)
        rows["ingested_at"].append(now)
        rows["issue_number"].append(c.issue_number)
        rows["state"].append(c.state)
        rows["is_comment"].append(c.is_comment)
        rows["parent_issue_number"].append(c.parent_issue_number)
        rows["author"].append(c.author)
        rows["labels"].append(_unwrap_list(c.labels, "labels"))
        rows["created_at"].append(c.created_at)
        rows["closed_at"].append(c.closed_at)
        rows["spark_versions_mentioned"].append(_unwrap_list(c.spark_versions_mentioned, "versions"))
        pr_list = _unwrap_list(c.linked_prs, "prs")
        rows["linked_prs"].append([int(p) for p in pr_list] if pr_list else [])

    return pa.table(
        rows,
        schema=pa.schema([
            pa.field("chunk_id", pa.utf8(), nullable=False),
            pa.field("content", pa.large_utf8(), nullable=False),
            pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("issue_number", pa.int64()),
            pa.field("state", pa.utf8()),
            pa.field("is_comment", pa.bool_()),
            pa.field("parent_issue_number", pa.int64()),
            pa.field("author", pa.utf8()),
            pa.field("labels", pa.list_(pa.utf8())),
            pa.field("created_at", pa.utf8()),
            pa.field("closed_at", pa.utf8()),
            pa.field("spark_versions_mentioned", pa.list_(pa.utf8())),
            pa.field("linked_prs", pa.list_(pa.int64())),
        ]),
    )
