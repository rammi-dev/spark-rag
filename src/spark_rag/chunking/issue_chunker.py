"""Chunk GitHub Issues + comments for Milvus ingestion.

Each issue body becomes one chunk, each comment becomes a separate chunk
linked to its parent issue. Extracts version references from labels,
milestone, and content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Reuse version extraction from SO chunker
from spark_rag.chunking.so_chunker import _extract_versions


@dataclass
class IssueChunk:
    content: str
    issue_number: int
    state: str  # "open" or "closed"
    is_comment: bool
    parent_issue_number: int
    author: str
    labels: dict  # {"labels": ["bug", "SQL"]}
    created_at: str  # ISO 8601
    closed_at: str  # ISO 8601 or ""
    spark_versions_mentioned: dict  # {"versions": ["4.1.0"]}
    linked_prs: dict  # {"prs": [12345]}

    def to_milvus_data(self, embedding: list[float]) -> dict:
        return {
            "embedding": embedding,
            "content": self.content[:65535],
            "issue_number": self.issue_number,
            "state": self.state[:16],
            "is_comment": self.is_comment,
            "parent_issue_number": self.parent_issue_number,
            "author": self.author[:128],
            "labels": self.labels,
            "created_at": self.created_at[:32],
            "closed_at": self.closed_at[:32],
            "spark_versions_mentioned": self.spark_versions_mentioned,
            "linked_prs": self.linked_prs,
        }


# Pattern to find PR references like #12345, GH-12345, or full URLs
_PR_PATTERN = re.compile(
    r"(?:#(\d{4,6}))|"
    r"(?:GH-(\d{4,6}))|"
    r"(?:github\.com/apache/spark/pull/(\d+))"
)


def _extract_pr_refs(text: str) -> list[int]:
    """Extract referenced PR numbers from text."""
    prs: set[int] = set()
    for match in _PR_PATTERN.finditer(text):
        for group in match.groups():
            if group:
                prs.add(int(group))
    return sorted(prs)


def _extract_versions_from_labels(labels: list[str], milestone: str | None) -> list[str]:
    """Extract Spark versions from issue labels and milestone."""
    versions: list[str] = []
    version_pattern = re.compile(r"(\d+\.\d+(?:\.\d+)?)")

    for label in labels:
        m = version_pattern.search(label)
        if m:
            versions.append(m.group(1))

    if milestone:
        m = version_pattern.search(milestone)
        if m:
            versions.append(m.group(1))

    return list(set(versions))


def chunk_issue(issue: dict) -> list[IssueChunk]:
    """Chunk a GitHub issue (from API response) into issue body + comment chunks.

    Args:
        issue: Dict from GitHub API /repos/{owner}/{repo}/issues endpoint.
               Expected keys:
               - number, title, body, state, user.login,
                 labels[].name, milestone.title,
                 created_at, closed_at
               - comments_data: list of comment dicts (fetched separately)
                 Each: {body, user.login, created_at}

    Returns:
        List of IssueChunks.
    """
    chunks: list[IssueChunk] = []

    number = issue["number"]
    title = issue.get("title", "")
    body = issue.get("body", "") or ""
    state = issue.get("state", "open")
    author = issue.get("user", {}).get("login", "")
    label_names = [l["name"] for l in issue.get("labels", [])]
    milestone_title = (issue.get("milestone") or {}).get("title")
    created_at = issue.get("created_at", "")
    closed_at = issue.get("closed_at") or ""

    issue_text = f"{title}\n\n{body}"

    # Versions from labels/milestone + content
    label_versions = _extract_versions_from_labels(label_names, milestone_title)
    content_versions = _extract_versions(issue_text)
    all_versions = list(set(label_versions + content_versions))

    pr_refs = _extract_pr_refs(issue_text)

    # Issue body chunk
    chunks.append(IssueChunk(
        content=issue_text,
        issue_number=number,
        state=state,
        is_comment=False,
        parent_issue_number=number,
        author=author,
        labels={"labels": label_names},
        created_at=created_at,
        closed_at=closed_at,
        spark_versions_mentioned={"versions": all_versions},
        linked_prs={"prs": pr_refs},
    ))

    # Comment chunks
    for comment in issue.get("comments_data", []):
        comment_body = comment.get("body", "") or ""
        comment_author = comment.get("user", {}).get("login", "")
        comment_created = comment.get("created_at", "")

        comment_versions = _extract_versions(comment_body)
        comment_prs = _extract_pr_refs(comment_body)

        chunks.append(IssueChunk(
            content=comment_body,
            issue_number=number,
            state=state,
            is_comment=True,
            parent_issue_number=number,
            author=comment_author,
            labels={"labels": label_names},
            created_at=comment_created,
            closed_at="",
            spark_versions_mentioned={"versions": comment_versions},
            linked_prs={"prs": comment_prs},
        ))

    return chunks
