"""GitHub Issues ingestion — fetch all apache/spark issues + comments.

Can be run standalone or called from an Airflow DAG.

Usage:
    uv run python -m spark_rag.ingestion.issues
    uv run python -m spark_rag.ingestion.issues --since 2025-01-01
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime

import requests

from spark_rag.chunking.issue_chunker import chunk_issue
from spark_rag.config import load_config
from spark_rag.lance.schemas import issue_chunks_to_table
from spark_rag.lance.store import LanceStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

GH_API_BASE = "https://api.github.com"


def fetch_issues(
    repo: str,
    state: str = "all",
    since: datetime | None = None,
    max_issues: int = 10000,
    include_comments: bool = True,
    token: str | None = None,
) -> list[dict]:
    """Fetch issues from GitHub API with pagination.

    Args:
        repo: Repo in "owner/repo" format.
        state: "all", "open", or "closed".
        since: Only fetch issues updated after this date.
        max_issues: Max issues to fetch.
        include_comments: Also fetch comments for each issue.
        token: GitHub personal access token (optional, for rate limits).

    Returns:
        List of issue dicts with comments_data added.
    """
    issues: list[dict] = []
    page = 1
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    params = {
        "state": state,
        "sort": "updated",
        "direction": "desc",
        "per_page": 100,
    }
    if since:
        params["since"] = since.isoformat() + "Z"

    while len(issues) < max_issues:
        params["page"] = page
        try:
            resp = requests.get(
                f"{GH_API_BASE}/repos/{repo}/issues",
                params=params, headers=headers, timeout=30,
            )
            resp.raise_for_status()
            items = resp.json()
        except Exception as e:
            logger.error("GitHub API error on page %d: %s", page, e)
            break

        if not items:
            break

        # Filter out pull requests (GitHub API returns PRs in issues endpoint)
        for item in items:
            if "pull_request" in item:
                continue
            issues.append(item)

        page += 1
        _check_rate_limit(resp.headers)
        time.sleep(0.5)

    issues = issues[:max_issues]
    logger.info("Fetched %d issues from %s", len(issues), repo)

    # Fetch comments
    if include_comments:
        for issue in issues:
            if issue.get("comments", 0) > 0:
                issue["comments_data"] = _fetch_comments(
                    repo, issue["number"], headers,
                )
            else:
                issue["comments_data"] = []

    return issues


def _fetch_comments(repo: str, issue_number: int, headers: dict) -> list[dict]:
    """Fetch all comments for an issue."""
    comments: list[dict] = []
    page = 1

    while True:
        try:
            resp = requests.get(
                f"{GH_API_BASE}/repos/{repo}/issues/{issue_number}/comments",
                params={"page": page, "per_page": 100},
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            items = resp.json()
        except Exception as e:
            logger.warning("Failed to fetch comments for issue #%d: %s", issue_number, e)
            break

        if not items:
            break

        comments.extend(items)
        page += 1
        _check_rate_limit(resp.headers)
        time.sleep(0.3)

    return comments


def _check_rate_limit(headers: dict) -> None:
    remaining = int(headers.get("X-RateLimit-Remaining", 100))
    if remaining < 50:
        reset_at = int(headers.get("X-RateLimit-Reset", 0))
        wait = max(0, reset_at - time.time()) + 1
        if remaining < 10:
            logger.warning("GitHub rate limit low (%d remaining), sleeping %ds", remaining, wait)
            time.sleep(min(wait, 60))


def main():
    parser = argparse.ArgumentParser(description="Ingest GitHub issues into Lance")
    parser.add_argument("--since", help="Fetch issues updated after this date (YYYY-MM-DD)")
    parser.add_argument("--max", type=int, default=None, help="Override max_issues from config")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--token", help="GitHub personal access token")
    args = parser.parse_args()

    cfg = load_config()
    issues_cfg = cfg.ingestion.github_issues
    since = datetime.fromisoformat(args.since) if args.since else None
    max_issues = args.max or issues_cfg.max_issues

    # Fetch
    issues = fetch_issues(
        repo=issues_cfg.repo,
        state=issues_cfg.state,
        since=since,
        max_issues=max_issues,
        include_comments=issues_cfg.include_comments,
        token=args.token,
    )

    # Chunk
    chunks_by_issue: dict[int, list] = {}
    total_chunks = 0
    for issue in issues:
        chunks = chunk_issue(issue)
        total_chunks += len(chunks)
        chunks_by_issue[issue["number"]] = chunks

    logger.info("Chunked %d issues → %d chunks", len(issues), total_chunks)

    if args.dry_run:
        logger.info("DRY RUN — not writing to Lance")
        return

    # Write to Lance (incremental by issue_number)
    store = LanceStore(cfg.lance, cfg.polaris)
    total = 0
    for issue_num, chunks in chunks_by_issue.items():
        table = issue_chunks_to_table(chunks)
        total += store.replace_by_id("spark_issues", "issue_number", issue_num, table)

    # Register table in Polaris (once after all inserts)
    if total > 0:
        from spark_rag.lance.polaris_helpers import register_table
        register_table(
            cfg.polaris.endpoint, store.token, cfg.polaris.catalog,
            cfg.polaris.namespace, "spark_issues", store._uri("spark_issues"),
        )

    logger.info("Wrote %d issue chunks to Lance", total)


if __name__ == "__main__":
    main()
