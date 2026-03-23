"""Git repo operations for Spark source ingestion.

Handles cloning and checkout for version-specific ingestion.
Shared by both code and docs ingestion.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CLONE_DIR = Path("/tmp/spark-rag-clone")


def ensure_repo(
    repo_url: str,
    clone_dir: Path = DEFAULT_CLONE_DIR,
) -> Path:
    """Clone the repo if not already present. Returns repo path."""
    if clone_dir.exists() and (clone_dir / ".git").exists():
        logger.info("Repo already cloned at %s", clone_dir)
        return clone_dir

    logger.info("Cloning %s → %s", repo_url, clone_dir)
    clone_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--no-checkout", "--filter=blob:none", repo_url, str(clone_dir)],
        check=True,
        capture_output=True,
    )
    return clone_dir


def checkout_version(repo_dir: Path, git_tag: str) -> None:
    """Checkout a specific git tag (sparse checkout — only fetches needed blobs)."""
    logger.info("Checking out %s", git_tag)
    subprocess.run(
        ["git", "checkout", git_tag],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )


def iter_files(
    repo_dir: Path,
    paths: list[str],
    extensions: set[str],
) -> list[Path]:
    """Find all files matching extensions under given paths.

    Args:
        repo_dir: Root of the git repo.
        paths: Relative paths to scan (e.g. ["sql/", "core/"]).
        extensions: File extensions to include (e.g. {".scala", ".java", ".py"}).

    Returns:
        List of absolute file paths.
    """
    files: list[Path] = []
    for rel_path in paths:
        search_dir = repo_dir / rel_path
        if not search_dir.exists():
            logger.warning("Path %s not found in repo", rel_path)
            continue
        for f in search_dir.rglob("*"):
            if f.is_file() and f.suffix in extensions:
                files.append(f)

    logger.info("Found %d files across %s", len(files), paths)
    return files
