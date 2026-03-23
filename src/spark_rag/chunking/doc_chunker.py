"""Chunk Spark documentation from Markdown source files.

Docs live in the apache/spark repo under docs/*.md. They use:
- Standard Markdown headings (# ## ###)
- Fenced code blocks (``` ```)
- Inline HTML tables (config reference)
- Jekyll include_example tags ({% include_example ... %})

Splits on heading boundaries. Code blocks and tables become separate chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

MAX_CHUNK_CHARS = 2048

# Regex for Spark config keys like spark.sql.shuffle.partitions
_CONFIG_PATTERN = re.compile(r"spark\.\w+(?:\.\w+)+")

# Match Markdown headings: # Heading, ## Heading, etc.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Match fenced code blocks: ```lang ... ```
_CODE_BLOCK_RE = re.compile(r"^```(\w*)\n(.*?)^```", re.MULTILINE | re.DOTALL)

# Match Jekyll include_example tags: {% include_example name path/to/file %}
# Captures: group(1)=name, group(2)=file_path
_INCLUDE_EXAMPLE_RE = re.compile(r"\{%\s*include_example\s+(\S+)\s+(\S+)\s*%\}")

# Match inline HTML tables (Spark docs embed config tables as raw HTML in .md)
_HTML_TABLE_RE = re.compile(r"<table.*?>.*?</table>", re.DOTALL | re.IGNORECASE)


@dataclass
class DocChunk:
    content: str
    doc_url: str  # relative path in repo, e.g. "docs/configuration.md"
    doc_section: str
    heading_hierarchy: list[str]
    content_type: str  # "prose", "code_example", "config_table"
    related_configs: dict = field(default_factory=lambda: {"configs": []})

    def to_milvus_data(self, spark_version: str, embedding: list[float]) -> dict:
        return {
            "embedding": embedding,
            "content": self.content[:65535],
            "spark_version": spark_version,
            "doc_url": self.doc_url[:512],
            "doc_section": self.doc_section[:256],
            "heading_hierarchy": self.heading_hierarchy,
            "content_type": self.content_type,
            "related_configs": self.related_configs,
        }


def _extract_configs(text: str) -> list[str]:
    """Find Spark configuration keys in text."""
    return list(set(_CONFIG_PATTERN.findall(text)))


def _html_table_to_text(html: str) -> str:
    """Convert an inline HTML table to readable text."""
    soup = BeautifulSoup(html, "html.parser")
    rows: list[str] = []
    for tr in soup.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _split_text(text: str) -> list[str]:
    """Split text exceeding MAX_CHUNK_CHARS.

    First tries splitting on paragraph boundaries (double newline).
    If a single paragraph is still too long, splits on single newlines (table rows).
    """
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    # First pass: split on double newlines
    paragraphs = text.split("\n\n")
    parts: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        # If a single paragraph exceeds limit, split it on single newlines
        if len(para) > MAX_CHUNK_CHARS:
            # Flush current
            if current:
                parts.append("\n\n".join(current))
                current = []
                current_len = 0
            # Split the oversized paragraph on newlines
            parts.extend(_split_lines(para))
            continue

        if current_len + len(para) > MAX_CHUNK_CHARS and current:
            parts.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += len(para) + 2

    if current:
        parts.append("\n\n".join(current))

    return parts


def _split_lines(text: str) -> list[str]:
    """Split on single newlines when paragraph-level split isn't enough."""
    lines = text.split("\n")
    parts: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        if current_len + len(line) > MAX_CHUNK_CHARS and current:
            parts.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += len(line) + 1

    if current:
        parts.append("\n".join(current))

    return parts


def chunk_markdown(
    content: str,
    file_path: str,
    examples_dir: str | None = None,
    examples_lookup: dict[str, str] | None = None,
) -> list[DocChunk]:
    """Parse a Markdown documentation file into chunks.

    Args:
        content: Raw Markdown text.
        file_path: Relative path in repo (e.g. "docs/configuration.md").
        examples_dir: Not used directly — examples resolved via examples_lookup.
        examples_lookup: Dict mapping example paths to their content.
            Used to resolve {% include_example path %} tags.

    Returns:
        List of DocChunks.
    """
    if not content or not content.strip():
        return []

    examples_lookup = examples_lookup or {}

    # Pre-process: resolve include_example tags
    def _resolve_example(match: re.Match) -> str:
        name = match.group(1)
        file_path_ref = match.group(2)
        # Try lookup by name first, then by file path
        code = examples_lookup.get(name) or examples_lookup.get(file_path_ref)
        if code:
            return f"```\n{code}\n```"
        return f"[Example: {name} ({file_path_ref})]"

    content = _INCLUDE_EXAMPLE_RE.sub(_resolve_example, content)

    # Extract all code blocks and HTML tables first, replace with placeholders
    code_blocks: list[tuple[str, str]] = []  # (language, code)
    html_tables: list[str] = []

    def _replace_code(match: re.Match) -> str:
        idx = len(code_blocks)
        code_blocks.append((match.group(1), match.group(2).strip()))
        return f"\n__CODE_BLOCK_{idx}__\n"

    def _replace_table(match: re.Match) -> str:
        idx = len(html_tables)
        html_tables.append(match.group(0))
        return f"\n__HTML_TABLE_{idx}__\n"

    processed = _CODE_BLOCK_RE.sub(_replace_code, content)
    processed = _HTML_TABLE_RE.sub(_replace_table, processed)

    # Now split by headings
    chunks: list[DocChunk] = []
    heading_stack: list[str] = []

    # Split content into sections by heading
    sections: list[tuple[int, str, str]] = []  # (level, heading_text, body)
    parts = _HEADING_RE.split(processed)

    # parts alternates: [pre-heading text, "#"*level, heading_text, body, "#"*level, ...]
    # First element is text before any heading
    if parts[0].strip():
        sections.append((0, "", parts[0]))

    i = 1
    while i < len(parts) - 2:
        level = len(parts[i])
        heading_text = parts[i + 1].strip()
        body = parts[i + 2] if i + 2 < len(parts) else ""
        sections.append((level, heading_text, body))
        i += 3

    for level, heading_text, body in sections:
        # Update heading hierarchy
        if level > 0:
            while len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(heading_text)

        section_name = heading_stack[0] if heading_stack else ""

        # Process body: split by placeholders
        lines = body.split("\n")
        prose_lines: list[str] = []

        def flush_prose():
            text = "\n".join(prose_lines).strip()
            if text and len(text) > 20:
                configs = _extract_configs(text)
                for part in _split_text(text):
                    chunks.append(DocChunk(
                        content=part,
                        doc_url=file_path,
                        doc_section=section_name,
                        heading_hierarchy=list(heading_stack),
                        content_type="prose",
                        related_configs={"configs": configs},
                    ))
            prose_lines.clear()

        for line in lines:
            # Code block placeholder
            code_match = re.match(r"__CODE_BLOCK_(\d+)__", line.strip())
            if code_match:
                flush_prose()
                idx = int(code_match.group(1))
                lang, code = code_blocks[idx]
                if code:
                    configs = _extract_configs(code)
                    for part in _split_text(code):
                        chunks.append(DocChunk(
                            content=part,
                            doc_url=file_path,
                            doc_section=section_name,
                            heading_hierarchy=list(heading_stack),
                            content_type="code_example",
                            related_configs={"configs": configs},
                        ))
                continue

            # HTML table placeholder
            table_match = re.match(r"__HTML_TABLE_(\d+)__", line.strip())
            if table_match:
                flush_prose()
                idx = int(table_match.group(1))
                table_text = _html_table_to_text(html_tables[idx])
                if table_text.strip():
                    configs = _extract_configs(table_text)
                    for part in _split_text(table_text):
                        chunks.append(DocChunk(
                            content=part,
                            doc_url=file_path,
                            doc_section=section_name,
                            heading_hierarchy=list(heading_stack),
                            content_type="config_table",
                            related_configs={"configs": configs},
                        ))
                continue

            prose_lines.append(line)

        flush_prose()

    return chunks
