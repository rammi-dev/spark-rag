"""Claude API synthesis provider."""

from __future__ import annotations

import logging

import anthropic

from spark_rag.synthesis.base import SynthesisInput, SynthesisProvider

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a Spark expert assistant. You analyze Apache Spark code, logs, and questions \
using retrieved context from Spark source code, documentation, StackOverflow, and GitHub issues.

Provide a clear, actionable analysis that includes:
1. Root cause identification (what's happening and why)
2. Specific recommendations with code examples when applicable
3. References to relevant Spark documentation or configuration
4. Version-specific notes when the behavior differs across Spark versions

Be concise and direct. Prioritize actionable advice over general explanations."""


def _build_prompt(input: SynthesisInput) -> str:
    """Build the user prompt from input + retrieved context."""
    parts: list[str] = []

    # User input
    parts.append(f"## User Input ({input.input_type})\n\n{input.user_input}")

    # Detected patterns
    if input.detected_patterns:
        parts.append("\n## Detected Patterns\n")
        for p in input.detected_patterns:
            parts.append(f"- **{p.get('name', 'unknown')}** ({p.get('risk', '?')}): {p.get('description', '')}")

    # Retrieved context chunks
    top_hits = input.search_result.hits[:input.max_context_chunks]
    if top_hits:
        parts.append("\n## Retrieved Context\n")
        for i, hit in enumerate(top_hits, 1):
            source_label = hit.source.replace("spark_", "").replace("_", " ").title()
            meta_parts = []
            if "file_path" in hit.metadata:
                meta_parts.append(f"file: {hit.metadata['file_path']}")
            if "doc_url" in hit.metadata:
                meta_parts.append(f"doc: {hit.metadata['doc_url']}")
            if "question_id" in hit.metadata:
                meta_parts.append(f"SO #{hit.metadata['question_id']}")
            if "issue_number" in hit.metadata:
                meta_parts.append(f"issue #{hit.metadata['issue_number']}")
            if "spark_version" in hit.metadata:
                meta_parts.append(f"v{hit.metadata['spark_version']}")

            meta_str = f" ({', '.join(meta_parts)})" if meta_parts else ""
            content_preview = hit.content[:1500]  # limit per chunk
            parts.append(f"### [{i}] {source_label}{meta_str}\n```\n{content_preview}\n```")

    parts.append("\n## Task\n\nAnalyze the user's input using the retrieved context above. "
                 "Provide root cause, recommendations, and relevant references.")

    return "\n".join(parts)


class ClaudeSynthesis(SynthesisProvider):
    """Synthesis via Claude API (anthropic SDK).

    Reads ANTHROPIC_API_KEY from environment.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self.client = anthropic.AsyncAnthropic()  # reads API key from env

    async def analyze(self, input: SynthesisInput) -> str | None:
        prompt = _build_prompt(input)

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except anthropic.AuthenticationError:
            logger.error("Claude API authentication failed — check ANTHROPIC_API_KEY")
            return None
        except Exception as e:
            logger.error("Claude synthesis failed: %s", e)
            return None

    @property
    def name(self) -> str:
        return f"claude ({self.model})"
