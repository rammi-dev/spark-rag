"""No-op synthesis provider — returns None (retrieval-only mode)."""

from __future__ import annotations

from spark_rag.synthesis.base import SynthesisInput, SynthesisProvider


class NoopSynthesis(SynthesisProvider):
    """Returns None — used when synthesis is disabled."""

    async def analyze(self, input: SynthesisInput) -> None:
        return None

    @property
    def name(self) -> str:
        return "noop"
