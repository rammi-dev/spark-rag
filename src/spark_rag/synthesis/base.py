"""Abstract base for synthesis providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from spark_rag.milvus.search import SearchResult


@dataclass
class SynthesisInput:
    """Input to the synthesis provider."""
    user_input: str
    input_type: str  # "code", "logs", "question"
    detected_patterns: list[dict]  # from spark_patterns
    search_result: SearchResult
    max_context_chunks: int = 15


class SynthesisProvider(ABC):
    """Abstract interface for LLM synthesis."""

    @abstractmethod
    async def analyze(self, input: SynthesisInput) -> str | None:
        """Generate analysis from retrieved context.

        Returns:
            Written analysis string, or None if synthesis is disabled/unavailable.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        ...
