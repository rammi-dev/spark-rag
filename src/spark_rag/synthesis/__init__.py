"""Synthesis module — swappable LLM providers."""

from spark_rag.config import SynthesisConfig
from spark_rag.synthesis.base import SynthesisInput, SynthesisProvider
from spark_rag.synthesis.noop import NoopSynthesis


def create_provider(config: SynthesisConfig) -> SynthesisProvider:
    """Factory: create synthesis provider from config.

    Returns NoopSynthesis when disabled, ClaudeSynthesis when enabled
    with provider="claude".
    """
    if not config.enabled:
        return NoopSynthesis()

    if config.provider == "claude":
        from spark_rag.synthesis.claude import ClaudeSynthesis
        return ClaudeSynthesis(model=config.model)

    raise ValueError(f"Unknown synthesis provider: {config.provider}")
