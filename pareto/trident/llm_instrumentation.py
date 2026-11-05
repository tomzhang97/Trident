"""LLM instrumentation for token and latency tracking across pipeline modes."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from trident.llm_interface import LLMInterface, LLMOutput
else:
    try:
        from .llm_interface import LLMInterface, LLMOutput
    except ImportError:
        # Fallback for direct imports
        from trident.llm_interface import LLMInterface, LLMOutput


@dataclass
class LLMCallStats:
    """Statistics from a single LLM call."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float


@dataclass
class QueryStats:
    """Accumulated statistics across all LLM calls for a single query."""
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    num_calls: int = 0
    latency_ms: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


class InstrumentedLLM:
    """
    Wrapper around LLMInterface that tracks token usage and latency.
    This provides a unified interface for model-backed pipeline components.
    """

    def __init__(self, llm: LLMInterface):
        """
        Initialize instrumented LLM wrapper.

        Args:
            llm: The base LLMInterface instance to wrap
        """
        self.llm = llm

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the LLM's tokenizer."""
        return self.llm.compute_token_cost(text)

    def generate(
        self,
        prompt: str,
        qstats: Optional[QueryStats] = None,
        **gen_kwargs
    ) -> Tuple[str, LLMCallStats]:
        """
        Generate text and track statistics.

        Args:
            prompt: The prompt to generate from
            qstats: Optional QueryStats to update in-place
            **gen_kwargs: Additional generation arguments

        Returns:
            Tuple of (generated_text, call_stats)
        """
        t0 = time.perf_counter()

        # Count prompt tokens
        prompt_tokens = self._count_tokens(prompt)

        # Generate
        llm_output: LLMOutput = self.llm.generate(prompt, **gen_kwargs)
        text = llm_output.text

        # Count completion tokens
        completion_tokens = self._count_tokens(text)
        total_tokens = prompt_tokens + completion_tokens

        # Calculate latency
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Create stats
        stats = LLMCallStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms
        )

        # Update query stats if provided
        if qstats is not None:
            qstats.total_tokens += stats.total_tokens
            qstats.total_prompt_tokens += stats.prompt_tokens
            qstats.total_completion_tokens += stats.completion_tokens
            qstats.num_calls += 1
            qstats.latency_ms += stats.latency_ms

        return text, stats

    def build_rag_prompt(self, *args, **kwargs) -> str:
        """Pass through to underlying LLM."""
        return self.llm.build_rag_prompt(*args, **kwargs)

    def build_multi_hop_prompt(self, *args, **kwargs) -> str:
        """Pass through to underlying LLM."""
        return self.llm.build_multi_hop_prompt(*args, **kwargs)

    def extract_answer(self, *args, **kwargs) -> str:
        """Pass through to underlying LLM."""
        return self.llm.extract_answer(*args, **kwargs)

    def compute_token_cost(self, *args, **kwargs) -> int:
        """Pass through to underlying LLM."""
        return self.llm.compute_token_cost(*args, **kwargs)


def timed_llm_call(
    llm: InstrumentedLLM,
    prompt: str,
    qstats: QueryStats,
    **gen_kwargs
) -> str:
    """
    Convenience function to make a timed LLM call and update query stats.

    Args:
        llm: Instrumented LLM instance
        prompt: The prompt to generate from
        qstats: QueryStats to update in-place
        **gen_kwargs: Additional generation arguments

    Returns:
        Generated text
    """
    text, _ = llm.generate(prompt, qstats, **gen_kwargs)
    return text
