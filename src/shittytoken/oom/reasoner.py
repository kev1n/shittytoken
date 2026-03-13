"""
OOM reasoner: delegates to the PydanticAI OOM reasoning agent for diagnosis
and validated configuration proposals.

This module re-exports OOMReasoningError (the canonical exception) and
OOMContext (the context dataclass) for backward compatibility.

The actual LLM interaction is handled by agent.llm.oom_reasoning_agent.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from ..knowledge.client import KnowledgeGraph

logger = structlog.get_logger()


class OOMReasoningError(Exception):
    """Raised when LLM output violates safety constraints."""


@dataclass
class OOMContext:
    """All context passed to the LLM for OOM reasoning."""

    gpu_model_name: str
    gpu_vram_gb: int
    gpu_memory_free_gb: float
    model_id: str
    params_b: float
    active_params_b: float
    current_config: dict           # current Configuration as dict
    oom_type: str                  # "loading" | "runtime"
    raw_error: str
    prior_resolutions: list[dict]  # from KG prior_oom_resolutions query


async def reason_about_oom(
    ctx: OOMContext,
    kg: KnowledgeGraph,
    model: str = "claude-opus-4-6",
) -> dict:
    """
    Calls the PydanticAI OOM reasoning agent with OOM context.
    Returns the proposed config as a dict after validation.

    Raises OOMReasoningError if the agent fails or returns invalid output.
    """
    from ..agent.llm import reason_about_oom as _reason

    proposed = await _reason(
        gpu_model_name=ctx.gpu_model_name,
        gpu_vram_gb=ctx.gpu_vram_gb,
        gpu_memory_free_gb=ctx.gpu_memory_free_gb,
        model_id=ctx.model_id,
        params_b=ctx.params_b,
        active_params_b=ctx.active_params_b,
        current_config=ctx.current_config,
        oom_type=ctx.oom_type,
        raw_error=ctx.raw_error,
        prior_resolutions=ctx.prior_resolutions,
        kg=kg,
        model=model,
    )

    # Convert ProposedConfig back to dict for backward compatibility
    return proposed.model_dump(exclude={"reasoning"})
