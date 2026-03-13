"""
PydanticAI-based LLM agents for the orchestration system.

This module defines all LLM-powered reasoning agents used by the orchestrator:
- OOM diagnosis and configuration recovery
- Initial vLLM configuration proposals for new GPU/model combinations

Each agent uses PydanticAI's structured output, dependency injection, and
result validation to ensure safety constraints are never violated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import structlog
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext

from ..knowledge.client import KnowledgeGraph

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Structured output models
# ---------------------------------------------------------------------------


class ProposedConfig(BaseModel):
    """Validated vLLM configuration proposed by the LLM."""

    reasoning: str = Field(description="2-3 sentence explanation of the root cause and fix")
    tensor_parallel_size: int = Field(ge=1)
    max_model_len: int = Field(ge=4096)
    gpu_memory_utilization: float = Field(gt=0, lt=1.0)
    quantization: Optional[str] = Field(
        default=None,
        description='Quantization method: "awq", "fp8", "fp16", or null',
    )
    kv_cache_dtype: str = Field(default="auto", description='"auto" or "fp8"')
    max_num_seqs: int = Field(ge=1)
    enable_prefix_caching: Literal[True] = True
    enforce_eager: bool = False


# ---------------------------------------------------------------------------
# Dependency types
# ---------------------------------------------------------------------------


@dataclass
class OOMReasoningDeps:
    """Dependencies injected into the OOM reasoning agent."""

    kg: KnowledgeGraph
    oom_type: str  # "loading" | "runtime"
    current_max_model_len: int | None  # for loading-OOM validation


@dataclass
class ConfigProposalDeps:
    """Dependencies injected into the config proposal agent."""

    kg: KnowledgeGraph


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

OOM_SYSTEM_PROMPT = """\
You are an expert at diagnosing vLLM out-of-memory errors and proposing the \
smallest configuration change that resolves them while preserving maximum throughput.

You will receive:
1. GPU specifications
2. Current vLLM configuration that caused the OOM
3. OOM type (loading or runtime)
4. The raw error message
5. Historical resolutions that worked on this GPU

Memory formulas:
- Weight memory (GB) = params_b × bytes_per_param
  - FP16/BF16: bytes_per_param = 2
  - INT8/AWQ: bytes_per_param ≈ 1
  - INT4/GPTQ: bytes_per_param ≈ 0.5
- KV cache per token per layer (bytes) = 2 × kv_heads × head_dim × dtype_bytes
  - FP16: dtype_bytes = 2
  - FP8: dtype_bytes = 1

Fix priority order (use the FIRST fix that resolves the issue):
For LOADING OOM:
1. Add AWQ quantization (if not already quantized)
2. Increase tensor_parallel_size (if more GPUs available)
3. Use INT4 quantization
DO NOT reduce max_model_len for loading OOM.

For RUNTIME OOM:
1. Reduce max_model_len by 50%
2. Reduce max_num_seqs by 50%
3. Enable enforce_eager (frees 200MB–2GB of CUDA graph memory)
4. Switch kv_cache_dtype to fp8

Safety constraints (NEVER violate):
- gpu_memory_utilization must be < 1.0 (use 0.90 max)
- max_model_len must be >= 4096
- enable_prefix_caching must remain True"""

CONFIG_PROPOSAL_SYSTEM_PROMPT = """\
You are an expert at configuring vLLM for optimal throughput on specific GPU hardware.

Given a GPU model and LLM model, propose a conservative initial vLLM configuration \
that maximizes throughput while staying safely within memory limits.

Memory formulas:
- Weight memory (GB) = params_b × bytes_per_param
  - FP16/BF16: bytes_per_param = 2
  - INT8/AWQ: bytes_per_param ≈ 1
  - INT4/GPTQ: bytes_per_param ≈ 0.5
- KV cache per token per layer (bytes) = 2 × kv_heads × head_dim × dtype_bytes

Guidelines:
- Start conservative: gpu_memory_utilization=0.90
- Use AWQ quantization for models that don't fit in FP16
- For MoE models, only active parameters consume compute — total params determine memory
- enable_prefix_caching is always True (non-negotiable)
- Prefer max_model_len=8192 as a starting point unless memory is very tight

Safety constraints (NEVER violate):
- gpu_memory_utilization must be < 1.0
- max_model_len must be >= 4096
- enable_prefix_caching must remain True"""


# ---------------------------------------------------------------------------
# OOM reasoning agent
# ---------------------------------------------------------------------------

oom_reasoning_agent = Agent(
    deps_type=OOMReasoningDeps,
    output_type=ProposedConfig,
    system_prompt=OOM_SYSTEM_PROMPT,
)


@oom_reasoning_agent.output_validator
async def _validate_oom_result(
    ctx: RunContext[OOMReasoningDeps], result: ProposedConfig
) -> ProposedConfig:
    """Enforce loading-OOM constraint: cannot reduce max_model_len."""
    if ctx.deps.oom_type == "loading":
        current = ctx.deps.current_max_model_len
        if current is not None and result.max_model_len < current:
            raise ModelRetry(
                f"Cannot reduce max_model_len from {current} to "
                f"{result.max_model_len} for a LOADING OOM — weights are not "
                f"allocated by max_model_len. Use quantization or tensor parallelism instead."
            )
    return result


# ---------------------------------------------------------------------------
# Config proposal agent
# ---------------------------------------------------------------------------

config_proposal_agent = Agent(
    deps_type=ConfigProposalDeps,
    output_type=ProposedConfig,
    system_prompt=CONFIG_PROPOSAL_SYSTEM_PROMPT,
)


@config_proposal_agent.tool
async def lookup_similar_configs(
    ctx: RunContext[ConfigProposalDeps],
    gpu_model_name: str,
    model_id: str,
) -> str:
    """Look up the best known configuration for a GPU/model pair from the knowledge graph."""
    config = await ctx.deps.kg.best_config_for(gpu_model_name, model_id)
    if config is None:
        return "No prior configurations found for this GPU/model combination."
    return (
        f"Best known config: tensor_parallel_size={config.tensor_parallel_size}, "
        f"max_model_len={config.max_model_len}, "
        f"gpu_memory_utilization={config.gpu_memory_utilization}, "
        f"quantization={config.quantization}, "
        f"kv_cache_dtype={config.kv_cache_dtype}, "
        f"max_num_seqs={config.max_num_seqs}, "
        f"enforce_eager={config.enforce_eager}"
    )


@config_proposal_agent.tool
async def lookup_oom_history(
    ctx: RunContext[ConfigProposalDeps],
    gpu_model_name: str,
    error_type: str,
) -> str:
    """Look up prior OOM resolutions for a GPU model to avoid known-bad configurations."""
    resolutions = await ctx.deps.kg.prior_oom_resolutions(gpu_model_name, error_type)
    if not resolutions:
        return "No prior OOM events recorded for this GPU."
    import json
    return json.dumps(resolutions, indent=2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def reason_about_oom(
    *,
    gpu_model_name: str,
    gpu_vram_gb: int,
    gpu_memory_free_gb: float,
    model_id: str,
    params_b: float,
    active_params_b: float,
    current_config: dict,
    oom_type: str,
    raw_error: str,
    prior_resolutions: list[dict],
    kg: KnowledgeGraph,
    model: str = "claude-opus-4-6",
) -> ProposedConfig:
    """
    Diagnose an OOM error and propose a configuration fix.

    Returns a validated ProposedConfig. Raises OOMReasoningError on failure.

    This replaces the old raw-Anthropic-API implementation with PydanticAI
    structured output and result validation.
    """
    import json

    from ..oom.reasoner import OOMReasoningError

    prior_text = (
        json.dumps(prior_resolutions, indent=2)
        if prior_resolutions
        else "None available"
    )
    user_prompt = (
        f"GPU: {gpu_model_name}\n"
        f"VRAM: {gpu_vram_gb} GB total, {gpu_memory_free_gb:.1f} GB free at OOM time\n"
        f"\n"
        f"Model: {model_id}\n"
        f"Parameters: {params_b}B total, {active_params_b}B active\n"
        f"\n"
        f"OOM type: {oom_type}\n"
        f"\n"
        f"Current configuration that caused OOM:\n"
        f"{json.dumps(current_config, indent=2)}\n"
        f"\n"
        f"Raw error message:\n"
        f"{raw_error}\n"
        f"\n"
        f"Historical resolutions that worked on this GPU for {oom_type} OOM:\n"
        f"{prior_text}\n"
        f"\n"
        f"Propose the smallest configuration change that will resolve this OOM."
    )

    deps = OOMReasoningDeps(
        kg=kg,
        oom_type=oom_type,
        current_max_model_len=current_config.get("max_model_len"),
    )

    try:
        result = await oom_reasoning_agent.run(
            user_prompt,
            deps=deps,
            model=f"anthropic:{model}",
        )
    except Exception as exc:
        raise OOMReasoningError(f"OOM reasoning failed: {exc}") from exc

    proposed = result.output

    logger.info(
        "oom_reasoner.reasoning",
        model_id=model_id,
        oom_type=oom_type,
        reasoning=proposed.reasoning,
    )

    return proposed


async def propose_initial_config(
    *,
    gpu_model_name: str,
    gpu_vram_gb: int,
    model_id: str,
    params_b: float,
    active_params_b: float,
    kg: KnowledgeGraph,
    model: str = "claude-opus-4-6",
) -> ProposedConfig:
    """
    Propose an initial vLLM configuration for a GPU/model pair with no prior data.

    The agent has access to KG tools to look up similar configs and OOM history,
    enabling it to make informed proposals even for new combinations.
    """
    user_prompt = (
        f"GPU: {gpu_model_name} ({gpu_vram_gb} GB VRAM)\n"
        f"Model: {model_id}\n"
        f"Parameters: {params_b}B total, {active_params_b}B active\n"
        f"\n"
        f"Propose a conservative initial vLLM configuration for this GPU/model combination. "
        f"Use the available tools to check for similar configurations and OOM history "
        f"before making your proposal."
    )

    deps = ConfigProposalDeps(kg=kg)

    result = await config_proposal_agent.run(
        user_prompt,
        deps=deps,
        model=f"anthropic:{model}",
    )

    proposed = result.output

    logger.info(
        "config_proposal.result",
        gpu_model=gpu_model_name,
        model_id=model_id,
        reasoning=proposed.reasoning,
    )

    return proposed
