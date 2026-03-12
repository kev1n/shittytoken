"""
OOM reasoner: calls the Anthropic API to diagnose root cause and propose
the smallest configuration change that resolves the OOM while maximising
throughput.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import anthropic
import structlog

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


SYSTEM_PROMPT = """You are an expert at diagnosing vLLM out-of-memory errors and proposing the smallest configuration change that resolves them while preserving maximum throughput.

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
- enable_prefix_caching must remain True

Respond ONLY with a valid JSON object matching this schema:
{
  "reasoning": "2-3 sentence explanation of the root cause and why this fix addresses it",
  "proposed_config": {
    "tensor_parallel_size": <int>,
    "max_model_len": <int>,
    "gpu_memory_utilization": <float, must be < 1.0>,
    "quantization": <"awq"|"fp8"|"fp16"|null>,
    "kv_cache_dtype": <"auto"|"fp8">,
    "max_num_seqs": <int>,
    "enable_prefix_caching": true,
    "enforce_eager": <bool>
  }
}"""


def build_user_prompt(ctx: OOMContext) -> str:
    """Renders the user turn of the OOM reasoning prompt with all context filled in."""
    prior_text = (
        json.dumps(ctx.prior_resolutions, indent=2)
        if ctx.prior_resolutions
        else "None available"
    )
    return (
        f"GPU: {ctx.gpu_model_name}\n"
        f"VRAM: {ctx.gpu_vram_gb} GB total, {ctx.gpu_memory_free_gb:.1f} GB free at OOM time\n"
        f"\n"
        f"Model: {ctx.model_id}\n"
        f"Parameters: {ctx.params_b}B total, {ctx.active_params_b}B active\n"
        f"\n"
        f"OOM type: {ctx.oom_type}\n"
        f"\n"
        f"Current configuration that caused OOM:\n"
        f"{json.dumps(ctx.current_config, indent=2)}\n"
        f"\n"
        f"Raw error message:\n"
        f"{ctx.raw_error}\n"
        f"\n"
        f"Historical resolutions that worked on this GPU for {ctx.oom_type} OOM:\n"
        f"{prior_text}\n"
        f"\n"
        f"Propose the smallest configuration change that will resolve this OOM."
    )


async def reason_about_oom(
    ctx: OOMContext,
    anthropic_client: anthropic.AsyncAnthropic,
    model: str = "claude-opus-4-6",
) -> dict:
    """
    Calls Anthropic API with OOM context.
    Returns the proposed_config dict after validation.

    Validation guards (raise OOMReasoningError if violated):
    - proposed_config.gpu_memory_utilization >= 1.0
    - proposed_config.max_model_len < 4096
    - oom_type == "loading" AND proposed max_model_len < current max_model_len
      (reducing max_model_len can't fix a loading OOM — weights aren't allocated yet)

    On JSON parse error or missing keys: raise OOMReasoningError.
    Logs the reasoning field at INFO level for observability.
    """
    user_prompt = build_user_prompt(ctx)

    try:
        response = await asyncio.wait_for(
            anthropic_client.messages.create(
                model=model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            ),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        raise OOMReasoningError("Anthropic API call timed out after 30s")
    except anthropic.APIError as exc:
        raise OOMReasoningError(f"Anthropic API error: {exc}") from exc

    raw_text = response.content[0].text

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise OOMReasoningError(
            f"LLM returned invalid JSON: {exc}\nRaw response: {raw_text!r}"
        ) from exc

    if "proposed_config" not in parsed:
        raise OOMReasoningError(
            f"LLM response missing 'proposed_config' key. Got keys: {list(parsed.keys())}"
        )

    reasoning = parsed.get("reasoning", "")
    logger.info(
        "oom_reasoner.reasoning",
        model_id=ctx.model_id,
        oom_type=ctx.oom_type,
        reasoning=reasoning,
    )

    proposed = parsed["proposed_config"]

    # Validate gpu_memory_utilization
    gpu_util = proposed.get("gpu_memory_utilization")
    if gpu_util is None:
        raise OOMReasoningError(
            "proposed_config missing 'gpu_memory_utilization'"
        )
    if gpu_util >= 1.0:
        raise OOMReasoningError(
            f"proposed gpu_memory_utilization={gpu_util} >= 1.0 (safety violation)"
        )

    # Validate max_model_len floor
    max_model_len = proposed.get("max_model_len")
    if max_model_len is None:
        raise OOMReasoningError(
            "proposed_config missing 'max_model_len'"
        )
    if max_model_len < 4096:
        raise OOMReasoningError(
            f"proposed max_model_len={max_model_len} < 4096 (safety violation)"
        )

    # Validate enable_prefix_caching
    if not proposed.get("enable_prefix_caching", True):
        raise OOMReasoningError(
            "proposed_config.enable_prefix_caching must be True — prefix caching is always required"
        )

    # Validate loading OOM constraint: must not reduce max_model_len
    if ctx.oom_type == "loading":
        current_max_model_len = ctx.current_config.get("max_model_len")
        if current_max_model_len is not None and max_model_len < current_max_model_len:
            raise OOMReasoningError(
                f"LLM proposed reducing max_model_len from {current_max_model_len} to "
                f"{max_model_len} for a LOADING OOM — this is a misclassification, "
                f"not a valid fix. Weights are not allocated by max_model_len."
            )

    return proposed
