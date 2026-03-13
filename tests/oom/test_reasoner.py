"""
Tests for oom/reasoner.py.

The PydanticAI agent is mocked at the agent.llm layer — no real API calls.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from shittytoken.oom.reasoner import (
    OOMContext,
    OOMReasoningError,
    reason_about_oom,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(oom_type: str = "runtime", current_max_model_len: int = 8192) -> OOMContext:
    return OOMContext(
        gpu_model_name="A100-80GB",
        gpu_vram_gb=80,
        gpu_memory_free_gb=10.0,
        model_id="meta-llama/Llama-3-8B",
        params_b=8.0,
        active_params_b=8.0,
        current_config={
            "tensor_parallel_size": 1,
            "max_model_len": current_max_model_len,
            "gpu_memory_utilization": 0.90,
            "quantization": None,
            "kv_cache_dtype": "auto",
            "max_num_seqs": 256,
            "enable_prefix_caching": True,
            "enforce_eager": False,
        },
        oom_type=oom_type,
        raw_error="torch.cuda.OutOfMemoryError: CUDA out of memory.",
        prior_resolutions=[],
    )


def _make_kg() -> MagicMock:
    """Returns a mock KnowledgeGraph."""
    return MagicMock()


def _valid_proposed_config(overrides: dict | None = None) -> dict:
    config = {
        "tensor_parallel_size": 1,
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.85,
        "quantization": None,
        "kv_cache_dtype": "auto",
        "max_num_seqs": 128,
        "enable_prefix_caching": True,
        "enforce_eager": True,
    }
    if overrides:
        config.update(overrides)
    return config


def _mock_proposed_config(overrides: dict | None = None):
    """Creates a mock ProposedConfig that agent.run() will return."""
    from shittytoken.agent.llm import ProposedConfig

    config = _valid_proposed_config(overrides)
    return ProposedConfig(reasoning="Test reasoning.", **config)


def _patch_agent_run(proposed_config=None, side_effect=None):
    """Patch the oom_reasoning_agent.run to return a mock result."""
    if side_effect:
        return patch(
            "shittytoken.agent.llm.oom_reasoning_agent.run",
            new=AsyncMock(side_effect=side_effect),
        )

    mock_result = MagicMock()
    mock_result.output = proposed_config or _mock_proposed_config()

    return patch(
        "shittytoken.agent.llm.oom_reasoning_agent.run",
        new=AsyncMock(return_value=mock_result),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_response_returns_proposed_config():
    """Valid agent response returns the proposed_config dict."""
    proposed = _mock_proposed_config()
    with _patch_agent_run(proposed):
        result = await reason_about_oom(ctx=_make_ctx(), kg=_make_kg())

    expected = _valid_proposed_config()
    assert result == expected


@pytest.mark.asyncio
async def test_gpu_memory_utilization_at_limit_raises():
    """gpu_memory_utilization=1.0 violates Pydantic Field(lt=1.0) → OOMReasoningError."""
    from pydantic import ValidationError

    with _patch_agent_run(side_effect=ValidationError.from_exception_data(
        title="ProposedConfig",
        line_errors=[],
    )):
        with pytest.raises(OOMReasoningError):
            await reason_about_oom(ctx=_make_ctx(), kg=_make_kg())


@pytest.mark.asyncio
async def test_max_model_len_below_floor_raises():
    """max_model_len=2048 violates Pydantic Field(ge=4096) → OOMReasoningError."""
    from pydantic import ValidationError

    with _patch_agent_run(side_effect=ValidationError.from_exception_data(
        title="ProposedConfig",
        line_errors=[],
    )):
        with pytest.raises(OOMReasoningError):
            await reason_about_oom(ctx=_make_ctx(), kg=_make_kg())


@pytest.mark.asyncio
async def test_agent_error_raises_oom_reasoning_error():
    """Any exception from the agent is wrapped as OOMReasoningError."""
    with _patch_agent_run(side_effect=RuntimeError("API down")):
        with pytest.raises(OOMReasoningError, match="OOM reasoning failed"):
            await reason_about_oom(ctx=_make_ctx(), kg=_make_kg())


@pytest.mark.asyncio
async def test_result_excludes_reasoning_field():
    """The dict returned by reason_about_oom excludes the 'reasoning' field."""
    with _patch_agent_run():
        result = await reason_about_oom(ctx=_make_ctx(), kg=_make_kg())

    assert "reasoning" not in result
    assert "max_model_len" in result
    assert "gpu_memory_utilization" in result
