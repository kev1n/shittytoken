"""
Tests for oom/reasoner.py.

All Anthropic API calls are mocked — no real network traffic.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

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


def _make_client(response_text: str) -> MagicMock:
    """Returns a mock AsyncAnthropic client that returns response_text."""
    content_block = MagicMock()
    content_block.text = response_text

    message = MagicMock()
    message.content = [content_block]

    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=message)
    return client


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


def _make_response(proposed: dict, reasoning: str = "Reducing max_model_len frees KV cache.") -> str:
    return json.dumps({"reasoning": reasoning, "proposed_config": proposed})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_response_returns_proposed_config():
    """Valid LLM response with a compliant config returns the proposed_config dict."""
    proposed = _valid_proposed_config()
    client = _make_client(_make_response(proposed))

    result = await reason_about_oom(ctx=_make_ctx(), anthropic_client=client)

    assert result == proposed


@pytest.mark.asyncio
async def test_gpu_memory_utilization_at_limit_raises():
    """gpu_memory_utilization=1.0 is a safety violation → OOMReasoningError."""
    proposed = _valid_proposed_config({"gpu_memory_utilization": 1.0})
    client = _make_client(_make_response(proposed))

    with pytest.raises(OOMReasoningError, match="gpu_memory_utilization"):
        await reason_about_oom(ctx=_make_ctx(), anthropic_client=client)


@pytest.mark.asyncio
async def test_max_model_len_below_floor_raises():
    """max_model_len=2048 is below the 4096 floor → OOMReasoningError."""
    proposed = _valid_proposed_config({"max_model_len": 2048})
    client = _make_client(_make_response(proposed))

    with pytest.raises(OOMReasoningError, match="max_model_len"):
        await reason_about_oom(ctx=_make_ctx(), anthropic_client=client)


@pytest.mark.asyncio
async def test_loading_oom_reduced_max_model_len_raises():
    """
    Loading OOM + proposed max_model_len < current max_model_len is a
    misclassification error → OOMReasoningError.

    Current max_model_len is 8192; proposed is 4096 — that's a reduction.
    """
    proposed = _valid_proposed_config({"max_model_len": 4096})
    client = _make_client(_make_response(proposed))

    ctx = _make_ctx(oom_type="loading", current_max_model_len=8192)

    with pytest.raises(OOMReasoningError, match="LOADING OOM"):
        await reason_about_oom(ctx=ctx, anthropic_client=client)


@pytest.mark.asyncio
async def test_malformed_json_raises():
    """Non-JSON LLM response → OOMReasoningError."""
    client = _make_client("Sorry, I cannot help with that.")

    with pytest.raises(OOMReasoningError, match="invalid JSON"):
        await reason_about_oom(ctx=_make_ctx(), anthropic_client=client)


@pytest.mark.asyncio
async def test_missing_proposed_config_key_raises():
    """JSON response without 'proposed_config' key → OOMReasoningError."""
    response = json.dumps({"reasoning": "Looks fine to me."})
    client = _make_client(response)

    with pytest.raises(OOMReasoningError, match="proposed_config"):
        await reason_about_oom(ctx=_make_ctx(), anthropic_client=client)
