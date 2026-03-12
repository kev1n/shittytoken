"""
Tests for oom/recovery.py.

Primary concern: the ORDERING INVARIANT — write_oom_event MUST be called
before destroy_fn. All KG methods and callables are mocked.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from shittytoken.knowledge.schema import Configuration
from shittytoken.oom.recovery import OOMRecovery
from shittytoken.oom.reasoner import OOMReasoningError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config() -> Configuration:
    return Configuration(
        tensor_parallel_size=1,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        quantization=None,
        kv_cache_dtype="auto",
        max_num_seqs=256,
        enable_prefix_caching=True,
        enforce_eager=False,
    )


def _make_kg(
    write_oom_event_return: str = "evt-abc123",
    write_configuration_return: str = "cfg-abc123",
    prior_resolutions_return: list | None = None,
) -> MagicMock:
    kg = MagicMock()
    kg.write_oom_event = AsyncMock(return_value=write_oom_event_return)
    kg.write_configuration = AsyncMock(return_value=write_configuration_return)
    kg.prior_oom_resolutions = AsyncMock(return_value=prior_resolutions_return or [])
    kg.update_oom_outcome = AsyncMock(return_value=None)
    return kg


def _make_anthropic_client(proposed_config: dict | None = None) -> MagicMock:
    """Returns a mock client whose reason_about_oom will return proposed_config."""
    client = MagicMock()
    # We patch reason_about_oom at the module level in tests, so the client
    # itself just needs to be a valid object reference.
    return client


_VALID_PROPOSED = {
    "tensor_parallel_size": 1,
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.85,
    "quantization": None,
    "kv_cache_dtype": "auto",
    "max_num_seqs": 128,
    "enable_prefix_caching": True,
    "enforce_eager": True,
}

_RECOVERY_KWARGS = dict(
    instance_id="inst-1",
    gpu_model_name="A100-80GB",
    gpu_vram_gb=80,
    gpu_memory_free_gb=10.0,
    model_id="meta-llama/Llama-3-8B",
    model_params_b=8.0,
    model_active_params_b=8.0,
    raw_error="torch.cuda.OutOfMemoryError: KV cache out of memory.",
    log_context="num_requests_running=128",
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_oom_event_called_before_destroy_fn():
    """
    CRITICAL ORDERING INVARIANT: write_oom_event must appear before destroy_fn
    in the call sequence.
    """
    call_order: list[str] = []

    kg = _make_kg()

    async def tracked_write_oom_event(*args, **kwargs):
        call_order.append("write_oom_event")
        return "evt-abc123"

    kg.write_oom_event = AsyncMock(side_effect=tracked_write_oom_event)
    kg.write_configuration = AsyncMock(return_value="cfg-abc123")
    kg.update_oom_outcome = AsyncMock(return_value=None)

    async def destroy_fn(instance_id: str) -> None:
        call_order.append("destroy_fn")

    async def provision_fn(config: Configuration) -> str:
        call_order.append("provision_fn")
        return "inst-new"

    current_config = _make_config()
    client = MagicMock()

    recovery = OOMRecovery(kg=kg, anthropic_client=client)

    with patch(
        "shittytoken.oom.recovery.reason_about_oom",
        new=AsyncMock(return_value=_VALID_PROPOSED),
    ):
        await recovery.recover(
            **_RECOVERY_KWARGS,
            current_config=current_config,
            destroy_fn=destroy_fn,
            provision_fn=provision_fn,
        )

    assert "write_oom_event" in call_order, "write_oom_event was never called"
    assert "destroy_fn" in call_order, "destroy_fn was never called"

    write_idx = call_order.index("write_oom_event")
    destroy_idx = call_order.index("destroy_fn")
    assert write_idx < destroy_idx, (
        f"write_oom_event (index {write_idx}) must come before "
        f"destroy_fn (index {destroy_idx}). Actual order: {call_order}"
    )


@pytest.mark.asyncio
async def test_successful_recovery_calls_update_oom_outcome_succeeded_true():
    """Successful provision → update_oom_outcome called with succeeded=True."""
    kg = _make_kg()
    current_config = _make_config()
    client = MagicMock()

    recovery = OOMRecovery(kg=kg, anthropic_client=client)

    with patch(
        "shittytoken.oom.recovery.reason_about_oom",
        new=AsyncMock(return_value=_VALID_PROPOSED),
    ):
        succeeded, new_config = await recovery.recover(
            **_RECOVERY_KWARGS,
            current_config=current_config,
            destroy_fn=AsyncMock(),
            provision_fn=AsyncMock(return_value="inst-new"),
        )

    assert succeeded is True
    assert new_config is not None

    kg.update_oom_outcome.assert_awaited_once()
    call_kwargs = kg.update_oom_outcome.call_args
    assert call_kwargs.kwargs.get("succeeded") is True or call_kwargs.args[1] is True


@pytest.mark.asyncio
async def test_failed_recovery_provision_returns_none_calls_update_oom_outcome_false():
    """provision_fn returns None → update_oom_outcome called with succeeded=False."""
    kg = _make_kg()
    current_config = _make_config()
    client = MagicMock()

    recovery = OOMRecovery(kg=kg, anthropic_client=client)

    with patch(
        "shittytoken.oom.recovery.reason_about_oom",
        new=AsyncMock(return_value=_VALID_PROPOSED),
    ):
        succeeded, new_config = await recovery.recover(
            **_RECOVERY_KWARGS,
            current_config=current_config,
            destroy_fn=AsyncMock(),
            provision_fn=AsyncMock(return_value=None),
        )

    assert succeeded is False
    assert new_config is None

    kg.update_oom_outcome.assert_awaited_once()
    call_kwargs = kg.update_oom_outcome.call_args
    assert (
        call_kwargs.kwargs.get("succeeded") is False
        or call_kwargs.args[1] is False
    )


@pytest.mark.asyncio
async def test_destroy_fn_raises_update_oom_outcome_still_called():
    """destroy_fn raises → update_oom_outcome is still called via try/finally."""
    kg = _make_kg()
    current_config = _make_config()
    client = MagicMock()

    recovery = OOMRecovery(kg=kg, anthropic_client=client)

    async def exploding_destroy(instance_id: str) -> None:
        raise RuntimeError("instance unreachable")

    with patch(
        "shittytoken.oom.recovery.reason_about_oom",
        new=AsyncMock(return_value=_VALID_PROPOSED),
    ):
        with pytest.raises(RuntimeError, match="instance unreachable"):
            await recovery.recover(
                **_RECOVERY_KWARGS,
                current_config=current_config,
                destroy_fn=exploding_destroy,
                provision_fn=AsyncMock(return_value="inst-new"),
            )

    kg.update_oom_outcome.assert_awaited_once()


@pytest.mark.asyncio
async def test_reasoning_error_calls_update_oom_outcome_false_no_destroy():
    """
    reason_about_oom raises OOMReasoningError → update_oom_outcome(succeeded=False)
    is called and destroy_fn is NOT called.
    """
    kg = _make_kg()
    current_config = _make_config()
    client = MagicMock()
    destroy_fn = AsyncMock()

    recovery = OOMRecovery(kg=kg, anthropic_client=client)

    with patch(
        "shittytoken.oom.recovery.reason_about_oom",
        new=AsyncMock(side_effect=OOMReasoningError("bad LLM output")),
    ):
        succeeded, new_config = await recovery.recover(
            **_RECOVERY_KWARGS,
            current_config=current_config,
            destroy_fn=destroy_fn,
            provision_fn=AsyncMock(return_value="inst-new"),
        )

    assert succeeded is False
    assert new_config is None

    # update_oom_outcome must be called with succeeded=False
    kg.update_oom_outcome.assert_awaited_once()
    call_kwargs = kg.update_oom_outcome.call_args
    assert (
        call_kwargs.kwargs.get("succeeded") is False
        or call_kwargs.args[1] is False
    )

    # destroy_fn must NOT have been called
    destroy_fn.assert_not_awaited()


@pytest.mark.asyncio
async def test_return_value_true_config_on_success():
    """recover() returns (True, Configuration) on successful provision."""
    kg = _make_kg()
    current_config = _make_config()
    client = MagicMock()

    recovery = OOMRecovery(kg=kg, anthropic_client=client)

    with patch(
        "shittytoken.oom.recovery.reason_about_oom",
        new=AsyncMock(return_value=_VALID_PROPOSED),
    ):
        succeeded, new_config = await recovery.recover(
            **_RECOVERY_KWARGS,
            current_config=current_config,
            destroy_fn=AsyncMock(),
            provision_fn=AsyncMock(return_value="inst-new"),
        )

    assert succeeded is True
    assert isinstance(new_config, Configuration)


@pytest.mark.asyncio
async def test_return_value_false_none_on_failure():
    """recover() returns (False, None) when provision_fn returns None."""
    kg = _make_kg()
    current_config = _make_config()
    client = MagicMock()

    recovery = OOMRecovery(kg=kg, anthropic_client=client)

    with patch(
        "shittytoken.oom.recovery.reason_about_oom",
        new=AsyncMock(return_value=_VALID_PROPOSED),
    ):
        succeeded, new_config = await recovery.recover(
            **_RECOVERY_KWARGS,
            current_config=current_config,
            destroy_fn=AsyncMock(),
            provision_fn=AsyncMock(return_value=None),
        )

    assert succeeded is False
    assert new_config is None
