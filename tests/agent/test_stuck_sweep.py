"""
Tests for Orchestrator._sweep_stuck_instances().

Covers:
- Stuck PROVISIONING instance (state_changed_at older than timeout) gets destroyed
- Stuck BENCHMARKING instance gets destroyed
- SERVING instance is never swept (even if old)
- Instance within timeout is NOT swept
- Multiple stuck instances all get swept
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from shittytoken.agent.orchestrator import Orchestrator
from shittytoken.agent.state_machine import InstanceRecord, InstanceState, InstanceStateMachine


def _make_orchestrator() -> Orchestrator:
    settings = MagicMock()
    kg = AsyncMock()
    gateway = AsyncMock()
    orch = Orchestrator(settings=settings, kg=kg, gateway=gateway)
    orch._session = AsyncMock()
    orch._provider = AsyncMock()
    return orch


def _make_instance(
    instance_id: str,
    state: InstanceState,
    state_changed_at: float,
) -> InstanceStateMachine:
    record = InstanceRecord(
        instance_id=instance_id,
        provider="vastai",
        gpu_model="RTX 3090",
        state=state,
        state_changed_at=state_changed_at,
    )
    sm = InstanceStateMachine(record)
    # Override internal state directly to avoid transition validation
    sm._record.state = state
    return sm


TIMEOUT = 1320  # default stuck_instance_timeout_s


# -----------------------------------------------------------------
# Test: stuck PROVISIONING instance gets destroyed
# -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_stuck_provisioning_instance_destroyed():
    """A PROVISIONING instance older than the timeout should be swept."""
    orch = _make_orchestrator()
    now = 100000.0
    sm = _make_instance("i-prov-1", InstanceState.PROVISIONING, now - TIMEOUT - 1)
    orch._instances["i-prov-1"] = sm

    with patch("shittytoken.agent.orchestrator.time") as mock_time, \
         patch.object(orch, "_destroy_instance", new_callable=AsyncMock) as mock_destroy:
        mock_time.time.return_value = now
        await orch._sweep_stuck_instances()

    assert sm.state == InstanceState.FAILED
    mock_destroy.assert_awaited_once_with(sm.record, sm)


# -----------------------------------------------------------------
# Test: stuck BENCHMARKING instance gets destroyed
# -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_stuck_benchmarking_instance_destroyed():
    """A BENCHMARKING instance older than the timeout should be swept."""
    orch = _make_orchestrator()
    now = 100000.0
    sm = _make_instance("i-bench-1", InstanceState.BENCHMARKING, now - TIMEOUT - 1)
    orch._instances["i-bench-1"] = sm

    with patch("shittytoken.agent.orchestrator.time") as mock_time, \
         patch.object(orch, "_destroy_instance", new_callable=AsyncMock) as mock_destroy:
        mock_time.time.return_value = now
        await orch._sweep_stuck_instances()

    assert sm.state == InstanceState.FAILED
    mock_destroy.assert_awaited_once_with(sm.record, sm)


# -----------------------------------------------------------------
# Test: SERVING instance is never swept
# -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_serving_instance_not_swept():
    """A SERVING instance should never be swept, even if very old."""
    orch = _make_orchestrator()
    now = 100000.0
    sm = _make_instance("i-serve-1", InstanceState.SERVING, now - TIMEOUT - 9999)
    orch._instances["i-serve-1"] = sm

    with patch("shittytoken.agent.orchestrator.time") as mock_time, \
         patch.object(orch, "_destroy_instance", new_callable=AsyncMock) as mock_destroy:
        mock_time.time.return_value = now
        await orch._sweep_stuck_instances()

    assert sm.state == InstanceState.SERVING
    mock_destroy.assert_not_awaited()


# -----------------------------------------------------------------
# Test: instance within timeout is NOT swept
# -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_instance_within_timeout_not_swept():
    """A PROVISIONING instance within the timeout should not be swept."""
    orch = _make_orchestrator()
    now = 100000.0
    # state_changed_at is only 60 seconds ago -- well within timeout
    sm = _make_instance("i-recent-1", InstanceState.PROVISIONING, now - 60)
    orch._instances["i-recent-1"] = sm

    with patch("shittytoken.agent.orchestrator.time") as mock_time, \
         patch.object(orch, "_destroy_instance", new_callable=AsyncMock) as mock_destroy:
        mock_time.time.return_value = now
        await orch._sweep_stuck_instances()

    assert sm.state == InstanceState.PROVISIONING
    mock_destroy.assert_not_awaited()


# -----------------------------------------------------------------
# Test: multiple stuck instances all get swept
# -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_stuck_instances_all_swept():
    """All stuck instances should be swept in a single pass."""
    orch = _make_orchestrator()
    now = 100000.0

    sm1 = _make_instance("i-prov-A", InstanceState.PROVISIONING, now - TIMEOUT - 10)
    sm2 = _make_instance("i-bench-B", InstanceState.BENCHMARKING, now - TIMEOUT - 500)
    sm3 = _make_instance("i-serve-C", InstanceState.SERVING, now - TIMEOUT - 999)
    sm4 = _make_instance("i-prov-D", InstanceState.PROVISIONING, now - 30)  # recent

    orch._instances = {
        "i-prov-A": sm1,
        "i-bench-B": sm2,
        "i-serve-C": sm3,
        "i-prov-D": sm4,
    }

    with patch("shittytoken.agent.orchestrator.time") as mock_time, \
         patch.object(orch, "_destroy_instance", new_callable=AsyncMock) as mock_destroy:
        mock_time.time.return_value = now
        await orch._sweep_stuck_instances()

    # sm1 and sm2 should be swept
    assert sm1.state == InstanceState.FAILED
    assert sm2.state == InstanceState.FAILED
    # sm3 (SERVING) and sm4 (recent PROVISIONING) should be untouched
    assert sm3.state == InstanceState.SERVING
    assert sm4.state == InstanceState.PROVISIONING

    assert mock_destroy.await_count == 2
    destroyed_ids = {call.args[0].instance_id for call in mock_destroy.await_args_list}
    assert destroyed_ids == {"i-prov-A", "i-bench-B"}
