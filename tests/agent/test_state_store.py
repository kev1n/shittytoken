"""
Tests for RedisStateStore — instance state persistence and recovery.

Covers:
- Round-trip: save an InstanceRecord, load_all, verify fields match
- Delete: save then delete, load_all returns empty
- Multiple records: load_all returns all saved records
- State enum deserialization
- Recovery logic: alive SERVING instance recovered, dead instance cleaned up
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis
import pytest
import pytest_asyncio

from shittytoken.agent.state_machine import (
    InstanceRecord,
    InstanceState,
    InstanceStateMachine,
)
from shittytoken.agent.state_store import RedisStateStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def store():
    """Create a RedisStateStore backed by fakeredis."""
    r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    s = RedisStateStore(redis=r)
    yield s
    await r.aclose()


def _make_record(**overrides) -> InstanceRecord:
    """Helper: create a test InstanceRecord with sensible defaults."""
    defaults = dict(
        instance_id="inst-001",
        provider="vastai",
        gpu_model="RTX 3090",
        ssh_host="192.168.1.1",
        ssh_port=22,
        ssh_user="root",
        http_port=8080,
        worker_url="http://192.168.1.1:8080",
        state=InstanceState.SERVING,
        config_id="cfg-abc",
        created_at=1700000000.0,
        state_changed_at=1700001000.0,
        cost_per_hour_usd=0.45,
    )
    defaults.update(overrides)
    return InstanceRecord(**defaults)


# ---------------------------------------------------------------------------
# Round-trip save/load
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_and_load_roundtrip(store: RedisStateStore):
    """Save a record, load_all, verify all fields match."""
    original = _make_record()
    await store.save(original)

    records = await store.load_all()
    assert len(records) == 1

    loaded = records[0]
    assert loaded.instance_id == original.instance_id
    assert loaded.provider == original.provider
    assert loaded.gpu_model == original.gpu_model
    assert loaded.ssh_host == original.ssh_host
    assert loaded.ssh_port == original.ssh_port
    assert loaded.ssh_user == original.ssh_user
    assert loaded.http_port == original.http_port
    assert loaded.worker_url == original.worker_url
    assert loaded.state == original.state
    assert loaded.config_id == original.config_id
    assert loaded.created_at == original.created_at
    assert loaded.state_changed_at == original.state_changed_at
    assert loaded.cost_per_hour_usd == original.cost_per_hour_usd


@pytest.mark.asyncio
async def test_save_with_none_http_port(store: RedisStateStore):
    """http_port=None should round-trip correctly."""
    original = _make_record(http_port=None)
    await store.save(original)

    records = await store.load_all()
    assert len(records) == 1
    assert records[0].http_port is None


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_removes_record(store: RedisStateStore):
    """After delete, load_all should return an empty list."""
    record = _make_record()
    await store.save(record)
    await store.delete(record.instance_id)

    records = await store.load_all()
    assert len(records) == 0


# ---------------------------------------------------------------------------
# Multiple records
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_all_multiple_records(store: RedisStateStore):
    """load_all returns all saved records."""
    r1 = _make_record(instance_id="inst-001")
    r2 = _make_record(instance_id="inst-002", gpu_model="RTX 4090")
    r3 = _make_record(instance_id="inst-003", state=InstanceState.PROVISIONING)

    await store.save(r1)
    await store.save(r2)
    await store.save(r3)

    records = await store.load_all()
    assert len(records) == 3

    ids = {r.instance_id for r in records}
    assert ids == {"inst-001", "inst-002", "inst-003"}


# ---------------------------------------------------------------------------
# State enum deserialization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_state_enum_deserialization(store: RedisStateStore):
    """Each InstanceState value should round-trip through Redis correctly."""
    for state in InstanceState:
        record = _make_record(instance_id=f"inst-{state.value}", state=state)
        await store.save(record)

    records = await store.load_all()
    states = {r.instance_id: r.state for r in records}

    for state in InstanceState:
        assert states[f"inst-{state.value}"] == state


# ---------------------------------------------------------------------------
# Overwrite on re-save
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_overwrites_existing(store: RedisStateStore):
    """Saving the same instance_id again overwrites all fields."""
    record = _make_record(state=InstanceState.PROVISIONING)
    await store.save(record)

    record.state = InstanceState.SERVING
    record.worker_url = "http://10.0.0.1:8080"
    await store.save(record)

    records = await store.load_all()
    assert len(records) == 1
    assert records[0].state == InstanceState.SERVING
    assert records[0].worker_url == "http://10.0.0.1:8080"


# ---------------------------------------------------------------------------
# Recovery logic (orchestrator integration)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recovery_restores_alive_serving_instance(store: RedisStateStore):
    """An alive SERVING instance with a worker_url should be recovered."""
    from shittytoken.agent.orchestrator import Orchestrator

    record = _make_record(
        state=InstanceState.SERVING,
        worker_url="http://10.0.0.1:8080",
    )
    await store.save(record)

    # Build orchestrator with mocks
    settings = MagicMock()
    kg = AsyncMock()
    gateway = AsyncMock()
    heartbeat = MagicMock()

    orch = Orchestrator(
        settings=settings, kg=kg, gateway=gateway, state_store=store,
    )
    # Inject mocks for the internals that _recover_instances uses
    orch._heartbeat_monitor = heartbeat

    mock_provider = AsyncMock()
    mock_provider.get_instance.return_value = {"id": record.instance_id, "status": "running"}
    orch._provider = mock_provider

    # Mock session.get() to return an async context manager with status=200
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_resp
    mock_session = MagicMock()
    mock_session.get.return_value = mock_ctx
    orch._session = mock_session

    await orch._recover_instances()

    assert record.instance_id in orch._instances
    heartbeat.register.assert_called_once_with(record.worker_url)
    gateway.register_worker.assert_called_once_with(record.worker_url)


@pytest.mark.asyncio
async def test_recovery_cleans_up_dead_instance(store: RedisStateStore):
    """A dead instance should be destroyed and deleted from state store."""
    from shittytoken.agent.orchestrator import Orchestrator

    record = _make_record(
        instance_id="dead-inst",
        state=InstanceState.SERVING,
        worker_url="http://10.0.0.1:8080",
    )
    await store.save(record)

    settings = MagicMock()
    kg = AsyncMock()
    gateway = AsyncMock()
    heartbeat = MagicMock()

    orch = Orchestrator(
        settings=settings, kg=kg, gateway=gateway, state_store=store,
    )
    orch._heartbeat_monitor = heartbeat

    mock_provider = AsyncMock()
    # Simulate dead instance: get_instance raises
    mock_provider.get_instance.side_effect = Exception("instance not found")
    orch._provider = mock_provider
    orch._session = AsyncMock()

    await orch._recover_instances()

    # Instance should NOT be in the orchestrator
    assert "dead-inst" not in orch._instances
    # Provider should have tried to destroy it
    mock_provider.destroy_instance.assert_called_once_with("dead-inst")
    # State store should be clean
    records = await store.load_all()
    assert len(records) == 0


@pytest.mark.asyncio
async def test_recovery_cleans_up_provisioning_instance(store: RedisStateStore):
    """Instances in PROVISIONING state should be cleaned up regardless of liveness."""
    from shittytoken.agent.orchestrator import Orchestrator

    record = _make_record(
        instance_id="prov-inst",
        state=InstanceState.PROVISIONING,
        worker_url="",
    )
    await store.save(record)

    settings = MagicMock()
    kg = AsyncMock()
    gateway = AsyncMock()
    heartbeat = MagicMock()

    orch = Orchestrator(
        settings=settings, kg=kg, gateway=gateway, state_store=store,
    )
    orch._heartbeat_monitor = heartbeat

    mock_provider = AsyncMock()
    # Even if instance is alive, PROVISIONING should be cleaned up
    mock_provider.get_instance.return_value = {"id": "prov-inst", "status": "running"}
    orch._provider = mock_provider
    orch._session = AsyncMock()

    await orch._recover_instances()

    assert "prov-inst" not in orch._instances
    mock_provider.destroy_instance.assert_called_once_with("prov-inst")
    records = await store.load_all()
    assert len(records) == 0
