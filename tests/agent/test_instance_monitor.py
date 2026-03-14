"""
Tests for the instance death monitor (formerly spot eviction monitor).

Covers:
- _check_instance_dead helper for each provider
- Full monitor loop with mocked instances
- Exception handling during polling
- State filtering (only SERVING/BENCHMARKING monitored)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shittytoken.agent.spot_monitor import _check_instance_dead, instance_death_monitor
from shittytoken.agent.state_machine import InstanceRecord, InstanceState, InstanceStateMachine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sm(
    instance_id: str = "inst-001",
    provider: str = "vastai",
    state: InstanceState = InstanceState.SERVING,
    worker_url: str = "http://1.2.3.4:8080",
) -> InstanceStateMachine:
    record = InstanceRecord(
        instance_id=instance_id,
        provider=provider,
        gpu_model="RTX 3090",
        worker_url=worker_url,
    )
    sm = InstanceStateMachine(record)
    # Force state without going through transition validation
    sm._record.state = state
    return sm


# ===================================================================
# _check_instance_dead unit tests
# ===================================================================


class TestCheckInstanceDead:
    """Direct tests for the provider-specific death detection helper."""

    # -- Vast.ai --

    def test_vastai_exited_is_dead(self):
        assert _check_instance_dead("vastai", {"actual_status": "exited"}) is True

    def test_vastai_error_is_dead(self):
        assert _check_instance_dead("vastai", {"actual_status": "error"}) is True

    def test_vastai_running_is_alive(self):
        assert _check_instance_dead("vastai", {"actual_status": "running"}) is False

    def test_vastai_loading_is_alive(self):
        assert _check_instance_dead("vastai", {"actual_status": "loading"}) is False

    def test_vastai_empty_status_is_alive(self):
        """Empty status string is treated as alive (instance may be initializing)."""
        assert _check_instance_dead("vastai", {"actual_status": ""}) is False

    def test_vastai_no_status_key_is_alive(self):
        """Missing actual_status key defaults to empty string, treated as alive."""
        assert _check_instance_dead("vastai", {"some_other_field": "value"}) is False

    # -- RunPod --

    def test_runpod_spot_eviction_is_dead(self):
        pod_info = {"desiredStatus": "EXITED", "podType": "INTERRUPTABLE"}
        assert _check_instance_dead("runpod", pod_info) is True

    def test_runpod_running_spot_is_alive(self):
        pod_info = {"desiredStatus": "RUNNING", "podType": "INTERRUPTABLE"}
        assert _check_instance_dead("runpod", pod_info) is False

    def test_runpod_on_demand_exited_is_alive(self):
        """On-demand pods exiting is not a spot eviction."""
        pod_info = {"desiredStatus": "EXITED", "podType": "ON_DEMAND"}
        assert _check_instance_dead("runpod", pod_info) is False

    # -- Empty info (any provider) --

    def test_empty_dict_is_dead(self):
        assert _check_instance_dead("vastai", {}) is True

    def test_none_is_dead(self):
        assert _check_instance_dead("runpod", None) is True

    # -- Unknown provider --

    def test_unknown_provider_with_info_is_alive(self):
        """Unknown providers with non-empty info are not considered dead."""
        assert _check_instance_dead("lambda", {"status": "running"}) is False

    def test_unknown_provider_empty_info_is_dead(self):
        assert _check_instance_dead("lambda", {}) is True


# ===================================================================
# Full monitor loop tests
# ===================================================================


class TestInstanceDeathMonitor:
    """Integration tests for the monitor loop with mocked dependencies."""

    @pytest.fixture
    def shutdown_event(self):
        return asyncio.Event()

    @pytest.fixture
    def provision_lock(self):
        return asyncio.Lock()

    @pytest.fixture
    def mock_provider(self):
        provider = AsyncMock()
        return provider

    @pytest.fixture
    def mock_heartbeat(self):
        hb = MagicMock()
        hb.deregister = MagicMock()
        return hb

    @pytest.fixture
    def mock_gateway(self):
        gw = AsyncMock()
        return gw

    @pytest.fixture
    def mock_on_reprovision(self):
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_vastai_exited_triggers_cleanup(
        self, shutdown_event, provision_lock, mock_provider, mock_heartbeat,
        mock_gateway, mock_on_reprovision,
    ):
        """Vast.ai instance with status 'exited' should be cleaned up."""
        sm = _make_sm(provider="vastai", state=InstanceState.SERVING)
        instances = {sm.record.instance_id: sm}
        snapshots = {}

        mock_provider.get_instance.return_value = {"actual_status": "exited"}

        # Stop the monitor after one iteration
        async def stop_after_poll(*args, **kwargs):
            shutdown_event.set()

        mock_on_reprovision.side_effect = stop_after_poll

        with patch("shittytoken.agent.spot_monitor.cfg", {"orchestrator": {"instance_poll_interval_sec": 0}}):
            await asyncio.wait_for(
                instance_death_monitor(
                    provider=mock_provider,
                    instances=instances,
                    snapshots=snapshots,
                    heartbeat_monitor=mock_heartbeat,
                    gateway=mock_gateway,
                    shutdown_event=shutdown_event,
                    provision_lock=provision_lock,
                    on_reprovision=mock_on_reprovision,
                    poll_interval_sec=0,
                ),
                timeout=5.0,
            )

        assert sm.state == InstanceState.FAILED
        assert sm.record.instance_id not in instances
        mock_heartbeat.deregister.assert_called_once_with(sm.record.worker_url)
        mock_gateway.deregister_worker.assert_called_once()

    @pytest.mark.asyncio
    async def test_vastai_running_left_alone(
        self, shutdown_event, provision_lock, mock_provider, mock_heartbeat,
        mock_gateway, mock_on_reprovision,
    ):
        """Vast.ai instance with status 'running' should not be touched."""
        sm = _make_sm(provider="vastai", state=InstanceState.SERVING)
        instances = {sm.record.instance_id: sm}
        snapshots = {}

        mock_provider.get_instance.return_value = {"actual_status": "running"}

        # Run one iteration then stop
        call_count = 0

        original_sleep = asyncio.sleep

        async def fake_sleep(secs):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                shutdown_event.set()
            await original_sleep(0)

        with patch("shittytoken.agent.spot_monitor.asyncio.sleep", side_effect=fake_sleep):
            with patch("shittytoken.agent.spot_monitor.cfg", {"orchestrator": {"instance_poll_interval_sec": 0}}):
                await asyncio.wait_for(
                    instance_death_monitor(
                        provider=mock_provider,
                        instances=instances,
                        snapshots=snapshots,
                        heartbeat_monitor=mock_heartbeat,
                        gateway=mock_gateway,
                        shutdown_event=shutdown_event,
                        provision_lock=provision_lock,
                        on_reprovision=mock_on_reprovision,
                        poll_interval_sec=0,
                    ),
                    timeout=5.0,
                )

        assert sm.state == InstanceState.SERVING
        assert sm.record.instance_id in instances
        mock_heartbeat.deregister.assert_not_called()

    @pytest.mark.asyncio
    async def test_vastai_loading_left_alone(
        self, shutdown_event, provision_lock, mock_provider, mock_heartbeat,
        mock_gateway, mock_on_reprovision,
    ):
        """Vast.ai instance with status 'loading' should not be touched."""
        sm = _make_sm(provider="vastai", state=InstanceState.SERVING)
        instances = {sm.record.instance_id: sm}
        snapshots = {}

        mock_provider.get_instance.return_value = {"actual_status": "loading"}

        call_count = 0
        original_sleep = asyncio.sleep

        async def fake_sleep(secs):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                shutdown_event.set()
            await original_sleep(0)

        with patch("shittytoken.agent.spot_monitor.asyncio.sleep", side_effect=fake_sleep):
            with patch("shittytoken.agent.spot_monitor.cfg", {"orchestrator": {"instance_poll_interval_sec": 0}}):
                await asyncio.wait_for(
                    instance_death_monitor(
                        provider=mock_provider,
                        instances=instances,
                        snapshots=snapshots,
                        heartbeat_monitor=mock_heartbeat,
                        gateway=mock_gateway,
                        shutdown_event=shutdown_event,
                        provision_lock=provision_lock,
                        on_reprovision=mock_on_reprovision,
                        poll_interval_sec=0,
                    ),
                    timeout=5.0,
                )

        assert sm.state == InstanceState.SERVING
        assert sm.record.instance_id in instances

    @pytest.mark.asyncio
    async def test_runpod_spot_eviction_triggers_cleanup(
        self, shutdown_event, provision_lock, mock_provider, mock_heartbeat,
        mock_gateway, mock_on_reprovision,
    ):
        """RunPod spot eviction should still trigger cleanup (regression test)."""
        sm = _make_sm(provider="runpod", state=InstanceState.SERVING)
        instances = {sm.record.instance_id: sm}
        snapshots = {}

        mock_provider.get_instance.return_value = {
            "desiredStatus": "EXITED",
            "podType": "INTERRUPTABLE",
        }

        async def stop_after_poll(*args, **kwargs):
            shutdown_event.set()

        mock_on_reprovision.side_effect = stop_after_poll

        with patch("shittytoken.agent.spot_monitor.cfg", {"orchestrator": {"instance_poll_interval_sec": 0}}):
            await asyncio.wait_for(
                instance_death_monitor(
                    provider=mock_provider,
                    instances=instances,
                    snapshots=snapshots,
                    heartbeat_monitor=mock_heartbeat,
                    gateway=mock_gateway,
                    shutdown_event=shutdown_event,
                    provision_lock=provision_lock,
                    on_reprovision=mock_on_reprovision,
                    poll_interval_sec=0,
                ),
                timeout=5.0,
            )

        assert sm.state == InstanceState.FAILED
        assert sm.record.instance_id not in instances

    @pytest.mark.asyncio
    async def test_empty_instance_info_triggers_cleanup(
        self, shutdown_event, provision_lock, mock_provider, mock_heartbeat,
        mock_gateway, mock_on_reprovision,
    ):
        """Empty dict from get_instance should trigger cleanup for any provider."""
        sm = _make_sm(provider="vastai", state=InstanceState.SERVING)
        instances = {sm.record.instance_id: sm}
        snapshots = {}

        mock_provider.get_instance.return_value = {}

        async def stop_after_poll(*args, **kwargs):
            shutdown_event.set()

        mock_on_reprovision.side_effect = stop_after_poll

        with patch("shittytoken.agent.spot_monitor.cfg", {"orchestrator": {"instance_poll_interval_sec": 0}}):
            await asyncio.wait_for(
                instance_death_monitor(
                    provider=mock_provider,
                    instances=instances,
                    snapshots=snapshots,
                    heartbeat_monitor=mock_heartbeat,
                    gateway=mock_gateway,
                    shutdown_event=shutdown_event,
                    provision_lock=provision_lock,
                    on_reprovision=mock_on_reprovision,
                    poll_interval_sec=0,
                ),
                timeout=5.0,
            )

        assert sm.state == InstanceState.FAILED
        assert sm.record.instance_id not in instances

    @pytest.mark.asyncio
    async def test_provisioning_state_not_monitored(
        self, shutdown_event, provision_lock, mock_provider, mock_heartbeat,
        mock_gateway, mock_on_reprovision,
    ):
        """Instance in PROVISIONING state should be skipped by the monitor."""
        sm = _make_sm(provider="vastai", state=InstanceState.PROVISIONING)
        instances = {sm.record.instance_id: sm}
        snapshots = {}

        mock_provider.get_instance.return_value = {"actual_status": "exited"}

        call_count = 0
        original_sleep = asyncio.sleep

        async def fake_sleep(secs):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                shutdown_event.set()
            await original_sleep(0)

        with patch("shittytoken.agent.spot_monitor.asyncio.sleep", side_effect=fake_sleep):
            with patch("shittytoken.agent.spot_monitor.cfg", {"orchestrator": {"instance_poll_interval_sec": 0}}):
                await asyncio.wait_for(
                    instance_death_monitor(
                        provider=mock_provider,
                        instances=instances,
                        snapshots=snapshots,
                        heartbeat_monitor=mock_heartbeat,
                        gateway=mock_gateway,
                        shutdown_event=shutdown_event,
                        provision_lock=provision_lock,
                        on_reprovision=mock_on_reprovision,
                        poll_interval_sec=0,
                    ),
                    timeout=5.0,
                )

        # Should still be PROVISIONING — monitor skipped it
        assert sm.state == InstanceState.PROVISIONING
        assert sm.record.instance_id in instances
        mock_provider.get_instance.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_instance_exception_does_not_crash(
        self, shutdown_event, provision_lock, mock_provider, mock_heartbeat,
        mock_gateway, mock_on_reprovision,
    ):
        """provider.get_instance raising an exception should not crash the monitor."""
        sm = _make_sm(provider="vastai", state=InstanceState.SERVING)
        instances = {sm.record.instance_id: sm}
        snapshots = {}

        mock_provider.get_instance.side_effect = RuntimeError("API timeout")

        call_count = 0
        original_sleep = asyncio.sleep

        async def fake_sleep(secs):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                shutdown_event.set()
            await original_sleep(0)

        with patch("shittytoken.agent.spot_monitor.asyncio.sleep", side_effect=fake_sleep):
            with patch("shittytoken.agent.spot_monitor.cfg", {"orchestrator": {"instance_poll_interval_sec": 0}}):
                await asyncio.wait_for(
                    instance_death_monitor(
                        provider=mock_provider,
                        instances=instances,
                        snapshots=snapshots,
                        heartbeat_monitor=mock_heartbeat,
                        gateway=mock_gateway,
                        shutdown_event=shutdown_event,
                        provision_lock=provision_lock,
                        on_reprovision=mock_on_reprovision,
                        poll_interval_sec=0,
                    ),
                    timeout=5.0,
                )

        # Instance should be untouched — error was caught gracefully
        assert sm.state == InstanceState.SERVING
        assert sm.record.instance_id in instances
        mock_heartbeat.deregister.assert_not_called()
