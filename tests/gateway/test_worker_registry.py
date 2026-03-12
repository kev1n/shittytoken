"""
Unit tests for gateway/worker_registry.py.

RouterManager is replaced by an AsyncMock so no real process or network
I/O occurs.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shittytoken.common.prometheus import parse_prometheus_text
from shittytoken.gateway.worker_registry import (
    CircuitBreakerState,
    WorkerEntry,
    WorkerRegistry,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_router_manager():
    """A RouterManager stand-in whose reload() is an AsyncMock."""
    mgr = MagicMock()
    mgr.reload = AsyncMock(return_value=None)
    return mgr


@pytest.fixture
def registry(mock_router_manager):
    return WorkerRegistry(router_manager=mock_router_manager)


# ---------------------------------------------------------------------------
# add_worker
# ---------------------------------------------------------------------------


class TestAddWorker:
    async def test_add_worker_calls_reload(self, registry, mock_router_manager):
        await registry.add_worker("http://w1:8000")
        mock_router_manager.reload.assert_called_once_with(["http://w1:8000"])

    async def test_add_worker_appears_in_list(self, registry):
        await registry.add_worker("http://w1:8000")
        workers = registry.list_workers()
        assert len(workers) == 1
        assert workers[0].url == "http://w1:8000"

    async def test_add_worker_duplicate_raises_value_error(self, registry):
        await registry.add_worker("http://w1:8000")
        with pytest.raises(ValueError, match="already registered"):
            await registry.add_worker("http://w1:8000")

    async def test_add_multiple_workers_reload_includes_all(
        self, registry, mock_router_manager
    ):
        await registry.add_worker("http://w1:8000")
        await registry.add_worker("http://w2:8000")
        # Second reload should include both URLs.
        last_call_args = mock_router_manager.reload.call_args[0][0]
        assert set(last_call_args) == {"http://w1:8000", "http://w2:8000"}

    async def test_add_worker_sets_registered_at(self, registry):
        before = time.time()
        await registry.add_worker("http://w1:8000")
        after = time.time()
        entry = registry.get_worker("http://w1:8000")
        assert entry is not None
        assert before <= entry.registered_at <= after


# ---------------------------------------------------------------------------
# remove_worker
# ---------------------------------------------------------------------------


class TestRemoveWorker:
    async def test_remove_worker_no_drain_calls_reload_immediately(
        self, registry, mock_router_manager
    ):
        await registry.add_worker("http://w1:8000")
        mock_router_manager.reload.reset_mock()

        await registry.remove_worker("http://w1:8000", drain=False)

        mock_router_manager.reload.assert_called_once_with([])

    async def test_remove_worker_no_drain_no_polling(
        self, registry, mock_router_manager
    ):
        await registry.add_worker("http://w1:8000")
        with patch.object(
            registry, "_poll_drain_complete", new_callable=AsyncMock
        ) as mock_poll:
            await registry.remove_worker("http://w1:8000", drain=False)
            mock_poll.assert_not_called()

    async def test_remove_worker_drain_calls_poll(
        self, registry, mock_router_manager
    ):
        await registry.add_worker("http://w1:8000")
        with patch.object(
            registry, "_poll_drain_complete", new_callable=AsyncMock
        ) as mock_poll:
            await registry.remove_worker("http://w1:8000", drain=True)
            mock_poll.assert_called_once_with("http://w1:8000")

    async def test_add_then_remove_list_is_empty(self, registry):
        await registry.add_worker("http://w1:8000")
        await registry.remove_worker("http://w1:8000", drain=False)
        assert registry.list_workers() == []

    async def test_remove_unknown_worker_raises_key_error(self, registry):
        with pytest.raises(KeyError, match="not found"):
            await registry.remove_worker("http://unknown:8000", drain=False)

    async def test_remove_worker_reload_excludes_removed_url(
        self, registry, mock_router_manager
    ):
        await registry.add_worker("http://w1:8000")
        await registry.add_worker("http://w2:8000")
        mock_router_manager.reload.reset_mock()

        await registry.remove_worker("http://w1:8000", drain=False)

        mock_router_manager.reload.assert_called_once_with(["http://w2:8000"])


# ---------------------------------------------------------------------------
# update_circuit_state
# ---------------------------------------------------------------------------


class TestUpdateCircuitState:
    async def test_update_circuit_state_changes_entry(self, registry):
        await registry.add_worker("http://w1:8000")
        registry.update_circuit_state("http://w1:8000", CircuitBreakerState.OPEN)
        entry = registry.get_worker("http://w1:8000")
        assert entry is not None
        assert entry.circuit_state == CircuitBreakerState.OPEN

    async def test_update_circuit_state_closed_to_half_open(self, registry):
        await registry.add_worker("http://w1:8000")
        registry.update_circuit_state("http://w1:8000", CircuitBreakerState.HALF_OPEN)
        entry = registry.get_worker("http://w1:8000")
        assert entry is not None
        assert entry.circuit_state == CircuitBreakerState.HALF_OPEN

    async def test_update_circuit_state_unknown_url_raises_key_error(self, registry):
        with pytest.raises(KeyError):
            registry.update_circuit_state("http://ghost:8000", CircuitBreakerState.OPEN)


# ---------------------------------------------------------------------------
# list_workers / get_worker
# ---------------------------------------------------------------------------


class TestListAndGetWorkers:
    async def test_list_workers_returns_snapshot(self, registry):
        await registry.add_worker("http://w1:8000")
        await registry.add_worker("http://w2:8000")
        workers = registry.list_workers()
        assert len(workers) == 2
        urls = {w.url for w in workers}
        assert urls == {"http://w1:8000", "http://w2:8000"}

    async def test_get_worker_returns_none_for_unknown(self, registry):
        assert registry.get_worker("http://unknown:8000") is None

    async def test_get_worker_returns_entry(self, registry):
        await registry.add_worker("http://w1:8000")
        entry = registry.get_worker("http://w1:8000")
        assert isinstance(entry, WorkerEntry)
        assert entry.url == "http://w1:8000"


# ---------------------------------------------------------------------------
# WorkerEntry defaults
# ---------------------------------------------------------------------------


class TestWorkerEntryDefaults:
    def test_default_circuit_state_is_closed(self):
        entry = WorkerEntry(url="http://w1:8000", registered_at=time.time())
        assert entry.circuit_state == CircuitBreakerState.CLOSED

    def test_default_not_draining(self):
        entry = WorkerEntry(url="http://w1:8000", registered_at=time.time())
        assert entry.draining is False

    def test_default_consecutive_failures_zero(self):
        entry = WorkerEntry(url="http://w1:8000", registered_at=time.time())
        assert entry.consecutive_failures == 0


# ---------------------------------------------------------------------------
# parse_prometheus_text (internal helper — tested directly)
# ---------------------------------------------------------------------------


class TestParsePrometheusText:
    def test_parses_simple_metric(self):
        text = "num_requests_running 3\n"
        result = parse_prometheus_text(text)
        assert result["num_requests_running"] == 3.0

    def test_skips_comment_lines(self):
        text = "# HELP num_requests_running foo\nnum_requests_running 5\n"
        result = parse_prometheus_text(text)
        assert result["num_requests_running"] == 5.0
        assert "# HELP num_requests_running foo" not in result

    def test_strips_label_block(self):
        text = 'vllm:gpu_cache_usage_perc{model_name="qwen"} 0.72\n'
        result = parse_prometheus_text(text)
        assert "vllm:gpu_cache_usage_perc" in result
        assert result["vllm:gpu_cache_usage_perc"] == pytest.approx(0.72)

    def test_skips_blank_lines(self):
        text = "\n\nnum_requests_running 1\n\n"
        result = parse_prometheus_text(text)
        assert result["num_requests_running"] == 1.0

    def test_empty_text_returns_empty_dict(self):
        assert parse_prometheus_text("") == {}

    def test_last_occurrence_wins(self):
        text = "my_metric 1\nmy_metric 2\n"
        result = parse_prometheus_text(text)
        assert result["my_metric"] == 2.0

    def test_non_numeric_value_skipped(self):
        text = "my_metric NaN\n"
        result = parse_prometheus_text(text)
        assert "my_metric" not in result
