"""
Tests for HeartbeatMonitor graceful drain on health failure.

Covers:
- Callback is NOT fired on the first failure (threshold not yet reached)
- Callback IS fired after N consecutive failures (default threshold=3)
- Failure counter resets to 0 on a successful health check
- deregister() cleans up the failure counter
- Custom threshold (e.g., threshold=2) works correctly
"""

from __future__ import annotations

import aiohttp
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from shittytoken.agent.health import HeartbeatMonitor


@pytest.fixture
def mock_session():
    return MagicMock(spec=aiohttp.ClientSession)


@pytest.fixture
def failure_callback():
    return AsyncMock()


@pytest.fixture
def monitor(mock_session, failure_callback):
    return HeartbeatMonitor(
        session=mock_session,
        health_check_interval_s=5,
        on_failure=failure_callback,
    )


def _mock_response(status: int):
    """Create a mock aiohttp response with the given status code."""
    resp = AsyncMock()
    resp.status = status
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


# -----------------------------------------------------------------
# Test: callback NOT fired on first failure
# -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_callback_not_fired_on_first_failure(monitor, mock_session, failure_callback):
    """A single health check failure should NOT trigger the callback."""
    mock_session.get.return_value = _mock_response(503)

    await monitor._check_worker("http://worker:8000")

    failure_callback.assert_not_called()


@pytest.mark.asyncio
async def test_callback_not_fired_below_threshold(monitor, mock_session, failure_callback):
    """Failures below the threshold (default 3) should NOT trigger the callback."""
    mock_session.get.return_value = _mock_response(503)

    await monitor._check_worker("http://worker:8000")
    await monitor._check_worker("http://worker:8000")

    failure_callback.assert_not_called()


# -----------------------------------------------------------------
# Test: callback IS fired after N consecutive failures
# -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_callback_fired_at_threshold(monitor, mock_session, failure_callback):
    """After 3 consecutive failures (default threshold), callback should fire."""
    mock_session.get.return_value = _mock_response(503)

    for _ in range(3):
        await monitor._check_worker("http://worker:8000")

    failure_callback.assert_awaited_once_with("http://worker:8000")


@pytest.mark.asyncio
async def test_callback_fires_on_connection_error(monitor, mock_session, failure_callback):
    """Connection errors also count toward the failure threshold."""
    mock_session.get.side_effect = aiohttp.ClientConnectionError("refused")

    for _ in range(3):
        await monitor._check_worker("http://worker:8000")

    failure_callback.assert_awaited_once_with("http://worker:8000")


# -----------------------------------------------------------------
# Test: counter resets on successful health check
# -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_counter_resets_on_success(monitor, mock_session, failure_callback):
    """A successful health check should reset the failure counter to 0."""
    # Two failures
    mock_session.get.return_value = _mock_response(503)
    await monitor._check_worker("http://worker:8000")
    await monitor._check_worker("http://worker:8000")

    # One success resets
    mock_session.get.return_value = _mock_response(200)
    await monitor._check_worker("http://worker:8000")

    assert monitor._failure_counts["http://worker:8000"] == 0

    # Two more failures — still below threshold
    mock_session.get.return_value = _mock_response(503)
    await monitor._check_worker("http://worker:8000")
    await monitor._check_worker("http://worker:8000")

    failure_callback.assert_not_called()


@pytest.mark.asyncio
async def test_full_reset_cycle(monitor, mock_session, failure_callback):
    """After reset, it takes another full threshold of failures to trigger callback."""
    url = "http://worker:8000"

    # 2 failures, then success
    mock_session.get.return_value = _mock_response(500)
    await monitor._check_worker(url)
    await monitor._check_worker(url)
    mock_session.get.return_value = _mock_response(200)
    await monitor._check_worker(url)

    # Now 3 more failures should trigger
    mock_session.get.return_value = _mock_response(500)
    for _ in range(3):
        await monitor._check_worker(url)

    failure_callback.assert_awaited_once_with(url)


# -----------------------------------------------------------------
# Test: deregister cleans up failure counter
# -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_deregister_cleans_up_failure_count(monitor, mock_session):
    """deregister() should remove the URL from _failure_counts."""
    mock_session.get.return_value = _mock_response(503)
    url = "http://worker:8000"

    monitor.register(url)
    await monitor._check_worker(url)
    assert url in monitor._failure_counts

    monitor.deregister(url)
    assert url not in monitor._failure_counts
    assert url not in monitor._workers


# -----------------------------------------------------------------
# Test: custom threshold
# -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_custom_threshold(mock_session, failure_callback):
    """A custom failure_threshold=2 should fire callback after 2 failures."""
    monitor = HeartbeatMonitor(
        session=mock_session,
        health_check_interval_s=5,
        failure_threshold=2,
        on_failure=failure_callback,
    )
    mock_session.get.return_value = _mock_response(503)

    await monitor._check_worker("http://worker:8000")
    failure_callback.assert_not_called()

    await monitor._check_worker("http://worker:8000")
    failure_callback.assert_awaited_once_with("http://worker:8000")


@pytest.mark.asyncio
async def test_custom_threshold_one(mock_session, failure_callback):
    """threshold=1 should fire callback on the very first failure (old behavior)."""
    monitor = HeartbeatMonitor(
        session=mock_session,
        health_check_interval_s=5,
        failure_threshold=1,
        on_failure=failure_callback,
    )
    mock_session.get.return_value = _mock_response(503)

    await monitor._check_worker("http://worker:8000")
    failure_callback.assert_awaited_once_with("http://worker:8000")
