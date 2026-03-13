"""
Tests for non-streaming proxy retry logic.

Verifies that when an upstream worker connection fails, non-streaming
requests retry with a different worker while streaming requests do not.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from shittytoken.gateway.proxy import handle_chat_completions
from shittytoken.gateway.worker_pool import WorkerPool, WorkerState
from shittytoken.gateway.routing_policy import CacheAwarePolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app(pool: WorkerPool, session: MagicMock) -> web.Application:
    """Build a minimal aiohttp app wired for proxy tests."""
    app = web.Application()
    app["worker_pool"] = pool
    app["upstream_session"] = session
    app["routing_policy"] = CacheAwarePolicy()
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    return app


def _chat_body(stream: bool = False) -> dict:
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": stream,
    }


def _mock_response(status: int = 200, body: dict | None = None) -> MagicMock:
    """Create a mock aiohttp ClientResponse."""
    resp = MagicMock()
    resp.status = status
    resp.content_type = "application/json"
    body = body or {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 10}}
    resp.text = AsyncMock(return_value=json.dumps(body))
    resp.release = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# WorkerPool.select exclude tests
# ---------------------------------------------------------------------------


class TestPoolSelectExclude:
    """Verify that WorkerPool.select filters out excluded URLs."""

    def test_exclude_filters_worker(self):
        pool = WorkerPool()
        pool._workers = {
            "http://w1:8000": WorkerState(url="http://w1:8000"),
            "http://w2:8000": WorkerState(url="http://w2:8000"),
        }
        result = pool.select("key", exclude={"http://w1:8000"})
        assert result is not None
        assert result.url == "http://w2:8000"

    def test_exclude_all_returns_none(self):
        pool = WorkerPool()
        pool._workers = {
            "http://w1:8000": WorkerState(url="http://w1:8000"),
        }
        result = pool.select("key", exclude={"http://w1:8000"})
        assert result is None

    def test_exclude_none_returns_worker(self):
        pool = WorkerPool()
        pool._workers = {
            "http://w1:8000": WorkerState(url="http://w1:8000"),
        }
        result = pool.select("key", exclude=None)
        assert result is not None

    def test_exclude_empty_set_returns_worker(self):
        pool = WorkerPool()
        pool._workers = {
            "http://w1:8000": WorkerState(url="http://w1:8000"),
        }
        result = pool.select("key", exclude=set())
        assert result is not None

    def test_exclude_nonexistent_url_keeps_all(self):
        pool = WorkerPool()
        pool._workers = {
            "http://w1:8000": WorkerState(url="http://w1:8000"),
            "http://w2:8000": WorkerState(url="http://w2:8000"),
        }
        result = pool.select("key", exclude={"http://w3:8000"})
        assert result is not None


# ---------------------------------------------------------------------------
# Proxy retry integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def worker_a() -> WorkerState:
    return WorkerState(url="http://worker-a:8000")


@pytest.fixture
def worker_b() -> WorkerState:
    return WorkerState(url="http://worker-b:8000")


@pytest.fixture
def pool_two_workers(worker_a, worker_b) -> WorkerPool:
    pool = WorkerPool()
    pool._workers = {
        worker_a.url: worker_a,
        worker_b.url: worker_b,
    }
    return pool


@pytest.fixture
def pool_one_worker(worker_a) -> WorkerPool:
    pool = WorkerPool()
    pool._workers = {worker_a.url: worker_a}
    return pool


class TestNonStreamingRetry:
    """Non-streaming requests should retry once on upstream connection error."""

    async def test_retry_succeeds_with_second_worker(self, pool_two_workers, worker_a, worker_b):
        """First worker fails, second succeeds -> 200."""
        mock_session = MagicMock()
        good_resp = _mock_response(200)
        # First call raises, second succeeds
        mock_session.post = AsyncMock(
            side_effect=[aiohttp.ClientError("conn refused"), good_resp]
        )

        app = _make_app(pool_two_workers, mock_session)

        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json=_chat_body(stream=False),
            )
            assert resp.status == 200
            data = await resp.json()
            assert "usage" in data

    async def test_retry_fails_both_workers_returns_502(self, pool_two_workers):
        """Both workers fail -> 502."""
        mock_session = MagicMock()
        mock_session.post = AsyncMock(
            side_effect=[
                aiohttp.ClientError("conn refused"),
                aiohttp.ClientError("conn refused again"),
            ]
        )

        app = _make_app(pool_two_workers, mock_session)

        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json=_chat_body(stream=False),
            )
            assert resp.status == 502

    async def test_no_retry_worker_available_returns_502(self, pool_one_worker):
        """Only one worker and it fails -> 502 (no retry candidate)."""
        mock_session = MagicMock()
        mock_session.post = AsyncMock(
            side_effect=aiohttp.ClientError("conn refused")
        )

        app = _make_app(pool_one_worker, mock_session)

        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json=_chat_body(stream=False),
            )
            assert resp.status == 502


class TestStreamingNoRetry:
    """Streaming requests should NOT retry on upstream connection error."""

    async def test_streaming_returns_502_without_retry(self, pool_two_workers):
        """Streaming request with connection error -> 502 immediately, no retry."""
        mock_session = MagicMock()
        mock_session.post = AsyncMock(
            side_effect=aiohttp.ClientError("conn refused")
        )

        app = _make_app(pool_two_workers, mock_session)

        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json=_chat_body(stream=True),
            )
            assert resp.status == 502
            # Should have been called only once (no retry)
            assert mock_session.post.call_count == 1


class TestFirstWorkerSucceeds:
    """When the first worker succeeds, no retry logic is triggered."""

    async def test_no_retry_on_success(self, pool_two_workers):
        mock_session = MagicMock()
        good_resp = _mock_response(200)
        mock_session.post = AsyncMock(return_value=good_resp)

        app = _make_app(pool_two_workers, mock_session)

        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json=_chat_body(stream=False),
            )
            assert resp.status == 200
            assert mock_session.post.call_count == 1
