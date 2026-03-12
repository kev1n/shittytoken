"""
Tests for src/shittytoken/benchmark/sse_client.py.

Uses aiohttp.TestServer backed by mock_openai_server.create_app().
"""

import pytest
import aiohttp
from aiohttp import web

from tests.benchmark import mock_openai_server
from shittytoken.benchmark.sse_client import send_chat_completion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MESSAGES = [{"role": "user", "content": "Hello"}]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_mock_counters():
    """Reset global counters before every test."""
    mock_openai_server.reset_counters()
    yield
    mock_openai_server.reset_counters()


@pytest.fixture()
def normal_app():
    """App with near-zero failure rate."""
    mock_openai_server.MOCK_FAIL_RATE = 0.0
    mock_openai_server.MOCK_NUM_OUTPUT_TOKENS = 10
    mock_openai_server.MOCK_TTFT_MS = 50
    mock_openai_server.MOCK_TOKEN_RATE_PER_SEC = 200
    return mock_openai_server.create_app()


@pytest.fixture()
def always_fail_app():
    """App that always returns HTTP 500."""
    mock_openai_server.MOCK_FAIL_RATE = 1.0
    mock_openai_server.MOCK_NUM_OUTPUT_TOKENS = 10
    mock_openai_server.MOCK_TTFT_MS = 50
    mock_openai_server.MOCK_TOKEN_RATE_PER_SEC = 200
    return mock_openai_server.create_app()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path(normal_app):
    """A successful request returns success=True, positive TTFT, and non-empty output."""
    async with aiohttp.test_utils.TestServer(normal_app) as server:
        base_url = f"http://{server.host}:{server.port}"
        connector = aiohttp.TCPConnector(limit=4)
        async with aiohttp.ClientSession(connector=connector) as session:
            result = await send_chat_completion(
                session=session,
                base_url=base_url,
                messages=_MESSAGES,
                max_tokens=50,
                session_id="test-happy",
                request_timeout=30.0,
                first_token_timeout=10.0,
            )

    assert result.success is True, f"Expected success, got error: {result.error}"
    assert result.ttft_sec is not None
    assert result.ttft_sec > 0
    assert result.output_text != ""
    assert result.tokens_generated > 0
    assert result.error is None


@pytest.mark.asyncio
async def test_http_500_returns_failure(always_fail_app):
    """When the server returns HTTP 500, success=False and ttft_sec=None."""
    async with aiohttp.test_utils.TestServer(always_fail_app) as server:
        base_url = f"http://{server.host}:{server.port}"
        connector = aiohttp.TCPConnector(limit=4)
        async with aiohttp.ClientSession(connector=connector) as session:
            result = await send_chat_completion(
                session=session,
                base_url=base_url,
                messages=_MESSAGES,
                max_tokens=50,
                session_id="test-fail",
                request_timeout=30.0,
                first_token_timeout=10.0,
            )

    assert result.success is False
    assert result.ttft_sec is None
    assert result.error is not None


@pytest.mark.asyncio
async def test_done_signal_terminates_stream(normal_app):
    """tokens_generated matches MOCK_NUM_OUTPUT_TOKENS when the stream ends normally."""
    expected_tokens = mock_openai_server.MOCK_NUM_OUTPUT_TOKENS

    async with aiohttp.test_utils.TestServer(normal_app) as server:
        base_url = f"http://{server.host}:{server.port}"
        connector = aiohttp.TCPConnector(limit=4)
        async with aiohttp.ClientSession(connector=connector) as session:
            result = await send_chat_completion(
                session=session,
                base_url=base_url,
                messages=_MESSAGES,
                max_tokens=200,
                session_id="test-done",
                request_timeout=30.0,
                first_token_timeout=10.0,
            )

    assert result.success is True
    assert result.tokens_generated == expected_tokens, (
        f"Expected {expected_tokens} tokens, got {result.tokens_generated}"
    )
