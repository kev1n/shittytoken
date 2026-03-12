"""
Mock vLLM-compatible HTTP server for use in unit tests and docker-compose.

Simulates:
- POST /v1/chat/completions  — SSE streaming with configurable TTFT, token rate,
  failure rate, and output length.
- GET  /metrics              — Prometheus text with all 5 vLLM metrics.
- GET  /v1/models            — Minimal models list.
- GET  /health               — 200 OK.

All behaviour is controlled by module-level globals so that tests can monkeypatch
them without environment variables (env vars provide defaults for standalone use).
"""

import asyncio
import json
import os
import time

import aiohttp.web

# ---------------------------------------------------------------------------
# Configurable parameters (environment variable defaults for standalone mode).
# ---------------------------------------------------------------------------

MOCK_TTFT_MS: int = int(os.getenv("MOCK_TTFT_MS", "150"))
MOCK_TOKEN_RATE_PER_SEC: int = int(os.getenv("MOCK_TOKEN_RATE_PER_SEC", "80"))
MOCK_CACHE_HIT_RATE: float = float(os.getenv("MOCK_CACHE_HIT_RATE", "0.65"))
MOCK_FAIL_RATE: float = float(os.getenv("MOCK_FAIL_RATE", "0.01"))
MOCK_NUM_OUTPUT_TOKENS: int = int(os.getenv("MOCK_NUM_OUTPUT_TOKENS", "50"))

# ---------------------------------------------------------------------------
# Global counters (for /metrics).
# ---------------------------------------------------------------------------

_requests_total: int = 0
_cache_queries: int = 0
_cache_hits: int = 0
_requests_running: int = 0


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


async def handle_chat_completions(
    request: aiohttp.web.Request,
) -> aiohttp.web.StreamResponse:
    """
    At MOCK_FAIL_RATE probability: returns HTTP 500.
    Otherwise:
      1. Sleep MOCK_TTFT_MS/1000 seconds (simulates TTFT).
      2. Stream MOCK_NUM_OUTPUT_TOKENS SSE chunks at MOCK_TOKEN_RATE_PER_SEC.
      3. Send "data: [DONE]".
    Updates global counters throughout.
    """
    import random

    global _requests_total, _cache_queries, _cache_hits, _requests_running

    _requests_total += 1
    _cache_queries += 1

    # Simulate cache hit (for metrics counter accuracy).
    import random as _random

    if _random.random() < MOCK_CACHE_HIT_RATE:
        _cache_hits += 1

    # Simulate failure.
    if _random.random() < MOCK_FAIL_RATE:
        raise aiohttp.web.HTTPInternalServerError(text="mock internal server error")

    _requests_running += 1
    try:
        # Simulate TTFT delay.
        await asyncio.sleep(MOCK_TTFT_MS / 1000.0)

        response = aiohttp.web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)

        # Stream tokens at the configured rate.
        token_interval_sec = 1.0 / max(MOCK_TOKEN_RATE_PER_SEC, 1)
        for i in range(MOCK_NUM_OUTPUT_TOKENS):
            chunk = {
                "id": f"mock-{_requests_total}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "mock-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"tok{i} "},
                        "finish_reason": None,
                    }
                ],
            }
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
            if i < MOCK_NUM_OUTPUT_TOKENS - 1:
                await asyncio.sleep(token_interval_sec)

        # Final done sentinel.
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response
    finally:
        _requests_running -= 1


async def handle_metrics(request: aiohttp.web.Request) -> aiohttp.web.Response:
    """Returns Prometheus text format with all 5 required vLLM metrics."""
    body = (
        f"# HELP vllm:prefix_cache_queries_total Total prefix cache queries\n"
        f"# TYPE vllm:prefix_cache_queries_total counter\n"
        f"vllm:prefix_cache_queries_total {_cache_queries}\n"
        f"# HELP vllm:prefix_cache_hits_total Total prefix cache hits\n"
        f"# TYPE vllm:prefix_cache_hits_total counter\n"
        f"vllm:prefix_cache_hits_total {_cache_hits}\n"
        f"# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage fraction\n"
        f"# TYPE vllm:gpu_cache_usage_perc gauge\n"
        f"vllm:gpu_cache_usage_perc 0.42\n"
        f"# HELP vllm:num_requests_running Requests currently executing\n"
        f"# TYPE vllm:num_requests_running gauge\n"
        f"vllm:num_requests_running {_requests_running}\n"
        f"# HELP vllm:num_requests_waiting Requests in queue\n"
        f"# TYPE vllm:num_requests_waiting gauge\n"
        f"vllm:num_requests_waiting 0\n"
    )
    return aiohttp.web.Response(
        text=body,
        headers={"Content-Type": "text/plain; version=0.0.4; charset=utf-8"},
    )


async def handle_models(request: aiohttp.web.Request) -> aiohttp.web.Response:
    return aiohttp.web.json_response(
        {"data": [{"id": "mock-model", "object": "model"}]}
    )


async def handle_health(request: aiohttp.web.Request) -> aiohttp.web.Response:
    return aiohttp.web.Response(status=200)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> aiohttp.web.Application:
    app = aiohttp.web.Application()
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/metrics", handle_metrics)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/health", handle_health)
    return app


def reset_counters() -> None:
    """Reset all global counters — call this in test setUp / fixture teardown."""
    global _requests_total, _cache_queries, _cache_hits, _requests_running
    _requests_total = 0
    _cache_queries = 0
    _cache_hits = 0
    _requests_running = 0


if __name__ == "__main__":
    aiohttp.web.run_app(create_app(), host="0.0.0.0", port=8000)
