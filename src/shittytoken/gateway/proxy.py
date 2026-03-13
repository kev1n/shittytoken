"""
SSE pass-through proxy — hot path for every inference request.

Proxies POST /v1/chat/completions to a selected vLLM worker using
cache-aware routing, streaming SSE chunks back to the client without
buffering.  Extracts token usage from the final SSE chunk for metrics.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Awaitable, TYPE_CHECKING

import aiohttp
from aiohttp import web

import structlog

if TYPE_CHECKING:
    from shittytoken.gateway.worker_pool import WorkerPool

from shittytoken.gateway.routing_policy import CacheAwarePolicy
from shittytoken.gateway import prom_metrics

logger = structlog.get_logger()

_UPSTREAM_TIMEOUT = aiohttp.ClientTimeout(total=300, connect=10)


async def handle_chat_completions(request: web.Request) -> web.StreamResponse:
    """Proxy POST /v1/chat/completions with SSE streaming."""
    pool: WorkerPool = request.app["worker_pool"]
    session: aiohttp.ClientSession = request.app["upstream_session"]
    policy: CacheAwarePolicy = request.app["routing_policy"]

    # ---- read client body ------------------------------------------------
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        logger.warning("proxy.invalid_json")
        return web.json_response(
            {"error": {"message": "Request body must be valid JSON"}},
            status=400,
        )

    # Build per-request usage callback from billing pipeline.
    on_usage: Callable[[int, int], Awaitable[None]] | Callable[[int, int], None] | None = None
    pipeline = request.app.get("billing_pipeline")
    user_id: str | None = request.get("user_id")
    key_hash: str | None = request.get("key_hash")
    request_id: str | None = request.get("request_id")
    if pipeline is not None and user_id is not None and key_hash is not None:
        model_id = body.get("model", "unknown")

        async def _billing_callback(prompt_tokens: int, completion_tokens: int) -> None:
            await pipeline.publish_usage(
                user_id=user_id,
                key_hash=key_hash,
                model=model_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=0,
                request_id=request_id,
            )
            # Record actual tokens in rate limiter
            billing_redis = request.app.get("billing_redis")
            if billing_redis is not None:
                await billing_redis.record_tokens(key_hash, prompt_tokens + completion_tokens)

        on_usage = _billing_callback
    else:
        on_usage = request.app.get("on_usage")

    prom_metrics.inc_active()

    # ---- routing ---------------------------------------------------------
    messages = body.get("messages", [])
    prefix_key = request.headers.get("X-Session-ID")
    if prefix_key is None:
        prefix_key = policy.compute_prefix_key(messages)

    worker = pool.select(prefix_key)
    if worker is None:
        logger.error("proxy.no_workers")
        return web.json_response(
            {"error": {"message": "No healthy workers available"}},
            status=503,
        )

    # ---- prepare upstream request ----------------------------------------
    upstream_body = dict(body)
    upstream_body["stream"] = True
    upstream_body.setdefault("stream_options", {})["include_usage"] = True

    upstream_url = f"{worker.url.rstrip('/')}/v1/chat/completions"

    # ---- open upstream connection ----------------------------------------
    try:
        upstream_resp = await session.post(
            upstream_url,
            json=upstream_body,
            timeout=_UPSTREAM_TIMEOUT,
        )
    except (aiohttp.ClientError, TimeoutError) as exc:
        logger.error("proxy.upstream_connect_error", worker=worker.url, error=str(exc))
        return web.json_response(
            {"error": {"message": "Upstream worker unavailable"}},
            status=502,
        )

    if upstream_resp.status != 200:
        error_body = await upstream_resp.text()
        logger.warning(
            "proxy.upstream_http_error",
            worker=worker.url,
            status=upstream_resp.status,
            body=error_body[:500],
        )
        upstream_resp.release()
        return web.Response(
            text=error_body,
            status=upstream_resp.status,
            content_type=upstream_resp.content_type or "application/json",
        )

    # ---- stream SSE back to client ---------------------------------------
    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)

    last_data_line: str | None = None
    prompt_tokens = 0
    completion_tokens = 0

    try:
        async for raw_chunk in upstream_resp.content:
            # Write the raw bytes to client immediately (no buffering).
            await response.write(raw_chunk)

            # Parse lines from the chunk to track the last data payload.
            text = raw_chunk.decode("utf-8", errors="replace")
            for line in text.split("\n"):
                line = line.rstrip("\r")
                if line.startswith("data: "):
                    data_str = line[len("data: "):]
                    if data_str == "[DONE]":
                        # Stream finished — parse usage from buffered last data line.
                        if last_data_line is not None:
                            try:
                                final_chunk = json.loads(last_data_line)
                                usage = final_chunk.get("usage", {})
                                prompt_tokens = usage.get("prompt_tokens", 0)
                                completion_tokens = usage.get("completion_tokens", 0)
                            except (json.JSONDecodeError, AttributeError):
                                logger.warning(
                                    "proxy.usage_parse_error",
                                    raw=last_data_line[:200],
                                )
                    else:
                        last_data_line = data_str
    except Exception as exc:
        logger.error(
            "proxy.mid_stream_error",
            worker=worker.url,
            error=str(exc),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        # Close client connection on mid-stream failure.
        await response.write_eof()
        upstream_resp.release()
        return response

    upstream_resp.release()

    # ---- metrics & billing ------------------------------------------------
    prom_metrics.dec_active()
    prom_metrics.inc_request("POST", 200)
    if prompt_tokens or completion_tokens:
        prom_metrics.add_tokens(prompt_tokens, completion_tokens)

    if on_usage is not None and (prompt_tokens or completion_tokens):
        try:
            result = on_usage(prompt_tokens, completion_tokens)
            if hasattr(result, "__await__"):
                await result
        except Exception:
            logger.warning("proxy.on_usage_callback_error", exc_info=True)

    await response.write_eof()
    return response


async def handle_models(request: web.Request) -> web.Response:
    """Proxy GET /v1/models to any healthy worker (non-streaming)."""
    pool: WorkerPool = request.app["worker_pool"]
    session: aiohttp.ClientSession = request.app["upstream_session"]

    worker = pool.select("models")
    if worker is None:
        return web.json_response(
            {"error": {"message": "No healthy workers available"}},
            status=503,
        )

    upstream_url = f"{worker.url.rstrip('/')}/v1/models"

    try:
        async with session.get(
            upstream_url,
            timeout=aiohttp.ClientTimeout(total=30, connect=10),
        ) as upstream_resp:
            body = await upstream_resp.text()
            return web.Response(
                text=body,
                status=upstream_resp.status,
                content_type=upstream_resp.content_type or "application/json",
            )
    except (aiohttp.ClientError, TimeoutError) as exc:
        logger.error("proxy.models_upstream_error", worker=worker.url, error=str(exc))
        return web.json_response(
            {"error": {"message": "Upstream worker unavailable"}},
            status=502,
        )
