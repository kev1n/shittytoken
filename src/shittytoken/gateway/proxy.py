"""
SSE pass-through proxy — hot path for every inference request.

Proxies POST /v1/chat/completions to a selected vLLM worker using
cache-aware routing, streaming SSE chunks back to the client without
buffering.  Extracts token usage from the final SSE chunk for metrics.
"""

from __future__ import annotations

import collections
import json
import time
from typing import Any, Callable, Awaitable, TYPE_CHECKING

import aiohttp
from aiohttp import web

import structlog

if TYPE_CHECKING:
    from shittytoken.gateway.worker_pool import WorkerPool

from shittytoken.gateway.routing_policy import CacheAwarePolicy
from shittytoken.gateway import prom_metrics

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Per-request ring buffer — most recent N requests for /admin/requests
# ---------------------------------------------------------------------------

_REQUEST_LOG: collections.deque[dict] = collections.deque(maxlen=200)

# Pricing constants ($/token) — adjust if cached tokens are priced differently
_PRICE_INPUT = 2.5 / 100 / 1_000_000   # $0.025 per 1M input tokens
_PRICE_CACHED = 2.5 / 100 / 1_000_000  # same rate for now — change if discounted
_PRICE_OUTPUT = 15.0 / 100 / 1_000_000  # $0.15 per 1M output tokens


async def _log_request(
    request_id: str | None,
    worker_url: str,
    duration_sec: float,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
    status: int,
) -> None:
    """Append a completed request to the ring buffer."""
    uncached = prompt_tokens - cached_tokens
    cost = uncached * _PRICE_INPUT + cached_tokens * _PRICE_CACHED + completion_tokens * _PRICE_OUTPUT
    entry = {
        "ts": time.time(),
        "request_id": request_id,
        "worker": worker_url,
        "status": status,
        "duration_ms": round(duration_sec * 1000, 1),
        "uncached_tokens": uncached,
        "cached_tokens": cached_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": round(cost, 8),
    }
    _REQUEST_LOG.append(entry)


def get_request_log() -> list[dict]:
    """Return a copy of the ring buffer (newest last)."""
    return list(_REQUEST_LOG)


def _upstream_unavailable() -> web.Response:
    return web.json_response(
        {"error": {"message": "Upstream worker unavailable"}},
        status=502,
    )

_UPSTREAM_TIMEOUT = aiohttp.ClientTimeout(total=300, connect=10)

UsageCallback = Callable[[int, int], Awaitable[None]] | Callable[[int, int], None] | None


async def _record_usage(
    prompt_tokens: int,
    completion_tokens: int,
    on_usage: UsageCallback,
    cached_tokens: int = 0,
) -> None:
    """Record token metrics and invoke the billing callback (if any)."""
    if not (prompt_tokens or completion_tokens):
        return
    prom_metrics.add_tokens(prompt_tokens, completion_tokens, cached_tokens)
    if on_usage is not None:
        try:
            result = on_usage(prompt_tokens, completion_tokens)
            if hasattr(result, "__await__"):
                await result
        except Exception:
            logger.warning("proxy.on_usage_callback_error", exc_info=True)


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

    # Normalize message roles — some SDKs (agno) send "developer" which
    # vLLM doesn't understand.  Rewrite to "system".
    messages = body.get("messages", [])
    for msg in messages:
        if msg.get("role") == "developer":
            msg["role"] = "system"

    # Build per-request usage callback from billing pipeline.
    on_usage: UsageCallback = None
    pipeline = request.app.get("billing_pipeline")
    user_id: str | None = request.get("user_id")
    key_hash: str | None = request.get("key_hash")
    request_id: str | None = request.get("request_id")
    if pipeline is not None and user_id is not None and key_hash is not None:
        model_id = body.get("model", "unknown")

        async def _billing_callback(prompt_tokens: int, completion_tokens: int) -> None:
            latency_ms = int((time.monotonic() - request_start) * 1000)
            await pipeline.publish_usage(
                user_id=user_id,
                key_hash=key_hash,
                model=model_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
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
    request_start = time.monotonic()

    # ---- routing ---------------------------------------------------------
    prefix_key = request.headers.get("X-Session-ID")
    if prefix_key is None:
        prefix_key = policy.compute_prefix_key(messages)

    worker = pool.select(prefix_key)
    if worker is None:
        prom_metrics.dec_active()
        logger.error("proxy.no_workers")
        return web.json_response(
            {"error": {"message": "No healthy workers available"}},
            status=503,
        )
    prom_metrics.inc_worker_request(worker.url)

    # ---- prepare upstream request ----------------------------------------
    upstream_body = dict(body)
    client_streaming = body.get("stream", False)
    if client_streaming:
        upstream_body["stream"] = True
        upstream_body.setdefault("stream_options", {})["include_usage"] = True

    upstream_url = f"{worker.url.rstrip('/')}/v1/chat/completions"

    # ---- open upstream connection ----------------------------------------
    gateway_overhead = time.monotonic() - request_start
    prom_metrics.observe_overhead(gateway_overhead)

    try:
        upstream_resp = await session.post(
            upstream_url,
            json=upstream_body,
            timeout=_UPSTREAM_TIMEOUT,
        )
    except (aiohttp.ClientError, TimeoutError) as exc:
        logger.warning("proxy.upstream_connect_error", worker=worker.url, error=str(exc))
        # Retry once with a different worker (non-streaming only)
        if not client_streaming:
            retry_worker = pool.select(prefix_key, exclude={worker.url})
            if retry_worker is not None:
                retry_url = f"{retry_worker.url.rstrip('/')}/v1/chat/completions"
                logger.info("proxy.retry_attempt", original=worker.url, retry=retry_worker.url)
                try:
                    upstream_resp = await session.post(
                        retry_url,
                        json=upstream_body,
                        timeout=_UPSTREAM_TIMEOUT,
                    )
                except (aiohttp.ClientError, TimeoutError) as retry_exc:
                    prom_metrics.dec_active()
                    logger.error("proxy.retry_failed", worker=retry_worker.url, error=str(retry_exc))
                    return _upstream_unavailable()
                # Update worker reference for logging
                worker = retry_worker
                upstream_url = retry_url
            else:
                prom_metrics.dec_active()
                logger.error("proxy.no_retry_worker")
                return _upstream_unavailable()
        else:
            prom_metrics.dec_active()
            logger.error("proxy.upstream_connect_error_no_retry", worker=worker.url)
            return _upstream_unavailable()

    if upstream_resp.status != 200:
        error_body = await upstream_resp.text()
        logger.warning(
            "proxy.upstream_http_error",
            worker=worker.url,
            status=upstream_resp.status,
            body=error_body[:500],
        )
        upstream_resp.release()
        prom_metrics.dec_active()
        return web.Response(
            text=error_body,
            status=upstream_resp.status,
            content_type=upstream_resp.content_type or "application/json",
        )

    # ---- non-streaming: return JSON response directly --------------------
    if not client_streaming:
        resp_body = await upstream_resp.text()
        upstream_resp.release()
        prom_metrics.dec_active()
        prom_metrics.inc_request("POST", 200)
        duration = time.monotonic() - request_start
        prom_metrics.observe_latency(duration)
        prompt_tok = 0
        completion_tok = 0
        cached_tok = 0
        try:
            resp_json = json.loads(resp_body)
            usage = resp_json.get("usage", {})
            ptd = usage.get("prompt_tokens_details") or {}
            cached_tok = ptd.get("cached_tokens", 0)
            prompt_tok = usage.get("prompt_tokens", 0)
            completion_tok = usage.get("completion_tokens", 0)
            await _record_usage(prompt_tok, completion_tok, on_usage, cached_tokens=cached_tok)
        except (json.JSONDecodeError, AttributeError):
            pass
        await _log_request(request_id, worker.url, duration, prompt_tok, completion_tok, cached_tok, 200)
        return web.Response(
            text=resp_body,
            status=200,
            content_type="application/json",
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
    try:
        await response.prepare(request)
    except Exception:
        upstream_resp.release()
        prom_metrics.dec_active()
        raise

    last_data_line: str | None = None
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    stream_error = False

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
                                ptd = usage.get("prompt_tokens_details") or {}
                                cached_tokens = ptd.get("cached_tokens", 0)
                            except (json.JSONDecodeError, AttributeError):
                                logger.warning(
                                    "proxy.usage_parse_error",
                                    raw=last_data_line[:200],
                                )
                    else:
                        last_data_line = data_str
    except Exception as exc:
        stream_error = True
        logger.error(
            "proxy.mid_stream_error",
            worker=worker.url,
            error=str(exc),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    finally:
        upstream_resp.release()
        prom_metrics.dec_active()
        if not stream_error:
            prom_metrics.inc_request("POST", 200)
        duration = time.monotonic() - request_start
        prom_metrics.observe_latency(duration)
        await _record_usage(prompt_tokens, completion_tokens, on_usage, cached_tokens=cached_tokens)
        await _log_request(request_id, worker.url, duration, prompt_tokens, completion_tokens, cached_tokens, 200 if not stream_error else 500)

    try:
        await response.write_eof()
    except Exception:
        pass
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
        return _upstream_unavailable()
