"""
Admin API — internal endpoints for worker pool management.

Protected by an optional admin token (``X-Admin-Token`` header).
When ``app["admin_token"]`` is ``None`` the check is skipped (dev mode).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiohttp import web

import structlog

if TYPE_CHECKING:
    from shittytoken.gateway.worker_pool import WorkerPool

logger = structlog.get_logger()


def _check_admin_token(request: web.Request) -> web.Response | None:
    """Return a 403 Response if the admin token is invalid, else None.

    When ``admin_token`` is not set, only requests from loopback (127.0.0.1 /
    ::1) are allowed — this permits the orchestrator (same host) to push
    metrics while blocking external access.
    """
    expected: str | None = request.app.get("admin_token")
    provided = request.headers.get("X-Admin-Token")
    if expected and provided == expected:
        return None  # valid token
    if expected and provided != expected:
        return web.json_response(
            {"error": {"message": "Forbidden: invalid admin token"}},
            status=403,
        )
    # No admin_token configured — restrict to loopback only
    peername = request.transport.get_extra_info("peername") if request.transport else None
    remote_ip = peername[0] if peername else ""
    if remote_ip in ("127.0.0.1", "::1", ""):
        return None
    return web.json_response(
        {"error": {"message": "Forbidden: admin endpoints require authentication"}},
        status=403,
    )


async def add_worker(request: web.Request) -> web.Response:
    """POST /admin/workers {"url": "http://host:port"} -> 201 Created."""
    denied = _check_admin_token(request)
    if denied is not None:
        return denied

    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            {"error": {"message": "Request body must be valid JSON"}},
            status=400,
        )

    url: str | None = body.get("url")
    if not url:
        return web.json_response(
            {"error": {"message": "Missing required field: url"}},
            status=400,
        )

    pool: WorkerPool = request.app["worker_pool"]
    try:
        await pool.add_worker(url)
    except ValueError as exc:
        return web.json_response(
            {"error": {"message": str(exc)}},
            status=409,
        )

    logger.info("admin.worker_added", url=url)
    return web.json_response({"url": url, "status": "added"}, status=201)


async def remove_worker(request: web.Request) -> web.Response:
    """DELETE /admin/workers {"url": "http://host:port"} -> 200 OK."""
    denied = _check_admin_token(request)
    if denied is not None:
        return denied

    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            {"error": {"message": "Request body must be valid JSON"}},
            status=400,
        )

    url: str | None = body.get("url")
    if not url:
        return web.json_response(
            {"error": {"message": "Missing required field: url"}},
            status=400,
        )

    pool: WorkerPool = request.app["worker_pool"]
    try:
        await pool.remove_worker(url)
    except KeyError as exc:
        return web.json_response(
            {"error": {"message": str(exc)}},
            status=404,
        )

    logger.info("admin.worker_removed", url=url)
    return web.json_response({"url": url, "status": "removed"}, status=200)


async def list_workers(request: web.Request) -> web.Response:
    """GET /admin/workers -> 200 [{url, requests_running, kv_cache_pct, healthy}]."""
    denied = _check_admin_token(request)
    if denied is not None:
        return denied

    pool: WorkerPool = request.app["worker_pool"]
    workers = pool.list_workers()

    result = [
        {
            "url": w.url,
            "requests_running": w.requests_running,
            "kv_cache_pct": w.kv_cache_pct,
            "healthy": w.healthy,
        }
        for w in workers
    ]

    return web.json_response(result, status=200)


async def update_orchestrator_metrics(request: web.Request) -> web.Response:
    """POST /admin/metrics {"instances": {"serving": 1}, "scale_events": {"scale_up": 2}}

    Called by the orchestrator to push its state into the gateway's metrics.
    """
    denied = _check_admin_token(request)
    if denied is not None:
        return denied

    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            {"error": {"message": "Request body must be valid JSON"}},
            status=400,
        )

    from shittytoken.gateway import prom_metrics

    instances = body.get("instances")
    if instances and isinstance(instances, dict):
        prom_metrics.set_instance_counts(instances)

    scale_events = body.get("scale_events")
    if scale_events and isinstance(scale_events, dict):
        for event_type, count in scale_events.items():
            prom_metrics._scale_events[event_type] = count

    # Cost metrics pushed from orchestrator's CostTracker
    cost = body.get("cost")
    if cost and isinstance(cost, dict):
        prom_metrics.set_cost_metrics(
            hourly_burn_usd=cost.get("hourly_burn_usd", 0.0),
            cumulative_cost_usd=cost.get("cumulative_cost_usd", 0.0),
        )

    return web.json_response({"status": "ok"}, status=200)


async def list_requests(request: web.Request) -> web.Response:
    """GET /admin/requests -> 200 [{ts, request_id, worker, status, duration_ms, ...}].

    Returns the most recent 200 requests (newest last) with per-request
    token breakdown and cost.
    """
    denied = _check_admin_token(request)
    if denied is not None:
        return denied

    from shittytoken.gateway.proxy import get_request_log

    return web.json_response(get_request_log(), status=200)
