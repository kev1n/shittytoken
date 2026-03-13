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
    """Return a 403 Response if the admin token is invalid, else None."""
    expected: str | None = request.app.get("admin_token")
    if expected is None:
        return None  # dev mode — no auth
    provided = request.headers.get("X-Admin-Token")
    if provided != expected:
        return web.json_response(
            {"error": {"message": "Forbidden: invalid admin token"}},
            status=403,
        )
    return None


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
