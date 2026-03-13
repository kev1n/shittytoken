import asyncio

import aiohttp
from aiohttp import web
import structlog

from ..config import cfg
from .worker_pool import WorkerPool
from .routing_policy import CacheAwarePolicy
from .proxy import handle_chat_completions, handle_models
from .admin_api import add_worker, remove_worker, list_workers
from .prom_metrics import handle_metrics
from .middleware import request_id_middleware

logger = structlog.get_logger()


async def handle_health(request: web.Request) -> web.Response:
    return web.Response(text="ok\n", content_type="text/plain")


async def create_router_app(
    admin_token: str | None = None,
) -> web.Application:
    """
    Create the custom router aiohttp application.

    Mounts routes:
    - POST /v1/chat/completions -> proxy handler (SSE streaming)
    - GET /v1/models -> proxy handler (non-streaming)
    - GET /health -> simple 200 response
    - GET /metrics -> Prometheus metrics
    - POST /admin/workers -> add worker
    - DELETE /admin/workers -> remove worker
    - GET /admin/workers -> list workers

    Stores shared state in app dict:
    - app["worker_pool"] = WorkerPool instance
    - app["admin_token"] = admin token for admin API auth
    - app["upstream_session"] = aiohttp.ClientSession for upstream connections
    """
    app = web.Application(middlewares=[request_id_middleware])

    # Create shared state
    policy = CacheAwarePolicy()
    pool = WorkerPool(policy=policy)
    app["worker_pool"] = pool
    app["routing_policy"] = policy
    app["admin_token"] = admin_token

    # Lifecycle hooks
    async def on_startup(app: web.Application) -> None:
        connector = aiohttp.TCPConnector(limit=256)
        session = aiohttp.ClientSession(connector=connector)
        app["upstream_session"] = session
        app["_scraper_task"] = asyncio.ensure_future(pool.run_scraper())
        logger.info("router_app_started")

    async def on_cleanup(app: web.Application) -> None:
        app["_scraper_task"].cancel()
        try:
            await app["_scraper_task"]
        except asyncio.CancelledError:
            pass
        await app["upstream_session"].close()
        logger.info("router_app_stopped")

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    # Mount routes
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/metrics", handle_metrics)
    app.router.add_post("/admin/workers", add_worker)
    app.router.add_delete("/admin/workers", remove_worker)
    app.router.add_get("/admin/workers", list_workers)

    return app
