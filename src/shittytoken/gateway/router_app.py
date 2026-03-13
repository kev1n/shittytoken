import asyncio
import os

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
from .auth import auth_middleware
from .billing_db import BillingDB
from .billing import BillingManager

logger = structlog.get_logger()

# Billing config defaults
_billing_cfg = cfg.get("gateway", {}).get("billing", {})
_auth_cfg = cfg.get("gateway", {}).get("auth", {})


async def handle_health(request: web.Request) -> web.Response:
    return web.Response(text="ok\n", content_type="text/plain")


async def create_router_app(
    admin_token: str | None = None,
) -> web.Application:
    """
    Create the custom router aiohttp application.

    Mounts routes, initialises billing DB, and wires auth + billing middleware.
    """
    middlewares = [request_id_middleware, auth_middleware]
    app = web.Application(middlewares=middlewares)

    # ── Shared state ──────────────────────────────────────────────────
    policy = CacheAwarePolicy()
    pool = WorkerPool(policy=policy)
    app["worker_pool"] = pool
    app["routing_policy"] = policy
    app["admin_token"] = admin_token
    app["auth_enabled"] = _auth_cfg.get("enabled", False)

    # ── Billing DB ────────────────────────────────────────────────────
    db_path = _billing_cfg.get("db_path", "data/billing.db")
    billing_db = BillingDB(db_path)
    app["billing_db"] = billing_db

    # ── Billing Manager ───────────────────────────────────────────────
    pricing = _billing_cfg.get("pricing", {})
    stripe_sync = _billing_cfg.get("stripe_sync_enabled", False)
    billing_mgr = BillingManager(
        db=billing_db,
        pricing=pricing,
        stripe_sync_enabled=stripe_sync,
    )
    app["billing_manager"] = billing_mgr

    # ── Lifecycle hooks ───────────────────────────────────────────────
    async def on_startup(app: web.Application) -> None:
        # Ensure data directory exists for SQLite
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        await billing_db.init()

        connector = aiohttp.TCPConnector(limit=256)
        session = aiohttp.ClientSession(connector=connector)
        app["upstream_session"] = session
        app["_scraper_task"] = asyncio.ensure_future(pool.run_scraper())
        app["_flush_task"] = asyncio.ensure_future(
            billing_mgr.run_flush_loop(
                interval_sec=_billing_cfg.get("flush_interval_s", 1.0),
            )
        )
        if stripe_sync:
            app["_stripe_sync_task"] = asyncio.ensure_future(
                billing_mgr.run_stripe_sync()
            )
        logger.info("router_app_started", auth_enabled=app["auth_enabled"])

    async def on_cleanup(app: web.Application) -> None:
        for task_key in ("_scraper_task", "_flush_task", "_stripe_sync_task"):
            task = app.get(task_key)
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await app["upstream_session"].close()
        await billing_db.close()
        logger.info("router_app_stopped")

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    # ── Routes ────────────────────────────────────────────────────────
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/metrics", handle_metrics)
    app.router.add_post("/admin/workers", add_worker)
    app.router.add_delete("/admin/workers", remove_worker)
    app.router.add_get("/admin/workers", list_workers)

    return app
