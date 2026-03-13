"""
Application factory for the custom aiohttp router.

Wires together: worker pool, routing policy, proxy, admin API,
auth middleware (Redis-backed), and the billing pipeline
(Redis hot-path + Postgres ledger + usage event consumer).
"""

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

logger = structlog.get_logger()

_gateway_cfg = cfg.get("gateway", {})
_billing_cfg = _gateway_cfg.get("billing", {})
_auth_cfg = _gateway_cfg.get("auth", {})


async def handle_health(request: web.Request) -> web.Response:
    return web.Response(text="ok\n", content_type="text/plain")


async def create_router_app(
    admin_token: str | None = None,
) -> web.Application:
    """Create the custom router aiohttp application."""
    middlewares = [request_id_middleware, auth_middleware]
    app = web.Application(middlewares=middlewares)

    # ── Core routing state ────────────────────────────────────────────
    policy = CacheAwarePolicy()
    pool = WorkerPool(policy=policy)
    app["worker_pool"] = pool
    app["routing_policy"] = policy
    app["admin_token"] = admin_token
    app["auth_enabled"] = _auth_cfg.get("enabled", False)

    # ── Lifecycle hooks ───────────────────────────────────────────────
    async def on_startup(app: web.Application) -> None:
        # Upstream HTTP session for proxying to vLLM workers
        connector = aiohttp.TCPConnector(limit=256)
        app["upstream_session"] = aiohttp.ClientSession(connector=connector)

        # Background worker metrics scraper
        app["_scraper_task"] = asyncio.ensure_future(pool.run_scraper())

        # ── Billing infrastructure (only if auth enabled) ────────────
        if app["auth_enabled"]:
            from shittytoken.billing.postgres import BillingPostgres
            from shittytoken.billing.redis_cache import BillingRedis
            from shittytoken.billing.usage_pipeline import (
                BillingPipeline,
                InProcessConsumer,
                InProcessPublisher,
            )
            from shittytoken.billing.reconciler import Reconciler

            pg_dsn = _billing_cfg.get("postgres_dsn", "postgresql://localhost/shittytoken")
            redis_url = _billing_cfg.get("redis_url", "redis://localhost:6379/0")
            pricing = _billing_cfg.get("pricing", {})

            billing_pg = await BillingPostgres.create(pg_dsn)
            billing_redis = await BillingRedis.create(redis_url)

            app["billing_postgres"] = billing_pg
            app["billing_redis"] = billing_redis

            # Usage pipeline (in-process queue; swap for Kafka when ready)
            publisher = InProcessPublisher()
            consumer = InProcessConsumer(publisher)
            pipeline = BillingPipeline(
                publisher=publisher,
                consumer=consumer,
                postgres=billing_pg,
                redis=billing_redis,
                pricing=pricing,
            )
            app["billing_pipeline"] = pipeline

            # Background tasks
            app["_consumer_task"] = asyncio.ensure_future(pipeline.run_consumer())
            reconciler = Reconciler(
                postgres=billing_pg,
                redis=billing_redis,
                interval_sec=_billing_cfg.get("reconcile_interval_s", 60.0),
            )
            app["_reconciler_task"] = asyncio.ensure_future(reconciler.run())

            logger.info("billing_started", pg_dsn=pg_dsn, redis_url=redis_url)
        else:
            logger.info("billing_disabled")

        logger.info("router_app_started", auth_enabled=app["auth_enabled"])

    async def on_cleanup(app: web.Application) -> None:
        # Cancel background tasks
        for task_key in (
            "_scraper_task", "_consumer_task", "_reconciler_task",
        ):
            task = app.get(task_key)
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        await app["upstream_session"].close()

        # Cleanup billing resources
        pipeline = app.get("billing_pipeline")
        if pipeline is not None:
            await pipeline.close()
        billing_pg = app.get("billing_postgres")
        if billing_pg is not None:
            await billing_pg.close()
        billing_redis = app.get("billing_redis")
        if billing_redis is not None:
            await billing_redis.close()

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
