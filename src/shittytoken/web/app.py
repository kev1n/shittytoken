"""
Web application factory — assembles middleware, templates, routes, and DB connections.
"""
from __future__ import annotations

import base64
from pathlib import Path

import aiohttp_jinja2
import jinja2
import redis.asyncio as aioredis
import structlog
from aiohttp import web
from aiohttp_session import setup as setup_session
from aiohttp_session.redis_storage import RedisStorage

from ..config import Settings
from ..billing.postgres import BillingPostgres
from ..billing.redis_cache import BillingRedis
from .routes import setup_routes
from .stripe_webhook import setup_webhook_routes

logger = structlog.get_logger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"


async def _on_startup(app: web.Application) -> None:
    """Connect to Postgres and Redis on startup."""
    postgres_dsn = app["_postgres_dsn"]
    redis_url = app["_redis_url"]

    app["billing_pg"] = await BillingPostgres.create(postgres_dsn)
    app["billing_redis"] = await BillingRedis.create(redis_url)
    logger.info("web.db_connected")


async def _on_cleanup(app: web.Application) -> None:
    """Close DB connections on shutdown."""
    if "billing_pg" in app:
        await app["billing_pg"].close()
    if "billing_redis" in app:
        await app["billing_redis"].close()
    if "_session_redis" in app:
        await app["_session_redis"].aclose()
    logger.info("web.db_closed")


async def create_web_app(
    postgres_dsn: str, redis_url: str
) -> web.Application:
    """Build and return the aiohttp web application."""
    settings = Settings()
    app = web.Application()

    # Store config for startup hook
    app["_postgres_dsn"] = postgres_dsn
    app["_redis_url"] = redis_url
    app["settings"] = settings

    # Session middleware (Redis-backed)
    session_redis = aioredis.from_url(redis_url, decode_responses=False)
    app["_session_redis"] = session_redis
    secret_key = base64.urlsafe_b64decode(
        base64.urlsafe_b64encode(settings.web_session_secret.encode().ljust(32, b"\0"))[:44]
    )
    setup_session(app, RedisStorage(session_redis, cookie_name="st_session", max_age=86400 * 7))

    # Jinja2 templates
    aiohttp_jinja2.setup(
        app,
        loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
    )

    # Static files
    app.router.add_static("/static", _STATIC_DIR, name="static")

    # Routes
    setup_routes(app)
    setup_webhook_routes(app)

    # Lifecycle hooks
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)

    return app
