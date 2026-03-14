"""
Authentication and rate-limiting middleware for the custom router.

Uses Redis for sub-millisecond hot-path operations:
- API key lookup (cached in Redis with 5min TTL, Postgres is source of truth)
- Rate limiting (RPM/TPM via sorted sets and counters)
- Balance check (Redis atomic read, reconciled periodically with Postgres)

Falls back to Postgres if Redis cache misses.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import structlog
from aiohttp import web

if TYPE_CHECKING:
    from shittytoken.billing.postgres import BillingPostgres
    from shittytoken.billing.redis_cache import BillingRedis

logger = structlog.get_logger()

# Paths that bypass authentication entirely.
_PUBLIC_PATHS = frozenset({"/health", "/metrics"})


@web.middleware
async def auth_middleware(
    request: web.Request, handler
) -> web.StreamResponse:
    """Authenticate API keys, enforce rate limits, and check billing balance.

    On success, sets ``request["user_id"]`` and ``request["key_hash"]`` for
    downstream handlers.
    """
    # Skip auth for public paths
    if request.path in _PUBLIC_PATHS:
        return await handler(request)

    # Admin paths require a valid admin token (checked per-handler via
    # _check_admin_token in admin_api.py), not a Bearer API key.
    if request.path.startswith("/admin"):
        return await handler(request)

    app = request.app
    if not app.get("auth_enabled"):
        return await handler(request)

    billing_redis: BillingRedis = app["billing_redis"]
    billing_pg: BillingPostgres = app["billing_postgres"]

    # ── Extract Bearer token ─────────────────────────────────────────
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise web.HTTPUnauthorized(
            text="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = auth_header.split(" ", 1)
    if len(parts) != 2 or parts[0] != "Bearer":
        raise web.HTTPUnauthorized(
            text="Authorization header must use Bearer scheme",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = parts[1]
    key_hash = hashlib.sha256(token.encode()).hexdigest()

    # ── API key lookup (Redis cache → Postgres fallback) ─────────────
    key_data = await billing_redis.get_cached_api_key(key_hash)
    if key_data is None:
        # Cache miss — look up in Postgres and cache
        api_key = await billing_pg.lookup_api_key(key_hash)
        if api_key is None:
            logger.warning("auth.key_not_found", key_hash=key_hash[:12])
            raise web.HTTPUnauthorized(text="Invalid API key")
        key_data = {
            "user_id": api_key.user_id,
            "rate_limit_rpm": api_key.rate_limit_rpm,
            "rate_limit_tpm": api_key.rate_limit_tpm,
            "is_active": api_key.is_active,
        }
        await billing_redis.cache_api_key(key_hash, key_data, ttl=300)

    if not key_data.get("is_active", False):
        logger.warning("auth.key_inactive", key_hash=key_hash[:12])
        raise web.HTTPUnauthorized(text="API key is inactive")

    user_id: str = key_data["user_id"]

    # ── Rate-limit check (Redis) ─────────────────────────────────────
    rpm_limit = key_data.get("rate_limit_rpm", 1500)
    if not await billing_redis.check_rate_limit_rpm(key_hash, rpm_limit):
        logger.warning("auth.rate_limit_rpm", key_hash=key_hash[:12], limit=rpm_limit)
        raise web.HTTPTooManyRequests(
            text="Rate limit exceeded (RPM)",
            headers={"Retry-After": "5"},
        )

    # ── Balance check (Redis atomic read) ────────────────────────────
    balance = await billing_redis.get_balance(user_id)
    if balance <= 0:
        logger.warning("auth.insufficient_balance", user_id=user_id, balance=balance)
        raise web.HTTPPaymentRequired(text="Insufficient balance")

    # ── Attach identity to request ───────────────────────────────────
    request["user_id"] = user_id
    request["key_hash"] = key_hash

    # Record the request in rate limiter
    await billing_redis.record_request(key_hash)

    return await handler(request)
