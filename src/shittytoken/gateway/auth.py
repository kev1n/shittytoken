from __future__ import annotations

import hashlib
import time
from collections import deque
from typing import TYPE_CHECKING

import structlog
from aiohttp import web

if TYPE_CHECKING:
    from shittytoken.gateway.billing_db import BillingDB

logger = structlog.get_logger()

# Paths that bypass authentication entirely.
_PUBLIC_PATHS = frozenset({"/health", "/metrics"})


class ApiKeyCache:
    """In-memory LRU-ish cache for API key lookups (TTL-based, not size-based)."""

    def __init__(self, ttl: float = 30.0) -> None:
        self._ttl = ttl
        # key_hash -> (result_dict, fetched_at)
        self._store: dict[str, tuple[dict, float]] = {}

    async def get(self, key_hash: str, db: BillingDB) -> dict | None:
        """Return cached key record if fresh, otherwise re-fetch from *db*."""
        entry = self._store.get(key_hash)
        now = time.monotonic()
        if entry is not None:
            result, fetched_at = entry
            if now - fetched_at < self._ttl:
                return result

        result = await db.lookup_api_key(key_hash)
        if result is not None:
            self._store[key_hash] = (result, now)
        else:
            # Negative results are not cached — a missing key might be added soon.
            self._store.pop(key_hash, None)
        return result

    def invalidate(self, key_hash: str) -> None:
        """Remove *key_hash* from the cache."""
        self._store.pop(key_hash, None)


class _BalanceCache:
    """Thin TTL cache for user balance lookups (5 s TTL)."""

    def __init__(self, ttl: float = 5.0) -> None:
        self._ttl = ttl
        # user_id -> (balance, fetched_at)
        self._store: dict[str, tuple[float, float]] = {}

    async def get(self, user_id: str, db: BillingDB) -> float:
        entry = self._store.get(user_id)
        now = time.monotonic()
        if entry is not None:
            balance, fetched_at = entry
            if now - fetched_at < self._ttl:
                return balance

        balance = await db.get_balance(user_id)
        self._store[user_id] = (balance, now)
        return balance


class SlidingWindowCounter:
    """Per-key sliding-window rate limiter for RPM and TPM.

    RPM tracks request timestamps in a deque, pruning entries older than 60 s.
    TPM tracks ``(timestamp, token_count)`` tuples with the same pruning window.
    """

    def __init__(self) -> None:
        # key_hash -> deque of timestamps
        self._rpm: dict[str, deque[float]] = {}
        # key_hash -> deque of (timestamp, token_count)
        self._tpm: dict[str, deque[tuple[float, int]]] = {}

    # ------------------------------------------------------------------
    # RPM helpers
    # ------------------------------------------------------------------

    def _prune_rpm(self, key_hash: str) -> deque[float]:
        dq = self._rpm.setdefault(key_hash, deque())
        cutoff = time.monotonic() - 60.0
        while dq and dq[0] < cutoff:
            dq.popleft()
        return dq

    def check_rpm(self, key_hash: str, limit: int) -> bool:
        """Return ``True`` if the request is allowed under the RPM *limit*."""
        dq = self._prune_rpm(key_hash)
        return len(dq) < limit

    def record_rpm(self, key_hash: str) -> None:
        """Record a single request for *key_hash*."""
        dq = self._rpm.setdefault(key_hash, deque())
        dq.append(time.monotonic())

    # ------------------------------------------------------------------
    # TPM helpers
    # ------------------------------------------------------------------

    def _prune_tpm(self, key_hash: str) -> deque[tuple[float, int]]:
        dq = self._tpm.setdefault(key_hash, deque())
        cutoff = time.monotonic() - 60.0
        while dq and dq[0][0] < cutoff:
            dq.popleft()
        return dq

    def check_tpm(self, key_hash: str, limit: int, tokens: int) -> bool:
        """Return ``True`` if consuming *tokens* is allowed under the TPM *limit*."""
        dq = self._prune_tpm(key_hash)
        current = sum(t for _, t in dq)
        return current + tokens <= limit

    def record_tpm(self, key_hash: str, tokens: int) -> None:
        """Record *tokens* consumed by *key_hash*."""
        dq = self._tpm.setdefault(key_hash, deque())
        dq.append((time.monotonic(), tokens))


# Module-level singletons — shared across the lifetime of the process.
_key_cache = ApiKeyCache(ttl=30.0)
_balance_cache = _BalanceCache(ttl=5.0)
rate_limiter = SlidingWindowCounter()


@web.middleware
async def auth_middleware(
    request: web.Request, handler
) -> web.StreamResponse:
    """Authenticate API keys, enforce rate limits, and check billing balance.

    On success, sets ``request["user_id"]`` and ``request["key_hash"]`` for
    downstream handlers.
    """
    # ------------------------------------------------------------------ #
    # Skip auth for public / admin paths
    # ------------------------------------------------------------------ #
    if request.path in _PUBLIC_PATHS or request.path.startswith("/admin"):
        return await handler(request)

    app = request.app

    if not app.get("auth_enabled"):
        return await handler(request)

    db: BillingDB = app["billing_db"]

    # ------------------------------------------------------------------ #
    # Extract and validate bearer token
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Key lookup (cached)
    # ------------------------------------------------------------------ #
    key_record = await _key_cache.get(key_hash, db)

    if key_record is None:
        logger.warning("auth_key_not_found", key_hash=key_hash[:12])
        raise web.HTTPUnauthorized(text="Invalid API key")

    if not key_record.get("is_active", False):
        logger.warning("auth_key_inactive", key_hash=key_hash[:12])
        raise web.HTTPUnauthorized(text="API key is inactive")

    user_id: str = key_record["user_id"]

    # ------------------------------------------------------------------ #
    # Rate-limit checks (RPM / TPM)
    # ------------------------------------------------------------------ #
    rpm_limit = key_record.get("rate_limit_rpm", 60)
    if not rate_limiter.check_rpm(key_hash, rpm_limit):
        logger.warning("rate_limit_rpm", key_hash=key_hash[:12], limit=rpm_limit)
        raise web.HTTPTooManyRequests(
            text="Rate limit exceeded (RPM)",
            headers={"Retry-After": "5"},
        )

    tpm_limit = key_record.get("rate_limit_tpm", 100_000)
    # Estimate input tokens from content-length as a rough pre-check.
    estimated_tokens = max(1, request.content_length or 0) // 4
    if not rate_limiter.check_tpm(key_hash, tpm_limit, estimated_tokens):
        logger.warning("rate_limit_tpm", key_hash=key_hash[:12], limit=tpm_limit)
        raise web.HTTPTooManyRequests(
            text="Rate limit exceeded (TPM)",
            headers={"Retry-After": "10"},
        )

    # ------------------------------------------------------------------ #
    # Balance check (cached, 5 s TTL)
    # ------------------------------------------------------------------ #
    balance = await _balance_cache.get(user_id, db)
    if balance <= 0:
        logger.warning("auth_insufficient_balance", user_id=user_id)
        raise web.HTTPPaymentRequired(text="Insufficient balance")

    # ------------------------------------------------------------------ #
    # Attach identity to request for downstream handlers
    # ------------------------------------------------------------------ #
    request["user_id"] = user_id
    request["key_hash"] = key_hash

    # Record the request in the rate-limiter window.
    rate_limiter.record_rpm(key_hash)

    return await handler(request)
