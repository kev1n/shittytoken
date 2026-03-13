"""
Reconciler — periodic Redis <-> PostgreSQL balance sync.

Catches drift from crashes, Redis evictions, or network issues.
Runs every N seconds (configurable, default 60s).

Also handles credit block expiration.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from shittytoken.billing.postgres import BillingPostgres
    from shittytoken.billing.redis_cache import BillingRedis

log = structlog.get_logger(__name__)


class Reconciler:
    """Periodically reconciles Redis balance cache with PostgreSQL source of truth."""

    def __init__(
        self,
        postgres: BillingPostgres,
        redis: BillingRedis,
        interval_sec: float = 60.0,
    ) -> None:
        self._postgres = postgres
        self._redis = redis
        self._interval = interval_sec

    async def run(self) -> None:
        """Run forever, reconciling on interval."""
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self.reconcile_all()
            except Exception:
                log.exception("reconciler.run_error")

    async def reconcile_all(self) -> None:
        """
        1. Expire any credit blocks past their expires_at
        2. For each user with active credit blocks:
           - Compute balance from Postgres (sum of remaining_cents)
           - Compare with Redis balance
           - If different, update Redis to match Postgres
        3. Log any drift found
        """
        # 1. Expire blocks
        await self.expire_blocks()

        # 2. Get all users with active credit blocks and reconcile
        active_users = await self._postgres.get_users_with_active_blocks()
        drift_count = 0
        total_drift = 0

        for user_id in active_users:
            drift = await self.reconcile_user(user_id)
            if drift is not None:
                drift_count += 1
                total_drift += abs(drift)

        if drift_count > 0:
            log.warning(
                "reconciler.drift_detected",
                users_with_drift=drift_count,
                total_abs_drift_cents=total_drift,
            )
        else:
            log.debug("reconciler.no_drift")

    async def reconcile_user(self, user_id: str) -> int | None:
        """Reconcile a single user. Returns drift amount (cents) or None if no drift."""
        pg_balance = await self._postgres.get_balance(user_id)
        redis_balance = await self._redis.get_balance(user_id)
        if pg_balance != redis_balance:
            log.warning(
                "reconciler.drift",
                user_id=user_id,
                pg_balance=pg_balance,
                redis_balance=redis_balance,
                drift=redis_balance - pg_balance,
            )
            await self._redis.set_balance(user_id, pg_balance)
            return redis_balance - pg_balance
        return None

    async def expire_blocks(self) -> int:
        """Delegate to postgres.expire_blocks(). Update Redis for affected users."""
        count = await self._postgres.expire_blocks()
        if count > 0:
            log.info("reconciler.blocks_expired", count=count)
        return count
