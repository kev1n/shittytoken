"""
Billing / usage manager for the aiohttp LLM inference router.

Receives token-usage callbacks from the SSE proxy, queues events for
batch persistence, deducts user balances, and optionally syncs charges
to Stripe.
"""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

import structlog

if TYPE_CHECKING:
    from shittytoken.gateway.billing_db import BillingDB

log = structlog.get_logger(__name__)

# Default pricing: 100 cents ($1.00) per 1 M tokens for any model not
# explicitly listed in the pricing dict.
_DEFAULT_PRICE_PER_1M = 100.0


@dataclass
class UsageEvent:
    """Single token-usage event produced by the proxy layer."""

    api_key_hash: str
    user_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    cost_cents: int
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "api_key_hash": self.api_key_hash,
            "user_id": self.user_id,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "cost_cents": self.cost_cents,
            "created_at": self.created_at,
        }


class BillingManager:
    """Manages usage recording, balance deduction, and optional Stripe sync."""

    def __init__(
        self,
        db: BillingDB,
        pricing: dict[str, float] | None = None,
        stripe_sync_enabled: bool = False,
    ) -> None:
        self._db = db
        self._pricing = pricing or {}
        self._stripe_sync_enabled = stripe_sync_enabled
        self._queue: asyncio.Queue[UsageEvent] = asyncio.Queue()

        # Balance cache: user_id -> (balance_cents, fetched_at_monotonic)
        self._balance_cache: dict[str, tuple[int, float]] = {}
        self._balance_cache_ttl = 5.0  # seconds

    # ------------------------------------------------------------------
    # Cost computation
    # ------------------------------------------------------------------

    def compute_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> int:
        """Return cost in cents.  Minimum 1 cent per request."""
        total_tokens = prompt_tokens + completion_tokens
        price_per_1m = self._pricing.get(model, _DEFAULT_PRICE_PER_1M)
        cost = math.ceil(total_tokens * price_per_1m / 1_000_000)
        return max(cost, 1)

    # ------------------------------------------------------------------
    # Usage recording
    # ------------------------------------------------------------------

    async def record_usage(
        self,
        user_id: str,
        key_hash: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
    ) -> None:
        """Create a UsageEvent and enqueue it for batch flushing."""
        cost = self.compute_cost(model, prompt_tokens, completion_tokens)
        event = UsageEvent(
            api_key_hash=key_hash,
            user_id=user_id,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cost_cents=cost,
        )
        self._queue.put_nowait(event)
        log.debug(
            "billing.event_queued",
            event_id=event.event_id,
            user_id=user_id,
            model=model,
            total_tokens=event.total_tokens,
            cost_cents=cost,
        )

    # ------------------------------------------------------------------
    # Background flush loop
    # ------------------------------------------------------------------

    async def run_flush_loop(self, interval_sec: float = 1.0) -> None:
        """Drain the event queue periodically and persist in batch."""
        log.info("billing.flush_loop_started", interval_sec=interval_sec)
        while True:
            await asyncio.sleep(interval_sec)
            events: list[UsageEvent] = []
            while not self._queue.empty():
                try:
                    events.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if not events:
                continue
            try:
                await self._db.batch_record_usage(
                    [e.to_dict() for e in events],
                )
                # Invalidate cached balances for affected users.
                for e in events:
                    self._balance_cache.pop(e.user_id, None)
                log.info("billing.flushed", count=len(events))
            except Exception:
                log.exception("billing.flush_error", count=len(events))
                # Re-enqueue so events are not lost.
                for e in events:
                    self._queue.put_nowait(e)

    # ------------------------------------------------------------------
    # Stripe sync loop
    # ------------------------------------------------------------------

    async def run_stripe_sync(self, interval_sec: float = 60.0) -> None:
        """Sync unsynced usage events to Stripe as balance transactions."""
        if not self._stripe_sync_enabled:
            return

        try:
            import stripe  # noqa: F811
        except ImportError:
            log.warning("billing.stripe_not_installed")
            return

        log.info("billing.stripe_sync_started", interval_sec=interval_sec)
        while True:
            await asyncio.sleep(interval_sec)
            try:
                events = await self._db.get_unsynced_events(limit=100)
                if not events:
                    continue

                # Group events by user for efficient Stripe calls.
                by_user: dict[str, list[dict]] = {}
                for ev in events:
                    by_user.setdefault(ev["user_id"], []).append(ev)

                synced_ids: list[str] = []
                for user_id, user_events in by_user.items():
                    total_cents = sum(e["cost_cents"] for e in user_events)
                    # Look up the Stripe customer ID — we store it on the
                    # users table but we only have user_id here; the proxy
                    # passes it through.  For now we use user_id as the
                    # Stripe customer ID (projects typically set user_id =
                    # cus_xxx).
                    try:
                        stripe.Customer.create_balance_transaction(
                            user_id,
                            amount=-total_cents,
                            currency="usd",
                            description=f"shittytoken usage: {len(user_events)} request(s)",
                        )
                        synced_ids.extend(e["event_id"] for e in user_events)
                    except Exception:
                        log.exception(
                            "billing.stripe_sync_user_error",
                            user_id=user_id,
                        )

                if synced_ids:
                    await self._db.mark_synced(synced_ids)
                    log.info("billing.stripe_synced", count=len(synced_ids))
            except Exception:
                log.exception("billing.stripe_sync_error")

    # ------------------------------------------------------------------
    # Balance check (cached)
    # ------------------------------------------------------------------

    async def check_balance(self, user_id: str) -> bool:
        """Return True if the user has a positive balance.

        Uses a simple in-memory cache with a 5-second TTL to avoid
        hitting the database on every request.
        """
        now = time.monotonic()
        cached = self._balance_cache.get(user_id)
        if cached is not None:
            balance, fetched_at = cached
            if now - fetched_at < self._balance_cache_ttl:
                return balance > 0

        balance = await self._db.get_balance(user_id)
        self._balance_cache[user_id] = (balance, now)
        return balance > 0

    # ------------------------------------------------------------------
    # Callback factory for the proxy layer
    # ------------------------------------------------------------------

    def on_usage_callback(
        self,
        user_id: str,
        key_hash: str,
        model: str,
    ) -> Callable:
        """Return an async callback compatible with the proxy's on_usage hook.

        Usage in proxy setup::

            app["on_usage"] = billing_mgr.on_usage_callback(
                user_id, key_hash, model,
            )
        """

        async def callback(prompt_tokens: int, completion_tokens: int) -> None:
            await self.record_usage(
                user_id=user_id,
                key_hash=key_hash,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=0,  # latency injected by caller if available
            )

        return callback
