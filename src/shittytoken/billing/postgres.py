from __future__ import annotations

import json
import uuid
from datetime import datetime

import asyncpg
import structlog

from shittytoken.billing.models import (
    ApiKey,
    CreditBlock,
    LedgerEvent,
    UsageEvent,
    User,
)

logger = structlog.get_logger(__name__)

_SCHEMA = """\
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT NOT NULL UNIQUE,
    stripe_customer_id TEXT UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS api_keys (
    key_hash TEXT PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id),
    name TEXT,
    rate_limit_rpm INTEGER NOT NULL DEFAULT 60,
    rate_limit_tpm INTEGER NOT NULL DEFAULT 100000,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);

CREATE TABLE IF NOT EXISTS credit_blocks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    original_cents INTEGER NOT NULL,
    remaining_cents INTEGER NOT NULL CHECK (remaining_cents >= 0),
    source TEXT NOT NULL,
    stripe_payment_intent_id TEXT,
    purchased_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_credit_blocks_user_fifo ON credit_blocks(user_id, purchased_at);
CREATE INDEX IF NOT EXISTS idx_credit_blocks_expiry ON credit_blocks(expires_at) WHERE expires_at IS NOT NULL;

CREATE TABLE IF NOT EXISTS ledger_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    event_type TEXT NOT NULL,
    amount_cents INTEGER NOT NULL,
    credit_block_id UUID REFERENCES credit_blocks(id),
    request_id TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_ledger_user_created ON ledger_events(user_id, created_at);

CREATE TABLE IF NOT EXISTS usage_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id TEXT NOT NULL UNIQUE,
    user_id UUID NOT NULL,
    api_key_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    cost_cents INTEGER NOT NULL,
    latency_ms INTEGER,
    request_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_usage_events_user ON usage_events(user_id, created_at);
"""


def _user_from_row(row: asyncpg.Record) -> User:
    return User(
        id=str(row["id"]),
        email=row["email"],
        stripe_customer_id=row["stripe_customer_id"],
        created_at=row["created_at"],
    )


def _api_key_from_row(row: asyncpg.Record) -> ApiKey:
    return ApiKey(
        key_hash=row["key_hash"],
        user_id=str(row["user_id"]),
        name=row["name"],
        rate_limit_rpm=row["rate_limit_rpm"],
        rate_limit_tpm=row["rate_limit_tpm"],
        is_active=row["is_active"],
        created_at=row["created_at"],
    )


def _credit_block_from_row(row: asyncpg.Record) -> CreditBlock:
    return CreditBlock(
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        original_cents=row["original_cents"],
        remaining_cents=row["remaining_cents"],
        source=row["source"],
        stripe_payment_intent_id=row["stripe_payment_intent_id"],
        purchased_at=row["purchased_at"],
        expires_at=row["expires_at"],
        created_at=row["created_at"],
    )


def _ledger_event_from_row(row: asyncpg.Record) -> LedgerEvent:
    return LedgerEvent(
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        event_type=row["event_type"],
        amount_cents=row["amount_cents"],
        credit_block_id=str(row["credit_block_id"]) if row["credit_block_id"] else None,
        request_id=row["request_id"],
        metadata=json.loads(row["metadata"]) if row["metadata"] else None,
        created_at=row["created_at"],
    )


class BillingPostgres:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @classmethod
    async def create(cls, dsn: str) -> BillingPostgres:
        """Create pool and init schema."""
        pool = await asyncpg.create_pool(dsn)
        instance = cls(pool)
        await instance._init_schema()
        logger.info("billing_postgres_ready", dsn=dsn)
        return instance

    async def _init_schema(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(_SCHEMA)

    async def close(self) -> None:
        await self._pool.close()

    # ── User ops ──────────────────────────────────────────────────────

    async def create_user(
        self, email: str, stripe_customer_id: str | None = None
    ) -> User:
        row = await self._pool.fetchrow(
            "INSERT INTO users (email, stripe_customer_id) "
            "VALUES ($1, $2) RETURNING *",
            email,
            stripe_customer_id,
        )
        return _user_from_row(row)

    async def get_user_by_email(self, email: str) -> User | None:
        row = await self._pool.fetchrow(
            "SELECT * FROM users WHERE email = $1", email
        )
        return _user_from_row(row) if row else None

    async def get_user(self, user_id: str) -> User | None:
        row = await self._pool.fetchrow(
            "SELECT * FROM users WHERE id = $1", uuid.UUID(user_id)
        )
        return _user_from_row(row) if row else None

    # ── API key ops ───────────────────────────────────────────────────

    async def create_api_key(
        self, key_hash: str, user_id: str, name: str | None = None
    ) -> ApiKey:
        row = await self._pool.fetchrow(
            "INSERT INTO api_keys (key_hash, user_id, name) "
            "VALUES ($1, $2, $3) RETURNING *",
            key_hash,
            uuid.UUID(user_id),
            name,
        )
        return _api_key_from_row(row)

    async def lookup_api_key(self, key_hash: str) -> ApiKey | None:
        row = await self._pool.fetchrow(
            "SELECT * FROM api_keys WHERE key_hash = $1 AND is_active = true",
            key_hash,
        )
        return _api_key_from_row(row) if row else None

    async def deactivate_api_key(self, key_hash: str) -> None:
        await self._pool.execute(
            "UPDATE api_keys SET is_active = false WHERE key_hash = $1",
            key_hash,
        )

    # ── Credit block ops ──────────────────────────────────────────────

    async def create_credit_block(
        self,
        user_id: str,
        amount_cents: int,
        source: str,
        stripe_payment_intent_id: str | None = None,
        expires_at: datetime | None = None,
    ) -> CreditBlock:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "INSERT INTO credit_blocks "
                    "(user_id, original_cents, remaining_cents, source, "
                    "stripe_payment_intent_id, expires_at) "
                    "VALUES ($1, $2, $3, $4, $5, $6) RETURNING *",
                    uuid.UUID(user_id),
                    amount_cents,
                    amount_cents,
                    source,
                    stripe_payment_intent_id,
                    expires_at,
                )
                block = _credit_block_from_row(row)
                await conn.execute(
                    "INSERT INTO ledger_events "
                    "(user_id, event_type, amount_cents, credit_block_id) "
                    "VALUES ($1, 'credit_purchase', $2, $3)",
                    uuid.UUID(user_id),
                    amount_cents,
                    row["id"],
                )
                return block

    async def get_active_blocks(self, user_id: str) -> list[CreditBlock]:
        rows = await self._pool.fetch(
            "SELECT * FROM credit_blocks "
            "WHERE user_id = $1 AND remaining_cents > 0 "
            "AND (expires_at IS NULL OR expires_at > now()) "
            "ORDER BY purchased_at ASC",
            uuid.UUID(user_id),
        )
        return [_credit_block_from_row(r) for r in rows]

    async def deduct_credits_fifo(
        self,
        user_id: str,
        amount_cents: int,
        request_id: str | None = None,
    ) -> int:
        """Deduct amount from credit blocks in FIFO order.

        Writes ledger events for each block touched.
        Returns total actually deducted (may be < amount if insufficient).
        Uses a transaction with row-level locks (SELECT ... FOR UPDATE).
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                blocks = await conn.fetch(
                    "SELECT id, remaining_cents FROM credit_blocks "
                    "WHERE user_id = $1 AND remaining_cents > 0 "
                    "AND (expires_at IS NULL OR expires_at > now()) "
                    "ORDER BY purchased_at ASC FOR UPDATE",
                    uuid.UUID(user_id),
                )
                remaining = amount_cents
                for block in blocks:
                    if remaining <= 0:
                        break
                    deduct = min(remaining, block["remaining_cents"])
                    await conn.execute(
                        "UPDATE credit_blocks SET remaining_cents = remaining_cents - $1 "
                        "WHERE id = $2",
                        deduct,
                        block["id"],
                    )
                    await conn.execute(
                        "INSERT INTO ledger_events "
                        "(user_id, event_type, amount_cents, credit_block_id, request_id) "
                        "VALUES ($1, 'usage_deduction', $2, $3, $4)",
                        uuid.UUID(user_id),
                        -deduct,
                        block["id"],
                        request_id,
                    )
                    remaining -= deduct
                return amount_cents - remaining

    async def get_balance(self, user_id: str) -> int:
        """Sum remaining_cents across active (unexpired, remaining > 0) blocks."""
        row = await self._pool.fetchrow(
            "SELECT COALESCE(SUM(remaining_cents), 0) AS balance "
            "FROM credit_blocks "
            "WHERE user_id = $1 AND remaining_cents > 0 "
            "AND (expires_at IS NULL OR expires_at > now())",
            uuid.UUID(user_id),
        )
        return int(row["balance"])

    async def expire_blocks(self) -> int:
        """Set remaining_cents = 0 for expired blocks. Write ledger events. Returns count expired."""
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                rows = await conn.fetch(
                    "SELECT id, user_id, remaining_cents FROM credit_blocks "
                    "WHERE expires_at IS NOT NULL AND expires_at <= now() "
                    "AND remaining_cents > 0 "
                    "FOR UPDATE"
                )
                for row in rows:
                    await conn.execute(
                        "UPDATE credit_blocks SET remaining_cents = 0 WHERE id = $1",
                        row["id"],
                    )
                    await conn.execute(
                        "INSERT INTO ledger_events "
                        "(user_id, event_type, amount_cents, credit_block_id) "
                        "VALUES ($1, 'expiration', $2, $3)",
                        row["user_id"],
                        -row["remaining_cents"],
                        row["id"],
                    )
                return len(rows)

    async def get_users_with_active_blocks(self) -> list[str]:
        """Return user IDs that have at least one active credit block."""
        rows = await self._pool.fetch(
            "SELECT DISTINCT user_id FROM credit_blocks "
            "WHERE remaining_cents > 0 "
            "AND (expires_at IS NULL OR expires_at > now())"
        )
        return [str(r["user_id"]) for r in rows]

    # ── Ledger ────────────────────────────────────────────────────────

    async def get_ledger(
        self, user_id: str, limit: int = 100, offset: int = 0
    ) -> list[LedgerEvent]:
        rows = await self._pool.fetch(
            "SELECT * FROM ledger_events "
            "WHERE user_id = $1 ORDER BY created_at DESC "
            "LIMIT $2 OFFSET $3",
            uuid.UUID(user_id),
            limit,
            offset,
        )
        return [_ledger_event_from_row(r) for r in rows]

    async def reconstruct_balance(self, user_id: str) -> int:
        """Sum all ledger events for user. Should match get_balance()."""
        row = await self._pool.fetchrow(
            "SELECT COALESCE(SUM(amount_cents), 0) AS balance "
            "FROM ledger_events WHERE user_id = $1",
            uuid.UUID(user_id),
        )
        return int(row["balance"])

    # ── Usage events ──────────────────────────────────────────────────

    async def record_usage_event(self, event: UsageEvent) -> None:
        await self._pool.execute(
            "INSERT INTO usage_events "
            "(event_id, user_id, api_key_hash, model, prompt_tokens, "
            "completion_tokens, total_tokens, cost_cents, latency_ms, request_id) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) "
            "ON CONFLICT (event_id) DO NOTHING",
            event.event_id,
            uuid.UUID(event.user_id),
            event.api_key_hash,
            event.model,
            event.prompt_tokens,
            event.completion_tokens,
            event.total_tokens,
            event.cost_cents,
            event.latency_ms,
            event.request_id,
        )

    async def batch_record_usage(self, events: list[UsageEvent]) -> None:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                for event in events:
                    await conn.execute(
                        "INSERT INTO usage_events "
                        "(event_id, user_id, api_key_hash, model, prompt_tokens, "
                        "completion_tokens, total_tokens, cost_cents, latency_ms, request_id) "
                        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) "
                        "ON CONFLICT (event_id) DO NOTHING",
                        event.event_id,
                        uuid.UUID(event.user_id),
                        event.api_key_hash,
                        event.model,
                        event.prompt_tokens,
                        event.completion_tokens,
                        event.total_tokens,
                        event.cost_cents,
                        event.latency_ms,
                        event.request_id,
                    )
