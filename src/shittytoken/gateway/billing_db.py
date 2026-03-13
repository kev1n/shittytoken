from __future__ import annotations

import aiosqlite
import structlog

log = structlog.get_logger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    email TEXT NOT NULL,
    stripe_customer_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS api_keys (
    key_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    name TEXT,
    rate_limit_rpm INTEGER NOT NULL DEFAULT 60,
    rate_limit_tpm INTEGER NOT NULL DEFAULT 100000,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS balances (
    user_id TEXT PRIMARY KEY REFERENCES users(user_id),
    balance_cents INTEGER NOT NULL DEFAULT 0,
    last_updated TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS usage_events (
    event_id TEXT PRIMARY KEY,
    api_key_hash TEXT NOT NULL,
    user_id TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    latency_ms INTEGER,
    cost_cents INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    stripe_synced INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_events_user_created ON usage_events(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_usage_events_stripe_synced ON usage_events(stripe_synced);
"""


class BillingDB:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA_SQL)
        await self._db.commit()
        log.info("billing_db.initialized", db_path=self.db_path)

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None
            log.info("billing_db.closed")

    # ------------------------------------------------------------------
    # API key operations
    # ------------------------------------------------------------------

    async def lookup_api_key(self, key_hash: str) -> dict | None:
        cursor = await self._db.execute(
            "SELECT key_hash, user_id, name, rate_limit_rpm, rate_limit_tpm, is_active "
            "FROM api_keys WHERE key_hash = ?",
            (key_hash,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "key_hash": row["key_hash"],
            "user_id": row["user_id"],
            "name": row["name"],
            "rate_limit_rpm": row["rate_limit_rpm"],
            "rate_limit_tpm": row["rate_limit_tpm"],
            "is_active": row["is_active"],
        }

    async def create_api_key(
        self,
        key_hash: str,
        user_id: str,
        name: str | None = None,
    ) -> None:
        await self._db.execute(
            "INSERT INTO api_keys (key_hash, user_id, name) VALUES (?, ?, ?)",
            (key_hash, user_id, name),
        )
        await self._db.commit()
        log.info("billing_db.api_key_created", user_id=user_id)

    # ------------------------------------------------------------------
    # User operations
    # ------------------------------------------------------------------

    async def create_user(
        self,
        user_id: str,
        email: str,
        stripe_customer_id: str | None = None,
    ) -> None:
        await self._db.execute(
            "INSERT INTO users (user_id, email, stripe_customer_id) VALUES (?, ?, ?)",
            (user_id, email, stripe_customer_id),
        )
        await self._db.commit()
        log.info("billing_db.user_created", user_id=user_id)

    # ------------------------------------------------------------------
    # Balance operations
    # ------------------------------------------------------------------

    async def get_balance(self, user_id: str) -> int:
        cursor = await self._db.execute(
            "SELECT balance_cents FROM balances WHERE user_id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return 0
        return row["balance_cents"]

    async def set_balance(self, user_id: str, balance_cents: int) -> None:
        await self._db.execute(
            "INSERT INTO balances (user_id, balance_cents, last_updated) "
            "VALUES (?, ?, datetime('now')) "
            "ON CONFLICT(user_id) DO UPDATE SET balance_cents = excluded.balance_cents, "
            "last_updated = excluded.last_updated",
            (user_id, balance_cents),
        )
        await self._db.commit()
        log.info("billing_db.balance_set", user_id=user_id, balance_cents=balance_cents)

    async def deduct_balance(self, user_id: str, amount_cents: int) -> int:
        await self._db.execute("BEGIN IMMEDIATE")
        try:
            cursor = await self._db.execute(
                "UPDATE balances SET balance_cents = balance_cents - ?, last_updated = datetime('now') "
                "WHERE user_id = ? RETURNING balance_cents",
                (amount_cents, user_id),
            )
            row = await cursor.fetchone()
            if row is None:
                await self._db.execute("ROLLBACK")
                raise ValueError(f"No balance row for user {user_id}")
            new_balance = row["balance_cents"]
            await self._db.execute("COMMIT")
        except Exception:
            try:
                await self._db.execute("ROLLBACK")
            except Exception:
                pass
            raise
        log.info(
            "billing_db.balance_deducted",
            user_id=user_id,
            amount_cents=amount_cents,
            new_balance=new_balance,
        )
        return new_balance

    # ------------------------------------------------------------------
    # Usage event operations
    # ------------------------------------------------------------------

    async def record_usage(self, event: dict) -> None:
        await self._db.execute(
            "INSERT INTO usage_events "
            "(event_id, api_key_hash, user_id, model, prompt_tokens, completion_tokens, "
            "total_tokens, latency_ms, cost_cents, stripe_synced) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event["event_id"],
                event["api_key_hash"],
                event["user_id"],
                event["model"],
                event.get("prompt_tokens"),
                event.get("completion_tokens"),
                event.get("total_tokens"),
                event.get("latency_ms"),
                event["cost_cents"],
                event.get("stripe_synced", 0),
            ),
        )
        await self._db.commit()

    async def batch_record_usage(self, events: list[dict]) -> None:
        await self._db.execute("BEGIN IMMEDIATE")
        try:
            for event in events:
                await self._db.execute(
                    "INSERT INTO usage_events "
                    "(event_id, api_key_hash, user_id, model, prompt_tokens, completion_tokens, "
                    "total_tokens, latency_ms, cost_cents, stripe_synced) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        event["event_id"],
                        event["api_key_hash"],
                        event["user_id"],
                        event["model"],
                        event.get("prompt_tokens"),
                        event.get("completion_tokens"),
                        event.get("total_tokens"),
                        event.get("latency_ms"),
                        event["cost_cents"],
                        event.get("stripe_synced", 0),
                    ),
                )
                await self._db.execute(
                    "UPDATE balances SET balance_cents = balance_cents - ?, "
                    "last_updated = datetime('now') WHERE user_id = ?",
                    (event["cost_cents"], event["user_id"]),
                )
            await self._db.execute("COMMIT")
            log.info("billing_db.batch_recorded", count=len(events))
        except Exception:
            try:
                await self._db.execute("ROLLBACK")
            except Exception:
                pass
            raise

    async def get_unsynced_events(self, limit: int = 100) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT event_id, api_key_hash, user_id, model, prompt_tokens, "
            "completion_tokens, total_tokens, latency_ms, cost_cents, created_at, stripe_synced "
            "FROM usage_events WHERE stripe_synced = 0 ORDER BY created_at LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "event_id": r["event_id"],
                "api_key_hash": r["api_key_hash"],
                "user_id": r["user_id"],
                "model": r["model"],
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "total_tokens": r["total_tokens"],
                "latency_ms": r["latency_ms"],
                "cost_cents": r["cost_cents"],
                "created_at": r["created_at"],
                "stripe_synced": r["stripe_synced"],
            }
            for r in rows
        ]

    async def mark_synced(self, event_ids: list[str]) -> None:
        if not event_ids:
            return
        placeholders = ",".join("?" for _ in event_ids)
        await self._db.execute(
            f"UPDATE usage_events SET stripe_synced = 1 WHERE event_id IN ({placeholders})",
            event_ids,
        )
        await self._db.commit()
        log.info("billing_db.marked_synced", count=len(event_ids))
