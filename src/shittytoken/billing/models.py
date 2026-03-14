from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class User:
    id: str  # UUID as string
    email: str
    stripe_customer_id: str | None = None
    created_at: datetime | None = None


@dataclass
class ApiKey:
    key_hash: str  # SHA-256 of plaintext
    user_id: str
    name: str | None = None
    rate_limit_rpm: int = 1500
    rate_limit_tpm: int = 100_000
    is_active: bool = True
    created_at: datetime | None = None


@dataclass
class CreditBlock:
    """Each top-up creates a discrete credit block. Deduct FIFO (oldest first).
    Handles promotional credits, multiple purchases, and expiration."""

    id: str  # UUID
    user_id: str
    original_cents: int
    remaining_cents: int
    source: str  # 'stripe_checkout', 'promotional', 'manual', 'refund'
    stripe_payment_intent_id: str | None = None
    purchased_at: datetime | None = None
    expires_at: datetime | None = None  # None = never expires
    created_at: datetime | None = None


@dataclass
class LedgerEvent:
    """Append-only ledger entry. The balance is always reconstructable from events.
    This is the system of record."""

    id: str  # UUID
    user_id: str
    event_type: str  # 'credit_purchase', 'usage_deduction', 'expiration', 'adjustment', 'refund'
    amount_cents: int  # positive = credit, negative = debit
    credit_block_id: str | None = None
    request_id: str | None = None  # links to the API request that caused this
    metadata: dict | None = None
    created_at: datetime | None = None


@dataclass
class UsageEvent:
    """Published after each API request completes. Consumed by the billing pipeline."""

    event_id: str
    user_id: str
    api_key_hash: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int  # computed
    cost_cents: float  # fractional cents for sub-cent precision
    latency_ms: int
    request_id: str | None = None
    created_at: str | None = None  # ISO format

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "user_id": self.user_id,
            "api_key_hash": self.api_key_hash,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_cents": self.cost_cents,
            "latency_ms": self.latency_ms,
            "request_id": self.request_id,
            "created_at": self.created_at,
        }
