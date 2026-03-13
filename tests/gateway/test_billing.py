"""Tests for the billing subsystem: models, pipeline, auth middleware, cost computation."""

from __future__ import annotations

import hashlib
import math

import pytest
from aiohttp import web

from shittytoken.billing.models import UsageEvent, CreditBlock, LedgerEvent
from shittytoken.billing.usage_pipeline import BillingPipeline


# ======================================================================
# 1. Model tests
# ======================================================================


class TestUsageEvent:
    def test_to_dict_roundtrip(self):
        event = UsageEvent(
            event_id="evt-1",
            user_id="u1",
            api_key_hash="abc",
            model="test-model",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_cents=1,
            latency_ms=200,
            request_id="req-1",
            created_at="2026-01-01T00:00:00Z",
        )
        d = event.to_dict()
        assert d["event_id"] == "evt-1"
        assert d["total_tokens"] == 150
        assert d["cost_cents"] == 1

        # Reconstruct from dict
        event2 = UsageEvent(**d)
        assert event2.event_id == event.event_id
        assert event2.total_tokens == event.total_tokens


# ======================================================================
# 2. Cost computation tests
# ======================================================================


class TestCostComputation:
    def test_default_pricing(self):
        # Default is 100 cents per 1M tokens.
        # 1000 tokens => ceil(1000 * 100 / 1_000_000) = ceil(0.1) = 1 cent.
        cost = BillingPipeline.compute_cost("some-model", 500, 500)
        assert cost == 1

    def test_custom_pricing(self):
        pricing = {"test": 500.0}
        # 10000 tokens at 500 cents/1M => ceil(10000 * 500 / 1_000_000) = 5 cents.
        cost = BillingPipeline.compute_cost("test", 5000, 5000, pricing)
        assert cost == 5

    def test_minimum_1_cent(self):
        # Even 1 token => max(ceil(1 * 100 / 1_000_000), 1) = 1 cent.
        cost = BillingPipeline.compute_cost("any-model", 1, 0)
        assert cost == 1

    def test_large_request(self):
        # 1M tokens at $1/1M = 100 cents.
        cost = BillingPipeline.compute_cost("model", 500_000, 500_000)
        assert cost == 100

    def test_zero_tokens_still_1_cent(self):
        # Edge case: 0 tokens should still be minimum 1 cent.
        cost = BillingPipeline.compute_cost("model", 0, 0)
        assert cost == 1


# ======================================================================
# 3. Auth middleware tests (with mocked Redis/Postgres)
# ======================================================================


class FakeBillingRedis:
    """Minimal fake for testing auth middleware without real Redis."""

    def __init__(self):
        self._balances: dict[str, int] = {}
        self._api_keys: dict[str, dict] = {}
        self._rpm_counts: dict[str, int] = {}

    async def get_cached_api_key(self, key_hash: str) -> dict | None:
        return self._api_keys.get(key_hash)

    async def cache_api_key(self, key_hash: str, key_data: dict, ttl: int = 300) -> None:
        self._api_keys[key_hash] = key_data

    async def get_balance(self, user_id: str) -> int:
        return self._balances.get(user_id, 0)

    async def check_rate_limit_rpm(self, key_hash: str, limit: int) -> bool:
        count = self._rpm_counts.get(key_hash, 0)
        return count < limit

    async def check_rate_limit_tpm(self, key_hash: str, limit: int, estimated_tokens: int = 0) -> bool:
        return True

    async def record_request(self, key_hash: str) -> None:
        self._rpm_counts[key_hash] = self._rpm_counts.get(key_hash, 0) + 1


class FakeBillingPostgres:
    """Minimal fake for testing auth middleware Postgres fallback."""

    def __init__(self):
        self._api_keys: dict[str, object] = {}

    async def lookup_api_key(self, key_hash: str):
        return self._api_keys.get(key_hash)


class FakeApiKey:
    def __init__(self, user_id, is_active=True, rpm=60, tpm=100_000):
        self.key_hash = ""
        self.user_id = user_id
        self.name = "test"
        self.rate_limit_rpm = rpm
        self.rate_limit_tpm = tpm
        self.is_active = is_active


async def _test_handler(request: web.Request) -> web.Response:
    user_id = request.get("user_id", "anonymous")
    return web.Response(text=user_id)


async def _health_handler(request: web.Request) -> web.Response:
    return web.Response(text="ok")


@pytest.fixture
def fake_redis():
    return FakeBillingRedis()


@pytest.fixture
def fake_pg():
    return FakeBillingPostgres()


@pytest.fixture
async def app(fake_redis, fake_pg):
    from shittytoken.gateway.auth import auth_middleware

    app = web.Application(middlewares=[auth_middleware])
    app["billing_redis"] = fake_redis
    app["billing_postgres"] = fake_pg
    app["auth_enabled"] = True
    app.router.add_post("/v1/chat/completions", _test_handler)
    app.router.add_get("/health", _health_handler)
    return app


@pytest.fixture
async def client(app, aiohttp_client):
    return await aiohttp_client(app)


class TestAuthMiddleware:
    async def test_health_skips_auth(self, client):
        resp = await client.get("/health")
        assert resp.status == 200

    async def test_missing_auth_header(self, client):
        resp = await client.post("/v1/chat/completions")
        assert resp.status == 401

    async def test_invalid_bearer_format(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Basic xxx"},
        )
        assert resp.status == 401

    async def test_unknown_api_key(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer unknown-key"},
        )
        assert resp.status == 401

    async def test_inactive_key(self, client, fake_redis):
        key_hash = hashlib.sha256(b"sk-inactive").hexdigest()
        fake_redis._api_keys[key_hash] = {
            "user_id": "u1",
            "rate_limit_rpm": 60,
            "rate_limit_tpm": 100_000,
            "is_active": False,
        }
        resp = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-inactive"},
        )
        assert resp.status == 401

    async def test_valid_key_success(self, client, fake_redis):
        key_hash = hashlib.sha256(b"sk-valid").hexdigest()
        fake_redis._api_keys[key_hash] = {
            "user_id": "u1",
            "rate_limit_rpm": 60,
            "rate_limit_tpm": 100_000,
            "is_active": True,
        }
        fake_redis._balances["u1"] = 1000

        resp = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-valid"},
        )
        assert resp.status == 200
        body = await resp.text()
        assert body == "u1"

    async def test_insufficient_balance(self, client, fake_redis):
        key_hash = hashlib.sha256(b"sk-broke").hexdigest()
        fake_redis._api_keys[key_hash] = {
            "user_id": "u1",
            "rate_limit_rpm": 60,
            "rate_limit_tpm": 100_000,
            "is_active": True,
        }
        fake_redis._balances["u1"] = 0

        resp = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-broke"},
        )
        assert resp.status == 402

    async def test_auth_disabled_skips_all(self, client, app):
        app["auth_enabled"] = False
        resp = await client.post("/v1/chat/completions")
        assert resp.status == 200

    async def test_postgres_fallback_on_cache_miss(self, client, fake_redis, fake_pg):
        """When Redis cache misses, auth falls back to Postgres and caches the result."""
        key_hash = hashlib.sha256(b"sk-fallback").hexdigest()
        # Not in Redis, but in Postgres
        fake_pg._api_keys[key_hash] = FakeApiKey(user_id="u2")
        fake_redis._balances["u2"] = 500

        resp = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-fallback"},
        )
        assert resp.status == 200

        # Should now be cached in Redis
        assert key_hash in fake_redis._api_keys
