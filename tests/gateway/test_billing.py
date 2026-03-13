"""Tests for the billing subsystem: billing_db, auth middleware, and BillingManager."""

from __future__ import annotations

import hashlib

import pytest
from aiohttp import web

from shittytoken.gateway.auth import auth_middleware, _key_cache, _balance_cache
from shittytoken.gateway.billing import BillingManager
from shittytoken.gateway.billing_db import BillingDB


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
async def db(tmp_path):
    db = BillingDB(str(tmp_path / "test.db"))
    await db.init()
    yield db
    await db.close()


# ======================================================================
# 1. BillingDB tests
# ======================================================================


class TestBillingDB:
    async def test_create_user_and_lookup(self, db: BillingDB):
        await db.create_user("u1", "u1@example.com", stripe_customer_id="cus_123")
        key_hash = hashlib.sha256(b"sk-test-123").hexdigest()
        await db.create_api_key(key_hash, "u1", name="test-key")

        result = await db.lookup_api_key(key_hash)
        assert result is not None
        assert result["key_hash"] == key_hash
        assert result["user_id"] == "u1"
        assert result["name"] == "test-key"
        assert result["rate_limit_rpm"] == 60
        assert result["rate_limit_tpm"] == 100_000
        assert result["is_active"] == 1

    async def test_lookup_nonexistent_key(self, db: BillingDB):
        result = await db.lookup_api_key("nonexistent_hash")
        assert result is None

    async def test_set_and_get_balance(self, db: BillingDB):
        await db.create_user("u1", "u1@example.com")
        await db.set_balance("u1", 500)
        balance = await db.get_balance("u1")
        assert balance == 500

    async def test_deduct_balance(self, db: BillingDB):
        await db.create_user("u1", "u1@example.com")
        await db.set_balance("u1", 1000)
        new_balance = await db.deduct_balance("u1", 300)
        assert new_balance == 700
        assert await db.get_balance("u1") == 700

    async def test_deduct_balance_no_row(self, db: BillingDB):
        with pytest.raises(ValueError, match="No balance row"):
            await db.deduct_balance("unknown_user", 100)

    async def test_record_usage_and_batch(self, db: BillingDB):
        await db.create_user("u1", "u1@example.com")
        await db.set_balance("u1", 1000)

        events = [
            {
                "event_id": "evt-1",
                "api_key_hash": "keyhash1",
                "user_id": "u1",
                "model": "gpt-4",
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "latency_ms": 200,
                "cost_cents": 10,
            },
            {
                "event_id": "evt-2",
                "api_key_hash": "keyhash1",
                "user_id": "u1",
                "model": "gpt-4",
                "prompt_tokens": 200,
                "completion_tokens": 100,
                "total_tokens": 300,
                "latency_ms": 300,
                "cost_cents": 20,
            },
        ]
        await db.batch_record_usage(events)

        # Balance should be deducted by sum of costs (10 + 20 = 30).
        assert await db.get_balance("u1") == 970

        # Events should appear as unsynced.
        unsynced = await db.get_unsynced_events()
        event_ids = {e["event_id"] for e in unsynced}
        assert "evt-1" in event_ids
        assert "evt-2" in event_ids

    async def test_mark_synced(self, db: BillingDB):
        await db.create_user("u1", "u1@example.com")
        await db.set_balance("u1", 1000)

        events = [
            {
                "event_id": "evt-1",
                "api_key_hash": "keyhash1",
                "user_id": "u1",
                "model": "gpt-4",
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "latency_ms": 200,
                "cost_cents": 10,
            },
        ]
        await db.batch_record_usage(events)

        await db.mark_synced(["evt-1"])
        unsynced = await db.get_unsynced_events()
        assert len(unsynced) == 0

    async def test_set_balance_upsert(self, db: BillingDB):
        await db.create_user("u1", "u1@example.com")
        await db.set_balance("u1", 500)
        assert await db.get_balance("u1") == 500

        await db.set_balance("u1", 999)
        assert await db.get_balance("u1") == 999


# ======================================================================
# 2. Auth middleware tests
# ======================================================================


async def _test_handler(request: web.Request) -> web.Response:
    """Simple handler that returns the user_id set by auth middleware."""
    user_id = request.get("user_id", "anonymous")
    return web.Response(text=user_id)


async def _health_handler(request: web.Request) -> web.Response:
    return web.Response(text="ok")


@pytest.fixture
async def app(db):
    app = web.Application(middlewares=[auth_middleware])
    app["billing_db"] = db
    app["auth_enabled"] = True
    app.router.add_post("/v1/chat/completions", _test_handler)
    app.router.add_get("/health", _health_handler)
    return app


@pytest.fixture
async def client(app, aiohttp_client):
    return await aiohttp_client(app)


@pytest.fixture(autouse=True)
def _clear_auth_caches():
    """Clear module-level caches between tests to avoid cross-test pollution."""
    _key_cache._store.clear()
    _balance_cache._store.clear()
    yield
    _key_cache._store.clear()
    _balance_cache._store.clear()


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
            headers={"Authorization": "Bearer unknown-key-value"},
        )
        assert resp.status == 401

    async def test_inactive_key(self, client, db: BillingDB):
        await db.create_user("u1", "u1@example.com")
        key_hash = hashlib.sha256(b"sk-inactive").hexdigest()
        await db.create_api_key(key_hash, "u1", name="inactive-key")
        # Manually deactivate the key.
        await db._db.execute(
            "UPDATE api_keys SET is_active = 0 WHERE key_hash = ?", (key_hash,)
        )
        await db._db.commit()

        resp = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-inactive"},
        )
        assert resp.status == 401

    async def test_valid_key_success(self, client, db: BillingDB):
        await db.create_user("u1", "u1@example.com")
        key_hash = hashlib.sha256(b"sk-valid").hexdigest()
        await db.create_api_key(key_hash, "u1", name="valid-key")
        await db.set_balance("u1", 1000)

        resp = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-valid"},
        )
        assert resp.status == 200
        body = await resp.text()
        assert body == "u1"

    async def test_insufficient_balance(self, client, db: BillingDB):
        await db.create_user("u1", "u1@example.com")
        key_hash = hashlib.sha256(b"sk-broke").hexdigest()
        await db.create_api_key(key_hash, "u1", name="broke-key")
        await db.set_balance("u1", 0)

        resp = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-broke"},
        )
        assert resp.status == 402

    async def test_auth_disabled_skips_all(self, client, app):
        app["auth_enabled"] = False
        resp = await client.post("/v1/chat/completions")
        assert resp.status == 200


# ======================================================================
# 3. BillingManager tests
# ======================================================================


class TestBillingManager:
    def test_compute_cost_default_pricing(self):
        mgr = BillingManager.__new__(BillingManager)
        mgr._pricing = {}
        # Default is 100 cents per 1M tokens.
        # 1000 tokens => ceil(1000 * 100 / 1_000_000) = ceil(0.1) = 1 cent.
        cost = mgr.compute_cost("some-model", prompt_tokens=500, completion_tokens=500)
        assert cost == 1

    def test_compute_cost_custom_pricing(self):
        mgr = BillingManager.__new__(BillingManager)
        mgr._pricing = {"test": 500.0}
        # 10000 tokens at 500 cents/1M => ceil(10000 * 500 / 1_000_000) = ceil(5.0) = 5 cents.
        cost = mgr.compute_cost("test", prompt_tokens=5000, completion_tokens=5000)
        assert cost == 5

    def test_compute_cost_minimum_1_cent(self):
        mgr = BillingManager.__new__(BillingManager)
        mgr._pricing = {}
        # Even 1 token => max(ceil(1 * 100 / 1_000_000), 1) = max(1, 1) = 1 cent.
        cost = mgr.compute_cost("any-model", prompt_tokens=1, completion_tokens=0)
        assert cost == 1

    async def test_record_and_flush(self, db: BillingDB):
        await db.create_user("u1", "u1@example.com")
        await db.set_balance("u1", 1000)

        key_hash = hashlib.sha256(b"sk-flush-test").hexdigest()
        await db.create_api_key(key_hash, "u1")

        mgr = BillingManager(db)
        await mgr.record_usage(
            user_id="u1",
            key_hash=key_hash,
            model="gpt-4",
            prompt_tokens=500,
            completion_tokens=500,
            latency_ms=100,
        )

        # Manually drain the queue and flush (simulating the flush loop).
        events = []
        while not mgr._queue.empty():
            events.append(mgr._queue.get_nowait())
        assert len(events) == 1

        await db.batch_record_usage([e.to_dict() for e in events])

        # Verify event is in DB.
        unsynced = await db.get_unsynced_events()
        assert len(unsynced) == 1
        assert unsynced[0]["user_id"] == "u1"

        # Verify balance was deducted.
        balance = await db.get_balance("u1")
        assert balance < 1000
