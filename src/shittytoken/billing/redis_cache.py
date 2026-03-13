from __future__ import annotations

import json
import time
import uuid

import redis.asyncio as aioredis
import structlog

log = structlog.get_logger(__name__)

# Lua script for atomic check-and-deduct.
# Returns {1, new_balance} on success, {0, current_balance} on insufficient funds,
# or {0, 0} if the key does not exist.
_CHECK_AND_DEDUCT_LUA = """
local bal = redis.call('GET', KEYS[1])
if not bal then return {0, 0} end
bal = tonumber(bal)
local cost = tonumber(ARGV[1])
if bal < cost then return {0, bal} end
local new_bal = redis.call('DECRBY', KEYS[1], cost)
return {1, new_bal}
"""


class BillingRedis:
    """Redis hot-path for billing: balance checks, API key cache, rate limits.

    Key layout:
        balance:{user_id}        -> integer cents (total available)
        apikey:{key_hash}        -> JSON {user_id, rate_limit_rpm, rate_limit_tpm, is_active}
        ratelimit:rpm:{key_hash} -> sorted set of request timestamps
        ratelimit:tpm:{key_hash}:{minute_bucket} -> token count for that minute
    """

    def __init__(self, redis: aioredis.Redis) -> None:
        self._redis = redis
        self._check_and_deduct_script = self._redis.register_script(
            _CHECK_AND_DEDUCT_LUA
        )

    @classmethod
    async def create(cls, url: str = "redis://localhost:6379/0") -> BillingRedis:
        """Connect to Redis."""
        r = aioredis.from_url(url, decode_responses=True)
        log.info("billing_redis.connected", url=url)
        return cls(redis=r)

    async def close(self) -> None:
        """Close the Redis connection."""
        await self._redis.aclose()
        log.info("billing_redis.closed")

    # ── Balance (hot path) ─────────────────────────────────────────────

    async def get_balance(self, user_id: str) -> int:
        """GET balance:{user_id}. Returns 0 if not set."""
        val = await self._redis.get(f"balance:{user_id}")
        if val is None:
            return 0
        return int(val)

    async def set_balance(self, user_id: str, balance_cents: int) -> None:
        """SET balance:{user_id} to exact value. Used by reconciliation."""
        await self._redis.set(f"balance:{user_id}", balance_cents)

    async def deduct_balance(self, user_id: str, amount_cents: int) -> int:
        """DECRBY balance:{user_id} amount_cents.

        Returns new balance (may be negative -- caller handles overdraft).
        This is the atomic hot-path operation.
        """
        new_balance = await self._redis.decrby(f"balance:{user_id}", amount_cents)
        return int(new_balance)

    async def credit_balance(self, user_id: str, amount_cents: int) -> int:
        """INCRBY balance:{user_id}. Used when adding credits."""
        new_balance = await self._redis.incrby(f"balance:{user_id}", amount_cents)
        return int(new_balance)

    async def check_and_deduct(
        self, user_id: str, amount_cents: int
    ) -> tuple[bool, int]:
        """Atomic check-and-deduct using a Lua script.

        Returns (allowed, new_balance).
        If balance < cost, does NOT deduct and returns (False, current_balance).
        """
        result = await self._check_and_deduct_script(
            keys=[f"balance:{user_id}"],
            args=[amount_cents],
        )
        allowed = bool(result[0])
        balance = int(result[1])
        return allowed, balance

    # ── API key cache ──────────────────────────────────────────────────

    async def cache_api_key(
        self, key_hash: str, key_data: dict, ttl: int = 300
    ) -> None:
        """SET apikey:{key_hash} with TTL. key_data is JSON-serializable dict."""
        await self._redis.set(
            f"apikey:{key_hash}", json.dumps(key_data), ex=ttl
        )

    async def get_cached_api_key(self, key_hash: str) -> dict | None:
        """GET apikey:{key_hash}. Returns parsed dict or None if expired/missing."""
        val = await self._redis.get(f"apikey:{key_hash}")
        if val is None:
            return None
        return json.loads(val)

    async def invalidate_api_key(self, key_hash: str) -> None:
        """DEL apikey:{key_hash}."""
        await self._redis.delete(f"apikey:{key_hash}")

    # ── Rate limiting ──────────────────────────────────────────────────

    async def check_rate_limit_rpm(self, key_hash: str, limit: int) -> bool:
        """Sliding window RPM check using sorted set.

        Prunes entries older than 60s, then counts remaining entries.
        Returns True if under limit.
        Does NOT record the request -- call record_request() after successful auth.
        """
        redis_key = f"ratelimit:rpm:{key_hash}"
        now = time.time()
        cutoff = now - 60

        pipe = self._redis.pipeline()
        pipe.zremrangebyscore(redis_key, "-inf", cutoff)
        pipe.zcard(redis_key)
        results = await pipe.execute()

        current_count = results[1]
        return current_count < limit

    async def record_request(self, key_hash: str) -> None:
        """Add current timestamp to the RPM sorted set. ZADD + EXPIRE 120s."""
        redis_key = f"ratelimit:rpm:{key_hash}"
        now = time.time()
        # Use timestamp with a random suffix to avoid collisions from
        # concurrent requests arriving in the same microsecond.
        member = f"{now}:{uuid.uuid4().hex[:8]}"

        pipe = self._redis.pipeline()
        pipe.zadd(redis_key, {member: now})
        pipe.expire(redis_key, 120)
        await pipe.execute()

    async def check_rate_limit_tpm(
        self, key_hash: str, limit: int, estimated_tokens: int = 0
    ) -> bool:
        """Check if adding estimated_tokens would exceed TPM limit.

        Uses a simple key with 60s TTL: ratelimit:tpm:{key_hash}:{minute_bucket}.
        Returns True if under limit.
        """
        minute_bucket = int(time.time()) // 60
        redis_key = f"ratelimit:tpm:{key_hash}:{minute_bucket}"

        current = await self._redis.get(redis_key)
        current_tokens = int(current) if current is not None else 0

        return (current_tokens + estimated_tokens) <= limit

    async def record_tokens(self, key_hash: str, tokens: int) -> None:
        """Record actual tokens used. INCRBY ratelimit:tpm:{key_hash}:{minute_bucket}."""
        minute_bucket = int(time.time()) // 60
        redis_key = f"ratelimit:tpm:{key_hash}:{minute_bucket}"

        pipe = self._redis.pipeline()
        pipe.incrby(redis_key, tokens)
        pipe.expire(redis_key, 120)
        await pipe.execute()
