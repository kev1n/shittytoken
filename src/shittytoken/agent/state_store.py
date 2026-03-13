"""
RedisStateStore — persists InstanceRecord state to Redis so the orchestrator
can recover running instances after a restart.

Key layout:
    shittytoken:instances:{instance_id} -> hash of InstanceRecord fields

This prevents orphaned cloud instances when the orchestrator process restarts.
"""

from __future__ import annotations

import redis.asyncio as aioredis
import structlog

from .state_machine import InstanceRecord, InstanceState

logger = structlog.get_logger()

# Fields that are stored as floats in the hash.
_FLOAT_FIELDS = {"created_at", "state_changed_at", "cost_per_hour_usd"}
# Fields that are stored as ints in the hash.
_INT_FIELDS = {"ssh_port", "http_port"}

_KEY_PREFIX = "shittytoken:instances:"


class RedisStateStore:
    """Persists InstanceRecord objects as Redis hashes for crash recovery."""

    def __init__(self, redis: aioredis.Redis) -> None:
        self._redis = redis

    @classmethod
    async def create(cls, url: str = "redis://localhost:6379/0") -> RedisStateStore:
        """Connect to Redis and return a new store instance."""
        r = aioredis.from_url(url, decode_responses=True)
        await r.ping()  # fail fast if Redis is unreachable
        logger.info("state_store.connected", url=url)
        return cls(redis=r)

    async def close(self) -> None:
        """Close the Redis connection."""
        await self._redis.aclose()
        logger.info("state_store.closed")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def save(self, record: InstanceRecord) -> None:
        """Persist an InstanceRecord as a Redis hash (HSET all fields)."""
        key = f"{_KEY_PREFIX}{record.instance_id}"
        mapping = {
            "instance_id": record.instance_id,
            "provider": record.provider,
            "gpu_model": record.gpu_model,
            "ssh_host": record.ssh_host,
            "ssh_port": str(record.ssh_port),
            "ssh_user": record.ssh_user,
            "http_port": str(record.http_port) if record.http_port is not None else "",
            "worker_url": record.worker_url,
            "state": record.state.value,
            "config_id": record.config_id,
            "created_at": str(record.created_at),
            "state_changed_at": str(record.state_changed_at),
            "cost_per_hour_usd": str(record.cost_per_hour_usd),
        }
        await self._redis.hset(key, mapping=mapping)
        logger.debug("state_store.saved", instance_id=record.instance_id)

    async def delete(self, instance_id: str) -> None:
        """Remove an instance record from Redis."""
        key = f"{_KEY_PREFIX}{instance_id}"
        await self._redis.delete(key)
        logger.debug("state_store.deleted", instance_id=instance_id)

    async def load_all(self) -> list[InstanceRecord]:
        """SCAN for all instance keys, HGETALL each, deserialize to InstanceRecord."""
        records: list[InstanceRecord] = []
        cursor = 0
        while True:
            cursor, keys = await self._redis.scan(
                cursor=cursor, match=f"{_KEY_PREFIX}*", count=100
            )
            for key in keys:
                data = await self._redis.hgetall(key)
                if not data:
                    continue
                try:
                    record = _deserialize(data)
                    records.append(record)
                except Exception:
                    logger.warning(
                        "state_store.deserialize_failed",
                        key=key,
                        exc_info=True,
                    )
            if cursor == 0:
                break
        return records


def _deserialize(data: dict[str, str]) -> InstanceRecord:
    """Convert a Redis hash dict back into an InstanceRecord."""
    http_port_raw = data.get("http_port", "")
    http_port = int(http_port_raw) if http_port_raw else None

    return InstanceRecord(
        instance_id=data["instance_id"],
        provider=data["provider"],
        gpu_model=data["gpu_model"],
        ssh_host=data.get("ssh_host", ""),
        ssh_port=int(data.get("ssh_port", "22")),
        ssh_user=data.get("ssh_user", "root"),
        http_port=http_port,
        worker_url=data.get("worker_url", ""),
        state=InstanceState(data["state"]),
        config_id=data.get("config_id", ""),
        created_at=float(data.get("created_at", "0")),
        state_changed_at=float(data.get("state_changed_at", "0")),
        cost_per_hour_usd=float(data.get("cost_per_hour_usd", "0.0")),
    )
