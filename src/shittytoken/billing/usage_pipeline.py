"""
Usage event pipeline.

Publishes usage events after each API request completes.  A consumer
drains the queue, deducts credits (FIFO from Postgres), and updates
the Redis balance cache.

When Kafka is configured, events go through a Kafka topic for durability
and horizontal scaling.  Otherwise, an in-process asyncio.Queue is used.
"""

from __future__ import annotations

import asyncio
import json
import math
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol

import structlog

from shittytoken.billing.models import UsageEvent

if TYPE_CHECKING:
    from shittytoken.billing.postgres import BillingPostgres
    from shittytoken.billing.redis_cache import BillingRedis

log = structlog.get_logger(__name__)


class UsagePublisher(Protocol):
    """Interface for publishing usage events."""

    async def publish(self, event: UsageEvent) -> None: ...
    async def close(self) -> None: ...


class UsageConsumer(Protocol):
    """Interface for consuming usage events."""

    async def consume(self, handler: Callable[[UsageEvent], Awaitable[None]]) -> None: ...
    async def close(self) -> None: ...


class InProcessPublisher:
    """In-process asyncio.Queue publisher. Use when Kafka is not configured."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[UsageEvent] = asyncio.Queue()

    async def publish(self, event: UsageEvent) -> None:
        self._queue.put_nowait(event)

    async def close(self) -> None:
        pass


class InProcessConsumer:
    """Drains from the InProcessPublisher's queue."""

    def __init__(self, publisher: InProcessPublisher) -> None:
        self._publisher = publisher

    async def consume(self, handler: Callable[[UsageEvent], Awaitable[None]]) -> None:
        """Run forever, draining events and calling handler."""
        while True:
            # Batch: drain up to 100 events, then process
            events: list[UsageEvent] = []
            try:
                event = await asyncio.wait_for(
                    self._publisher._queue.get(), timeout=1.0
                )
                events.append(event)
            except asyncio.TimeoutError:
                continue
            # Drain remaining without blocking
            while not self._publisher._queue.empty() and len(events) < 100:
                try:
                    events.append(self._publisher._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            for evt in events:
                try:
                    await handler(evt)
                except Exception:
                    log.exception("usage_consumer.handler_error", event_id=evt.event_id)

    async def close(self) -> None:
        pass


class KafkaPublisher:
    """Kafka publisher. Requires aiokafka."""

    def __init__(self, producer: object) -> None:
        self._producer = producer
        self._topic = "shittytoken.usage"

    @classmethod
    async def create(
        cls, bootstrap_servers: str, topic: str = "shittytoken.usage"
    ) -> KafkaPublisher:
        try:
            from aiokafka import AIOKafkaProducer
        except ImportError:
            raise ImportError("aiokafka is required for Kafka support: uv add aiokafka")
        producer = AIOKafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode(),
        )
        await producer.start()
        inst = cls(producer)
        inst._topic = topic
        return inst

    async def publish(self, event: UsageEvent) -> None:
        await self._producer.send(  # type: ignore[union-attr]
            self._topic, event.to_dict(), key=event.user_id.encode()
        )

    async def close(self) -> None:
        await self._producer.stop()  # type: ignore[union-attr]


class KafkaConsumer:
    """Kafka consumer. Requires aiokafka."""

    def __init__(self, consumer: object) -> None:
        self._consumer = consumer

    @classmethod
    async def create(
        cls,
        bootstrap_servers: str,
        group_id: str = "shittytoken-billing",
        topic: str = "shittytoken.usage",
    ) -> KafkaConsumer:
        try:
            from aiokafka import AIOKafkaConsumer
        except ImportError:
            raise ImportError("aiokafka is required for Kafka support: uv add aiokafka")
        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda v: json.loads(v),
        )
        await consumer.start()
        return cls(consumer)

    async def consume(self, handler: Callable[[UsageEvent], Awaitable[None]]) -> None:
        """Run forever, consuming from Kafka topic."""
        async for msg in self._consumer:  # type: ignore[union-attr]
            event_data = json.loads(msg.value) if isinstance(msg.value, (str, bytes)) else msg.value
            event = UsageEvent(**event_data)
            await handler(event)

    async def close(self) -> None:
        await self._consumer.stop()  # type: ignore[union-attr]


class BillingPipeline:
    """Orchestrates the full billing pipeline: publish -> consume -> deduct.

    Wires together the publisher, consumer, Postgres, and Redis.
    """

    def __init__(
        self,
        publisher: UsagePublisher,
        consumer: UsageConsumer,
        postgres: BillingPostgres,
        redis: BillingRedis,
        pricing: dict[str, float] | None = None,
    ) -> None:
        self._publisher = publisher
        self._consumer = consumer
        self._postgres = postgres
        self._redis = redis
        self._pricing = pricing

    @staticmethod
    def compute_cost(
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        pricing: dict[str, float] | None = None,
    ) -> int:
        """Compute cost in cents. Default $1/1M tokens. Minimum 1 cent."""
        total = prompt_tokens + completion_tokens
        price_per_1m = (pricing or {}).get(model, 100.0)
        return max(math.ceil(total * price_per_1m / 1_000_000), 1)

    async def publish_usage(
        self,
        user_id: str,
        key_hash: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        request_id: str | None = None,
    ) -> None:
        """Create UsageEvent with computed cost and publish."""
        cost = self.compute_cost(model, prompt_tokens, completion_tokens, self._pricing)
        event = UsageEvent(
            event_id=uuid.uuid4().hex,
            user_id=user_id,
            api_key_hash=key_hash,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_cents=cost,
            latency_ms=latency_ms,
            request_id=request_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        await self._publisher.publish(event)
        log.debug(
            "usage_pipeline.published",
            event_id=event.event_id,
            user_id=user_id,
            cost_cents=cost,
        )

    async def _handle_event(self, event: UsageEvent) -> None:
        """Process a single usage event.

        1. Record in Postgres usage_events table
        2. Deduct from credit blocks FIFO via Postgres
        3. Update Redis balance to match Postgres

        If Postgres deduction succeeds but Redis update fails,
        reconciliation will fix it.
        """
        # 1. Record event
        await self._postgres.record_usage_event(event)

        # 2. Deduct credits FIFO
        deducted = await self._postgres.deduct_credits_fifo(
            user_id=event.user_id,
            amount_cents=event.cost_cents,
            request_id=event.request_id,
        )
        if deducted < event.cost_cents:
            log.warning(
                "usage_pipeline.insufficient_credits",
                event_id=event.event_id,
                user_id=event.user_id,
                requested=event.cost_cents,
                deducted=deducted,
            )

        # 3. Update Redis balance to match Postgres
        try:
            pg_balance = await self._postgres.get_balance(event.user_id)
            await self._redis.set_balance(event.user_id, pg_balance)
        except Exception:
            log.exception(
                "usage_pipeline.redis_update_failed",
                event_id=event.event_id,
                user_id=event.user_id,
            )

    async def run_consumer(self) -> None:
        """Start the consumer loop. Each event:
        1. Record in Postgres usage_events table
        2. Deduct from credit blocks FIFO via Postgres
        3. Update Redis balance to match Postgres

        If Postgres deduction succeeds but Redis update fails,
        reconciliation will fix it.
        """
        await self._consumer.consume(self._handle_event)

    async def close(self) -> None:
        await self._publisher.close()
        await self._consumer.close()
