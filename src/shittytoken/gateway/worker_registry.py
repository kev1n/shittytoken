"""
WorkerRegistry — active worker pool with circuit-breaker state and drain support.

All worker add/remove/list operations go through Python functions only —
no shell scripts.  Pool changes trigger router_manager.reload() automatically.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum

import aiohttp
import structlog

from shittytoken.common.prometheus import parse_prometheus_text

logger = structlog.get_logger()

# How long to wait between drain-poll iterations (seconds).
_DRAIN_POLL_INTERVAL_SEC = 5.0


class CircuitBreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class WorkerEntry:
    url: str
    registered_at: float
    circuit_state: CircuitBreakerState = CircuitBreakerState.CLOSED
    consecutive_failures: int = 0
    last_health_check: float = 0.0
    draining: bool = False


class WorkerRegistry:
    """
    Manages the active worker pool.

    Calling add_worker / remove_worker always triggers router_manager.reload()
    so the vLLM Router reflects the new pool immediately.
    """

    def __init__(self, router_manager) -> None:  # type: RouterManager
        self._router_manager = router_manager
        self._workers: dict[str, WorkerEntry] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def add_worker(self, url: str) -> None:
        """
        Add *url* to the pool and trigger a router reload.

        Raises ValueError if *url* is already registered.
        """
        async with self._lock:
            if url in self._workers:
                raise ValueError(f"Worker already registered: {url}")

            self._workers[url] = WorkerEntry(
                url=url,
                registered_at=time.time(),
            )
            pool_size = len(self._workers)
            active_urls = self._active_urls()

        await self._router_manager.reload(active_urls)
        logger.info("worker_added", url=url, pool_size=pool_size)

    async def remove_worker(self, url: str, drain: bool = True) -> None:
        """
        Remove *url* from the pool and trigger a router reload.

        drain=True  — mark the worker as draining, poll its /metrics until
                      num_requests_running == 0 (or timeout), then remove.
        drain=False — remove immediately without waiting for in-flight
                      requests to complete.

        Raises KeyError if *url* is not registered.
        """
        async with self._lock:
            if url not in self._workers:
                raise KeyError(f"Worker not found: {url}")

            if drain:
                self._workers[url].draining = True

        if drain:
            await self._poll_drain_complete(url)

        async with self._lock:
            self._workers.pop(url, None)
            active_urls = self._active_urls()

        await self._router_manager.reload(active_urls)
        logger.info("worker_removed", url=url, drain=drain)

    def list_workers(self) -> list[WorkerEntry]:
        """Return a snapshot of the current worker pool."""
        return list(self._workers.values())

    def get_worker(self, url: str) -> WorkerEntry | None:
        return self._workers.get(url)

    def update_circuit_state(self, url: str, state: CircuitBreakerState) -> None:
        """
        Update the circuit-breaker state for *url*.

        Raises KeyError if *url* is not registered.
        """
        entry = self._workers.get(url)
        if entry is None:
            raise KeyError(f"Worker not found: {url}")
        entry.circuit_state = state
        logger.info("circuit_state_changed", url=url, state=state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _active_urls(self) -> list[str]:
        """Return URLs of non-draining workers (used for router reload)."""
        return [w.url for w in self._workers.values() if not w.draining]

    async def _poll_drain_complete(
        self,
        url: str,
        timeout_sec: float = 300.0,
    ) -> None:
        """
        Poll {url}/metrics every 5 seconds.  Stop when num_requests_running
        reaches 0 or *timeout_sec* elapses.  After timeout, proceed anyway so
        the caller can still remove the worker.

        Uses inline Prometheus text parsing — no external dependencies.
        """
        deadline = time.monotonic() + timeout_sec
        log = logger.bind(url=url)
        log.info("drain_poll_started", timeout_sec=timeout_sec)

        async with aiohttp.ClientSession() as session:
            while time.monotonic() < deadline:
                running = await self._fetch_requests_running(session, url)
                if running is not None and running == 0:
                    log.info("drain_complete", num_requests_running=0)
                    return
                log.debug(
                    "drain_poll_tick",
                    num_requests_running=running,
                    remaining_sec=round(deadline - time.monotonic(), 1),
                )
                await asyncio.sleep(_DRAIN_POLL_INTERVAL_SEC)

        log.warning("drain_timeout", timeout_sec=timeout_sec, action="proceeding_anyway")

    @staticmethod
    async def _fetch_requests_running(
        session: aiohttp.ClientSession,
        base_url: str,
    ) -> float | None:
        """
        Scrape {base_url}/metrics and return num_requests_running.
        Returns None on any error.
        """
        metrics_url = base_url.rstrip("/") + "/metrics"
        try:
            async with session.get(
                metrics_url,
                timeout=aiohttp.ClientTimeout(total=10.0),
            ) as resp:
                if resp.status != 200:
                    return None
                text = await resp.text()
                metrics = parse_prometheus_text(text)
                return metrics.get("num_requests_running")
        except Exception:  # noqa: BLE001 — intentional: errors during drain treated as unknown
            return None


