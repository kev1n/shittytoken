"""
Health checking utilities for vLLM worker instances.

wait_for_model_ready() polls /v1/models (not /health) because /health becomes
ready before the model weights are fully loaded.

HeartbeatMonitor runs as a background asyncio task and calls on_failure(url)
when a registered worker stops responding.
"""

from __future__ import annotations

import asyncio
import time

import aiohttp
import structlog

logger = structlog.get_logger()


async def wait_for_model_ready(
    base_url: str,
    session: aiohttp.ClientSession,
    timeout_sec: float = 600.0,
    poll_interval_sec: float = 5.0,
) -> bool:
    """
    Poll GET {base_url}/v1/models until it returns a non-empty data array.

    Returns True when the model is loaded, False on timeout.

    Uses /v1/models rather than /health because /health becomes ready before
    the model weights finish loading.
    """
    url = base_url.rstrip("/") + "/v1/models"
    deadline = time.monotonic() + timeout_sec
    start = time.monotonic()
    log = logger.bind(worker_url=base_url)
    attempt = 0

    log.info(
        "worker_model_poll_started",
        url=url,
        timeout_sec=timeout_sec,
    )

    while time.monotonic() < deadline:
        attempt += 1
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=10.0),
            ) as resp:
                if resp.status == 200:
                    try:
                        data = await resp.json()
                    except (ValueError, aiohttp.ContentTypeError) as exc:
                        log.debug("worker_model_poll_json_error", error=str(exc))
                        await asyncio.sleep(poll_interval_sec)
                        continue
                    models = data.get("data", [])
                    if not isinstance(models, list):
                        await asyncio.sleep(poll_interval_sec)
                        continue
                    if models:
                        elapsed = time.monotonic() - start
                        log.info(
                            "worker_model_ready",
                            model_count=len(models),
                            elapsed_s=round(elapsed, 1),
                            attempts=attempt,
                        )
                        return True
                else:
                    # Log non-200 periodically (every 30s)
                    if attempt % 6 == 0:
                        elapsed = time.monotonic() - start
                        log.info(
                            "worker_model_poll_waiting",
                            status=resp.status,
                            elapsed_s=round(elapsed, 1),
                            attempts=attempt,
                        )
        except aiohttp.ClientError as exc:
            # Log connection errors periodically (every 30s)
            if attempt % 6 == 0:
                elapsed = time.monotonic() - start
                log.info(
                    "worker_model_poll_waiting",
                    error=str(exc)[:100],
                    elapsed_s=round(elapsed, 1),
                    attempts=attempt,
                )

        await asyncio.sleep(poll_interval_sec)

    log.warning("worker_model_ready_timeout", timeout_sec=timeout_sec)
    return False


class HeartbeatMonitor:
    """
    Background task that periodically polls /health for all registered workers.

    When a worker fails a health check, on_failure(url) is called (if provided).

    Usage:
        monitor = HeartbeatMonitor(session, health_check_interval_s=30, on_failure=my_fn)
        monitor.register("http://worker:8000")
        task = asyncio.create_task(monitor.run())
        ...
        monitor.stop()
    """

    def __init__(
        self,
        session: aiohttp.ClientSession,
        health_check_interval_s: int = 30,
        failure_threshold: int = 3,
        on_failure=None,  # async callable(url: str) -> None
    ) -> None:
        self._session = session
        self._interval = health_check_interval_s
        self._failure_threshold = failure_threshold
        self._on_failure = on_failure
        self._workers: set[str] = set()
        self._failure_counts: dict[str, int] = {}
        self._running = False

    def register(self, url: str) -> None:
        """Add *url* to the set of monitored workers."""
        self._workers.add(url)
        logger.info("heartbeat_monitor_registered", url=url)

    def deregister(self, url: str) -> None:
        """Remove *url* from the monitored set (no-op if not present)."""
        self._workers.discard(url)
        self._failure_counts.pop(url, None)
        logger.info("heartbeat_monitor_deregistered", url=url)

    async def run(self) -> None:
        """
        Main loop — runs until stop() is called.

        Polls all registered workers every health_check_interval_s seconds.
        Workers that do not return HTTP 200 trigger on_failure().
        """
        self._running = True
        logger.info("heartbeat_monitor_started", interval_s=self._interval)

        while self._running:
            # Snapshot the current worker set to avoid mutation during iteration
            current_workers = list(self._workers)
            await asyncio.gather(
                *[self._check_worker(url) for url in current_workers],
                return_exceptions=True,
            )
            await asyncio.sleep(self._interval)

        logger.info("heartbeat_monitor_stopped")

    def stop(self) -> None:
        """Signal the run loop to exit after the current sleep."""
        self._running = False

    async def _check_worker(self, url: str) -> None:
        health_url = url.rstrip("/") + "/health"
        try:
            async with self._session.get(
                health_url,
                timeout=aiohttp.ClientTimeout(total=10.0),
            ) as resp:
                if resp.status == 200:
                    self._failure_counts[url] = 0
                else:
                    await self._handle_failure(url, reason=f"HTTP {resp.status}")
        except aiohttp.ClientError as exc:
            await self._handle_failure(url, reason=str(exc))

    async def _handle_failure(self, url: str, reason: str) -> None:
        self._failure_counts[url] = self._failure_counts.get(url, 0) + 1
        count = self._failure_counts[url]
        logger.warning(
            "heartbeat_check_failed",
            url=url,
            reason=reason,
            failure_count=count,
            threshold=self._failure_threshold,
        )
        if count == self._failure_threshold and self._on_failure is not None:
            await self._on_failure(url)
