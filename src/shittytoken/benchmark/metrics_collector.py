import asyncio
import time
from dataclasses import dataclass, field

import aiohttp
import structlog

from shittytoken.common.prometheus import parse_prometheus_text
from .constants import METRICS_SCRAPE_INTERVAL_SEC

logger = structlog.get_logger()


@dataclass
class MetricsScrape:
    timestamp: float  # time.monotonic()
    unix_ts: float  # time.time()
    phase: int  # 0=not started, 1/2/3
    prefix_cache_queries: float
    prefix_cache_hits: float
    kv_cache_usage_perc: float
    num_requests_running: float
    num_requests_waiting: float

    @property
    def prefix_cache_hit_rate(self) -> float:
        if self.prefix_cache_queries == 0:
            return 0.0
        return self.prefix_cache_hits / self.prefix_cache_queries


class MetricsCollector:
    """
    Scrapes vLLM Prometheus metrics at a fixed interval while a benchmark phase
    is running.  Each scrape is tagged with the current phase number so that
    per-phase statistics (e.g. delta-based cache hit rate) can be computed
    after the run.

    Usage:
        collector = MetricsCollector(metrics_url, session)
        task = asyncio.create_task(collector.run())
        collector.mark_phase(1)
        # ... run requests ...
        collector.mark_phase(2)
        # ...
        collector.stop()
        await task
        rate = collector.compute_delta_hit_rate(phase=2)
    """

    def __init__(self, metrics_url: str, session: aiohttp.ClientSession) -> None:
        self._metrics_url = metrics_url
        self._session = session
        self._scrapes: list[MetricsScrape] = []
        self._current_phase: int = 0
        self._stop_event: asyncio.Event = asyncio.Event()

    async def run(self) -> None:
        """Scrape metrics every METRICS_SCRAPE_INTERVAL_SEC until stop() is called."""
        logger.info("metrics_collector.run.start", url=self._metrics_url)
        while not self._stop_event.is_set():
            try:
                await self._scrape_once()
            except aiohttp.ClientError as exc:
                logger.warning("metrics_collector.scrape_client_error", error=str(exc))
            except asyncio.TimeoutError as exc:
                logger.warning("metrics_collector.scrape_timeout", error=str(exc))
            except Exception as exc:  # noqa: BLE001 — log-and-continue for resilience
                logger.warning(
                    "metrics_collector.scrape_unexpected_error", error=str(exc)
                )
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._stop_event.wait()),
                    timeout=METRICS_SCRAPE_INTERVAL_SEC,
                )
            except asyncio.TimeoutError:
                pass  # normal: interval elapsed, continue loop
        logger.info("metrics_collector.run.stopped")

    def stop(self) -> None:
        """Signal the run() coroutine to exit after the current scrape."""
        self._stop_event.set()

    def mark_phase(self, phase: int) -> None:
        """Tag subsequent scrapes with `phase`."""
        logger.info("metrics_collector.mark_phase", phase=phase)
        self._current_phase = phase

    @property
    def scrapes(self) -> list[MetricsScrape]:
        return list(self._scrapes)

    def scrapes_for_phase(self, phase: int) -> list[MetricsScrape]:
        return [s for s in self._scrapes if s.phase == phase]

    def compute_delta_hit_rate(self, phase: int) -> float:
        """
        delta(hits) / delta(queries) across all scrapes tagged with this phase.
        Returns 0.0 if fewer than 2 scrapes or delta(queries) == 0.

        IMPORTANT: delta-based, not point-in-time, to be independent of prior traffic.
        """
        phase_scrapes = self.scrapes_for_phase(phase)
        if len(phase_scrapes) < 2:
            return 0.0

        first = phase_scrapes[0]
        last = phase_scrapes[-1]
        delta_queries = last.prefix_cache_queries - first.prefix_cache_queries
        delta_hits = last.prefix_cache_hits - first.prefix_cache_hits
        if delta_queries <= 0:
            return 0.0
        return max(0.0, min(1.0, delta_hits / delta_queries))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _scrape_once(self) -> None:
        timeout = aiohttp.ClientTimeout(total=10, connect=5)
        async with self._session.get(self._metrics_url, timeout=timeout) as resp:
            text = await resp.text()
            if resp.status != 200:
                logger.warning("metrics_scrape_http_error", status=resp.status)
                return

        parsed = parse_prometheus_text(text)
        scrape = MetricsScrape(
            timestamp=time.monotonic(),
            unix_ts=time.time(),
            phase=self._current_phase,
            prefix_cache_queries=parsed.get("vllm:prefix_cache_queries_total", 0.0),
            prefix_cache_hits=parsed.get("vllm:prefix_cache_hits_total", 0.0),
            kv_cache_usage_perc=parsed.get("vllm:gpu_cache_usage_perc", 0.0),
            num_requests_running=parsed.get("vllm:num_requests_running", 0.0),
            num_requests_waiting=parsed.get("vllm:num_requests_waiting", 0.0),
        )
        self._scrapes.append(scrape)
        logger.debug(
            "metrics_collector.scraped",
            phase=scrape.phase,
            cache_hit_rate=scrape.prefix_cache_hit_rate,
            kv_usage=scrape.kv_cache_usage_perc,
        )
