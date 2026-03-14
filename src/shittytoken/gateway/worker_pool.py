"""
WorkerPool — in-process worker state management for the custom router.

Workers are vLLM instances accessed via HTTP.  The pool maintains live
load metrics (scraped in the background) and delegates worker selection
to a :class:`CacheAwarePolicy`.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import aiohttp
import structlog

from shittytoken.common.prometheus import parse_prometheus_text
from shittytoken.gateway.routing_policy import CacheAwarePolicy

logger = structlog.get_logger()

# Consecutive scrape failures before a worker is marked unhealthy.
_UNHEALTHY_THRESHOLD = 3


@dataclass
class WorkerState:
    url: str
    requests_running: int = 0     # from vLLM /metrics scrape
    requests_waiting: int = 0     # from vLLM /metrics scrape
    local_in_flight: int = 0      # tracked by proxy (inc on send, dec on response)
    kv_cache_pct: float = 0.0
    prefix_cache_hits: float = 0.0
    prefix_cache_queries: float = 0.0
    healthy: bool = True
    added_at: float = field(default_factory=time.monotonic)

    # vLLM histogram _sum/_count pairs (for computing averages)
    ttft_sum: float = 0.0
    ttft_count: float = 0.0
    itl_sum: float = 0.0
    itl_count: float = 0.0
    e2e_latency_sum: float = 0.0
    e2e_latency_count: float = 0.0
    queue_time_sum: float = 0.0
    queue_time_count: float = 0.0

    # vLLM counters (monotonic)
    prompt_tokens_total: float = 0.0
    generation_tokens_total: float = 0.0
    preemptions_total: float = 0.0
    request_success_total: float = 0.0


class WorkerPool:
    """Thread-safe (asyncio) worker pool with background metrics scraping."""

    def __init__(self, policy: CacheAwarePolicy | None = None) -> None:
        self._workers: dict[str, WorkerState] = {}
        self._lock = asyncio.Lock()
        self._policy = policy or CacheAwarePolicy()
        self._consecutive_failures: dict[str, int] = {}
        self._session: aiohttp.ClientSession | None = None

    # ------------------------------------------------------------------
    # Pool mutations
    # ------------------------------------------------------------------

    async def add(self, url: str) -> None:
        """Add a worker. Raises ValueError if already present."""
        async with self._lock:
            if url in self._workers:
                raise ValueError(f"Worker already in pool: {url}")
            self._workers[url] = WorkerState(url=url)
            self._consecutive_failures[url] = 0
        logger.info("worker_pool.added", url=url, pool_size=len(self._workers))

    async def remove(self, url: str) -> None:
        """Remove a worker. Raises KeyError if not found."""
        async with self._lock:
            if url not in self._workers:
                raise KeyError(f"Worker not in pool: {url}")
            del self._workers[url]
            self._consecutive_failures.pop(url, None)
        logger.info("worker_pool.removed", url=url, pool_size=len(self._workers))

    # ------------------------------------------------------------------
    # Selection & metrics
    # ------------------------------------------------------------------

    def select(self, prefix_key: str, exclude: set[str] | None = None) -> WorkerState | None:
        """Select the best worker for *prefix_key*, optionally excluding URLs."""
        workers = list(self._workers.values())
        if exclude:
            workers = [w for w in workers if w.url not in exclude]
        if not workers:
            return None
        return self._policy.select(prefix_key, workers)

    def report_metrics(
        self,
        url: str,
        requests_running: int,
        kv_cache_pct: float,
        requests_waiting: int = 0,
    ) -> None:
        """Update load metrics for a worker (called by background scraper)."""
        worker = self._workers.get(url)
        if worker is None:
            return
        worker.requests_running = requests_running
        worker.requests_waiting = requests_waiting
        worker.kv_cache_pct = kv_cache_pct

    def list_active(self) -> list[WorkerState]:
        """Return a snapshot of all workers."""
        return list(self._workers.values())

    # Aliases for admin_api / prom_metrics compatibility
    add_worker = add
    remove_worker = remove
    list_workers = list_active

    # ------------------------------------------------------------------
    # Background metrics scraper
    # ------------------------------------------------------------------

    async def run_scraper(self, interval_sec: float = 5.0) -> None:
        """Poll each worker's ``/metrics`` every *interval_sec*.

        Extracts ``num_requests_running`` and ``vllm:gpu_cache_usage_perc``.
        After 3 consecutive scrape failures a worker is marked unhealthy.
        A single successful scrape resets the failure counter and restores
        healthy status.
        """
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10.0),
        )
        try:
            while True:
                urls = list(self._workers.keys())
                await asyncio.gather(
                    *(self._scrape_worker(url) for url in urls),
                    return_exceptions=True,
                )
                await asyncio.sleep(interval_sec)
        finally:
            await self._session.close()
            self._session = None

    async def _scrape_worker(self, url: str) -> None:
        """Scrape a single worker and update its state."""
        assert self._session is not None  # noqa: S101
        metrics_url = url.rstrip("/") + "/metrics"
        try:
            async with self._session.get(metrics_url) as resp:
                if resp.status != 200:
                    self._record_failure(url)
                    return
                text = await resp.text()
        except Exception:  # noqa: BLE001
            self._record_failure(url)
            return

        parsed = parse_prometheus_text(text)
        requests_running = int(parsed.get("vllm:num_requests_running", parsed.get("num_requests_running", 0)))
        requests_waiting = int(parsed.get("vllm:num_requests_waiting", parsed.get("num_requests_waiting", 0)))
        kv_cache_pct = parsed.get(
            "vllm:kv_cache_usage_perc",                          # vLLM v1
            parsed.get("vllm:gpu_cache_usage_perc",              # vLLM v0
                        parsed.get("gpu_cache_usage_perc", 0.0)),
        )
        cache_hits = parsed.get("vllm:prefix_cache_hits_total", 0.0)
        cache_queries = parsed.get("vllm:prefix_cache_queries_total", 0.0)

        self.report_metrics(url, requests_running, kv_cache_pct, requests_waiting)
        worker = self._workers.get(url)
        if worker is not None:
            worker.prefix_cache_hits = cache_hits
            worker.prefix_cache_queries = cache_queries

            # vLLM histogram _sum/_count (for computing averages)
            worker.ttft_sum = parsed.get("vllm:time_to_first_token_seconds_sum", 0.0)
            worker.ttft_count = parsed.get("vllm:time_to_first_token_seconds_count", 0.0)
            worker.itl_sum = parsed.get("vllm:inter_token_latency_seconds_sum", 0.0)
            worker.itl_count = parsed.get("vllm:inter_token_latency_seconds_count", 0.0)
            worker.e2e_latency_sum = parsed.get("vllm:e2e_request_latency_seconds_sum", 0.0)
            worker.e2e_latency_count = parsed.get("vllm:e2e_request_latency_seconds_count", 0.0)
            worker.queue_time_sum = parsed.get("vllm:request_queue_time_seconds_sum", 0.0)
            worker.queue_time_count = parsed.get("vllm:request_queue_time_seconds_count", 0.0)

            # vLLM counters
            worker.prompt_tokens_total = parsed.get("vllm:prompt_tokens_total", 0.0)
            worker.generation_tokens_total = parsed.get("vllm:generation_tokens_total", 0.0)
            worker.preemptions_total = parsed.get("vllm:num_preemptions_total", 0.0)
            worker.request_success_total = parsed.get("vllm:request_success_total", 0.0)

        # Successful scrape — reset failure counter and mark healthy.
        self._consecutive_failures[url] = 0
        worker = self._workers.get(url)
        if worker is not None and not worker.healthy:
            worker.healthy = True
            logger.info("worker_pool.healthy_again", url=url)

    def _record_failure(self, url: str) -> None:
        """Increment failure counter; mark unhealthy after threshold."""
        self._consecutive_failures[url] = self._consecutive_failures.get(url, 0) + 1
        count = self._consecutive_failures[url]
        logger.warning(
            "worker_pool.scrape_failed",
            url=url,
            consecutive_failures=count,
        )
        if count >= _UNHEALTHY_THRESHOLD:
            worker = self._workers.get(url)
            if worker is not None and worker.healthy:
                worker.healthy = False
                logger.error("worker_pool.marked_unhealthy", url=url)
