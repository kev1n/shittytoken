"""
Metrics scraper — collects and aggregates vLLM Prometheus metrics across
all active workers.

Only three metrics are extracted from each worker:
  - num_requests_running
  - num_requests_waiting
  - gpu_cache_usage_perc

Workers that fail to respond contribute 0 to totals (not failure).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import aiohttp
import structlog

from shittytoken.common.prometheus import parse_prometheus_text

logger = structlog.get_logger()


@dataclass
class AggregateMetrics:
    total_requests_running: int
    total_requests_waiting: int
    avg_kv_cache_usage: float   # average across all workers (0.0–1.0)
    worker_count: int




async def scrape_worker_metrics(
    worker_url: str,
    session: aiohttp.ClientSession,
) -> dict[str, float]:
    """
    GET {worker_url}/metrics and extract the three metrics of interest.

    Returns {} on any error — never raises.
    """
    url = worker_url.rstrip("/") + "/metrics"
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=10.0),
        ) as resp:
            if resp.status != 200:
                logger.debug(
                    "metrics_scrape_non200",
                    worker_url=worker_url,
                    status=resp.status,
                )
                return {}
            text = await resp.text()
            return parse_prometheus_text(text)
    except aiohttp.ClientError as exc:
        logger.debug("metrics_scrape_error", worker_url=worker_url, error=str(exc))
        return {}


async def aggregate_metrics(
    worker_urls: list[str],
    session: aiohttp.ClientSession,
) -> AggregateMetrics:
    """
    Scrape all workers concurrently.

    Workers that fail to respond contribute 0 to totals but are still counted
    in worker_count (so the caller knows about the gap).

    avg_kv_cache_usage is averaged only over workers that actually reported
    the metric; 0.0 is returned if no workers reported it.
    """
    if not worker_urls:
        return AggregateMetrics(
            total_requests_running=0,
            total_requests_waiting=0,
            avg_kv_cache_usage=0.0,
            worker_count=0,
        )

    raw_results: list[dict[str, float]] = await asyncio.gather(
        *[scrape_worker_metrics(url, session) for url in worker_urls],
        return_exceptions=False,
    )

    total_running = 0
    total_waiting = 0
    kv_cache_values: list[float] = []

    for metrics in raw_results:
        total_running += int(metrics.get("num_requests_running", 0))
        total_waiting += int(metrics.get("num_requests_waiting", 0))
        if "gpu_cache_usage_perc" in metrics:
            kv_cache_values.append(metrics["gpu_cache_usage_perc"])

    avg_kv = sum(kv_cache_values) / len(kv_cache_values) if kv_cache_values else 0.0

    return AggregateMetrics(
        total_requests_running=total_running,
        total_requests_waiting=total_waiting,
        avg_kv_cache_usage=avg_kv,
        worker_count=len(worker_urls),
    )
