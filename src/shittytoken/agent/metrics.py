"""
Metrics scraper — collects and aggregates vLLM Prometheus metrics across
all active workers.

Extracts gauges (running/waiting/cache), histogram _sum/_count pairs
(TTFT, ITL, queue time), and counters (tokens, preemptions) from each
worker's /metrics endpoint.

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
class WorkerMetrics:
    """Per-worker metrics scraped from vLLM /metrics."""
    url: str
    requests_running: int = 0
    requests_waiting: int = 0
    kv_cache_pct: float = 0.0
    prefix_cache_hits: float = 0.0
    prefix_cache_queries: float = 0.0
    reachable: bool = True  # False if scrape failed

    # vLLM histogram _sum/_count (for delta-based averages)
    ttft_sum: float = 0.0
    ttft_count: float = 0.0
    itl_sum: float = 0.0
    itl_count: float = 0.0
    queue_time_sum: float = 0.0
    queue_time_count: float = 0.0

    # vLLM counters (monotonic, use deltas for rates)
    prompt_tokens_total: float = 0.0
    generation_tokens_total: float = 0.0
    preemptions_total: float = 0.0


@dataclass
class AggregateMetrics:
    total_requests_running: int
    total_requests_waiting: int
    avg_kv_cache_usage: float   # average across all workers (0.0–1.0)
    worker_count: int
    per_worker: list[WorkerMetrics] | None = None  # populated when available




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
    per_worker: list[WorkerMetrics] = []

    for url, metrics in zip(worker_urls, raw_results):
        reachable = len(metrics) > 0  # empty dict = scrape failed
        running = int(metrics.get("vllm:num_requests_running", metrics.get("num_requests_running", 0)))
        waiting = int(metrics.get("vllm:num_requests_waiting", metrics.get("num_requests_waiting", 0)))
        total_running += running
        total_waiting += waiting
        kv_pct = 0.0
        for kv_key in ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc", "gpu_cache_usage_perc"):
            if kv_key in metrics:
                kv_pct = metrics[kv_key]
                kv_cache_values.append(kv_pct)
                break
        per_worker.append(WorkerMetrics(
            url=url,
            requests_running=running,
            requests_waiting=waiting,
            kv_cache_pct=kv_pct,
            prefix_cache_hits=metrics.get("vllm:prefix_cache_hits_total", 0.0),
            prefix_cache_queries=metrics.get("vllm:prefix_cache_queries_total", 0.0),
            reachable=reachable,
            ttft_sum=metrics.get("vllm:time_to_first_token_seconds_sum", 0.0),
            ttft_count=metrics.get("vllm:time_to_first_token_seconds_count", 0.0),
            itl_sum=metrics.get("vllm:inter_token_latency_seconds_sum", 0.0),
            itl_count=metrics.get("vllm:inter_token_latency_seconds_count", 0.0),
            queue_time_sum=metrics.get("vllm:request_queue_time_seconds_sum", 0.0),
            queue_time_count=metrics.get("vllm:request_queue_time_seconds_count", 0.0),
            prompt_tokens_total=metrics.get("vllm:prompt_tokens_total", 0.0),
            generation_tokens_total=metrics.get("vllm:generation_tokens_total", 0.0),
            preemptions_total=metrics.get("vllm:num_preemptions_total", 0.0),
        ))

    avg_kv = sum(kv_cache_values) / len(kv_cache_values) if kv_cache_values else 0.0

    return AggregateMetrics(
        total_requests_running=total_running,
        total_requests_waiting=total_waiting,
        avg_kv_cache_usage=avg_kv,
        worker_count=len(worker_urls),
        per_worker=per_worker,
    )
