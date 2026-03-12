"""
Tests for src/shittytoken/benchmark/metrics_collector.py.
"""

import asyncio
import time

import pytest
import aiohttp
import aiohttp.test_utils
import aiohttp.web

from shittytoken.benchmark.metrics_collector import MetricsCollector, MetricsScrape


# ---------------------------------------------------------------------------
# Synthetic Prometheus text helpers
# ---------------------------------------------------------------------------

_VALID_PROMETHEUS_TEXT = """\
# HELP vllm:prefix_cache_queries_total Total prefix cache queries
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total 100
# HELP vllm:prefix_cache_hits_total Total prefix cache hits
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total 65
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc 0.42
# HELP vllm:num_requests_running Requests running
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 3
# HELP vllm:num_requests_waiting Requests waiting
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting 1
"""

_PROMETHEUS_TEXT_WITH_LABELS = """\
# HELP vllm:prefix_cache_queries_total Total
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{model="qwen"} 200
vllm:prefix_cache_hits_total{model="qwen"} 130
vllm:gpu_cache_usage_perc{model="qwen"} 0.55
vllm:num_requests_running{model="qwen"} 5
vllm:num_requests_waiting{model="qwen"} 2
"""

_PROMETHEUS_TEXT_COMMENT_ONLY = """\
# HELP some_metric A metric
# TYPE some_metric gauge
# This is just a comment line
"""


# ---------------------------------------------------------------------------
# parse_prometheus_text
# ---------------------------------------------------------------------------


def test_parse_extracts_all_five_metrics():
    result = MetricsCollector.parse_prometheus_text(_VALID_PROMETHEUS_TEXT)
    assert result["vllm:prefix_cache_queries_total"] == 100.0
    assert result["vllm:prefix_cache_hits_total"] == 65.0
    assert result["vllm:gpu_cache_usage_perc"] == pytest.approx(0.42)
    assert result["vllm:num_requests_running"] == 3.0
    assert result["vllm:num_requests_waiting"] == 1.0


def test_parse_handles_comment_lines_without_error():
    # Should not raise; unknown metrics are simply ignored.
    result = MetricsCollector.parse_prometheus_text(_PROMETHEUS_TEXT_COMMENT_ONLY)
    assert result == {}


def test_parse_handles_metrics_with_labels():
    result = MetricsCollector.parse_prometheus_text(_PROMETHEUS_TEXT_WITH_LABELS)
    assert result["vllm:prefix_cache_queries_total"] == 200.0
    assert result["vllm:prefix_cache_hits_total"] == 130.0
    assert result["vllm:gpu_cache_usage_perc"] == pytest.approx(0.55)
    assert result["vllm:num_requests_running"] == 5.0
    assert result["vllm:num_requests_waiting"] == 2.0


# ---------------------------------------------------------------------------
# compute_delta_hit_rate
# ---------------------------------------------------------------------------


def _make_scrape(phase: int, queries: float, hits: float) -> MetricsScrape:
    return MetricsScrape(
        timestamp=time.monotonic(),
        unix_ts=time.time(),
        phase=phase,
        prefix_cache_queries=queries,
        prefix_cache_hits=hits,
        kv_cache_usage_perc=0.0,
        num_requests_running=0.0,
        num_requests_waiting=0.0,
    )


def test_compute_delta_hit_rate_correct_value():
    """delta hits = 13-5=8, delta queries = 20-10=10 → rate = 0.8"""
    collector = MetricsCollector.__new__(MetricsCollector)
    collector._scrapes = [
        _make_scrape(phase=1, queries=10, hits=5),
        _make_scrape(phase=1, queries=20, hits=13),
    ]

    rate = collector.compute_delta_hit_rate(phase=1)
    assert rate == pytest.approx(0.8)


def test_compute_delta_hit_rate_fewer_than_two_scrapes():
    collector = MetricsCollector.__new__(MetricsCollector)
    collector._scrapes = [
        _make_scrape(phase=1, queries=10, hits=5),
    ]

    assert collector.compute_delta_hit_rate(phase=1) == 0.0


def test_compute_delta_hit_rate_no_scrapes():
    collector = MetricsCollector.__new__(MetricsCollector)
    collector._scrapes = []
    assert collector.compute_delta_hit_rate(phase=1) == 0.0


def test_compute_delta_hit_rate_zero_delta_queries():
    """If queries didn't change, returns 0.0 (avoids division by zero)."""
    collector = MetricsCollector.__new__(MetricsCollector)
    collector._scrapes = [
        _make_scrape(phase=2, queries=50, hits=20),
        _make_scrape(phase=2, queries=50, hits=25),  # queries unchanged
    ]
    assert collector.compute_delta_hit_rate(phase=2) == 0.0


# ---------------------------------------------------------------------------
# mark_phase integration with scrapes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mark_phase_tags_subsequent_scrapes():
    """
    When mark_phase(2) is called, scrapes taken after that point carry phase=2.
    Uses a TestServer that serves synthetic Prometheus text so the collector
    can actually scrape.
    """

    async def handle_metrics(req: aiohttp.web.Request) -> aiohttp.web.Response:
        return aiohttp.web.Response(text=_VALID_PROMETHEUS_TEXT, content_type="text/plain")

    app = aiohttp.web.Application()
    app.router.add_get("/metrics", handle_metrics)

    async with aiohttp.test_utils.TestServer(app) as server:
        metrics_url = f"http://{server.host}:{server.port}/metrics"
        connector = aiohttp.TCPConnector(limit=4)
        async with aiohttp.ClientSession(connector=connector) as session:
            collector = MetricsCollector(metrics_url, session)

            # Phase 1: scrape once manually (bypass run() for determinism).
            collector.mark_phase(1)
            await collector._scrape_once()  # type: ignore[attr-defined]

            # Transition to phase 2 and scrape again.
            collector.mark_phase(2)
            await collector._scrape_once()  # type: ignore[attr-defined]

    scrapes = collector.scrapes
    assert len(scrapes) == 2
    assert scrapes[0].phase == 1
    assert scrapes[1].phase == 2
