import asyncio
import time

import aiohttp
import structlog

from .metrics_collector import MetricsCollector
from .constants import LONG_CONTEXT_ENABLED
from .phases import (
    run_phase_1_cold_cache,
    run_phase_2_warmup,
    run_phase_3_sustained,
    run_phase_4_long_context,
)
from .request_generator import VirtualUserPool
from .results_analyzer import evaluate_benchmark
from .schema import BenchmarkResult, BenchmarkVerdict, FailReason

logger = structlog.get_logger()


async def run_benchmark(
    worker_url: str,
    model_id: str,
    gpu_model: str,
    raw_config: dict,
    level_duration_sec: float = 30.0,
) -> BenchmarkResult:
    """
    Main benchmark entry point. Called by the agent.

    Lifecycle:
    1. Creates aiohttp.ClientSession with TCPConnector(limit=128)
    2. Creates VirtualUserPool and MetricsCollector
    3. Starts MetricsCollector as asyncio.Task
    4. Runs phase 1 → awaits
    5. Runs phase 2 → awaits
    6. Runs phase 3 + sweep → awaits
    7. Stops MetricsCollector
    8. Calls evaluate_benchmark() → returns BenchmarkResult

    NEVER raises. Any top-level exception returns:
    BenchmarkResult(verdict=FAIL, fail_reasons=[...], ...)
    with a partial or empty result set.

    Logs: {"event": "benchmark_started", "worker_url": ..., "model_id": ...}
    Logs: {"event": "benchmark_complete", "verdict": ..., "duration_sec": ...}
    """
    started_at = time.time()
    logger.info(
        "benchmark_started",
        worker_url=worker_url,
        model_id=model_id,
        gpu_model=gpu_model,
    )

    metrics_url = f"{worker_url.rstrip('/')}/metrics"
    connector = aiohttp.TCPConnector(limit=128)
    collector_task: asyncio.Task | None = None

    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            pool = VirtualUserPool()
            collector = MetricsCollector(metrics_url, session)
            collector_task = asyncio.create_task(collector.run())

            try:
                phase1 = await run_phase_1_cold_cache(
                    pool, collector, session, worker_url
                )
                phase2 = await run_phase_2_warmup(pool, collector, session, worker_url)
                phase3, concurrency_sweep = await run_phase_3_sustained(
                    pool,
                    collector,
                    session,
                    worker_url,
                    level_duration_sec=level_duration_sec,
                )

                # Phase 4: long-context agent flow (optional)
                phase4 = None
                long_context_steps = None
                if LONG_CONTEXT_ENABLED:
                    phase4, long_context_steps = await run_phase_4_long_context(
                        collector, session, worker_url
                    )
            finally:
                collector.stop()
                if collector_task is not None:
                    await collector_task

            completed_at = time.time()
            result = evaluate_benchmark(
                worker_url=worker_url,
                model_id=model_id,
                gpu_model=gpu_model,
                phase1=phase1,
                phase2=phase2,
                phase3=phase3,
                concurrency_sweep=concurrency_sweep,
                raw_config=raw_config,
                started_at=started_at,
                completed_at=completed_at,
            )

    except Exception as exc:  # noqa: BLE001
        completed_at = time.time()
        duration_sec = completed_at - started_at
        logger.error(
            "benchmark_error",
            error=str(exc),
            exc_info=True,
            duration_sec=duration_sec,
        )
        from .schema import PhaseMetrics

        empty_phase = PhaseMetrics(
            phase_number=0,
            duration_sec=0.0,
            ttft_samples=[],
            cache_hit_rate_timeseries=[],
            kv_cache_usage_timeseries=[],
            failed_request_count=0,
            total_request_count=0,
        )
        result = BenchmarkResult(
            worker_url=worker_url,
            model_id=model_id,
            gpu_model=gpu_model,
            started_at=started_at,
            completed_at=completed_at,
            verdict=BenchmarkVerdict.FAIL,
            fail_reasons=[FailReason.HIGH_ERROR_RATE],
            deltanet_cache_suspect=False,
            phase_metrics=[empty_phase],
            concurrency_sweep=[],
            cold_cache_baseline_ttft_p95=float("inf"),
            warm_cache_ttft_p95_at_concurrency_1=float("inf"),
            peak_throughput_tokens_per_sec=0.0,
            raw_config=raw_config,
        )

    duration_sec = result.completed_at - result.started_at

    # ── Summary log (easy to grep for monitoring) ─────────────────────────
    summary = {
        "verdict": result.verdict.value,
        "duration_sec": round(duration_sec, 1),
        "cold_ttft_p95": round(result.cold_cache_baseline_ttft_p95, 2),
        "warm_ttft_p95": round(result.warm_cache_ttft_p95_at_concurrency_1, 2),
        "peak_tps": round(result.peak_throughput_tokens_per_sec, 1),
    }
    if result.fail_reasons:
        summary["fail_reasons"] = [r.value for r in result.fail_reasons]
    if result.concurrency_sweep:
        summary["sweep"] = {
            p.concurrency: {
                "tps": round(p.throughput_tokens_per_sec, 1),
                "ttft_p95": round(p.ttft_p95_sec, 2),
                "failed": p.failed_request_count,
            }
            for p in result.concurrency_sweep
        }

    logger.info("benchmark_complete", **summary)
    return result
