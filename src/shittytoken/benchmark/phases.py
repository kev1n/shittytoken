import asyncio
import time
from dataclasses import dataclass

import aiohttp
import structlog

from .constants import (
    CONCURRENCY_LEVELS,
    FIRST_TOKEN_TIMEOUT_SEC,
    LONG_CONTEXT_OUTPUT_TOKENS,
    LONG_CONTEXT_REQUEST_TIMEOUT_SEC,
    LONG_CONTEXT_STEPS_TOKENS,
    REQUEST_TIMEOUT_SEC,
    WARM_UP_REQUESTS_PER_PREFIX,
)
from .metrics_collector import MetricsCollector
from .request_generator import VirtualUserPool
from .results_analyzer import compute_throughput_tps, compute_ttft_percentile
from .schema import ConcurrencyPoint, PhaseMetrics
from .sse_client import RequestResult, send_chat_completion
from .workloads import WorkloadProfile, make_system_prompt, make_query

logger = structlog.get_logger()


def _build_phase_metrics(
    phase_number: int,
    duration_sec: float,
    results: list[RequestResult],
    collector: MetricsCollector,
) -> PhaseMetrics:
    """Build a PhaseMetrics from raw results plus collector timeseries."""
    ttft_samples: list[float] = []
    failed = 0
    for r in results:
        if r.success:
            if r.ttft_sec is not None:
                ttft_samples.append(r.ttft_sec)
            # success=True but ttft_sec=None means we got a valid response
            # with no content/reasoning tokens — don't penalize as failure
        else:
            ttft_samples.append(float("inf"))
            failed += 1

    phase_scrapes = collector.scrapes_for_phase(phase_number)
    cache_hit_ts = [(s.unix_ts, s.prefix_cache_hit_rate) for s in phase_scrapes]
    kv_cache_ts = [(s.unix_ts, s.kv_cache_usage_perc) for s in phase_scrapes]

    return PhaseMetrics(
        phase_number=phase_number,
        duration_sec=duration_sec,
        ttft_samples=ttft_samples,
        cache_hit_rate_timeseries=cache_hit_ts,
        kv_cache_usage_timeseries=kv_cache_ts,
        failed_request_count=failed,
        total_request_count=len(results),
    )


async def run_phase_1_cold_cache(
    pool: VirtualUserPool,
    collector: MetricsCollector,
    session: aiohttp.ClientSession,
    base_url: str,
    num_unique_sessions: int = 20,
) -> PhaseMetrics:
    """
    Phase 1: Send exactly one request per unique prefix session (cold cache misses).
    Concurrency = 1 (sequential) to get clean baseline TTFTs.

    All requests use pool.next_request() which creates new sessions (since pool is empty).
    Calls pool.record_reply() after each successful request.
    Calls collector.mark_phase(1) at the start.

    PhaseMetrics.ttft_samples includes float('inf') for any failed requests.
    """
    collector.mark_phase(1)
    logger.info("phase_1.start", num_unique_sessions=num_unique_sessions)

    start = time.monotonic()
    results: list[RequestResult] = []

    for i in range(num_unique_sessions):
        req = pool.next_request()
        result = await send_chat_completion(
            session,
            base_url,
            req.messages,
            300,
            req.session_id,
            REQUEST_TIMEOUT_SEC,
            FIRST_TOKEN_TIMEOUT_SEC,
        )
        results.append(result)
        if result.success and result.output_text:
            pool.record_reply(req.session_id, result.output_text)
        logger.debug("phase_1.request_done", index=i, success=result.success)

    duration_sec = time.monotonic() - start
    logger.info(
        "phase_1.complete",
        total=len(results),
        duration_sec=duration_sec,
    )
    return _build_phase_metrics(1, duration_sec, results, collector)


async def run_phase_2_warmup(
    pool: VirtualUserPool,
    collector: MetricsCollector,
    session: aiohttp.ClientSession,
    base_url: str,
    requests_per_prefix: int = WARM_UP_REQUESTS_PER_PREFIX,
) -> PhaseMetrics:
    """
    Phase 2: Extend each existing session requests_per_prefix times.
    Uses concurrency=4 to populate cache without thrashing.

    Iterates pool's existing sessions (call pool.session_count() to know how many).
    For each session, issues requests_per_prefix follow-up turns.
    Calls collector.mark_phase(2) at the start.
    """
    collector.mark_phase(2)
    session_count = pool.session_count()
    logger.info(
        "phase_2.start",
        session_count=session_count,
        requests_per_prefix=requests_per_prefix,
    )

    start = time.monotonic()
    results: list[RequestResult] = []
    sem = asyncio.Semaphore(4)

    async def one_request() -> None:
        async with sem:
            req = pool.next_request()
            result = await send_chat_completion(
                session,
                base_url,
                req.messages,
                300,
                req.session_id,
                REQUEST_TIMEOUT_SEC,
                FIRST_TOKEN_TIMEOUT_SEC,
            )
            results.append(result)
            if result.success and result.output_text:
                pool.record_reply(req.session_id, result.output_text)

    total_requests = session_count * requests_per_prefix
    tasks = [asyncio.create_task(one_request()) for _ in range(total_requests)]
    await asyncio.gather(*tasks, return_exceptions=True)

    duration_sec = time.monotonic() - start
    logger.info(
        "phase_2.complete",
        total=len(results),
        duration_sec=duration_sec,
    )
    return _build_phase_metrics(2, duration_sec, results, collector)


async def run_phase_3_sustained(
    pool: VirtualUserPool,
    collector: MetricsCollector,
    session: aiohttp.ClientSession,
    base_url: str,
    level_duration_sec: float = 30.0,
) -> tuple[PhaseMetrics, list[ConcurrencyPoint]]:
    """
    Phase 3: Concurrency sweep across CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32, 64].
    At each level, maintains exactly N concurrent requests for level_duration_sec seconds
    using asyncio.Semaphore(N).

    Calls collector.mark_phase(3) at the start.

    For each concurrency level:
    - Run _run_at_concurrency(N, pool, session, base_url, level_duration_sec)
    - Compute ConcurrencyPoint from results + collector.compute_delta_hit_rate(3)

    Returns (aggregate PhaseMetrics, list[ConcurrencyPoint] one per level).
    The aggregate PhaseMetrics combines all results across all concurrency levels.
    """
    collector.mark_phase(3)
    logger.info(
        "phase_3.start",
        concurrency_levels=CONCURRENCY_LEVELS,
        level_duration_sec=level_duration_sec,
    )

    start = time.monotonic()
    all_results: list[RequestResult] = []
    concurrency_points: list[ConcurrencyPoint] = []

    for concurrency in CONCURRENCY_LEVELS:
        logger.info("phase_3.level_start", concurrency=concurrency)
        level_start = time.monotonic()
        level_results = await _run_at_concurrency(
            concurrency, pool, session, base_url, level_duration_sec
        )
        level_duration = time.monotonic() - level_start

        all_results.extend(level_results)

        # Compute stats for this concurrency point.
        ttft_samples = [
            r.ttft_sec if (r.success and r.ttft_sec is not None) else float("inf")
            for r in level_results
        ]
        failed = sum(1 for r in level_results if not r.success)
        total = len(level_results)

        throughput = compute_throughput_tps(level_results, level_duration)

        point = ConcurrencyPoint(
            concurrency=concurrency,
            ttft_p50_sec=compute_ttft_percentile(ttft_samples, 50),
            ttft_p95_sec=compute_ttft_percentile(ttft_samples, 95),
            throughput_tokens_per_sec=throughput,
            # hit_rate is set to 0.0 as placeholder — backfilled after all levels
            # complete so the aggregate delta covers the full phase-3 window.
            prefix_cache_hit_rate=0.0,
            failed_request_count=failed,
            total_request_count=total,
        )
        concurrency_points.append(point)
        logger.info(
            "phase_3.level_complete",
            concurrency=concurrency,
            total=total,
            failed=failed,
            throughput=throughput,
        )

    # Compute aggregate hit rate now that all phase-3 scrapes are available.
    aggregate_hit_rate = collector.compute_delta_hit_rate(3)
    for point in concurrency_points:
        point.prefix_cache_hit_rate = aggregate_hit_rate

    duration_sec = time.monotonic() - start
    logger.info(
        "phase_3.complete",
        duration_sec=duration_sec,
        total=len(all_results),
        aggregate_hit_rate=aggregate_hit_rate,
    )

    aggregate = _build_phase_metrics(3, duration_sec, all_results, collector)
    return aggregate, concurrency_points


async def _run_at_concurrency(
    concurrency: int,
    pool: VirtualUserPool,
    session: aiohttp.ClientSession,
    base_url: str,
    duration_sec: float,
) -> list[RequestResult]:
    """
    Maintains exactly `concurrency` in-flight requests using asyncio.Semaphore.
    Spawns tasks continuously until deadline, then awaits all pending tasks.

    For each completed successful request, calls pool.record_reply() so the
    next request from that session will hit the prefix cache.
    """
    sem = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []
    loop = asyncio.get_running_loop()
    deadline = loop.time() + duration_sec
    tasks: list[asyncio.Task] = []

    async def one_request() -> None:
        async with sem:
            req = pool.next_request()
            result = await send_chat_completion(
                session,
                base_url,
                req.messages,
                300,
                req.session_id,
                REQUEST_TIMEOUT_SEC,
                FIRST_TOKEN_TIMEOUT_SEC,
            )
            results.append(result)
            if result.success and result.output_text:
                pool.record_reply(req.session_id, result.output_text)

    # Maximum pending (not-yet-done) tasks is bounded to avoid creating millions
    # of queued coroutines in tight loops at high concurrency levels.
    _max_pending = concurrency * 4

    while loop.time() < deadline:
        # Prune completed tasks to keep the pending count accurate.
        tasks = [t for t in tasks if not t.done()]
        if len(tasks) < _max_pending:
            task = asyncio.create_task(one_request())
            tasks.append(task)
        await asyncio.sleep(0)  # yield to let semaphore-gated tasks progress

    await asyncio.gather(*tasks, return_exceptions=True)
    return results


# ---------------------------------------------------------------------------
# Phase 4: Long-context agent flow
# ---------------------------------------------------------------------------


@dataclass
class LongContextStep:
    target_context_tokens: int
    actual_context_chars: int
    ttft_sec: float | None
    total_duration_sec: float
    tokens_generated: int
    success: bool
    error: str | None


async def run_phase_4_long_context(
    collector: MetricsCollector,
    session: aiohttp.ClientSession,
    base_url: str,
) -> tuple[PhaseMetrics, list[LongContextStep]]:
    """
    Phase 4: Agent-style long-context ramp.

    Simulates an agentic workflow where context grows over multiple turns:
    a large system prompt + accumulating assistant/user turns that push the
    context window from 8k tokens up to 131k tokens.

    Each step sends ONE request at the target context length and records TTFT.
    """
    collector.mark_phase(4)
    steps = LONG_CONTEXT_STEPS_TOKENS
    logger.info("phase_4.start", context_steps=steps)

    start = time.monotonic()
    results: list[RequestResult] = []
    step_details: list[LongContextStep] = []

    # Build a single conversation that grows across steps.
    # Start with a large coding system prompt, then pad context with
    # synthetic "prior agent turns" to reach the target token count.
    messages: list[dict] = [
        {"role": "system", "content": make_system_prompt(WorkloadProfile.CODING, 2000)},
    ]
    current_context_chars = 2000 * 4  # approximate

    for step_tokens in steps:
        target_chars = step_tokens * 4
        # Pad with synthetic prior turns to reach the target context size
        chars_needed = target_chars - current_context_chars
        if chars_needed > 0:
            # Add padding as prior "agent thinking" turns
            pad_text = make_system_prompt(WorkloadProfile.DOCUMENT_QA, chars_needed // 4)
            messages.append({"role": "assistant", "content": pad_text})
            current_context_chars += len(pad_text)

        # Add the actual user query for this step
        query = make_query(WorkloadProfile.CODING, 200)
        messages.append({"role": "user", "content": query})
        current_context_chars += len(query)

        logger.info(
            "phase_4.step_start",
            target_tokens=step_tokens,
            context_chars=current_context_chars,
            num_messages=len(messages),
        )

        result = await send_chat_completion(
            session,
            base_url,
            list(messages),  # copy
            LONG_CONTEXT_OUTPUT_TOKENS,
            f"long-context-{step_tokens}",
            LONG_CONTEXT_REQUEST_TIMEOUT_SEC,
            LONG_CONTEXT_REQUEST_TIMEOUT_SEC,  # TTFT can be slow for long context
        )
        results.append(result)

        step = LongContextStep(
            target_context_tokens=step_tokens,
            actual_context_chars=current_context_chars,
            ttft_sec=result.ttft_sec,
            total_duration_sec=result.total_duration_sec,
            tokens_generated=result.tokens_generated,
            success=result.success,
            error=result.error,
        )
        step_details.append(step)

        logger.info(
            "phase_4.step_complete",
            target_tokens=step_tokens,
            success=result.success,
            ttft_sec=round(result.ttft_sec, 2) if result.ttft_sec else None,
            duration_sec=round(result.total_duration_sec, 2),
            tokens_generated=result.tokens_generated,
            error=result.error[:100] if result.error else None,
        )

        # Append the model's reply so context grows naturally
        if result.success and result.output_text:
            messages.append({"role": "assistant", "content": result.output_text})
            current_context_chars += len(result.output_text)

    duration_sec = time.monotonic() - start
    logger.info(
        "phase_4.complete",
        duration_sec=round(duration_sec, 1),
        steps_passed=sum(1 for s in step_details if s.success),
        steps_total=len(step_details),
    )

    phase_metrics = _build_phase_metrics(4, duration_sec, results, collector)
    return phase_metrics, step_details
