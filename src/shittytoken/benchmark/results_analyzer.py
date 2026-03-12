import math
import time

from .constants import (
    DELTANET_CACHE_FLAG_THRESHOLD,
    MAX_COLD_CACHE_TTFT_P95,
    MAX_FAILED_REQUEST_RATE,
    MIN_PREFIX_CACHE_HIT_RATE_PHASE3,
    MIN_THROUGHPUT_CONCURRENCY_1,
)
from .schema import (
    BenchmarkResult,
    BenchmarkVerdict,
    ConcurrencyPoint,
    FailReason,
    PhaseMetrics,
)


def compute_ttft_percentile(samples: list[float], percentile: float) -> float:
    """
    Computes the given percentile of TTFT samples.
    Failed requests contribute float('inf') — included in the distribution.
    Returns float('inf') if all samples are inf or list is empty.
    Uses linear interpolation.
    """
    if not samples:
        return float("inf")

    sorted_samples = sorted(samples)
    n = len(sorted_samples)

    # All-inf case.
    if all(math.isinf(s) for s in sorted_samples):
        return float("inf")

    # Linear interpolation (same as numpy's default).
    # index into [0, n-1] range.
    index = (percentile / 100.0) * (n - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))

    if lower == upper:
        return sorted_samples[lower]

    frac = index - lower
    low_val = sorted_samples[lower]
    high_val = sorted_samples[upper]

    # If either endpoint is inf, the interpolated result is inf.
    if math.isinf(low_val) or math.isinf(high_val):
        return float("inf")

    return low_val + frac * (high_val - low_val)


def compute_throughput_tps(results: list, duration_sec: float) -> float:
    """
    Computes tokens per second from a list of RequestResult objects.
    Only counts tokens from successful requests (success=True).
    duration_sec is the elapsed time of the measurement window.
    """
    if duration_sec <= 0:
        return 0.0
    total_tokens = sum(r.tokens_generated for r in results if r.success)
    return total_tokens / duration_sec


def evaluate_benchmark(
    worker_url: str,
    model_id: str,
    gpu_model: str,
    phase1: PhaseMetrics,
    phase2: PhaseMetrics,
    phase3: PhaseMetrics,
    concurrency_sweep: list[ConcurrencyPoint],
    raw_config: dict,
    started_at: float,
    completed_at: float,
) -> BenchmarkResult:
    """
    Applies all pass/fail criteria. Multiple fail_reasons can be present.
    A result can have deltanet_cache_suspect=True regardless of verdict.

    Criteria applied:
    1. Throughput: concurrency_sweep[0].throughput_tokens_per_sec >= MIN_THROUGHPUT_CONCURRENCY_1
    2. Cold TTFT P95: compute_ttft_percentile(phase1.ttft_samples, 95) <= MAX_COLD_CACHE_TTFT_P95
    3. Cache hit rate: phase3 cache hit rate >= MIN_PREFIX_CACHE_HIT_RATE_PHASE3
       (use ConcurrencyPoint at concurrency=1 for this, or aggregate phase3 hit rate)
    4. Error rate: phase3 failed / total <= MAX_FAILED_REQUEST_RATE

    DeltaNet suspect flag: if phase3 hit rate < DELTANET_CACHE_FLAG_THRESHOLD → deltanet_cache_suspect=True

    cold_cache_baseline_ttft_p95: from phase1
    warm_cache_ttft_p95_at_concurrency_1: from concurrency_sweep[0] (concurrency=1 point)
    peak_throughput_tokens_per_sec: max across all ConcurrencyPoints
    """
    fail_reasons: list[FailReason] = []

    # --- Criterion 1: Throughput at concurrency=1 ---
    throughput_at_c1 = (
        concurrency_sweep[0].throughput_tokens_per_sec if concurrency_sweep else 0.0
    )
    if throughput_at_c1 < MIN_THROUGHPUT_CONCURRENCY_1:
        fail_reasons.append(FailReason.LOW_THROUGHPUT)

    # --- Criterion 2: Cold cache TTFT P95 ---
    cold_p95 = compute_ttft_percentile(phase1.ttft_samples, 95)
    if cold_p95 > MAX_COLD_CACHE_TTFT_P95:
        fail_reasons.append(FailReason.HIGH_COLD_TTFT)

    # --- Criterion 3: Phase 3 prefix cache hit rate ---
    # Use the weighted aggregate across all ConcurrencyPoints to capture the full
    # phase-3 picture.  A per-level point may have too few metrics scrapes to be
    # representative (e.g. when level_duration_sec equals the scrape interval).
    phase3_hit_rate: float
    if concurrency_sweep:
        total_reqs = sum(p.total_request_count for p in concurrency_sweep)
        if total_reqs > 0:
            phase3_hit_rate = sum(
                p.prefix_cache_hit_rate * p.total_request_count
                for p in concurrency_sweep
            ) / total_reqs
        else:
            # Fall back to simple mean when no requests were recorded.
            phase3_hit_rate = sum(
                p.prefix_cache_hit_rate for p in concurrency_sweep
            ) / len(concurrency_sweep)
    else:
        phase3_hit_rate = 0.0

    if phase3_hit_rate < MIN_PREFIX_CACHE_HIT_RATE_PHASE3:
        fail_reasons.append(FailReason.LOW_CACHE_HIT_RATE)

    # --- Criterion 4: Phase 3 error rate ---
    phase3_total = phase3.total_request_count
    phase3_failed = phase3.failed_request_count
    error_rate = (phase3_failed / phase3_total) if phase3_total > 0 else 0.0
    if error_rate > MAX_FAILED_REQUEST_RATE:
        fail_reasons.append(FailReason.HIGH_ERROR_RATE)

    # --- DeltaNet suspect flag ---
    deltanet_cache_suspect = phase3_hit_rate < DELTANET_CACHE_FLAG_THRESHOLD

    # --- Summary metrics ---
    warm_cache_ttft_p95_at_c1 = (
        concurrency_sweep[0].ttft_p95_sec if concurrency_sweep else float("inf")
    )
    peak_throughput = (
        max(p.throughput_tokens_per_sec for p in concurrency_sweep)
        if concurrency_sweep
        else 0.0
    )

    verdict = BenchmarkVerdict.PASS if not fail_reasons else BenchmarkVerdict.FAIL

    return BenchmarkResult(
        worker_url=worker_url,
        model_id=model_id,
        gpu_model=gpu_model,
        started_at=started_at,
        completed_at=completed_at,
        verdict=verdict,
        fail_reasons=fail_reasons,
        deltanet_cache_suspect=deltanet_cache_suspect,
        phase_metrics=[phase1, phase2, phase3],
        concurrency_sweep=concurrency_sweep,
        cold_cache_baseline_ttft_p95=cold_p95,
        warm_cache_ttft_p95_at_concurrency_1=warm_cache_ttft_p95_at_c1,
        peak_throughput_tokens_per_sec=peak_throughput,
        raw_config=raw_config,
    )
