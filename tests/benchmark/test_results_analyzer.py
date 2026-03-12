"""
Pure unit tests for results_analyzer — no I/O.
All PhaseMetrics and ConcurrencyPoint objects are constructed synthetically.
"""
import math
import time

import pytest

from shittytoken.benchmark.constants import (
    DELTANET_CACHE_FLAG_THRESHOLD,
    MAX_COLD_CACHE_TTFT_P95,
    MAX_FAILED_REQUEST_RATE,
    MIN_PREFIX_CACHE_HIT_RATE_PHASE3,
    MIN_THROUGHPUT_CONCURRENCY_1,
)
from shittytoken.benchmark.results_analyzer import (
    compute_ttft_percentile,
    evaluate_benchmark,
)
from shittytoken.benchmark.schema import (
    BenchmarkVerdict,
    ConcurrencyPoint,
    FailReason,
    PhaseMetrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_phase(
    phase_number: int = 1,
    ttft_samples: list[float] | None = None,
    failed: int = 0,
    total: int = 20,
) -> PhaseMetrics:
    if ttft_samples is None:
        ttft_samples = [0.5] * total
    return PhaseMetrics(
        phase_number=phase_number,
        duration_sec=10.0,
        ttft_samples=ttft_samples,
        cache_hit_rate_timeseries=[],
        kv_cache_usage_timeseries=[],
        failed_request_count=failed,
        total_request_count=total,
    )


def _make_concurrency_point(
    concurrency: int = 1,
    throughput: float = MIN_THROUGHPUT_CONCURRENCY_1 + 10.0,
    hit_rate: float = MIN_PREFIX_CACHE_HIT_RATE_PHASE3 + 0.1,
    failed: int = 0,
    total: int = 100,
    ttft_p50: float = 0.3,
    ttft_p95: float = 0.8,
) -> ConcurrencyPoint:
    return ConcurrencyPoint(
        concurrency=concurrency,
        ttft_p50_sec=ttft_p50,
        ttft_p95_sec=ttft_p95,
        throughput_tokens_per_sec=throughput,
        prefix_cache_hit_rate=hit_rate,
        failed_request_count=failed,
        total_request_count=total,
    )


def _good_evaluate(**overrides):
    """Call evaluate_benchmark with passing defaults, optionally overridden."""
    phase1 = overrides.pop("phase1", _make_phase(1, ttft_samples=[0.5] * 20))
    phase2 = overrides.pop("phase2", _make_phase(2))
    phase3 = overrides.pop("phase3", _make_phase(3, failed=0, total=700))
    concurrency_sweep = overrides.pop(
        "concurrency_sweep",
        [
            _make_concurrency_point(concurrency=c)
            for c in [1, 2, 4, 8, 16, 32, 64]
        ],
    )
    now = time.time()
    return evaluate_benchmark(
        worker_url="http://localhost:8000",
        model_id="test-model",
        gpu_model="A100",
        phase1=phase1,
        phase2=phase2,
        phase3=phase3,
        concurrency_sweep=concurrency_sweep,
        raw_config={},
        started_at=now - 60,
        completed_at=now,
        **overrides,
    )


# ---------------------------------------------------------------------------
# compute_ttft_percentile tests
# ---------------------------------------------------------------------------


def test_percentile_basic():
    """P95 of [1,2,3,4,5] should be approximately 4.8 (linear interpolation)."""
    result = compute_ttft_percentile([1.0, 2.0, 3.0, 4.0, 5.0], 95)
    assert abs(result - 4.8) < 1e-9


def test_percentile_with_inf_samples():
    """inf samples penalize P95 — result should be inf when enough infs exist."""
    result = compute_ttft_percentile([1.0, float("inf"), float("inf")], 95)
    assert math.isinf(result)


def test_percentile_empty_list():
    """Empty list returns float('inf')."""
    result = compute_ttft_percentile([], 95)
    assert math.isinf(result)


def test_percentile_all_finite():
    """All finite samples return a finite result."""
    result = compute_ttft_percentile([1.0, 2.0, 3.0], 50)
    assert math.isfinite(result)
    assert result == 2.0


def test_percentile_single_element():
    """Single-element list returns that element for any percentile."""
    assert compute_ttft_percentile([3.5], 50) == 3.5
    assert compute_ttft_percentile([3.5], 0) == 3.5
    assert compute_ttft_percentile([3.5], 100) == 3.5


# ---------------------------------------------------------------------------
# evaluate_benchmark tests
# ---------------------------------------------------------------------------


def test_evaluate_all_passing():
    """All passing criteria → PASS verdict, empty fail_reasons."""
    result = _good_evaluate()
    assert result.verdict == BenchmarkVerdict.PASS
    assert result.fail_reasons == []


def test_evaluate_low_throughput():
    """Low throughput at concurrency=1 → FAIL with LOW_THROUGHPUT."""
    low_sweep = [
        _make_concurrency_point(
            concurrency=c,
            throughput=MIN_THROUGHPUT_CONCURRENCY_1 - 1.0 if c == 1 else MIN_THROUGHPUT_CONCURRENCY_1 + 10.0,
        )
        for c in [1, 2, 4, 8, 16, 32, 64]
    ]
    result = _good_evaluate(concurrency_sweep=low_sweep)
    assert result.verdict == BenchmarkVerdict.FAIL
    assert FailReason.LOW_THROUGHPUT in result.fail_reasons


def test_evaluate_high_cold_ttft():
    """High cold TTFT P95 → FAIL with HIGH_COLD_TTFT."""
    # All samples above threshold.
    bad_samples = [MAX_COLD_CACHE_TTFT_P95 + 1.0] * 20
    bad_phase1 = _make_phase(1, ttft_samples=bad_samples)
    result = _good_evaluate(phase1=bad_phase1)
    assert result.verdict == BenchmarkVerdict.FAIL
    assert FailReason.HIGH_COLD_TTFT in result.fail_reasons


def test_evaluate_low_cache_hit_rate():
    """Low cache hit rate → FAIL with LOW_CACHE_HIT_RATE."""
    low_hit_rate = MIN_PREFIX_CACHE_HIT_RATE_PHASE3 - 0.01
    # Must be above deltanet threshold to not conflate with deltanet test.
    low_hit_rate = max(low_hit_rate, DELTANET_CACHE_FLAG_THRESHOLD + 0.005)
    low_sweep = [
        _make_concurrency_point(concurrency=c, hit_rate=low_hit_rate)
        for c in [1, 2, 4, 8, 16, 32, 64]
    ]
    result = _good_evaluate(concurrency_sweep=low_sweep)
    assert result.verdict == BenchmarkVerdict.FAIL
    assert FailReason.LOW_CACHE_HIT_RATE in result.fail_reasons


def test_evaluate_high_error_rate():
    """High error rate in phase3 → FAIL with HIGH_ERROR_RATE."""
    # Set failed/total so error rate exceeds MAX_FAILED_REQUEST_RATE.
    bad_phase3 = _make_phase(3, failed=50, total=100)
    result = _good_evaluate(phase3=bad_phase3)
    assert result.verdict == BenchmarkVerdict.FAIL
    assert FailReason.HIGH_ERROR_RATE in result.fail_reasons


def test_deltanet_cache_suspect_set():
    """Cache hit rate below DELTANET_CACHE_FLAG_THRESHOLD → deltanet_cache_suspect=True."""
    tiny_hit_rate = DELTANET_CACHE_FLAG_THRESHOLD / 2.0  # well below threshold
    low_sweep = [
        _make_concurrency_point(concurrency=c, hit_rate=tiny_hit_rate)
        for c in [1, 2, 4, 8, 16, 32, 64]
    ]
    result = _good_evaluate(concurrency_sweep=low_sweep)
    assert result.deltanet_cache_suspect is True


def test_deltanet_cache_suspect_regardless_of_verdict():
    """
    deltanet_cache_suspect can be True on a FAIL result too.
    Combine zero hit rate with otherwise-passing criteria to test the flag in isolation.
    """
    # Use a hit rate of 0.0 (below deltanet threshold) but still ensure the verdict
    # fails on cache hit rate as well — the flag is independent of verdict.
    zero_sweep = [
        _make_concurrency_point(concurrency=c, hit_rate=0.0)
        for c in [1, 2, 4, 8, 16, 32, 64]
    ]
    result = _good_evaluate(concurrency_sweep=zero_sweep)
    assert result.deltanet_cache_suspect is True
    # Verdict is FAIL because hit rate is 0 (below MIN_PREFIX_CACHE_HIT_RATE_PHASE3)
    assert result.verdict == BenchmarkVerdict.FAIL


def test_deltanet_cache_suspect_false_when_above_threshold():
    """Hit rate above DELTANET_CACHE_FLAG_THRESHOLD → deltanet_cache_suspect=False."""
    good_sweep = [
        _make_concurrency_point(concurrency=c, hit_rate=DELTANET_CACHE_FLAG_THRESHOLD + 0.1)
        for c in [1, 2, 4, 8, 16, 32, 64]
    ]
    result = _good_evaluate(concurrency_sweep=good_sweep)
    assert result.deltanet_cache_suspect is False


def test_multiple_failures():
    """Multiple failures in same benchmark → multiple entries in fail_reasons."""
    # Low throughput + high cold TTFT.
    low_sweep = [
        _make_concurrency_point(
            concurrency=c,
            throughput=MIN_THROUGHPUT_CONCURRENCY_1 - 5.0,
        )
        for c in [1, 2, 4, 8, 16, 32, 64]
    ]
    bad_phase1 = _make_phase(1, ttft_samples=[MAX_COLD_CACHE_TTFT_P95 + 2.0] * 20)
    result = _good_evaluate(phase1=bad_phase1, concurrency_sweep=low_sweep)
    assert result.verdict == BenchmarkVerdict.FAIL
    assert FailReason.LOW_THROUGHPUT in result.fail_reasons
    assert FailReason.HIGH_COLD_TTFT in result.fail_reasons
    assert len(result.fail_reasons) >= 2


def test_summary_metrics_populated():
    """cold_cache_baseline_ttft_p95 and peak_throughput are populated correctly."""
    phase1_samples = [1.0, 2.0, 3.0]
    phase1 = _make_phase(1, ttft_samples=phase1_samples)
    sweep = [
        _make_concurrency_point(concurrency=c, throughput=float(c * 10))
        for c in [1, 2, 4, 8, 16, 32, 64]
    ]
    result = _good_evaluate(phase1=phase1, concurrency_sweep=sweep)
    # cold_cache_baseline_ttft_p95 should be P95 of phase1 samples.
    expected_p95 = compute_ttft_percentile(phase1_samples, 95)
    assert abs(result.cold_cache_baseline_ttft_p95 - expected_p95) < 1e-9
    # peak throughput should be from the highest-concurrency point (64 * 10 = 640).
    assert result.peak_throughput_tokens_per_sec == 640.0
    # warm_cache_ttft_p95_at_concurrency_1 is from sweep[0].
    assert result.warm_cache_ttft_p95_at_concurrency_1 == sweep[0].ttft_p95_sec
