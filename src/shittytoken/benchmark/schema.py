from dataclasses import dataclass, field
from enum import Enum


class BenchmarkVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"


class FailReason(str, Enum):
    LOW_THROUGHPUT = "low_throughput"
    HIGH_COLD_TTFT = "high_cold_cache_ttft"
    LOW_CACHE_HIT_RATE = "low_prefix_cache_hit_rate"
    DELTANET_CACHE_SUSPECT = "deltanet_prefix_caching_suspect"
    HIGH_ERROR_RATE = "high_failed_request_rate"


@dataclass
class ConcurrencyPoint:
    concurrency: int
    ttft_p50_sec: float
    ttft_p95_sec: float
    throughput_tokens_per_sec: float
    prefix_cache_hit_rate: float
    failed_request_count: int
    total_request_count: int


@dataclass
class PhaseMetrics:
    phase_number: int
    duration_sec: float
    ttft_samples: list[float]  # float('inf') for failed requests
    cache_hit_rate_timeseries: list[tuple[float, float]]
    kv_cache_usage_timeseries: list[tuple[float, float]]
    failed_request_count: int
    total_request_count: int


@dataclass
class BenchmarkResult:
    worker_url: str
    model_id: str
    gpu_model: str
    started_at: float
    completed_at: float
    verdict: BenchmarkVerdict
    fail_reasons: list[FailReason]
    deltanet_cache_suspect: bool
    phase_metrics: list[PhaseMetrics]
    concurrency_sweep: list[ConcurrencyPoint]
    cold_cache_baseline_ttft_p95: float
    warm_cache_ttft_p95_at_concurrency_1: float
    peak_throughput_tokens_per_sec: float
    raw_config: dict
