"""
Benchmark constants — all loaded from config.yml.
"""

from ..config import benchmark_cfg

_b = benchmark_cfg()

# Throughput gate (tokens/sec at concurrency=1)
MIN_THROUGHPUT_CONCURRENCY_1: float = _b["min_throughput_tps"]

# TTFT gates (seconds)
MAX_COLD_CACHE_TTFT_P95: float = _b["max_cold_cache_ttft_p95"]
MAX_WARM_CACHE_TTFT_P95: float = _b["max_warm_cache_ttft_p95"]

# Cache behavior
MIN_PREFIX_CACHE_HIT_RATE_PHASE3: float = _b["min_prefix_cache_hit_rate_phase3"]
DELTANET_CACHE_FLAG_THRESHOLD: float = _b["deltanet_cache_flag_threshold"]

# Reliability
MAX_FAILED_REQUEST_RATE: float = _b["max_failed_request_rate"]

# Benchmark shape
METRICS_SCRAPE_INTERVAL_SEC: float = _b["metrics_scrape_interval_sec"]
WARM_UP_REQUESTS_PER_PREFIX: int = _b["warm_up_requests_per_prefix"]
CONCURRENCY_LEVELS: list[int] = _b["concurrency_levels"]

# Timeouts
REQUEST_TIMEOUT_SEC: float = _b["request_timeout_sec"]
FIRST_TOKEN_TIMEOUT_SEC: float = _b["first_token_timeout_sec"]

# Virtual users
RETURNING_USER_FRACTION: float = _b["returning_user_fraction"]
MAX_SESSIONS: int = _b["max_sessions"]

# Long-context agent flow (phase 4)
_lc = _b.get("long_context", {})
LONG_CONTEXT_ENABLED: bool = _lc.get("enabled", False)
LONG_CONTEXT_STEPS_TOKENS: list[int] = _lc.get(
    "context_steps_tokens", [8192, 16384, 32768, 65536, 131072]
)
LONG_CONTEXT_OUTPUT_TOKENS: int = _lc.get("output_tokens_per_step", 512)
LONG_CONTEXT_REQUEST_TIMEOUT_SEC: float = _lc.get("request_timeout_sec", 300.0)
