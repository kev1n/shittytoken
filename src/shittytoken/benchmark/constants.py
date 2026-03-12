# Throughput gate (tokens/sec at concurrency=1)
MIN_THROUGHPUT_CONCURRENCY_1: float = 20.0

# TTFT gates (seconds)
MAX_COLD_CACHE_TTFT_P95: float = 8.0
MAX_WARM_CACHE_TTFT_P95: float = 4.0

# Cache behavior
MIN_PREFIX_CACHE_HIT_RATE_PHASE3: float = 0.30
DELTANET_CACHE_FLAG_THRESHOLD: float = 0.01

# Reliability
MAX_FAILED_REQUEST_RATE: float = 0.02

# Benchmark shape
METRICS_SCRAPE_INTERVAL_SEC: float = 2.0
WARM_UP_REQUESTS_PER_PREFIX: int = 7
CONCURRENCY_LEVELS: list[int] = [1, 2, 4, 8, 16, 32, 64]

# Timeouts
REQUEST_TIMEOUT_SEC: float = 120.0
FIRST_TOKEN_TIMEOUT_SEC: float = 30.0

# Virtual users
RETURNING_USER_FRACTION: float = 0.80
MAX_SESSIONS: int = 200
