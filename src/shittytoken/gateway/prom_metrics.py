"""
Prometheus metrics endpoint — counters, gauges, and histograms for the gateway.

Uses simple module-level dicts/ints.  Thread-safety is not a concern
because the gateway runs as a single-process async application.
"""

from __future__ import annotations

import collections
import time
from typing import TYPE_CHECKING

from aiohttp import web

if TYPE_CHECKING:
    from shittytoken.gateway.worker_pool import WorkerPool

# ---------------------------------------------------------------------------
# Module-level counters / gauges
# ---------------------------------------------------------------------------

_requests_total: dict[str, int] = {}  # key: "method:status" -> count
_requests_active: int = 0
_tokens_total: dict[str, int] = {"prompt": 0, "completion": 0, "cached": 0}

# Latency histogram buckets (seconds)
_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, float("inf"))
_latency_bucket_counts: list[int] = [0] * len(_LATENCY_BUCKETS)
_latency_sum: float = 0.0
_latency_count: int = 0

# Gateway overhead histogram (time spent in gateway before first upstream byte)
_overhead_bucket_counts: list[int] = [0] * len(_LATENCY_BUCKETS)
_overhead_sum: float = 0.0
_overhead_count: int = 0

# TTFT histogram (time to first token from upstream, excludes gateway overhead)
_TTFT_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, float("inf"))
_ttft_bucket_counts: list[int] = [0] * len(_TTFT_BUCKETS)
_ttft_sum: float = 0.0
_ttft_count: int = 0
# Rolling window for p95 computation (last 200 observations)
_ttft_window: collections.deque[float] = collections.deque(maxlen=200)

# Per-worker request counters
_worker_requests_total: dict[str, int] = {}  # url -> count

# Orchestrator scaling events
_scale_events: dict[str, int] = {}  # "scale_up" / "scale_down" -> count
_instances_total: dict[str, int] = {}  # state -> count (provisioning, benchmarking, serving, destroyed)

# Cost metrics (pushed from orchestrator)
_hourly_burn_usd: float = 0.0
_cumulative_cost_usd: float = 0.0


# ---------------------------------------------------------------------------
# Counter helpers
# ---------------------------------------------------------------------------

def inc_request(method: str, status: int) -> None:
    key = f"{method}:{status}"
    _requests_total[key] = _requests_total.get(key, 0) + 1


def inc_active() -> None:
    global _requests_active
    _requests_active += 1


def dec_active() -> None:
    global _requests_active
    _requests_active = max(0, _requests_active - 1)


def add_tokens(prompt: int, completion: int, cached: int = 0) -> None:
    _tokens_total["prompt"] += prompt
    _tokens_total["completion"] += completion
    _tokens_total["cached"] += cached


def observe_latency(duration_sec: float) -> None:
    """Record a request's total duration in the latency histogram."""
    global _latency_sum, _latency_count
    _latency_sum += duration_sec
    _latency_count += 1
    for i, boundary in enumerate(_LATENCY_BUCKETS):
        if duration_sec <= boundary:
            _latency_bucket_counts[i] += 1
            break


def observe_ttft(duration_sec: float) -> None:
    """Record time-to-first-token in the TTFT histogram and rolling window."""
    global _ttft_sum, _ttft_count
    _ttft_sum += duration_sec
    _ttft_count += 1
    for i, boundary in enumerate(_TTFT_BUCKETS):
        if duration_sec <= boundary:
            _ttft_bucket_counts[i] += 1
            break
    _ttft_window.append(duration_sec)


def get_ttft_p95() -> float | None:
    """Return the p95 TTFT from the rolling window, or None if no data."""
    if not _ttft_window:
        return None
    sorted_vals = sorted(_ttft_window)
    idx = int(len(sorted_vals) * 0.95)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def observe_overhead(duration_sec: float) -> None:
    """Record gateway overhead (auth + routing, before upstream)."""
    global _overhead_sum, _overhead_count
    _overhead_sum += duration_sec
    _overhead_count += 1
    for i, boundary in enumerate(_LATENCY_BUCKETS):
        if duration_sec <= boundary:
            _overhead_bucket_counts[i] += 1
            break


def inc_worker_request(worker_url: str) -> None:
    """Increment per-worker request counter."""
    _worker_requests_total[worker_url] = _worker_requests_total.get(worker_url, 0) + 1


def record_scale_event(event_type: str) -> None:
    """Record a scale_up or scale_down event."""
    _scale_events[event_type] = _scale_events.get(event_type, 0) + 1


def set_instance_counts(counts: dict[str, int]) -> None:
    """Update instance state counts from orchestrator."""
    global _instances_total
    _instances_total = dict(counts)


def set_cost_metrics(hourly_burn_usd: float, cumulative_cost_usd: float) -> None:
    """Update cost metrics pushed from orchestrator."""
    global _hourly_burn_usd, _cumulative_cost_usd
    _hourly_burn_usd = hourly_burn_usd
    _cumulative_cost_usd = cumulative_cost_usd


# ---------------------------------------------------------------------------
# Prometheus text exposition
# ---------------------------------------------------------------------------

def _histogram_lines(name: str, help_text: str, buckets: tuple, counts: list[int], total_sum: float, total_count: int) -> list[str]:
    """Generate Prometheus histogram lines."""
    lines = [
        f"# HELP {name} {help_text}",
        f"# TYPE {name} histogram",
    ]
    cumulative = 0
    for i, boundary in enumerate(buckets):
        cumulative += counts[i]
        le = f"+Inf" if boundary == float("inf") else f"{boundary}"
        lines.append(f'{name}_bucket{{le="{le}"}} {cumulative}')
    lines.append(f"{name}_sum {total_sum:.6f}")
    lines.append(f"{name}_count {total_count}")
    return lines


async def handle_metrics(request: web.Request) -> web.Response:
    """GET /metrics -> Prometheus text format."""
    lines: list[str] = []

    # -- shittytoken_requests_total ----------------------------------------
    lines.append("# HELP shittytoken_requests_total Total completed HTTP requests.")
    lines.append("# TYPE shittytoken_requests_total counter")
    for key, count in sorted(_requests_total.items()):
        method, status = key.split(":", 1)
        lines.append(
            f'shittytoken_requests_total{{method="{method}",status="{status}"}} {count}'
        )

    # -- shittytoken_requests_active ---------------------------------------
    lines.append("# HELP shittytoken_requests_active Currently in-flight requests.")
    lines.append("# TYPE shittytoken_requests_active gauge")
    lines.append(f"shittytoken_requests_active {_requests_active}")

    # -- shittytoken_tokens_total ------------------------------------------
    lines.append("# HELP shittytoken_tokens_total Total tokens processed.")
    lines.append("# TYPE shittytoken_tokens_total counter")
    lines.append(
        f'shittytoken_tokens_total{{type="prompt"}} {_tokens_total["prompt"]}'
    )
    uncached = _tokens_total["prompt"] - _tokens_total["cached"]
    lines.append(
        f'shittytoken_tokens_total{{type="uncached"}} {uncached}'
    )
    lines.append(
        f'shittytoken_tokens_total{{type="cached"}} {_tokens_total["cached"]}'
    )
    lines.append(
        f'shittytoken_tokens_total{{type="completion"}} {_tokens_total["completion"]}'
    )

    # -- shittytoken_request_duration_seconds (histogram) ------------------
    lines.extend(_histogram_lines(
        "shittytoken_request_duration_seconds",
        "Total request duration from client perspective (seconds).",
        _LATENCY_BUCKETS, _latency_bucket_counts, _latency_sum, _latency_count,
    ))

    # -- shittytoken_gateway_overhead_seconds (histogram) ------------------
    lines.extend(_histogram_lines(
        "shittytoken_gateway_overhead_seconds",
        "Gateway overhead before first upstream byte (auth + routing, seconds).",
        _LATENCY_BUCKETS, _overhead_bucket_counts, _overhead_sum, _overhead_count,
    ))

    # -- shittytoken_ttft_seconds (histogram) --------------------------------
    lines.extend(_histogram_lines(
        "shittytoken_ttft_seconds",
        "Time to first token from upstream (seconds, excludes gateway overhead).",
        _TTFT_BUCKETS, _ttft_bucket_counts, _ttft_sum, _ttft_count,
    ))

    # -- Per-worker metrics from pool --------------------------------------
    pool: WorkerPool | None = request.app.get("worker_pool")
    total_running = 0
    total_waiting = 0
    if pool is not None:
        workers = pool.list_workers()

        # Per-worker requests running
        lines.append("# HELP shittytoken_worker_requests_running Requests currently running on each worker.")
        lines.append("# TYPE shittytoken_worker_requests_running gauge")
        for w in workers:
            total_running += w.requests_running
            total_waiting += w.requests_waiting
            lines.append(f'shittytoken_worker_requests_running{{url="{w.url}"}} {w.requests_running}')

        # Aggregate
        lines.append("# HELP num_requests_running Total requests running across all workers.")
        lines.append("# TYPE num_requests_running gauge")
        lines.append(f"num_requests_running {total_running}")
        lines.append("# HELP num_requests_waiting Requests waiting in queue.")
        lines.append("# TYPE num_requests_waiting gauge")
        lines.append(f"num_requests_waiting {total_waiting}")

        # Per-worker KV cache usage
        lines.append("# HELP shittytoken_worker_kv_cache_pct KV cache usage percentage per worker.")
        lines.append("# TYPE shittytoken_worker_kv_cache_pct gauge")
        for w in workers:
            lines.append(f'shittytoken_worker_kv_cache_pct{{url="{w.url}"}} {w.kv_cache_pct:.4f}')

        # Per-worker health
        lines.append("# HELP shittytoken_worker_health Worker health status (1=healthy, 0=unhealthy).")
        lines.append("# TYPE shittytoken_worker_health gauge")
        for w in workers:
            health_val = 1 if w.healthy else 0
            lines.append(f'shittytoken_worker_health{{url="{w.url}"}} {health_val}')

        # Worker count (healthy only — this is what matters for capacity)
        healthy_count = sum(1 for w in workers if w.healthy)
        lines.append("# HELP shittytoken_workers_total Number of healthy workers.")
        lines.append("# TYPE shittytoken_workers_total gauge")
        lines.append(f"shittytoken_workers_total {healthy_count}")

        # Prefix cache (scraped from workers)
        total_cache_hits = sum(w.prefix_cache_hits for w in workers)
        total_cache_queries = sum(w.prefix_cache_queries for w in workers)
        lines.append("# HELP shittytoken_prefix_cache_hits_total Prefix cache hits (tokens).")
        lines.append("# TYPE shittytoken_prefix_cache_hits_total counter")
        lines.append(f"shittytoken_prefix_cache_hits_total {total_cache_hits}")
        lines.append("# HELP shittytoken_prefix_cache_queries_total Prefix cache queries (tokens).")
        lines.append("# TYPE shittytoken_prefix_cache_queries_total counter")
        lines.append(f"shittytoken_prefix_cache_queries_total {total_cache_queries}")

        # Per-worker request totals
        lines.append("# HELP shittytoken_worker_requests_total Total requests routed to each worker.")
        lines.append("# TYPE shittytoken_worker_requests_total counter")
        for url, count in sorted(_worker_requests_total.items()):
            lines.append(f'shittytoken_worker_requests_total{{url="{url}"}} {count}')

        # -- vLLM engine metrics (scraped from workers, re-exposed as aggregates) --

        # Aggregate token counters (for throughput via rate())
        total_prompt_tokens = sum(w.prompt_tokens_total for w in workers)
        total_gen_tokens = sum(w.generation_tokens_total for w in workers)
        lines.append("# HELP shittytoken_vllm_prompt_tokens_total Total prompt tokens processed by vLLM.")
        lines.append("# TYPE shittytoken_vllm_prompt_tokens_total counter")
        lines.append(f"shittytoken_vllm_prompt_tokens_total {total_prompt_tokens}")
        lines.append("# HELP shittytoken_vllm_generation_tokens_total Total generation tokens produced by vLLM.")
        lines.append("# TYPE shittytoken_vllm_generation_tokens_total counter")
        lines.append(f"shittytoken_vllm_generation_tokens_total {total_gen_tokens}")

        # Aggregate preemptions (memory pressure indicator)
        total_preemptions = sum(w.preemptions_total for w in workers)
        lines.append("# HELP shittytoken_vllm_preemptions_total Total KV cache preemption events.")
        lines.append("# TYPE shittytoken_vllm_preemptions_total counter")
        lines.append(f"shittytoken_vllm_preemptions_total {total_preemptions}")

        # Aggregate completed requests
        total_success = sum(w.request_success_total for w in workers)
        lines.append("# HELP shittytoken_vllm_request_success_total Total completed requests in vLLM.")
        lines.append("# TYPE shittytoken_vllm_request_success_total counter")
        lines.append(f"shittytoken_vllm_request_success_total {total_success}")

        # Aggregate TTFT (from vLLM engine, more accurate than gateway measurement)
        total_ttft_sum = sum(w.ttft_sum for w in workers)
        total_ttft_count = sum(w.ttft_count for w in workers)
        lines.append("# HELP shittytoken_vllm_ttft_seconds_sum Sum of vLLM engine TTFT across all workers.")
        lines.append("# TYPE shittytoken_vllm_ttft_seconds_sum counter")
        lines.append(f"shittytoken_vllm_ttft_seconds_sum {total_ttft_sum:.6f}")
        lines.append("# HELP shittytoken_vllm_ttft_seconds_count Count of vLLM engine TTFT observations.")
        lines.append("# TYPE shittytoken_vllm_ttft_seconds_count counter")
        lines.append(f"shittytoken_vllm_ttft_seconds_count {total_ttft_count}")

        # Aggregate ITL (inter-token latency — decode speed)
        total_itl_sum = sum(w.itl_sum for w in workers)
        total_itl_count = sum(w.itl_count for w in workers)
        lines.append("# HELP shittytoken_vllm_itl_seconds_sum Sum of vLLM inter-token latency across all workers.")
        lines.append("# TYPE shittytoken_vllm_itl_seconds_sum counter")
        lines.append(f"shittytoken_vllm_itl_seconds_sum {total_itl_sum:.6f}")
        lines.append("# HELP shittytoken_vllm_itl_seconds_count Count of vLLM ITL observations.")
        lines.append("# TYPE shittytoken_vllm_itl_seconds_count counter")
        lines.append(f"shittytoken_vllm_itl_seconds_count {total_itl_count}")

        # Aggregate queue time
        total_queue_sum = sum(w.queue_time_sum for w in workers)
        total_queue_count = sum(w.queue_time_count for w in workers)
        lines.append("# HELP shittytoken_vllm_queue_time_seconds_sum Sum of vLLM queue wait time.")
        lines.append("# TYPE shittytoken_vllm_queue_time_seconds_sum counter")
        lines.append(f"shittytoken_vllm_queue_time_seconds_sum {total_queue_sum:.6f}")
        lines.append("# HELP shittytoken_vllm_queue_time_seconds_count Count of vLLM queue time observations.")
        lines.append("# TYPE shittytoken_vllm_queue_time_seconds_count counter")
        lines.append(f"shittytoken_vllm_queue_time_seconds_count {total_queue_count}")
    else:
        lines.append("# HELP num_requests_running Total requests running across workers.")
        lines.append("# TYPE num_requests_running gauge")
        lines.append("num_requests_running 0")
        lines.append("# HELP num_requests_waiting Requests waiting in queue.")
        lines.append("# TYPE num_requests_waiting gauge")
        lines.append("num_requests_waiting 0")

    # -- Orchestrator / cost metrics ---------------------------------------
    lines.append("# HELP shittytoken_hourly_burn_usd Current hourly burn rate in USD.")
    lines.append("# TYPE shittytoken_hourly_burn_usd gauge")
    lines.append(f"shittytoken_hourly_burn_usd {_hourly_burn_usd:.4f}")
    lines.append("# HELP shittytoken_cumulative_cost_usd Total cumulative cost in USD.")
    lines.append("# TYPE shittytoken_cumulative_cost_usd gauge")
    lines.append(f"shittytoken_cumulative_cost_usd {_cumulative_cost_usd:.4f}")

    # -- Instance state counts (from orchestrator) -------------------------
    if _instances_total:
        lines.append("# HELP shittytoken_instances Instances by state.")
        lines.append("# TYPE shittytoken_instances gauge")
        for state, count in sorted(_instances_total.items()):
            lines.append(f'shittytoken_instances{{state="{state}"}} {count}')

    # -- Scaling events ----------------------------------------------------
    if _scale_events:
        lines.append("# HELP shittytoken_scale_events_total Orchestrator scaling events.")
        lines.append("# TYPE shittytoken_scale_events_total counter")
        for event_type, count in sorted(_scale_events.items()):
            lines.append(f'shittytoken_scale_events_total{{type="{event_type}"}} {count}')

    body = "\n".join(lines) + "\n"
    return web.Response(text=body, content_type="text/plain; version=0.0.4")
