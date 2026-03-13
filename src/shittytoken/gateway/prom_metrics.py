"""
Prometheus metrics endpoint — lightweight counters for the custom router.

Uses simple module-level dicts/ints.  Thread-safety is not a concern
because the gateway runs as a single-process async application.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiohttp import web

if TYPE_CHECKING:
    from shittytoken.gateway.worker_pool import WorkerPool

# ---------------------------------------------------------------------------
# Module-level counters
# ---------------------------------------------------------------------------

_requests_total: dict[str, int] = {}  # key: "method:status" -> count
_requests_active: int = 0
_tokens_total: dict[str, int] = {"prompt": 0, "completion": 0}


# ---------------------------------------------------------------------------
# Counter helpers
# ---------------------------------------------------------------------------

def inc_request(method: str, status: int) -> None:
    """Increment the completed-request counter for *method*:*status*."""
    key = f"{method}:{status}"
    _requests_total[key] = _requests_total.get(key, 0) + 1


def inc_active() -> None:
    """Increment the active-request gauge."""
    global _requests_active
    _requests_active += 1


def dec_active() -> None:
    """Decrement the active-request gauge."""
    global _requests_active
    _requests_active -= 1


def add_tokens(prompt: int, completion: int) -> None:
    """Add token counts to the running totals."""
    _tokens_total["prompt"] += prompt
    _tokens_total["completion"] += completion


# ---------------------------------------------------------------------------
# Prometheus text exposition
# ---------------------------------------------------------------------------

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
    lines.append(
        f'shittytoken_tokens_total{{type="completion"}} {_tokens_total["completion"]}'
    )

    # -- Aggregate worker stats from pool ----------------------------------
    pool: WorkerPool | None = request.app.get("worker_pool")
    total_running = 0
    if pool is not None:
        workers = pool.list_workers()

        lines.append("# HELP num_requests_running Total requests running across workers.")
        lines.append("# TYPE num_requests_running gauge")
        for w in workers:
            total_running += w.requests_running
        lines.append(f"num_requests_running {total_running}")

        lines.append("# HELP num_requests_waiting Requests waiting in queue (always 0, we don't queue).")
        lines.append("# TYPE num_requests_waiting gauge")
        lines.append("num_requests_waiting 0")

        # -- Per-worker health ---------------------------------------------
        lines.append("# HELP shittytoken_worker_health Worker health status (1=healthy, 0=unhealthy).")
        lines.append("# TYPE shittytoken_worker_health gauge")
        for w in workers:
            health_val = 1 if w.healthy else 0
            lines.append(f'shittytoken_worker_health{{url="{w.url}"}} {health_val}')
    else:
        lines.append("# HELP num_requests_running Total requests running across workers.")
        lines.append("# TYPE num_requests_running gauge")
        lines.append("num_requests_running 0")
        lines.append("# HELP num_requests_waiting Requests waiting in queue.")
        lines.append("# TYPE num_requests_waiting gauge")
        lines.append("num_requests_waiting 0")

    body = "\n".join(lines) + "\n"
    return web.Response(text=body, content_type="text/plain; version=0.0.4")
