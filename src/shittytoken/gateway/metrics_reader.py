"""
metrics_reader — scrape vLLM Router Prometheus metrics at port 29000.

Designed to be called from the Orchestration Agent to read aggregate gateway
metrics (e.g. num_requests_running, routing weights) without depending on
any third-party Prometheus client library.
"""

import math

import aiohttp

ROUTER_METRICS_URL = "http://localhost:29000/metrics"


async def read_router_metrics(session: aiohttp.ClientSession) -> dict[str, float]:
    """
    Scrape the vLLM Router Prometheus metrics endpoint at port 29000.

    Returns a dict mapping metric name (without label blocks) to float value.
    Returns an empty dict on any network or parse error — never raises.
    """
    try:
        async with session.get(
            ROUTER_METRICS_URL,
            timeout=aiohttp.ClientTimeout(total=10.0),
        ) as resp:
            if resp.status != 200:
                return {}
            text = await resp.text()
            return _parse_prometheus_text(text)
    except Exception:  # noqa: BLE001 — callers expect {} on any failure
        return {}


def _parse_prometheus_text(text: str) -> dict[str, float]:
    """
    Minimal Prometheus text-format parser.  Uses stdlib only.

    Rules:
    - Lines starting with '#' are comments — skipped.
    - Blank lines are skipped.
    - Line format: metric_name[{labels}] value [timestamp]
    - Label blocks are stripped; only the base metric name is kept.
    - Non-numeric values (NaN, Inf, etc.) are skipped.
    - On duplicate names, the last occurrence wins.
    """
    result: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        raw_name = parts[0]
        raw_value = parts[1]
        # Strip optional label block: name{k="v",...} → name
        brace = raw_name.find("{")
        name = raw_name[:brace] if brace != -1 else raw_name
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if math.isfinite(value):
            result[name] = value
    return result
