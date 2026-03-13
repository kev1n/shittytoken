"""
metrics_reader — scrape vLLM Router Prometheus metrics.

Designed to be called from the Orchestration Agent to read aggregate gateway
metrics (e.g. num_requests_running, routing weights) without depending on
any third-party Prometheus client library.
"""

import aiohttp

from shittytoken.common.prometheus import parse_prometheus_text
from shittytoken.config import cfg

_router_cfg = cfg["gateway"]["router"]
ROUTER_METRICS_URL = f"http://localhost:{_router_cfg['metrics_port']}/metrics"


async def read_router_metrics(session: aiohttp.ClientSession) -> dict[str, float]:
    """
    Scrape the vLLM Router Prometheus metrics endpoint.

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
            return parse_prometheus_text(text)
    except Exception:  # noqa: BLE001 — callers expect {} on any failure
        return {}
