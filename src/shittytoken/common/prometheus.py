"""Shared Prometheus text format parser. Single source of truth."""
import math
import re
import structlog

logger = structlog.get_logger()

_METRIC_LINE_RE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([^\s]+)')


def parse_prometheus_text(text: str) -> dict[str, float]:
    """
    Parse Prometheus text exposition format.
    Skips comment lines (# HELP, # TYPE).
    Skips non-finite values (NaN, Inf).
    Returns dict of metric_name -> float.
    """
    result: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE_RE.match(line)
        if not match:
            continue
        name = match.group(1)
        try:
            value = float(match.group(2))
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        # Counters (_total suffix) with label variants need summing;
        # gauges and histogram _sum/_count are single-valued so last-write is fine.
        if name.endswith("_total") and name in result:
            result[name] += value
        else:
            result[name] = value
    return result
