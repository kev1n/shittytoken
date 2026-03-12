"""
Gateway package — vLLM Router process management, nginx config rendering,
worker pool registry, and Prometheus metrics scraping.
"""

from shittytoken.gateway.nginx_config import render_nginx_config, write_nginx_config
from shittytoken.gateway.router_manager import RouterManager
from shittytoken.gateway.worker_registry import WorkerRegistry, WorkerEntry, CircuitBreakerState
from shittytoken.gateway.metrics_reader import read_router_metrics

__all__ = [
    "render_nginx_config",
    "write_nginx_config",
    "RouterManager",
    "WorkerRegistry",
    "WorkerEntry",
    "CircuitBreakerState",
    "read_router_metrics",
]
