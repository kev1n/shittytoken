"""
GatewayClient — the orchestrator's interface to the gateway subsystem.

Wraps WorkerRegistry so the orchestrator never imports gateway internals
directly.  All gateway mutations go through this class.
"""

from __future__ import annotations

import structlog

from ..gateway.worker_registry import WorkerRegistry

logger = structlog.get_logger()


class GatewayClient:
    """
    Thin adapter between the orchestration agent and WorkerRegistry.

    The orchestrator calls only this class; it never touches WorkerRegistry
    or RouterManager directly.
    """

    def __init__(self, registry: WorkerRegistry) -> None:
        self._registry = registry

    async def register_worker(self, url: str) -> None:
        """
        Add *url* to the active worker pool.

        Logs {"event": "worker_registered", "url": url} on success.
        Raises ValueError (from WorkerRegistry) if the URL is already registered.
        """
        await self._registry.add_worker(url)
        logger.info("worker_registered", url=url)

    async def deregister_worker(self, url: str, drain: bool = True) -> None:
        """
        Remove *url* from the active worker pool.

        drain=True (default) waits for in-flight requests to complete before
        removing the worker from the router.

        Logs {"event": "worker_deregistered", "url": url} on success.
        Raises KeyError (from WorkerRegistry) if the URL is not registered.
        """
        await self._registry.remove_worker(url, drain=drain)
        logger.info("worker_deregistered", url=url, drain=drain)

    def list_workers(self) -> list:
        """Return a snapshot of the current worker pool (list of WorkerEntry)."""
        return self._registry.list_workers()
