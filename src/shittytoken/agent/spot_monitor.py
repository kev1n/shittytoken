"""
Instance death monitor — background task for detecting provider-level instance death.

Supports all providers:
- RunPod: detects spot evictions (INTERRUPTABLE pods with EXITED status)
- Vast.ai: detects instances whose actual_status leaves "running"/"loading"
- Generic fallback: treats empty instance info as dead

This module is provider-agnostic in interface but the death detection
logic is delegated to provider-specific checks via _check_instance_dead().
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import structlog

from ..config import cfg
from .provisioner import RunPodProvider
from .state_machine import InstanceState

if TYPE_CHECKING:
    from .provisioner import GPUProvider
    from .health import HeartbeatMonitor
    from .gateway import GatewayClient

logger = structlog.get_logger()


def _check_instance_dead(provider_name: str, instance_info: dict) -> bool:
    """Check if an instance is dead based on provider-specific status."""
    if not instance_info:
        return True
    if provider_name == "runpod":
        return RunPodProvider.is_spot_eviction(instance_info)
    if provider_name == "vastai":
        status = instance_info.get("actual_status", "")
        return status not in ("running", "loading", "")
    return False


async def instance_death_monitor(
    *,
    provider: GPUProvider,
    instances: dict,
    snapshots: dict,
    heartbeat_monitor: HeartbeatMonitor,
    gateway: GatewayClient,
    shutdown_event: asyncio.Event,
    provision_lock: asyncio.Lock,
    on_reprovision: Callable[[], Awaitable[None]],
    on_state_delete: Callable[[str], Awaitable[None]] | None = None,
    on_cost_deregister: Callable[[str], None] | None = None,
    poll_interval_sec: int | None = None,
) -> None:
    """Background task that polls provider status to detect instance death.

    For RunPod spot instances, detects eviction by checking pod status.
    For Vast.ai, detects instances that have exited or errored.
    For any provider, treats unreachable instances (empty info) as dead.
    """
    if poll_interval_sec is None:
        poll_interval_sec = (
            cfg.get("orchestrator", {})
            .get("instance_poll_interval_sec", 15)
        )

    logger.info("instance_death_monitor_started", poll_interval_sec=poll_interval_sec)

    while not shutdown_event.is_set():
        try:
            await asyncio.sleep(poll_interval_sec)

            # Gather all provider polls concurrently
            candidates = [
                sm for sm in list(instances.values())
                if sm.state in (InstanceState.SERVING, InstanceState.BENCHMARKING)
            ]
            if not candidates:
                continue

            poll_results = await asyncio.gather(
                *(provider.get_instance(sm.record.instance_id) for sm in candidates),
                return_exceptions=True,
            )

            for sm, result in zip(candidates, poll_results):
                if isinstance(result, Exception):
                    logger.warning(
                        "instance_monitor_poll_error",
                        instance_id=sm.record.instance_id,
                        error=str(result),
                    )
                    continue

                is_dead = _check_instance_dead(sm.record.provider, result)
                if is_dead:
                    logger.warning(
                        "instance_death_detected",
                        instance_id=sm.record.instance_id,
                        gpu_model=sm.record.gpu_model,
                        provider=sm.record.provider,
                    )
                    if sm.state == InstanceState.SERVING and sm.record.worker_url:
                        heartbeat_monitor.deregister(sm.record.worker_url)
                        try:
                            # drain=False: instance is dead, can't drain in-flight requests
                            await gateway.deregister_worker(
                                sm.record.worker_url, drain=False
                            )
                        except KeyError:
                            pass

                    sm.transition(InstanceState.FAILED, reason="instance_death")
                    instances.pop(sm.record.instance_id, None)
                    snapshots.pop(sm.record.instance_id, None)

                    if on_state_delete is not None:
                        result = on_state_delete(sm.record.instance_id)
                        if inspect.isawaitable(result):
                            await result
                    if on_cost_deregister is not None:
                        result = on_cost_deregister(sm.record.instance_id)
                        if inspect.isawaitable(result):
                            await result

                    if not provision_lock.locked():
                        logger.info("instance_death_reprovision", instance_id=sm.record.instance_id)
                        asyncio.create_task(on_reprovision())

        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning("instance_death_monitor_error", error=str(exc))

    logger.info("instance_death_monitor_stopped")


# Backward-compatible alias
spot_eviction_monitor = instance_death_monitor
