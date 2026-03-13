"""
Spot eviction monitor — background task for detecting provider-level evictions.

Currently supports RunPod spot instances (INTERRUPTABLE pods). Vast.ai uses
on-demand instances, so eviction monitoring is not applicable.

This module is provider-agnostic in interface but the eviction detection
logic is delegated to the provider implementation.
"""

from __future__ import annotations

import asyncio
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


async def spot_eviction_monitor(
    *,
    provider: GPUProvider,
    instances: dict,
    idle_since: dict,
    last_running: dict,
    heartbeat_monitor: HeartbeatMonitor,
    gateway: GatewayClient,
    shutdown_event: asyncio.Event,
    provision_lock: asyncio.Lock,
    on_reprovision: callable,
    poll_interval_sec: int | None = None,
) -> None:
    """Background task that polls provider status to detect spot evictions.

    RunPod spot instances get SIGTERM with only 5s grace — no advance
    warning. This monitor detects eviction by checking if the pod
    status transitioned to EXITED while podType is INTERRUPTABLE.
    """
    if poll_interval_sec is None:
        poll_interval_sec = (
            cfg.get("orchestrator", {})
            .get("runpod", {})
            .get("spot_poll_interval_sec", 10)
        )

    logger.info("spot_eviction_monitor_started", poll_interval_sec=poll_interval_sec)

    while not shutdown_event.is_set():
        try:
            await asyncio.sleep(poll_interval_sec)

            for sm in list(instances.values()):
                if sm.record.provider != "runpod":
                    continue
                if sm.state not in (InstanceState.SERVING, InstanceState.BENCHMARKING):
                    continue

                pod_info = await provider.get_instance(sm.record.instance_id)
                if not pod_info:
                    continue

                if RunPodProvider.is_spot_eviction(pod_info):
                    logger.warning(
                        "spot_eviction_detected",
                        instance_id=sm.record.instance_id,
                        gpu_model=sm.record.gpu_model,
                    )
                    if sm.state == InstanceState.SERVING and sm.record.worker_url:
                        heartbeat_monitor.deregister(sm.record.worker_url)
                        try:
                            await gateway.deregister_worker(
                                sm.record.worker_url, drain=False
                            )
                        except KeyError:
                            pass

                    sm.transition(InstanceState.FAILED, reason="spot_eviction")
                    instances.pop(sm.record.instance_id, None)
                    idle_since.pop(sm.record.instance_id, None)
                    last_running.pop(sm.record.instance_id, None)

                    if not provision_lock.locked():
                        logger.info("spot_eviction_reprovision", instance_id=sm.record.instance_id)
                        asyncio.create_task(on_reprovision())

        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning("spot_eviction_monitor_error", error=str(exc))

    logger.info("spot_eviction_monitor_stopped")
