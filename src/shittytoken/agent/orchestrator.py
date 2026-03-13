"""
Orchestrator — main demand-detect and scaling control loop.

Lifecycle:
1. Startup: restore any previously registered workers from KnowledgeGraph.
2. Main loop (every metrics_poll_interval_s):
   a. Scrape aggregate metrics from all SERVING workers.
   b. Scale up if total_requests_waiting > scale_up_waiting_threshold.
   c. Scale down if a worker has been idle (0 requests) for scale_down_idle_seconds
      AND avg_kv_cache_usage < scale_down_cache_max.
3. SIGTERM: drain all workers, terminate all instances, exit cleanly.

Scale-up flow:
  provision → verify GPU → monitor startup → run benchmark → register with gateway

Scale-down flow:
  identify least-utilized worker → deregister (drain=True) → destroy instance
"""

from __future__ import annotations

import asyncio
import signal
import time
from typing import Optional

import aiohttp
import structlog

from ..config import Settings, cfg
from ..knowledge.client import KnowledgeGraph
from .gateway import GatewayClient
from .health import HeartbeatMonitor
from .metrics import AggregateMetrics, aggregate_metrics, scrape_worker_metrics
from .provisioner import GPUProvider, get_provider
from .qualification import provision_and_qualify
from .spot_monitor import spot_eviction_monitor
from .ssh import SSHManager
from .cost_tracker import CostTracker
from .state_machine import InstanceRecord, InstanceState, InstanceStateMachine
from .state_store import RedisStateStore

logger = structlog.get_logger()


class Orchestrator:
    """
    Autonomous GPU instance lifecycle manager.

    The orchestrator is the SOLE writer to KnowledgeGraph.
    All gateway mutations go through GatewayClient.
    """

    def __init__(
        self,
        settings: Settings,
        kg: KnowledgeGraph,
        gateway: GatewayClient,
        approval_fn=None,
        state_store: RedisStateStore | None = None,
    ) -> None:
        self._settings = settings
        self._kg = kg
        self._gateway = gateway
        self._approval_fn = approval_fn
        self._state_store = state_store

        # instance_id → InstanceStateMachine
        self._instances: dict[str, InstanceStateMachine] = {}
        # instance_id → last_idle timestamp (monotonic)
        self._idle_since: dict[str, float] = {}
        # instance_id → last known running-request count
        self._last_running: dict[str, int] = {}

        self._cost_tracker = CostTracker()

        self._provision_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._session: Optional[aiohttp.ClientSession] = None
        self._provider: Optional[GPUProvider] = None
        self._ssh_manager: Optional[SSHManager] = None
        self._heartbeat_monitor: Optional[HeartbeatMonitor] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._spot_monitor_task: Optional[asyncio.Task] = None
        self._provision_cooldown_until: float = 0.0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main control loop. Runs until SIGTERM or _shutdown_event is set."""
        self._session = aiohttp.ClientSession()
        self._ssh_manager = SSHManager(
            private_key_path=self._settings.ssh_private_key_path,
            keepalive_interval=cfg["ssh"]["keepalive_interval"],
        )
        self._heartbeat_monitor = HeartbeatMonitor(
            session=self._session,
            health_check_interval_s=cfg["orchestrator"]["health_check_interval_s"],
            failure_threshold=cfg["orchestrator"].get("health_failure_threshold", 3),
            on_failure=self._on_worker_failure,
        )
        self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor.run())

        # Start instance death monitor for all providers
        self._spot_monitor_task = asyncio.create_task(
            spot_eviction_monitor(
                provider=self._get_provider(),
                instances=self._instances,
                idle_since=self._idle_since,
                last_running=self._last_running,
                heartbeat_monitor=self._heartbeat_monitor,
                gateway=self._gateway,
                shutdown_event=self._shutdown_event,
                provision_lock=self._provision_lock,
                on_reprovision=self._guarded_provision,
                on_state_delete=self._delete_state,
                on_cost_deregister=lambda iid: self._cost_tracker.deregister(iid),
            )
        )

        loop = asyncio.get_running_loop()
        loop.add_signal_handler(
            signal.SIGTERM,
            lambda: asyncio.create_task(self._shutdown()),
        )

        # Recover instances from previous run
        if self._state_store:
            await self._recover_instances()

        logger.info("orchestrator_started")

        try:
            while not self._shutdown_event.is_set():
                await self._tick()
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=cfg["orchestrator"]["metrics_poll_interval_s"],
                    )
                except asyncio.TimeoutError:
                    pass
        finally:
            await self._cleanup()

    # ------------------------------------------------------------------
    # Main loop tick
    # ------------------------------------------------------------------

    async def _tick(self) -> None:
        """One iteration of the demand-detect and scaling loop."""
        serving_urls = [
            sm.record.worker_url
            for sm in self._instances.values()
            if sm.state == InstanceState.SERVING and sm.record.worker_url
        ]

        min_workers = cfg["orchestrator"].get("min_workers", 1)
        provisioning = self._provision_lock.locked()

        if not serving_urls:
            states = {}
            for sm in self._instances.values():
                s = sm.state.value
                states[s] = states.get(s, 0) + 1
            logger.info(
                "orchestrator_tick",
                serving_workers=0,
                instances=states if states else None,
            )
            now = time.monotonic()
            if now < self._provision_cooldown_until:
                remaining = round(self._provision_cooldown_until - now)
                logger.info("provision_cooldown", remaining_s=remaining)
            elif not provisioning:
                logger.info("min_workers_provision", min_workers=min_workers)
                asyncio.create_task(self._guarded_provision())
            await self._sweep_stuck_instances()
            self._cost_tracker.maybe_log_summary()
            return

        metrics = await aggregate_metrics(serving_urls, self._session)
        logger.info(
            "orchestrator_tick",
            serving_workers=metrics.worker_count,
            total_running=metrics.total_requests_running,
            total_waiting=metrics.total_requests_waiting,
            avg_kv_cache=round(metrics.avg_kv_cache_usage, 3),
        )

        # Scrape per-worker metrics concurrently
        serving_sms = [
            sm for sm in self._instances.values()
            if sm.state == InstanceState.SERVING and sm.record.worker_url
        ]
        if serving_sms:
            scrape_results = await asyncio.gather(
                *(scrape_worker_metrics(sm.record.worker_url, self._session) for sm in serving_sms),
                return_exceptions=True,
            )
            for sm, result in zip(serving_sms, scrape_results):
                if isinstance(result, Exception):
                    continue
                running = int(result.get("num_requests_running", 0))
                self._last_running[sm.record.instance_id] = running

        # Update idle tracking per instance
        now = time.monotonic()
        for sm in self._instances.values():
            if sm.state != InstanceState.SERVING:
                continue
            iid = sm.record.instance_id
            running = self._last_running.get(iid, 1)
            if running == 0:
                self._idle_since.setdefault(iid, now)
            else:
                self._idle_since.pop(iid, None)

        # Scale up
        _orch = cfg["orchestrator"]
        if metrics.total_requests_waiting > _orch["scale_up_waiting_threshold"]:
            if not self._provision_lock.locked():
                logger.info(
                    "scale_up_triggered",
                    waiting=metrics.total_requests_waiting,
                    threshold=_orch["scale_up_waiting_threshold"],
                )
                asyncio.create_task(self._guarded_provision())

        # Scale down (never below min_workers)
        elif len(serving_urls) > min_workers:
            await self._maybe_scale_down(metrics)

        await self._sweep_stuck_instances()
        self._cost_tracker.maybe_log_summary()

    # ------------------------------------------------------------------
    # Provisioning
    # ------------------------------------------------------------------

    async def _guarded_provision(self) -> None:
        """Serialize provisioning so only one scale-up runs at a time."""
        cooldown_s = cfg["orchestrator"].get("provision_cooldown_s", 60)
        async with self._provision_lock:
            try:
                record, sm = await provision_and_qualify(
                    kg=self._kg,
                    provider=self._get_provider(),
                    ssh_manager=self._ssh_manager,
                    session=self._session,
                    heartbeat_monitor=self._heartbeat_monitor,
                    gateway=self._gateway,
                    settings=self._settings,
                    approval_fn=self._approval_fn,
                )
                if record is not None and sm is not None:
                    self._instances[record.instance_id] = sm
                    await self._save_state(record)
                    if record.cost_per_hour_usd:
                        self._cost_tracker.register(record.instance_id, record.cost_per_hour_usd)
                elif sm is not None:
                    # Instance was created but qualification failed — destroy it
                    await self._destroy_instance(sm.record, sm)
                    self._provision_cooldown_until = time.monotonic() + cooldown_s
                    logger.info("provision_failed_cooldown", cooldown_s=cooldown_s)
                else:
                    self._provision_cooldown_until = time.monotonic() + cooldown_s
                    logger.info("provision_failed_cooldown", cooldown_s=cooldown_s)
            except Exception as exc:
                logger.error(
                    "provision_unhandled_exception",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                self._provision_cooldown_until = time.monotonic() + cooldown_s
                for sm in list(self._instances.values()):
                    if sm.state in (InstanceState.PROVISIONING, InstanceState.BENCHMARKING):
                        logger.info("provision_cleanup_orphan", instance_id=sm.record.instance_id, state=sm.state.value)
                        await self._destroy_instance(sm.record, sm)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    async def _save_state(self, record: InstanceRecord) -> None:
        """Persist instance record to Redis if a state store is configured."""
        if self._state_store:
            await self._state_store.save(record)

    async def _delete_state(self, instance_id: str) -> None:
        """Remove instance record from Redis if a state store is configured."""
        if self._state_store:
            await self._state_store.delete(instance_id)

    async def _recover_instances(self) -> None:
        """Reload instance records from Redis and recover or clean up each one."""
        records = await self._state_store.load_all()
        if not records:
            logger.info("recovery_no_instances")
            return

        logger.info("recovery_starting", count=len(records))
        provider = self._get_provider()

        # Verify all instances concurrently
        poll_results = await asyncio.gather(
            *(provider.get_instance(r.instance_id) for r in records),
            return_exceptions=True,
        )

        for record, result in zip(records, poll_results):
            if isinstance(result, Exception):
                instance_alive = False
                instance_info = None
            else:
                instance_info = result
                instance_alive = bool(instance_info)

            if (
                instance_alive
                and record.state == InstanceState.SERVING
                and record.worker_url
            ):
                # Verify the vLLM process is actually responding
                try:
                    async with self._session.get(
                        record.worker_url.rstrip("/") + "/health",
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        worker_healthy = resp.status == 200
                except Exception:
                    worker_healthy = False

                if worker_healthy:
                    # Recover: re-create state machine, register with monitors
                    sm = InstanceStateMachine(record)
                    self._instances[record.instance_id] = sm
                    self._heartbeat_monitor.register(record.worker_url)
                    await self._gateway.register_worker(record.worker_url)
                    if record.cost_per_hour_usd:
                        self._cost_tracker.register(record.instance_id, record.cost_per_hour_usd)
                    logger.info(
                        "recovery_instance_restored",
                        instance_id=record.instance_id,
                        worker_url=record.worker_url,
                    )
                    continue

                # Instance VM is alive but vLLM crashed — treat as dead
                logger.info(
                    "recovery_worker_unhealthy",
                    instance_id=record.instance_id,
                    worker_url=record.worker_url,
                )

            # Dead, in-progress, or unhealthy instance — clean up
            logger.info(
                "recovery_instance_cleanup",
                instance_id=record.instance_id,
                state=record.state.value,
                alive=instance_alive,
            )
            try:
                await provider.destroy_instance(record.instance_id)
            except Exception as exc:
                logger.warning(
                    "recovery_destroy_failed",
                    instance_id=record.instance_id,
                    error=str(exc),
                )
            await self._state_store.delete(record.instance_id)

    # ------------------------------------------------------------------
    # Stuck instance sweep
    # ------------------------------------------------------------------

    async def _sweep_stuck_instances(self) -> None:
        """Destroy instances stuck in PROVISIONING or BENCHMARKING too long."""
        stuck_timeout = cfg["orchestrator"].get("stuck_instance_timeout_s", 1320)
        now = time.time()

        for sm in list(self._instances.values()):
            if sm.state not in (InstanceState.PROVISIONING, InstanceState.BENCHMARKING):
                continue
            elapsed = now - sm.record.state_changed_at
            if elapsed > stuck_timeout:
                logger.warning(
                    "stuck_instance_sweep",
                    instance_id=sm.record.instance_id,
                    state=sm.state.value,
                    elapsed_s=round(elapsed, 1),
                    timeout_s=stuck_timeout,
                )
                sm.transition(InstanceState.FAILED, reason="stuck_timeout")
                await self._destroy_instance(sm.record, sm)

    # ------------------------------------------------------------------
    # Scale down
    # ------------------------------------------------------------------

    async def _maybe_scale_down(self, metrics: AggregateMetrics) -> None:
        """Scale down if a worker has been idle long enough and KV cache is low."""
        now = time.monotonic()
        idle_threshold = cfg["orchestrator"]["scale_down_idle_seconds"]

        for sm in list(self._instances.values()):
            if sm.state != InstanceState.SERVING:
                continue
            iid = sm.record.instance_id
            idle_start = self._idle_since.get(iid)
            if idle_start is None:
                continue
            idle_secs = now - idle_start
            if (
                idle_secs >= idle_threshold
                and metrics.avg_kv_cache_usage < cfg["orchestrator"]["scale_down_cache_max"]
            ):
                logger.info(
                    "scale_down_triggered",
                    instance_id=iid,
                    idle_secs=round(idle_secs, 1),
                    avg_kv_cache=round(metrics.avg_kv_cache_usage, 3),
                )
                await self._scale_down_one()
                return

    async def _scale_down_one(self) -> None:
        """Select the least-utilized SERVING worker and deregister + destroy it."""
        candidates = [
            sm for sm in self._instances.values()
            if sm.state == InstanceState.SERVING and sm.record.worker_url
        ]
        if not candidates:
            logger.info("scale_down_no_candidates")
            return

        target_sm = min(
            candidates,
            key=lambda sm: self._last_running.get(sm.record.instance_id, 0),
        )
        record = target_sm.record

        logger.info(
            "scale_down_selected",
            instance_id=record.instance_id,
            worker_url=record.worker_url,
        )

        target_sm.transition(InstanceState.DRAINING, reason="scale_down")
        self._heartbeat_monitor.deregister(record.worker_url)

        try:
            await self._gateway.deregister_worker(record.worker_url, drain=True)
        except KeyError:
            logger.warning("scale_down_deregister_not_found", url=record.worker_url)

        runtime_hours = (time.time() - record.created_at) / 3600.0
        await self._kg.write_final_instance_metrics(
            config_id=record.config_id,
            runtime_hours=runtime_hours,
            total_tokens=0,
            cost_usd=0.0,
        )

        await self._destroy_instance(record, target_sm)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def _shutdown(self) -> None:
        """Drain all SERVING workers, terminate all instances, close KG."""
        logger.info("orchestrator_shutdown_initiated")
        self._shutdown_event.set()

        for sm in list(self._instances.values()):
            if sm.state == InstanceState.SERVING:
                record = sm.record
                sm.transition(InstanceState.DRAINING, reason="sigterm_shutdown")
                if record.worker_url:
                    self._heartbeat_monitor.deregister(record.worker_url)
                    try:
                        await self._gateway.deregister_worker(record.worker_url, drain=True)
                    except KeyError:
                        pass

        for sm in list(self._instances.values()):
            if sm.state not in (InstanceState.TERMINATED, InstanceState.FAILED):
                await self._destroy_instance(sm.record, sm)

        await self._cleanup()
        logger.info("orchestrator_shutdown_complete")

    async def _cleanup(self) -> None:
        """Close background tasks and connections."""
        if self._heartbeat_monitor is not None:
            self._heartbeat_monitor.stop()
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        if self._spot_monitor_task is not None:
            self._spot_monitor_task.cancel()
            try:
                await self._spot_monitor_task
            except asyncio.CancelledError:
                pass
        if self._session is not None:
            await self._session.close()
        await self._kg.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_provider(self) -> GPUProvider:
        """Return the cached GPU provider instance."""
        if self._provider is None:
            provider_name = cfg["orchestrator"].get("provider", "vastai")
            self._provider = get_provider(
                provider_name=provider_name,
                vastai_api_key=self._settings.vastai_api_key,
                runpod_api_key=self._settings.runpod_api_key,
                session=self._session,
            )
        return self._provider

    async def _destroy_instance(
        self,
        record,
        sm: InstanceStateMachine,
    ) -> None:
        """Destroy the cloud instance and mark it TERMINATED (or FAILED)."""
        try:
            provider = self._get_provider()
            await provider.destroy_instance(record.instance_id)
        except Exception as exc:
            logger.error(
                "destroy_instance_failed",
                instance_id=record.instance_id,
                provider=record.provider,
                error=str(exc),
            )

        self._cost_tracker.deregister(record.instance_id)

        if sm.state not in (InstanceState.TERMINATED, InstanceState.FAILED):
            sm.transition(InstanceState.TERMINATED, reason="instance_destroyed")

        self._instances.pop(record.instance_id, None)
        self._idle_since.pop(record.instance_id, None)
        self._last_running.pop(record.instance_id, None)
        await self._delete_state(record.instance_id)

    async def _on_worker_failure(self, url: str) -> None:
        """Callback from HeartbeatMonitor when a worker fails health checks."""
        logger.warning("worker_health_failure", url=url)
        for sm in list(self._instances.values()):
            if sm.record.worker_url == url and sm.state == InstanceState.SERVING:
                sm.transition(InstanceState.DRAINING, reason="health_check_failure")
                self._heartbeat_monitor.deregister(url)
                try:
                    await self._gateway.deregister_worker(url, drain=True)
                except KeyError:
                    pass
                await self._destroy_instance(sm.record, sm)
                break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def interactive_approval(plan) -> bool:
    """Default HITL approval: prints the deployment plan and asks for confirmation."""
    print(plan.display())

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None, lambda: input("\n  Approve this deployment? [y/N] ").strip().lower()
    )
    approved = response in ("y", "yes")
    if not approved:
        print("  Deployment rejected.\n")
    return approved


async def main() -> None:
    """
    Process entry point.

    Reads settings from environment / .env, creates KnowledgeGraph and
    Orchestrator, runs the control loop.

    HITL approval is enabled by default. Set SHITTYTOKEN_AUTO_APPROVE=1
    to skip approval prompts.
    """
    import os

    from ..log import configure_logging
    from ..gateway.router_manager import RouterManager
    from ..gateway.worker_registry import WorkerRegistry

    configure_logging()
    log = structlog.get_logger()

    settings = Settings()
    log.info("orchestrator_main_starting", neo4j_uri=settings.neo4j_uri)

    kg = KnowledgeGraph(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    await kg.verify_connectivity()

    router_manager = RouterManager()
    registry = WorkerRegistry(router_manager=router_manager)
    gateway = GatewayClient(registry=registry)

    # State store for instance recovery across restarts
    state_store = None
    redis_url = cfg["orchestrator"].get("redis_url")
    if redis_url:
        state_store = await RedisStateStore.create(url=redis_url)
        log.info("state_store_enabled", redis_url=redis_url)

    auto_approve = os.getenv("SHITTYTOKEN_AUTO_APPROVE", "").strip() in ("1", "true")
    approval_fn = None if auto_approve else interactive_approval

    orchestrator = Orchestrator(
        settings=settings,
        kg=kg,
        gateway=gateway,
        approval_fn=approval_fn,
        state_store=state_store,
    )
    try:
        await orchestrator.run()
    finally:
        if state_store:
            await state_store.close()


if __name__ == "__main__":
    asyncio.run(main())
