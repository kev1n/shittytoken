"""
Orchestrator — main demand-detect and scaling control loop.

Lifecycle:
1. Startup: restore any previously registered workers from KnowledgeGraph.
2. Main loop (every metrics_poll_interval_s):
   a. Scrape aggregate metrics from all SERVING workers.
   b. Scale up if all workers are saturated, preemptions spike, or queue time is high.
   c. Scale down if a specific worker has been idle long enough and its cache is cold.
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


# ---------------------------------------------------------------------------
# Per-worker metrics snapshot (replaces N separate dicts)
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field as dc_field


@dataclass
class WorkerSnapshot:
    """All per-worker metrics the orchestrator tracks between ticks.

    One instance per serving worker, keyed by instance_id in
    ``Orchestrator._snapshots``.  Adding a new metric means adding a
    field here — no extra dicts, no extra cleanup code.
    """
    # Gauges (latest value)
    requests_running: int = 0
    kv_cache_pct: float = 0.0

    # Counters (cumulative — use deltas between ticks for rates)
    prefix_cache_hits: float = 0.0
    prefix_cache_queries: float = 0.0
    preemptions_total: float = 0.0
    prompt_tokens_total: float = 0.0
    generation_tokens_total: float = 0.0

    # Histogram sum/count pairs (cumulative — delta gives per-interval avg)
    ttft_sum: float = 0.0
    ttft_count: float = 0.0
    itl_sum: float = 0.0
    itl_count: float = 0.0
    queue_time_sum: float = 0.0
    queue_time_count: float = 0.0

    # Idle tracking
    idle_since: float | None = None  # monotonic timestamp when became idle


# ---------------------------------------------------------------------------
# Data-driven scale-up triggers
# ---------------------------------------------------------------------------

@dataclass
class ScaleTrigger:
    """A single scale-up condition, evaluated each tick."""
    name: str
    check: object  # Callable[[TickContext], bool]
    sustain_ticks: int = 1  # how many consecutive True ticks before firing
    _consecutive: int = dc_field(default=0, repr=False)

    def evaluate(self, ctx: "TickContext") -> bool:
        if self.check(ctx):
            self._consecutive += 1
        else:
            self._consecutive = 0
        return self._consecutive >= self.sustain_ticks

    def reset(self) -> None:
        self._consecutive = 0


@dataclass
class TickContext:
    """All data available to scale triggers on a given tick."""
    metrics: AggregateMetrics
    preemptions_delta: float
    gen_tokens_delta: float
    avg_queue_time_delta: float  # avg queue wait (seconds) this tick across workers
    min_requests_waiting: int    # min requests_waiting across all serving workers
    cfg: dict  # orchestrator config section


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

        # Consolidated per-worker metrics: instance_id → WorkerSnapshot
        self._snapshots: dict[str, WorkerSnapshot] = {}
        # Previous tick's snapshots for delta computation
        self._prev_snapshots: dict[str, WorkerSnapshot] = {}

        self._cost_tracker = CostTracker()

        self._provision_lock = asyncio.Lock()
        self._provision_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._session: Optional[aiohttp.ClientSession] = None
        self._provider: Optional[GPUProvider] = None
        self._ssh_manager: Optional[SSHManager] = None
        self._heartbeat_monitor: Optional[HeartbeatMonitor] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._spot_monitor_task: Optional[asyncio.Task] = None
        self._provision_cooldown_until: float = 0.0
        self._scale_events: dict[str, int] = {"scale_up": 0, "scale_down": 0}

        # Workers that failed their last metrics scrape
        self._unreachable_workers: set[str] = set()

        # Hysteresis: timestamp of last scale-up completion
        self._last_scale_up_at: float = 0.0

        # Reference to the gateway's worker pool for health checks
        self._worker_pool = None  # set in run() if available

        # Data-driven scale-up triggers — add new ones here
        self._scale_triggers: list[ScaleTrigger] = [
            ScaleTrigger(
                name="all_workers_saturated",
                # Every worker has waiting requests AND is near capacity
                # (running >= threshold per worker). max_num_seqs is 16,
                # optimal throughput around 10 concurrent requests.
                check=lambda ctx: (
                    ctx.min_requests_waiting > 0
                    and ctx.metrics.total_requests_running >= ctx.metrics.worker_count * ctx.cfg.get("saturation_running_threshold", 10)
                ),
                sustain_ticks=2,
            ),
            ScaleTrigger(
                name="preemptions",
                check=lambda ctx: ctx.preemptions_delta > ctx.cfg.get("scale_up_preemption_threshold", 2),
            ),
            ScaleTrigger(
                name="queue_time",
                check=lambda ctx: ctx.avg_queue_time_delta > ctx.cfg.get("scale_up_queue_time_threshold_s", 2.0),
                sustain_ticks=2,  # sustained high queue wait, not just one long prompt
            ),
        ]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main control loop. Runs until SIGTERM or _shutdown_event is set."""
        self._session = aiohttp.ClientSession()
        # Share session with worker registry for drain polling
        if hasattr(self._gateway, '_registry'):
            self._gateway._registry._session = self._session
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
                snapshots=self._snapshots,
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
                self._provision_task = asyncio.create_task(self._guarded_provision())
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

        # ── Update per-worker snapshots ──────────────────────────────────
        self._unreachable_workers.clear()
        self._prev_snapshots = dict(self._snapshots)  # save for delta computation
        url_to_iid: dict[str, str] = {}

        if metrics.per_worker:
            url_to_iid = {
                sm.record.worker_url: sm.record.instance_id
                for sm in self._instances.values()
                if sm.state == InstanceState.SERVING and sm.record.worker_url
            }
            for wm in metrics.per_worker:
                if not wm.reachable:
                    self._unreachable_workers.add(wm.url)
                iid = url_to_iid.get(wm.url)
                if iid:
                    snap = self._snapshots.get(iid, WorkerSnapshot())
                    snap.requests_running = wm.requests_running
                    snap.kv_cache_pct = wm.kv_cache_pct
                    snap.prefix_cache_hits = wm.prefix_cache_hits
                    snap.prefix_cache_queries = wm.prefix_cache_queries
                    snap.preemptions_total = wm.preemptions_total
                    snap.prompt_tokens_total = wm.prompt_tokens_total
                    snap.generation_tokens_total = wm.generation_tokens_total
                    snap.ttft_sum = wm.ttft_sum
                    snap.ttft_count = wm.ttft_count
                    snap.itl_sum = wm.itl_sum
                    snap.itl_count = wm.itl_count
                    snap.queue_time_sum = wm.queue_time_sum
                    snap.queue_time_count = wm.queue_time_count
                    self._snapshots[iid] = snap

        # ── Compute aggregate deltas ─────────────────────────────────────
        total_preemptions_delta = 0.0
        total_gen_tokens_delta = 0.0
        queue_time_delta_sum = 0.0
        queue_time_delta_count = 0.0
        for iid, snap in self._snapshots.items():
            prev = self._prev_snapshots.get(iid)
            if prev is None:
                continue
            if snap.preemptions_total > prev.preemptions_total:
                total_preemptions_delta += snap.preemptions_total - prev.preemptions_total
            if snap.generation_tokens_total > prev.generation_tokens_total:
                total_gen_tokens_delta += snap.generation_tokens_total - prev.generation_tokens_total
            # Queue time delta: compute avg queue wait for requests completed this tick
            qt_sum_delta = snap.queue_time_sum - prev.queue_time_sum
            qt_count_delta = snap.queue_time_count - prev.queue_time_count
            if qt_count_delta > 0 and qt_sum_delta >= 0:
                queue_time_delta_sum += qt_sum_delta
                queue_time_delta_count += qt_count_delta
        avg_queue_time_delta = queue_time_delta_sum / queue_time_delta_count if queue_time_delta_count > 0 else 0.0

        # Min requests_waiting across all serving workers (for all_workers_saturated trigger)
        min_requests_waiting = 0
        if metrics.per_worker:
            waiting_values = [wm.requests_waiting for wm in metrics.per_worker if wm.reachable]
            min_requests_waiting = min(waiting_values) if waiting_values else 0

        # ── Update idle tracking ─────────────────────────────────────────
        now = time.monotonic()
        for sm in self._instances.values():
            if sm.state != InstanceState.SERVING:
                continue
            iid = sm.record.instance_id
            snap = self._snapshots.get(iid)
            running = snap.requests_running if snap else 1
            if running == 0:
                if snap and snap.idle_since is None:
                    snap.idle_since = now
            elif snap:
                snap.idle_since = None

        # ── Scale up — data-driven triggers ──────────────────────────────
        _orch = cfg["orchestrator"]
        max_workers = _orch.get("max_workers", 10)

        ctx = TickContext(
            metrics=metrics,
            preemptions_delta=total_preemptions_delta,
            gen_tokens_delta=total_gen_tokens_delta,
            avg_queue_time_delta=avg_queue_time_delta,
            min_requests_waiting=min_requests_waiting,
            cfg=_orch,
        )

        fired_triggers = [t for t in self._scale_triggers if t.evaluate(ctx)]

        # Check if any workers are unhealthy (recovering from crash).
        unhealthy_workers = self._get_unhealthy_workers()
        if unhealthy_workers:
            logger.info(
                "scale_up_deferred_unhealthy",
                unhealthy=unhealthy_workers,
                waiting=metrics.total_requests_waiting,
            )
            fired_triggers = []

        if fired_triggers and len(serving_urls) < max_workers:
            if not self._provision_lock.locked():
                reason = fired_triggers[0].name  # first firing trigger is the reason
                logger.info(
                    "scale_up_triggered",
                    reason=reason,
                    all_triggers=[t.name for t in fired_triggers],
                    min_waiting=min_requests_waiting,
                    total_waiting=metrics.total_requests_waiting,
                    preemptions_delta=total_preemptions_delta,
                    avg_queue_time=round(avg_queue_time_delta, 3),
                )
                self._scale_events["scale_up"] += 1
                self._provision_task = asyncio.create_task(self._guarded_provision())

        # Scale down (never below min_workers)
        elif len(serving_urls) > min_workers:
            await self._maybe_scale_down(metrics)

        await self._sweep_stuck_instances()
        await self._maybe_evict_volumes()
        self._cost_tracker.maybe_log_summary()
        await self._push_metrics_to_gateway()

    def _get_unhealthy_workers(self) -> list[str]:
        """Return URLs of SERVING workers that are currently unreachable.

        Checks two signals:
        1. Heartbeat monitor consecutive failures (health endpoint down)
        2. Metrics scrape reachability (from last _tick's aggregate_metrics)
        """
        unhealthy = []
        for sm in self._instances.values():
            if sm.state != InstanceState.SERVING or not sm.record.worker_url:
                continue
            url = sm.record.worker_url
            # Check heartbeat failures
            if self._heartbeat_monitor is not None:
                failures = self._heartbeat_monitor.get_consecutive_failures(url)
                if failures >= 1:
                    unhealthy.append(url)
                    continue
            # Check last metrics scrape reachability
            if url in self._unreachable_workers:
                unhealthy.append(url)
        return unhealthy

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
                    self._last_scale_up_at = time.monotonic()
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
        """Reload instance records from Redis, scan provider for orphans, recover or clean up."""
        records = await self._state_store.load_all()
        provider = self._get_provider()
        known_ids = {r.instance_id for r in records}

        # Scan provider API for ALL running instances — catch orphans not in Redis
        try:
            all_provider_instances = await provider.list_all_instances()
        except Exception as exc:
            logger.warning("recovery_provider_scan_failed", error=str(exc))
            all_provider_instances = []

        orphan_ids = {
            str(inst.get("id"))
            for inst in all_provider_instances
            if str(inst.get("id")) not in known_ids
        }
        if orphan_ids:
            logger.warning(
                "recovery_orphans_found",
                count=len(orphan_ids),
                instance_ids=list(orphan_ids),
            )
            for orphan_id in orphan_ids:
                try:
                    await provider.destroy_instance(orphan_id)
                    logger.info("recovery_orphan_destroyed", instance_id=orphan_id)
                except Exception as exc:
                    logger.warning("recovery_orphan_destroy_failed", instance_id=orphan_id, error=str(exc))

        if not records:
            logger.info("recovery_no_instances")
            return

        logger.info("recovery_starting", count=len(records))

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
    # Metrics push
    # ------------------------------------------------------------------

    async def _push_metrics_to_gateway(self) -> None:
        """Push orchestrator state to the gateway's /admin/metrics endpoint."""
        counts: dict[str, int] = {}
        for sm in self._instances.values():
            s = sm.state.value
            counts[s] = counts.get(s, 0) + 1
        payload = {
            "instances": counts,
            "scale_events": dict(self._scale_events),
            "cost": {
                "hourly_burn_usd": self._cost_tracker.hourly_burn_usd,
                "cumulative_cost_usd": self._cost_tracker.cumulative_cost_usd,
            },
        }
        try:
            from ..gateway.router_manager import ROUTER_PORT
            url = f"http://localhost:{ROUTER_PORT}/admin/metrics"
            async with self._session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=2.0)) as resp:
                if resp.status != 200:
                    logger.debug("push_metrics_failed", status=resp.status)
        except Exception:
            pass  # Non-critical — don't spam logs

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

    # Volume eviction
    # ------------------------------------------------------------------

    async def _maybe_evict_volumes(self) -> None:
        """Periodically evict stale volumes (once per hour)."""
        vol_cfg = cfg.get("orchestrator", {}).get("volume_cache", {})
        if not vol_cfg.get("enabled", False):
            return

        now = time.monotonic()
        if now - getattr(self, "_last_volume_eviction", 0) < 3600:
            return
        self._last_volume_eviction = now

        eviction_days = vol_cfg.get("eviction_days", 3)
        provider = self._get_provider()
        if hasattr(provider, "evict_stale_volumes"):
            await provider.evict_stale_volumes(eviction_days)

    # ------------------------------------------------------------------
    # Scale down
    # ------------------------------------------------------------------

    async def _maybe_scale_down(self, metrics: AggregateMetrics) -> None:
        """Scale down if a worker has been idle long enough and its KV cache is low.

        Uses per-worker cache metrics (not global average) and hysteresis
        to prevent flapping with recent scale-ups.
        """
        now = time.monotonic()
        idle_threshold = cfg["orchestrator"]["scale_down_idle_seconds"]
        cache_max = cfg["orchestrator"]["scale_down_cache_max"]

        # Hysteresis: don't scale down within 5 min of last scale-up
        if now < self._last_scale_up_at + 300:
            return

        candidates = []
        for sm in list(self._instances.values()):
            if sm.state != InstanceState.SERVING:
                continue
            iid = sm.record.instance_id
            snap = self._snapshots.get(iid)
            if snap is None or snap.idle_since is None:
                continue
            idle_secs = now - snap.idle_since
            if idle_secs < idle_threshold:
                continue
            if snap.kv_cache_pct < cache_max:
                candidates.append((sm, idle_secs, snap))

        if not candidates:
            return

        # Pick the least-valuable worker: lowest prefix cache hit count, then fewest requests
        target_sm, idle_secs, snap = min(
            candidates,
            key=lambda t: (t[2].prefix_cache_hits, t[2].requests_running),
        )
        logger.info(
            "scale_down_triggered",
            instance_id=target_sm.record.instance_id,
            idle_secs=round(idle_secs, 1),
            worker_kv_cache=round(snap.kv_cache_pct, 3),
            cache_hits=snap.prefix_cache_hits,
        )
        self._scale_events["scale_down"] += 1
        await self._scale_down_one(target_sm)

    async def _scale_down_one(self, target_sm: InstanceStateMachine) -> None:
        """Deregister and destroy the selected worker."""
        record = target_sm.record

        if target_sm.state != InstanceState.SERVING or not record.worker_url:
            logger.info("scale_down_no_candidates")
            return

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

        # Cancel any in-flight provision to prevent orphaned instances
        if self._provision_task and not self._provision_task.done():
            self._provision_task.cancel()
            try:
                await self._provision_task
            except (asyncio.CancelledError, Exception):
                pass
            logger.info("orchestrator_provision_cancelled")

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
        self._snapshots.pop(record.instance_id, None)
        self._prev_snapshots.pop(record.instance_id, None)
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

    run_dir = configure_logging(component="orchestrator")
    log = structlog.get_logger()

    settings = Settings()
    log.info("orchestrator_main_starting", neo4j_uri=settings.neo4j_uri, run_dir=str(run_dir))

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
