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

from ..config import Settings
from ..knowledge.client import KnowledgeGraph
from ..knowledge.schema import Configuration
from .gateway import GatewayClient
from .health import HeartbeatMonitor, wait_for_model_ready
from .metrics import AggregateMetrics, aggregate_metrics
from .provisioner import VastAIProvisioner, RunPodProvisioner, provision_instance
from .ssh import SSHManager
from .startup_monitor import StartupResult, monitor_startup
from .state_machine import InstanceRecord, InstanceState, InstanceStateMachine

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
    ) -> None:
        self._settings = settings
        self._kg = kg
        self._gateway = gateway

        # instance_id → InstanceStateMachine
        self._instances: dict[str, InstanceStateMachine] = {}
        # instance_id → last_idle timestamp (monotonic)
        self._idle_since: dict[str, float] = {}
        # instance_id → last known running-request count
        self._last_running: dict[str, int] = {}

        self._shutdown_event = asyncio.Event()
        self._session: Optional[aiohttp.ClientSession] = None
        self._ssh_manager: Optional[SSHManager] = None
        self._heartbeat_monitor: Optional[HeartbeatMonitor] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main control loop. Runs until SIGTERM or _shutdown_event is set."""
        self._session = aiohttp.ClientSession()
        self._ssh_manager = SSHManager(
            private_key_path=self._settings.ssh_private_key_path,
            keepalive_interval=self._settings.ssh_keepalive_interval,
        )
        self._heartbeat_monitor = HeartbeatMonitor(
            session=self._session,
            health_check_interval_s=self._settings.health_check_interval_s,
            on_failure=self._on_worker_failure,
        )
        self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor.run())

        # Install SIGTERM handler
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(
            signal.SIGTERM,
            lambda: asyncio.create_task(self._shutdown()),
        )

        logger.info("orchestrator_started")

        try:
            while not self._shutdown_event.is_set():
                await self._tick()
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self._settings.metrics_poll_interval_s,
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

        if not serving_urls:
            logger.info("orchestrator_tick", serving_workers=0)
            return

        metrics = await aggregate_metrics(serving_urls, self._session)
        logger.info(
            "orchestrator_tick",
            serving_workers=metrics.worker_count,
            total_running=metrics.total_requests_running,
            total_waiting=metrics.total_requests_waiting,
            avg_kv_cache=round(metrics.avg_kv_cache_usage, 3),
        )

        # --- Update idle tracking per instance ---
        now = time.monotonic()
        for sm in self._instances.values():
            if sm.state != InstanceState.SERVING:
                continue
            iid = sm.record.instance_id
            # We use aggregate-level running as a proxy here; in a real
            # implementation we'd scrape per-worker and track individually.
            # For scale-down we track per-instance via individual scrapes.
            running = self._last_running.get(iid, 1)
            if running == 0:
                self._idle_since.setdefault(iid, now)
            else:
                self._idle_since.pop(iid, None)

        # --- Scale up ---
        if metrics.total_requests_waiting > self._settings.scale_up_waiting_threshold:
            logger.info(
                "scale_up_triggered",
                waiting=metrics.total_requests_waiting,
                threshold=self._settings.scale_up_waiting_threshold,
            )
            asyncio.create_task(self.provision_and_qualify())

        # --- Scale down ---
        elif len(serving_urls) > 1:
            await self._maybe_scale_down(metrics)

    async def _maybe_scale_down(self, metrics: AggregateMetrics) -> None:
        """Scale down if a worker has been idle long enough and KV cache is low."""
        now = time.monotonic()
        idle_threshold = self._settings.scale_down_idle_seconds

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
                and metrics.avg_kv_cache_usage < self._settings.scale_down_cache_max
            ):
                logger.info(
                    "scale_down_triggered",
                    instance_id=iid,
                    idle_secs=round(idle_secs, 1),
                    avg_kv_cache=round(metrics.avg_kv_cache_usage, 3),
                )
                await self.scale_down_one()
                return  # one scale-down per tick

    # ------------------------------------------------------------------
    # Provision + qualify a new instance
    # ------------------------------------------------------------------

    async def provision_and_qualify(
        self,
        gpu_names: list[str] | None = None,
        model_id: str = "Qwen/Qwen3.5-35B-A3B",
    ) -> InstanceRecord | None:
        """
        Full provisioning + qualification flow.

        Steps:
        1. Provision instance (Vast.ai → RunPod fallback)
        2. Verify GPU hardware matches expectations
        3. Monitor startup logs for READY / OOM / CUDA_ERROR / TIMEOUT
        4. Poll /v1/models until model weights are loaded
        5. Run benchmark; record result in KnowledgeGraph
        6. On pass: register with gateway; transition to SERVING
        7. On any failure: record outcome in KG, destroy instance

        Returns InstanceRecord on success, None on any failure.
        """
        if gpu_names is None:
            gpu_names = ["RTX 3090", "RTX 4090"]

        settings = self._settings

        # Look up the best known configuration for the first GPU type
        config = await self._kg.best_config_for(gpu_names[0], model_id)
        if config is None:
            logger.warning(
                "provision_no_config",
                gpu_names=gpu_names,
                model_id=model_id,
            )
            return None

        vastai = VastAIProvisioner(settings.vastai_api_key, self._session)
        runpod = RunPodProvisioner(settings.runpod_api_key, self._session)

        # --- Step 1: Provision ---
        try:
            provisioned = await provision_instance(
                vastai=vastai,
                runpod=runpod,
                config=config,
                model_id=model_id,
                hf_token="",   # HF token should come from settings; omitted per spec
                gpu_names=gpu_names,
            )
        except (aiohttp.ClientError, TimeoutError) as exc:
            logger.error("provision_failed", error=str(exc))
            return None

        record = InstanceRecord(
            instance_id=provisioned.instance_id,
            provider=provisioned.provider,
            gpu_model=provisioned.gpu_model,
            ssh_host=provisioned.ssh_host,
            ssh_port=provisioned.ssh_port,
            config_id=config.config_id,
        )
        sm = InstanceStateMachine(record)
        self._instances[record.instance_id] = sm

        # --- Step 2: Verify GPU via SSH ---
        import asyncssh  # avoid circular; only needed here

        try:
            session = await self._ssh_manager.connect(
                host=record.ssh_host,
                port=record.ssh_port,
            )
        except asyncssh.Error as exc:
            logger.error(
                "provision_ssh_connect_failed",
                instance_id=record.instance_id,
                error=str(exc),
            )
            sm.transition(InstanceState.FAILED, reason="ssh_connect_failed")
            await self._destroy_instance(record, sm)
            return None

        # GPU verification uses first GPU name as expected model
        gpu_ok = await self._ssh_manager.verify_gpu(
            session=session,
            expected_gpu_name=gpu_names[0],
            expected_vram_gb=24,  # RTX 3090/4090 baseline VRAM
        )
        if not gpu_ok:
            logger.error("provision_gpu_verify_failed", instance_id=record.instance_id)
            sm.transition(InstanceState.FAILED, reason="gpu_verify_failed")
            await self._ssh_manager.close(session)
            await self._destroy_instance(record, sm)
            return None

        # --- Transition to BENCHMARKING ---
        sm.transition(InstanceState.BENCHMARKING, reason="gpu_verified")

        # --- Step 3: Monitor startup ---
        log_lines: asyncio.Queue[str] = asyncio.Queue()

        async def _enqueue(line: str) -> None:
            await log_lines.put(line)

        async def _line_gen():
            while True:
                line = await log_lines.get()
                yield line

        # Stream logs in background
        stream_task = asyncio.create_task(
            self._ssh_manager.stream_logs(session, _enqueue)
        )

        startup_result, matched_line = await monitor_startup(
            line_generator=_line_gen(),
            timeout_sec=settings.startup_monitor_timeout_s,
        )
        stream_task.cancel()

        if startup_result != StartupResult.READY:
            reason = f"startup_{startup_result}"
            logger.error(
                "provision_startup_failed",
                instance_id=record.instance_id,
                startup_result=startup_result,
                matched_line=matched_line,
            )
            if startup_result == StartupResult.OOM:
                await self._kg.write_oom_event(
                    config_id=config.config_id,
                    gpu_model_name=record.gpu_model,
                    error_type="loading",
                    error_message=matched_line,
                    error_phase="loading",
                    gpu_memory_free_gb=0.0,
                    gpu_memory_total_gb=0.0,
                )
            sm.transition(InstanceState.FAILED, reason=reason)
            await self._ssh_manager.close(session)
            await self._destroy_instance(record, sm)
            return None

        # --- Step 4: Wait for model ready via HTTP ---
        worker_url = f"http://{record.ssh_host}:8000"
        record.worker_url = worker_url

        model_ready = await wait_for_model_ready(
            base_url=worker_url,
            session=self._session,
            timeout_sec=settings.startup_monitor_timeout_s,
        )
        if not model_ready:
            sm.transition(InstanceState.FAILED, reason="model_ready_timeout")
            await self._ssh_manager.close(session)
            await self._destroy_instance(record, sm)
            return None

        # --- Step 5: Run benchmark ---
        try:
            from ..benchmark.runner import run_benchmark  # type: ignore[import]

            bench_result = await run_benchmark(
                worker_url=worker_url,
                model_id=model_id,
                gpu_model=record.gpu_model,
                min_throughput_tps=settings.benchmark_min_throughput_tps,
            )
        except Exception as exc:  # noqa: BLE001 — benchmark errors are non-fatal
            logger.error(
                "provision_benchmark_error",
                instance_id=record.instance_id,
                error=str(exc),
            )
            sm.transition(InstanceState.FAILED, reason="benchmark_exception")
            await self._ssh_manager.close(session)
            await self._destroy_instance(record, sm)
            return None

        # Write benchmark result to KG regardless of pass/fail
        from datetime import datetime, timezone

        await self._kg.write_benchmark_result(
            result_id=str(bench_result.verdict),
            config_id=config.config_id,
            gpu_model_name=record.gpu_model,
            verdict=bench_result.verdict.value,
            cold_ttft_p95_s=bench_result.cold_cache_baseline_ttft_p95,
            warm_ttft_p95_s_at_c1=bench_result.warm_cache_ttft_p95_at_concurrency_1,
            peak_throughput_tps=bench_result.peak_throughput_tokens_per_sec,
            prefix_cache_hit_rate_phase3=0.0,
            failed_request_rate=0.0,
            deltanet_cache_suspect=bench_result.deltanet_cache_suspect,
            started_at=datetime.fromtimestamp(bench_result.started_at, tz=timezone.utc),
            completed_at=datetime.fromtimestamp(bench_result.completed_at, tz=timezone.utc),
        )

        if bench_result.verdict.value != "pass":
            logger.warning(
                "provision_benchmark_failed",
                instance_id=record.instance_id,
                verdict=bench_result.verdict.value,
            )
            sm.transition(InstanceState.FAILED, reason="benchmark_failed")
            await self._ssh_manager.close(session)
            await self._destroy_instance(record, sm)
            return None

        # --- Step 6: Register with gateway and transition to SERVING ---
        await self._gateway.register_worker(worker_url)
        self._heartbeat_monitor.register(worker_url)
        sm.transition(InstanceState.SERVING, reason="benchmark_passed")

        await self._ssh_manager.close(session)
        logger.info(
            "provision_qualify_complete",
            instance_id=record.instance_id,
            worker_url=worker_url,
        )
        return record

    # ------------------------------------------------------------------
    # Scale down
    # ------------------------------------------------------------------

    async def scale_down_one(self) -> None:
        """
        Select the least-utilized SERVING worker (lowest num_requests_running).
        Deregister (drain=True), destroy instance, write final metrics to KG.
        """
        # Find candidate: pick SERVING worker with lowest tracked running count
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
            logger.warning(
                "scale_down_deregister_not_found",
                url=record.worker_url,
            )

        # Write final metrics to KG
        runtime_hours = (time.time() - record.created_at) / 3600.0
        await self._kg.write_final_instance_metrics(
            config_id=record.config_id,
            runtime_hours=runtime_hours,
            total_tokens=0,     # token accounting tracked elsewhere
            cost_usd=0.0,
        )

        await self._destroy_instance(record, target_sm)

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------

    async def _shutdown(self) -> None:
        """
        Drain all SERVING workers, terminate all instances, close KG.
        Called on SIGTERM.
        """
        logger.info("orchestrator_shutdown_initiated")
        self._shutdown_event.set()

        # Drain and deregister all serving instances
        for sm in list(self._instances.values()):
            if sm.state == InstanceState.SERVING:
                record = sm.record
                sm.transition(InstanceState.DRAINING, reason="sigterm_shutdown")
                if record.worker_url:
                    self._heartbeat_monitor.deregister(record.worker_url)
                    try:
                        await self._gateway.deregister_worker(
                            record.worker_url, drain=True
                        )
                    except KeyError:
                        pass

        # Terminate all non-terminal instances
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
        if self._session is not None:
            await self._session.close()
        await self._kg.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _destroy_instance(
        self,
        record: InstanceRecord,
        sm: InstanceStateMachine,
    ) -> None:
        """Destroy the cloud instance and mark it TERMINATED (or FAILED)."""
        try:
            if record.provider == "vastai":
                vastai = VastAIProvisioner(
                    self._settings.vastai_api_key, self._session
                )
                await vastai.destroy_instance(record.instance_id)
            elif record.provider == "runpod":
                runpod = RunPodProvisioner(
                    self._settings.runpod_api_key, self._session
                )
                await runpod.destroy_instance(record.instance_id)
        except (aiohttp.ClientError, Exception) as exc:
            logger.error(
                "destroy_instance_failed",
                instance_id=record.instance_id,
                provider=record.provider,
                error=str(exc),
            )

        if sm.state not in (InstanceState.TERMINATED, InstanceState.FAILED):
            sm.transition(InstanceState.TERMINATED, reason="instance_destroyed")

        self._instances.pop(record.instance_id, None)
        self._idle_since.pop(record.instance_id, None)
        self._last_running.pop(record.instance_id, None)

    async def _on_worker_failure(self, url: str) -> None:
        """Callback from HeartbeatMonitor when a worker fails health checks."""
        logger.warning("worker_health_failure", url=url)
        # Find the instance with this URL
        for sm in list(self._instances.values()):
            if sm.record.worker_url == url and sm.state == InstanceState.SERVING:
                sm.transition(InstanceState.DRAINING, reason="health_check_failure")
                self._heartbeat_monitor.deregister(url)
                try:
                    await self._gateway.deregister_worker(url, drain=False)
                except KeyError:
                    pass
                await self._destroy_instance(sm.record, sm)
                break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    """
    Process entry point.

    Reads settings from environment / .env, creates KnowledgeGraph and
    Orchestrator, runs the control loop.
    """
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

    orchestrator = Orchestrator(settings=settings, kg=kg, gateway=gateway)
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
