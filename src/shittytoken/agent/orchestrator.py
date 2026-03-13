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

from ..config import Settings, cfg, primary_model_id, preferred_gpus
from ..knowledge.client import KnowledgeGraph
from ..knowledge.schema import Configuration
from .gateway import GatewayClient
from .health import HeartbeatMonitor, wait_for_model_ready
from .metrics import AggregateMetrics, aggregate_metrics, scrape_worker_metrics
from .provisioner import (
    DeploymentPlan,
    GPUProvider,
    RunPodProvider,
    VastAIProvider,
    build_deployment_plan,
    execute_deployment,
    get_provider,
)
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
        approval_fn=None,
    ) -> None:
        self._settings = settings
        self._kg = kg
        self._gateway = gateway
        # Optional HITL callback: async fn(DeploymentPlan) -> bool
        # When set, provision_and_qualify() presents the plan for approval
        # before renting any GPU. If the callback returns False, provisioning
        # is aborted. When None, all deployments are auto-approved.
        self._approval_fn = approval_fn

        # instance_id → InstanceStateMachine
        self._instances: dict[str, InstanceStateMachine] = {}
        # instance_id → last_idle timestamp (monotonic)
        self._idle_since: dict[str, float] = {}
        # instance_id → last known running-request count
        self._last_running: dict[str, int] = {}

        self._provision_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._session: Optional[aiohttp.ClientSession] = None
        self._provider: Optional[GPUProvider] = None
        self._ssh_manager: Optional[SSHManager] = None
        self._heartbeat_monitor: Optional[HeartbeatMonitor] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._spot_monitor_task: Optional[asyncio.Task] = None
        self._provision_cooldown_until: float = 0.0  # monotonic time

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
            on_failure=self._on_worker_failure,
        )
        self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor.run())

        # Start spot eviction monitor for RunPod instances
        provider_name = cfg["orchestrator"].get("provider", "vastai")
        if provider_name == "runpod":
            poll_sec = cfg["orchestrator"].get("runpod", {}).get("spot_poll_interval_sec", 10)
            self._spot_monitor_task = asyncio.create_task(
                self._spot_eviction_monitor(poll_sec)
            )

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
            # Log all instance states for monitoring
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
            return

        metrics = await aggregate_metrics(serving_urls, self._session)
        logger.info(
            "orchestrator_tick",
            serving_workers=metrics.worker_count,
            total_running=metrics.total_requests_running,
            total_waiting=metrics.total_requests_waiting,
            avg_kv_cache=round(metrics.avg_kv_cache_usage, 3),
        )

        # --- Scrape per-worker metrics for idle tracking ---
        for sm in self._instances.values():
            if sm.state == InstanceState.SERVING and sm.record.worker_url:
                worker_metrics = await scrape_worker_metrics(
                    sm.record.worker_url, self._session
                )
                running = int(worker_metrics.get("num_requests_running", 0))
                self._last_running[sm.record.instance_id] = running

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
        _orch = cfg["orchestrator"]
        if metrics.total_requests_waiting > _orch["scale_up_waiting_threshold"]:
            if not self._provision_lock.locked():
                logger.info(
                    "scale_up_triggered",
                    waiting=metrics.total_requests_waiting,
                    threshold=_orch["scale_up_waiting_threshold"],
                )
                asyncio.create_task(self._guarded_provision())

        # --- Scale down (never below min_workers) ---
        elif len(serving_urls) > min_workers:
            await self._maybe_scale_down(metrics)

    async def _guarded_provision(self) -> None:
        """Serialize provisioning so only one scale-up runs at a time."""
        async with self._provision_lock:
            try:
                result = await self.provision_and_qualify()
                if result is None:
                    # Provision failed gracefully — cooldown before retrying
                    self._provision_cooldown_until = time.monotonic() + 60
                    logger.info("provision_failed_cooldown", cooldown_s=60)
            except Exception as exc:
                logger.error(
                    "provision_unhandled_exception",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                self._provision_cooldown_until = time.monotonic() + 60
                # Clean up any instance that was created but not fully serving
                for sm in list(self._instances.values()):
                    if sm.state in (InstanceState.PROVISIONING, InstanceState.BENCHMARKING):
                        logger.info("provision_cleanup_orphan", instance_id=sm.record.instance_id, state=sm.state.value)
                        await self._destroy_instance(sm.record, sm)

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
                await self.scale_down_one()
                return  # one scale-down per tick

    # ------------------------------------------------------------------
    # Provision + qualify a new instance
    # ------------------------------------------------------------------

    async def provision_and_qualify(
        self,
        gpu_names: list[str] | None = None,
        model_id: str | None = None,
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
        if model_id is None:
            model_id = primary_model_id()
        if gpu_names is None:
            gpu_names = preferred_gpus()

        settings = self._settings

        # Look up the best known configuration for ANY preferred GPU type
        # Priority: benchmarked passing config > any seeded config > LLM proposal
        config_source = "knowledge_graph_benchmarked"
        config = None
        for gpu_name in gpu_names:
            config = await self._kg.best_config_for(gpu_name, model_id)
            if config is not None:
                break
        if config is None:
            config_source = "knowledge_graph_seed"
            for gpu_name in gpu_names:
                config = await self._kg.any_config_for(gpu_name, model_id)
                if config is not None:
                    break
        if config is None:
            # Last resort: ask the LLM to propose a config
            logger.info(
                "provision_no_config_proposing",
                gpu_names=gpu_names,
                model_id=model_id,
            )
            config = await self._propose_and_store_config(
                gpu_name=gpu_names[0], model_id=model_id
            )
            config_source = "llm_proposal"
            if config is None:
                logger.error(
                    "provision_config_proposal_failed",
                    gpu_names=gpu_names,
                    model_id=model_id,
                )
                return None

        provider = self._get_provider()

        # --- Step 1/7: Build deployment plan (searches offers, no spend) ---
        logger.info("provision_step", step="1/7", action="searching_offers", gpu_names=gpu_names)
        try:
            plan = await build_deployment_plan(
                provider=provider,
                config=config,
                model_id=model_id,
                gpu_names=gpu_names,
                config_source=config_source,
            )
        except (aiohttp.ClientError, TimeoutError, RuntimeError) as exc:
            logger.error("provision_plan_failed", error=str(exc))
            return None

        # --- Step 2/7: HITL approval gate ---
        logger.info("provision_step", step="2/7", action="awaiting_approval", offer_id=plan.offer.offer_id, gpu=f"{plan.offer.num_gpus}x {plan.offer.gpu_name}", cost=f"${plan.offer.cost_per_hour_usd:.4f}/hr")
        if self._approval_fn is not None:
            approved = await self._approval_fn(plan)
            if not approved:
                logger.info("provision_rejected_by_user", provider=plan.provider)
                return None

        # --- Step 3/7: Execute the approved plan (rents the instance) ---
        logger.info("provision_step", step="3/7", action="renting_instance", offer_id=plan.offer.offer_id)
        try:
            provisioned = await execute_deployment(
                plan=plan,
                provider=provider,
                hf_token=self._settings.huggingface_token,
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
            http_port=provisioned.http_port,
            config_id=config.config_id,
        )
        sm = InstanceStateMachine(record)
        self._instances[record.instance_id] = sm

        # --- Step 4/7: Verify GPU via SSH ---
        logger.info("provision_step", step="4/7", action="verifying_gpu", instance_id=record.instance_id, ssh_host=record.ssh_host)
        import asyncssh  # avoid circular; only needed here

        try:
            session = await self._ssh_manager.connect(
                host=record.ssh_host,
                port=record.ssh_port,
            )
        except (asyncssh.Error, OSError, asyncio.TimeoutError) as exc:
            logger.error(
                "provision_ssh_connect_failed",
                instance_id=record.instance_id,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            sm.transition(InstanceState.FAILED, reason="ssh_connect_failed")
            await self._destroy_instance(record, sm)
            return None

        # GPU verification uses the ACTUAL provisioned GPU, not the first preferred
        actual_gpu = record.gpu_model  # e.g. "RTX 4090" from the provisioner
        expected_vram_gb = await self._kg.gpu_vram_for(actual_gpu)
        if expected_vram_gb is None:
            logger.error(
                "provision_gpu_vram_unknown",
                gpu_name=actual_gpu,
            )
            sm.transition(InstanceState.FAILED, reason="gpu_vram_unknown")
            await self._ssh_manager.close(session)
            await self._destroy_instance(record, sm)
            return None

        gpu_ok = await self._ssh_manager.verify_gpu(
            session=session,
            expected_gpu_name=actual_gpu,
            expected_vram_gb=expected_vram_gb,
        )
        if not gpu_ok:
            logger.error("provision_gpu_verify_failed", instance_id=record.instance_id)
            sm.transition(InstanceState.FAILED, reason="gpu_verify_failed")
            await self._ssh_manager.close(session)
            await self._destroy_instance(record, sm)
            return None

        # --- Transition to BENCHMARKING ---
        sm.transition(InstanceState.BENCHMARKING, reason="gpu_verified")

        # --- Step 5/7: Monitor startup (may take several minutes for model download + load) ---
        logger.info("provision_step", step="5/7", action="monitoring_startup", instance_id=record.instance_id, msg="streaming VM logs — model download + weight loading can take several minutes")
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
            timeout_sec=cfg["orchestrator"]["startup_monitor_timeout_s"],
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

        # --- Step 6/7: Wait for model ready via HTTP ---
        logger.info("provision_step", step="6/7", action="waiting_for_model_ready", instance_id=record.instance_id)
        worker_url = self._build_worker_url(record)
        record.worker_url = worker_url

        model_ready = await wait_for_model_ready(
            base_url=worker_url,
            session=self._session,
            timeout_sec=cfg["orchestrator"]["startup_monitor_timeout_s"],
        )
        if not model_ready:
            sm.transition(InstanceState.FAILED, reason="model_ready_timeout")
            await self._ssh_manager.close(session)
            await self._destroy_instance(record, sm)
            return None

        # --- Step 7/7: Run benchmark ---
        logger.info("provision_step", step="7/7", action="running_benchmark", instance_id=record.instance_id, worker_url=worker_url)
        try:
            from ..benchmark.runner import run_benchmark  # type: ignore[import]

            bench_result = await run_benchmark(
                worker_url=worker_url,
                model_id=model_id,
                gpu_model=record.gpu_model,
                raw_config={
                    "tensor_parallel_size": config.tensor_parallel_size,
                    "max_model_len": config.max_model_len,
                    "gpu_memory_utilization": config.gpu_memory_utilization,
                    "quantization": config.quantization,
                    "kv_cache_dtype": config.kv_cache_dtype,
                    "max_num_seqs": config.max_num_seqs,
                    "enable_prefix_caching": config.enable_prefix_caching,
                    "enforce_eager": config.enforce_eager,
                },
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
        try:
            await self._gateway.register_worker(worker_url)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "gateway_register_failed",
                instance_id=record.instance_id,
                error=str(exc),
                msg="Worker passed benchmark but gateway registration failed. "
                    "Worker is still reachable directly at worker_url.",
            )
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

    async def _propose_and_store_config(
        self,
        gpu_name: str,
        model_id: str,
    ) -> Configuration | None:
        """
        Use the LLM to propose an initial config when no KG data exists.
        Writes the proposed config to the KG and returns it.
        Returns None if the proposal fails.
        """
        from .llm import propose_initial_config

        gpu_vram = await self._kg.gpu_vram_for(gpu_name)
        if gpu_vram is None:
            logger.error("propose_config_gpu_vram_unknown", gpu_name=gpu_name)
            return None

        model_params = await self._kg.llm_model_params(model_id)
        if model_params is None:
            logger.error("propose_config_model_params_unknown", model_id=model_id)
            return None

        params_b, active_params_b = model_params

        try:
            proposed = await propose_initial_config(
                gpu_model_name=gpu_name,
                gpu_vram_gb=gpu_vram,
                model_id=model_id,
                params_b=params_b,
                active_params_b=active_params_b,
                kg=self._kg,
                model=self._settings.agent_model,
            )
        except Exception as exc:
            logger.error("propose_config_failed", error=str(exc))
            return None

        config = Configuration(
            tensor_parallel_size=proposed.tensor_parallel_size,
            max_model_len=proposed.max_model_len,
            gpu_memory_utilization=proposed.gpu_memory_utilization,
            quantization=proposed.quantization,
            kv_cache_dtype=proposed.kv_cache_dtype,
            max_num_seqs=proposed.max_num_seqs,
            enable_prefix_caching=proposed.enable_prefix_caching,
            enforce_eager=proposed.enforce_eager,
        )
        await self._kg.write_configuration(config)

        logger.info(
            "propose_config_stored",
            config_id=config.config_id,
            gpu_name=gpu_name,
            model_id=model_id,
        )
        return config

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

    @staticmethod
    def _build_worker_url(record: InstanceRecord) -> str:
        """Build the HTTP URL to reach the vLLM server on a provisioned instance."""
        http_port = record.http_port or 8080
        return f"http://{record.ssh_host}:{http_port}"

    async def _spot_eviction_monitor(self, poll_interval_sec: int = 10) -> None:
        """Background task that polls RunPod pod status to detect spot evictions.

        RunPod spot instances get SIGTERM with only 5s grace — no advance
        warning.  This monitor detects eviction by checking if the pod
        status transitioned to EXITED while podType is INTERRUPTABLE.
        """
        logger.info("spot_eviction_monitor_started", poll_interval_sec=poll_interval_sec)
        provider = self._get_provider()
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(poll_interval_sec)

                for sm in list(self._instances.values()):
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
                            self._heartbeat_monitor.deregister(sm.record.worker_url)
                            try:
                                await self._gateway.deregister_worker(
                                    sm.record.worker_url, drain=False
                                )
                            except KeyError:
                                pass

                        sm.transition(InstanceState.FAILED, reason="spot_eviction")
                        self._instances.pop(sm.record.instance_id, None)
                        self._idle_since.pop(sm.record.instance_id, None)
                        self._last_running.pop(sm.record.instance_id, None)

                        if not self._provision_lock.locked():
                            logger.info("spot_eviction_reprovision", instance_id=sm.record.instance_id)
                            asyncio.create_task(self._guarded_provision())

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("spot_eviction_monitor_error", error=str(exc))

        logger.info("spot_eviction_monitor_stopped")

    async def _destroy_instance(
        self,
        record: InstanceRecord,
        sm: InstanceStateMachine,
    ) -> None:
        """Destroy the cloud instance and mark it TERMINATED (or FAILED)."""
        try:
            provider = self._get_provider()
            await provider.destroy_instance(record.instance_id)
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


async def interactive_approval(plan: DeploymentPlan) -> bool:
    """
    Default HITL approval: prints the deployment plan and asks for confirmation.

    Runs the blocking input() call in a thread executor so the event loop
    stays alive.
    """
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

    router_manager = RouterManager(
        static_models=[m["model_id"] for m in cfg["models"]["serving"]],
    )
    registry = WorkerRegistry(router_manager=router_manager)
    gateway = GatewayClient(registry=registry)

    auto_approve = os.getenv("SHITTYTOKEN_AUTO_APPROVE", "").strip() in ("1", "true")
    approval_fn = None if auto_approve else interactive_approval

    orchestrator = Orchestrator(
        settings=settings, kg=kg, gateway=gateway, approval_fn=approval_fn
    )
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
