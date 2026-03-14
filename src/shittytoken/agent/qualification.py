"""
Qualification pipeline — 7-step provision + qualify flow.

Extracted from orchestrator.py for maintainability. This module contains
the full lifecycle of bringing a new GPU instance from "rented" to "SERVING":

1. Build deployment plan (search offers)
2. HITL approval gate
3. Execute deployment (rent instance)
4. Verify GPU hardware
5. Monitor startup logs
6. Wait for model ready via HTTP
7. Run benchmark + register with gateway
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import aiohttp
import structlog

from ..config import cfg, primary_model_id, preferred_gpus
from ..knowledge.schema import Configuration
from .health import wait_for_model_ready
from .provisioner import (
    build_deployment_plan,
    build_worker_url,
    execute_deployment,
)
from .startup_monitor import StartupResult, monitor_startup
from .state_machine import InstanceRecord, InstanceState, InstanceStateMachine

if TYPE_CHECKING:
    from ..knowledge.client import KnowledgeGraph
    from .health import HeartbeatMonitor
    from .provisioner import GPUProvider
    from .ssh import SSHManager
    from .gateway import GatewayClient

logger = structlog.get_logger()


async def provision_and_qualify(
    *,
    kg: KnowledgeGraph,
    provider: GPUProvider,
    ssh_manager: SSHManager,
    session: aiohttp.ClientSession,
    heartbeat_monitor: HeartbeatMonitor,
    gateway: GatewayClient,
    settings: object,
    approval_fn=None,
    gpu_names: list[str] | None = None,
    model_id: str | None = None,
) -> tuple[InstanceRecord | None, InstanceStateMachine | None]:
    """
    Full provisioning + qualification flow.

    Returns (record, state_machine) on success, (None, None) on failure.
    On failure where an instance was created, returns (None, sm) so the
    caller can clean up.
    """
    if model_id is None:
        model_id = primary_model_id()
    if gpu_names is None:
        gpu_names = preferred_gpus()

    # Look up the best known configuration for ANY preferred GPU type
    config, config_source = await _resolve_config(kg, gpu_names, model_id, settings)
    if config is None:
        return None, None

    # --- Step 1/7: Build deployment plan ---
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
        return None, None

    # --- Step 2/7: HITL approval gate ---
    logger.info(
        "provision_step", step="2/7", action="awaiting_approval",
        offer_id=plan.offer.offer_id,
        gpu=f"{plan.offer.num_gpus}x {plan.offer.gpu_name}",
        cost=f"${plan.offer.cost_per_hour_usd:.4f}/hr",
    )
    if approval_fn is not None:
        approved = await approval_fn(plan)
        if not approved:
            logger.info("provision_rejected_by_user", provider=plan.provider)
            return None, None

    # --- Step 3/7: Execute the approved plan (rents the instance) ---
    logger.info("provision_step", step="3/7", action="renting_instance", offer_id=plan.offer.offer_id)
    try:
        provisioned = await execute_deployment(
            plan=plan,
            provider=provider,
            hf_token=settings.huggingface_token,
        )
    except (aiohttp.ClientError, TimeoutError) as exc:
        logger.error("provision_failed", error=str(exc))
        return None, None

    record = InstanceRecord(
        instance_id=provisioned.instance_id,
        provider=provisioned.provider,
        gpu_model=provisioned.gpu_model,
        ssh_host=provisioned.ssh_host,
        ssh_port=provisioned.ssh_port,
        ssh_user=provisioned.ssh_user,
        http_port=provisioned.http_port,
        config_id=config.config_id,
        cost_per_hour_usd=plan.offer.cost_per_hour_usd or 0.0,
    )
    sm = InstanceStateMachine(record)

    # --- Step 4/7: Verify GPU ---
    ssh_session = await _verify_gpu(record, sm, kg, ssh_manager, provider)
    if sm.state == InstanceState.FAILED:
        return None, sm

    # --- Transition to BENCHMARKING ---
    sm.transition(InstanceState.BENCHMARKING, reason="gpu_verified")

    # --- Step 5/7: Monitor startup ---
    if ssh_session is not None:
        startup_ok = await _monitor_startup_logs(
            record, sm, kg, ssh_manager, ssh_session, config
        )
        if not startup_ok:
            return None, sm
    else:
        logger.info(
            "provision_step", step="5/7", action="skipping_log_stream",
            instance_id=record.instance_id,
            msg="RunPod — will poll HTTP health instead",
        )

    # --- Step 6/7: Wait for model ready via HTTP ---
    logger.info("provision_step", step="6/7", action="waiting_for_model_ready", instance_id=record.instance_id)
    worker_url = build_worker_url(record)
    record.worker_url = worker_url

    model_ready = await wait_for_model_ready(
        base_url=worker_url,
        session=session,
        timeout_sec=cfg["orchestrator"]["startup_monitor_timeout_s"],
    )
    if not model_ready:
        sm.transition(InstanceState.FAILED, reason="model_ready_timeout")
        if ssh_session is not None:
            await ssh_manager.close(ssh_session)
        return None, sm

    # --- Step 7/7: Run benchmark (or skip in test_mode) ---
    test_mode = cfg.get("benchmark", {}).get("test_mode", False)
    if test_mode:
        logger.info(
            "provision_step", step="7/7", action="skipping_benchmark",
            instance_id=record.instance_id, worker_url=worker_url,
            reason="test_mode=true",
        )
    else:
        logger.info("provision_step", step="7/7", action="running_benchmark", instance_id=record.instance_id, worker_url=worker_url)
        bench_ok = await _run_benchmark(
            record, sm, kg, config, model_id, worker_url, ssh_manager, ssh_session
        )
        if not bench_ok:
            return None, sm

    # --- Register with gateway and transition to SERVING ---
    try:
        await gateway.register_worker(worker_url)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "gateway_register_failed",
            instance_id=record.instance_id,
            error=str(exc),
            msg="Worker passed benchmark but gateway registration failed.",
        )
    heartbeat_monitor.register(worker_url)
    sm.transition(InstanceState.SERVING, reason="benchmark_passed")

    if ssh_session is not None:
        await ssh_manager.close(ssh_session)
    logger.info(
        "provision_qualify_complete",
        instance_id=record.instance_id,
        worker_url=worker_url,
    )
    return record, sm


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _resolve_config(
    kg: KnowledgeGraph,
    gpu_names: list[str],
    model_id: str,
    settings: object,
) -> tuple[Configuration | None, str]:
    """Find the best config: benchmarked > seeded > LLM proposal."""
    config_source = "knowledge_graph_benchmarked"
    config = None
    for gpu_name in gpu_names:
        config = await kg.best_config_for(gpu_name, model_id)
        if config is not None:
            break
    if config is None:
        config_source = "knowledge_graph_seed"
        for gpu_name in gpu_names:
            config = await kg.any_config_for(gpu_name, model_id)
            if config is not None:
                break
    if config is None:
        logger.info("provision_no_config_proposing", gpu_names=gpu_names, model_id=model_id)
        config = await _propose_and_store_config(kg, gpu_names[0], model_id, settings)
        config_source = "llm_proposal"
        if config is None:
            logger.error("provision_config_proposal_failed", gpu_names=gpu_names, model_id=model_id)
            return None, config_source
    return config, config_source


async def _propose_and_store_config(
    kg: KnowledgeGraph,
    gpu_name: str,
    model_id: str,
    settings: object,
) -> Configuration | None:
    """Use the LLM to propose an initial config when no KG data exists."""
    from .llm import propose_initial_config

    gpu_vram = await kg.gpu_vram_for(gpu_name)
    if gpu_vram is None:
        logger.error("propose_config_gpu_vram_unknown", gpu_name=gpu_name)
        return None

    model_params = await kg.llm_model_params(model_id)
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
            kg=kg,
            model=settings.agent_model,
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
    await kg.write_configuration(config)
    logger.info("propose_config_stored", config_id=config.config_id, gpu_name=gpu_name, model_id=model_id)
    return config


async def _verify_gpu(
    record: InstanceRecord,
    sm: InstanceStateMachine,
    kg: KnowledgeGraph,
    ssh_manager: SSHManager,
    provider: GPUProvider,
):
    """Step 4/7: Verify GPU hardware. Returns SSH session or None."""
    if record.provider == "runpod":
        logger.info(
            "provision_step", step="4/7", action="verifying_gpu_via_api",
            instance_id=record.instance_id, gpu_model=record.gpu_model,
        )
        return None

    logger.info(
        "provision_step", step="4/7", action="verifying_gpu",
        instance_id=record.instance_id, ssh_host=record.ssh_host,
    )
    import asyncssh

    ssh_timeout = cfg.get("orchestrator", {}).get("ssh_ready_timeout_s", 600)
    max_ssh_attempts = ssh_timeout // 10
    session = None
    last_exc = None
    for attempt in range(max_ssh_attempts):
        try:
            session = await ssh_manager.connect(
                host=record.ssh_host,
                port=record.ssh_port,
                username=record.ssh_user,
            )
            break
        except (asyncssh.Error, OSError, asyncio.TimeoutError) as exc:
            last_exc = exc
            if attempt % 3 == 2:
                logger.info(
                    "provision_ssh_retrying",
                    instance_id=record.instance_id,
                    attempt=attempt + 1,
                    error=str(exc),
                )
            await asyncio.sleep(10)

    if session is None:
        logger.error(
            "provision_ssh_connect_failed",
            instance_id=record.instance_id,
            error=str(last_exc),
            error_type=type(last_exc).__name__,
            attempts=max_ssh_attempts,
        )
        sm.transition(InstanceState.FAILED, reason="ssh_connect_failed")
        return None

    actual_gpu = record.gpu_model
    expected_vram_gb = await kg.gpu_vram_for(actual_gpu)
    if expected_vram_gb is None:
        logger.error("provision_gpu_vram_unknown", gpu_name=actual_gpu)
        sm.transition(InstanceState.FAILED, reason="gpu_vram_unknown")
        await ssh_manager.close(session)
        return None

    gpu_ok = await ssh_manager.verify_gpu(
        session=session,
        expected_gpu_name=actual_gpu,
        expected_vram_gb=expected_vram_gb,
    )
    if not gpu_ok:
        logger.error("provision_gpu_verify_failed", instance_id=record.instance_id)
        sm.transition(InstanceState.FAILED, reason="gpu_verify_failed")
        await ssh_manager.close(session)
        return None

    return session


async def _monitor_startup_logs(
    record: InstanceRecord,
    sm: InstanceStateMachine,
    kg: KnowledgeGraph,
    ssh_manager: SSHManager,
    ssh_session,
    config: Configuration,
) -> bool:
    """Step 5/7: Stream and monitor startup logs. Returns True if READY."""
    logger.info(
        "provision_step", step="5/7", action="monitoring_startup",
        instance_id=record.instance_id,
        msg="streaming VM logs — model download + weight loading can take several minutes",
    )
    log_lines: asyncio.Queue[str] = asyncio.Queue()

    async def _enqueue(line: str) -> None:
        await log_lines.put(line)

    async def _line_gen():
        while True:
            line = await log_lines.get()
            yield line

    stream_task = asyncio.create_task(
        ssh_manager.stream_logs(ssh_session, _enqueue)
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
            await kg.write_oom_event(
                config_id=config.config_id,
                gpu_model_name=record.gpu_model,
                error_type="loading",
                error_message=matched_line,
                error_phase="loading",
                gpu_memory_free_gb=0.0,
                gpu_memory_total_gb=0.0,
            )
        sm.transition(InstanceState.FAILED, reason=reason)
        await ssh_manager.close(ssh_session)
        return False

    return True


async def _run_benchmark(
    record: InstanceRecord,
    sm: InstanceStateMachine,
    kg: KnowledgeGraph,
    config: Configuration,
    model_id: str,
    worker_url: str,
    ssh_manager: SSHManager,
    ssh_session,
) -> bool:
    """Step 7/7: Run benchmark and write results to KG. Returns True if passed."""
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
    except Exception as exc:  # noqa: BLE001
        logger.error("provision_benchmark_error", instance_id=record.instance_id, error=str(exc))
        sm.transition(InstanceState.FAILED, reason="benchmark_exception")
        if ssh_session is not None:
            await ssh_manager.close(ssh_session)
        return False

    from datetime import datetime, timezone

    await kg.write_benchmark_result(
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
        test_mode = cfg.get("benchmark", {}).get("test_mode", False)
        if test_mode:
            logger.warning(
                "provision_benchmark_failed_test_mode",
                instance_id=record.instance_id,
                verdict=bench_result.verdict.value,
                fail_reasons=[r.value for r in bench_result.fail_reasons],
                action="keeping instance (test_mode=true)",
            )
            # In test mode, treat as pass — keep the instance running
            return True
        else:
            logger.warning(
                "provision_benchmark_failed",
                instance_id=record.instance_id,
                verdict=bench_result.verdict.value,
            )
            sm.transition(InstanceState.FAILED, reason="benchmark_failed")
            if ssh_session is not None:
                await ssh_manager.close(ssh_session)
            return False

    return True
