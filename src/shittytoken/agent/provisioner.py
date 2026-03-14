"""
Provisioner — GPU instance lifecycle management.

build_vllm_command() converts a Configuration dataclass into a vllm serve
command string.  All vLLM parameters come from Configuration; nothing is
hard-coded here.

INVARIANTS (enforced by build_vllm_command):
- config.gpu_memory_utilization < 1.0
- config.enable_prefix_caching is True

Always injected:
- --enable-prefix-caching
- --host 0.0.0.0 --port 8080  (listen on all interfaces, tunnel port)

Provider abstraction:
- GPUProvider is the abstract interface
- VastAIProvider and RunPodProvider are the implementations
- Adding a new provider means subclassing GPUProvider
"""

from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import structlog

from ..config import cfg
from ..knowledge.schema import Configuration

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------


def build_vllm_command(config: Configuration, model_id: str) -> str:
    """
    Generate a vllm serve command string from a Configuration dataclass.

    INVARIANTS (raises ValueError if violated):
    - config.gpu_memory_utilization < 1.0
    - config.enable_prefix_caching is True

    Always injects --enable-prefix-caching.

    NOTE: --ipc=host is a Docker runtime flag, NOT a vLLM CLI flag.
    It must be passed to the container runtime (e.g. docker run --ipc=host).
    On Vast.ai, the container is already started with appropriate IPC settings.
    """
    if config.gpu_memory_utilization >= 1.0:
        raise ValueError(
            f"build_vllm_command invariant violated: "
            f"gpu_memory_utilization must be < 1.0, got {config.gpu_memory_utilization}"
        )
    if not config.enable_prefix_caching:
        raise ValueError(
            "build_vllm_command invariant violated: "
            "enable_prefix_caching must be True"
        )

    worker_port = cfg.get("vllm", {}).get("worker_port", 8080)

    parts: list[str] = [
        "vllm",
        "serve",
        model_id,
        "--tensor-parallel-size", str(config.tensor_parallel_size),
        "--max-model-len", str(config.max_model_len),
        "--gpu-memory-utilization", str(config.gpu_memory_utilization),
        "--max-num-seqs", str(config.max_num_seqs),
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--host", "0.0.0.0",
        "--port", str(worker_port),
        "--enable-auto-tool-choice",
        "--tool-call-parser", cfg.get("vllm", {}).get("tool_call_parser", "hermes"),
    ]

    _vllm = cfg.get("vllm", {})
    if _vllm.get("reasoning_parser"):
        parts += ["--reasoning-parser", _vllm["reasoning_parser"]]
    if _vllm.get("attention_backend"):
        parts += ["--attention-backend", _vllm["attention_backend"]]

    if config.quantization is not None:
        parts += ["--quantization", config.quantization]

    if config.kv_cache_dtype != "auto":
        parts += ["--kv-cache-dtype", config.kv_cache_dtype]

    if config.enforce_eager:
        parts.append("--enforce-eager")

    # Extra CLI args from config
    extra_args = _vllm.get("extra_args", [])
    if extra_args:
        parts += [str(a) for a in extra_args]

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class GPUOffer:
    """Provider-agnostic representation of a rentable GPU instance."""

    offer_id: str
    provider: str
    gpu_name: str
    num_gpus: int
    cost_per_hour_usd: float | None
    reliability: float | None

    # Networking
    inet_up_mbps: float | None = None
    inet_down_mbps: float | None = None
    inet_up_cost_per_gb: float | None = None
    inet_down_cost_per_gb: float | None = None

    # Interconnect
    pcie_bw_gbps: float | None = None
    bw_nvlink_gbps: float | None = None

    # Provider-specific extras
    dlperf: float | None = None
    raw: dict = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Composite score for ranking. Lower is better."""
        cost = self.cost_per_hour_usd or float("inf")

        # Bandwidth cost estimate (~0.1 GB/hr at scale)
        inet_up_cost = self.inet_up_cost_per_gb or 0
        bw_cost_per_hr = inet_up_cost * 0.1

        # Upload speed penalty: below 200 MB/s
        inet_up = self.inet_up_mbps or 0
        upload_penalty = max(0, (200 - inet_up) / 200) * 0.05

        # NVLink bonus for multi-GPU
        nvlink = self.bw_nvlink_gbps or 0
        nvlink_bonus = -0.03 if (self.num_gpus > 1 and nvlink > 0) else 0.0

        return cost + bw_cost_per_hr + upload_penalty + nvlink_bonus


@dataclass
class ProvisionedInstance:
    """A running instance with network access."""

    instance_id: str
    provider: str
    gpu_model: str
    ssh_host: str
    ssh_port: int
    status: str
    ssh_user: str = "root"  # RunPod uses {podHostId}@ssh.runpod.io
    http_port: int | None = None  # public port mapped to container 8080


@dataclass
class DeploymentPlan:
    """All details about a planned deployment, presented for human approval."""

    # Config source
    config_source: str

    # vLLM configuration
    tensor_parallel_size: int
    max_model_len: int
    gpu_memory_utilization: float
    quantization: str | None
    kv_cache_dtype: str
    max_num_seqs: int
    enable_prefix_caching: bool
    enforce_eager: bool

    # The full command that will be executed
    vllm_command: str

    # Offer details
    offer: GPUOffer

    # Model
    model_id: str

    def estimated_bandwidth_cost_per_mtok(self) -> float | None:
        """Rough estimate of bandwidth cost per million output tokens."""
        if self.offer.inet_up_cost_per_gb is None:
            return None
        bytes_per_token = 24  # ~4 bytes token + SSE framing
        gb_per_mtok = (bytes_per_token * 1_000_000) / (1024 ** 3)
        return gb_per_mtok * self.offer.inet_up_cost_per_gb

    def display(self) -> str:
        """Human-readable summary for approval."""
        o = self.offer
        lines = [
            "",
            "=" * 64,
            "  DEPLOYMENT PLAN",
            "=" * 64,
            "",
            f"  Provider:       {o.provider}",
            f"  Offer ID:       {o.offer_id}",
            f"  GPU:            {o.num_gpus}x {o.gpu_name}",
        ]
        if o.cost_per_hour_usd is not None:
            lines.append(f"  Compute cost:   ${o.cost_per_hour_usd:.4f}/hr")
        if o.reliability is not None:
            lines.append(f"  Reliability:    {o.reliability:.1%}")
        if o.dlperf is not None:
            lines.append(f"  DL perf score:  {o.dlperf:.1f}")

        # Networking
        lines.append("")
        lines.append("  Networking:")
        if o.inet_up_mbps is not None:
            lines.append(f"    Upload:       {o.inet_up_mbps:.0f} MB/s")
        if o.inet_down_mbps is not None:
            lines.append(f"    Download:     {o.inet_down_mbps:.0f} MB/s")
        if o.inet_up_cost_per_gb is not None:
            lines.append(f"    Upload cost:  ${o.inet_up_cost_per_gb:.4f}/GB")
        if o.inet_down_cost_per_gb is not None:
            lines.append(f"    Download cost: ${o.inet_down_cost_per_gb:.4f}/GB")
        bw_cost = self.estimated_bandwidth_cost_per_mtok()
        if bw_cost is not None:
            lines.append(f"    Est. BW cost: ${bw_cost:.6f}/Mtok (upload)")

        # Interconnect
        lines.append("")
        lines.append("  Interconnect:")
        if o.pcie_bw_gbps is not None:
            lines.append(f"    PCIe BW:      {o.pcie_bw_gbps:.1f} GB/s")
        if o.bw_nvlink_gbps is not None:
            nvlink_str = f"{o.bw_nvlink_gbps:.1f} GB/s" if o.bw_nvlink_gbps > 0 else "none"
            lines.append(f"    NVLink BW:    {nvlink_str}")
        elif o.num_gpus > 1:
            lines.append("    NVLink BW:    none (PCIe-only multi-GPU)")

        lines += [
            "",
            f"  Model:          {self.model_id}",
            "",
            "  vLLM Configuration:",
            f"    Config source:          {self.config_source}",
            f"    tensor_parallel_size:   {self.tensor_parallel_size}",
            f"    max_model_len:          {self.max_model_len}",
            f"    gpu_memory_utilization: {self.gpu_memory_utilization}",
            f"    quantization:           {self.quantization}",
            f"    kv_cache_dtype:         {self.kv_cache_dtype}",
            f"    max_num_seqs:           {self.max_num_seqs}",
            f"    enable_prefix_caching:  {self.enable_prefix_caching}",
            f"    enforce_eager:          {self.enforce_eager}",
            "",
            "  Command:",
            f"    {self.vllm_command}",
            "",
            "=" * 64,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------


class GPUProvider(abc.ABC):
    """Abstract interface for GPU cloud providers."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short provider name (e.g. 'vastai')."""

    @abc.abstractmethod
    async def find_offers(
        self,
        gpu_names: list[str],
        min_gpus: int,
    ) -> list[GPUOffer]:
        """Search for available GPU offers matching the criteria."""

    @abc.abstractmethod
    async def create_instance(
        self,
        offer: GPUOffer,
        vllm_command: str,
        hf_token: str,
        disk_gb: int = 50,
    ) -> ProvisionedInstance:
        """Rent an instance from an offer and wait for SSH readiness."""

    @abc.abstractmethod
    async def destroy_instance(self, instance_id: str) -> None:
        """Terminate a running instance."""

    @abc.abstractmethod
    async def get_instance(self, instance_id: str) -> dict:
        """Get current instance status."""

    @abc.abstractmethod
    async def list_all_instances(self) -> list[dict]:
        """List all instances currently rented by this account."""


# ---------------------------------------------------------------------------
# Vast.ai provider (uses vastai-sdk)
# ---------------------------------------------------------------------------


class VastAIProvider(GPUProvider):
    """Provisions and manages GPU instances on Vast.ai via the official SDK."""

    def __init__(self, api_key: str, session: aiohttp.ClientSession | None = None) -> None:
        from vastai import VastAI
        self._client = VastAI(api_key=api_key, raw=True)
        # session kept for interface compat but SDK handles its own HTTP
        self._session = session

    @property
    def name(self) -> str:
        return "vastai"

    async def find_offers(
        self,
        gpu_names: list[str],
        min_gpus: int = 2,
    ) -> list[GPUOffer]:
        """
        Search Vast.ai for GPU offers using the SDK query syntax.

        Returns offers sorted by composite score (lower is better).
        """
        _gpu_cfg = cfg["gpus"]

        # Build SDK query string — spaces in GPU names become underscores
        # SDK uses = for exact match (e.g. gpu_name=RTX_3090)
        if len(gpu_names) == 1:
            gpu_filter = f"gpu_name={gpu_names[0].replace(' ', '_')}"
        else:
            gpu_values = "[" + ",".join(n.replace(" ", "_") for n in gpu_names) + "]"
            gpu_filter = f"gpu_name in {gpu_values}"

        query_parts = [
            gpu_filter,
            f"num_gpus>={min_gpus}",
            f"gpu_total_ram>={_gpu_cfg['min_gpu_total_ram_gb']}",
            f"reliability>={_gpu_cfg['min_reliability']}",
            f"inet_up>={_gpu_cfg['min_inet_up_mbps']}",
            f"pcie_bw>={_gpu_cfg['min_pcie_bw_gbps']}",
            "rentable=True",
        ]
        query_str = " ".join(query_parts)

        logger.info(
            "vastai_searching",
            query=query_str,
            gpu_names=gpu_names,
            min_gpus=min_gpus,
        )

        # SDK is synchronous — run in thread pool
        raw_offers = await asyncio.to_thread(
            self._client.search_offers, query=query_str, type="on-demand"
        )

        if not isinstance(raw_offers, list):
            logger.warning("vastai_unexpected_response", response=str(raw_offers)[:200])
            return []

        # Client-side filters — SDK doesn't reliably enforce all query params
        gpu_names_lower = {n.lower() for n in gpu_names}
        raw_offers = [
            o for o in raw_offers
            if o.get("gpu_name", "").lower() in gpu_names_lower
            and int(o.get("num_gpus", 0)) >= min_gpus
            and float(o.get("gpu_frac", 1.0)) >= 1.0  # no shared GPUs
        ]

        offers = [self._to_gpu_offer(o) for o in raw_offers]
        offers.sort(key=lambda o: o.score)

        logger.info(
            "vastai_offers_found",
            gpu_names=gpu_names,
            count=len(offers),
            top_score=round(offers[0].score, 4) if offers else None,
        )
        return offers

    async def create_instance(
        self,
        offer: GPUOffer,
        vllm_command: str,
        hf_token: str,
        disk_gb: int | None = None,
    ) -> ProvisionedInstance:
        """
        Rent an instance via the SDK and poll until SSH is ready.
        If volume_cache is enabled, attaches or creates a persistent volume
        to cache model weights, torch.compile artifacts, and Triton kernels.
        """
        _vllm_cfg = cfg.get("vllm", {})
        docker_image = _vllm_cfg.get("docker_image", "vllm/vllm-openai:latest")
        if disk_gb is None:
            disk_gb = _vllm_cfg.get("disk_gb", 50)

        vol_cfg = cfg.get("orchestrator", {}).get("volume_cache", {})
        vol_enabled = vol_cfg.get("enabled", False)
        mount_path = vol_cfg.get("mount_path", "/cache")

        # Build env vars — redirect all caches to the volume mount if enabled
        env_parts = [f"-e HF_TOKEN={hf_token}"]
        if vol_enabled:
            env_parts += [
                f"-e HF_HOME={mount_path}/huggingface",
                f"-e VLLM_CACHE_ROOT={mount_path}/vllm",
                f"-e TRITON_CACHE_DIR={mount_path}/triton",
            ]
        for k, v in _vllm_cfg.get("env", {}).items():
            env_parts.append(f"-e {k}={v}")
        env_str = " ".join(env_parts)

        # Snapshot existing instance IDs so we can detect the new one
        existing_instances = await asyncio.to_thread(self._client.show_instances)
        existing_ids = set()
        if isinstance(existing_instances, list):
            existing_ids = {inst.get("id") for inst in existing_instances}

        # Prepend pip installs if configured
        pre_install = _vllm_cfg.get("pre_install", [])
        if pre_install:
            pip_cmd = "pip install " + " ".join(pre_install)
            startup_cmd = f"{pip_cmd} && {vllm_command}"
        else:
            startup_cmd = vllm_command

        # Volume handling — check for existing volume or create new one
        volume_info: dict[str, Any] | None = None
        if vol_enabled:
            machine_id = offer.raw.get("machine_id")
            if machine_id:
                existing_vol = await self.find_volume_for_machine(machine_id)
                if existing_vol:
                    vol_id = existing_vol["id"]
                    logger.info(
                        "vastai_volume_reusing",
                        volume_id=vol_id,
                        machine_id=machine_id,
                        disk_space=existing_vol.get("disk_space"),
                    )
                    volume_info = {
                        "mount_path": mount_path,
                        "create_new": False,
                        "volume_id": vol_id,
                    }
                else:
                    vol_offer_id = await self._find_volume_offer_for_machine(machine_id)
                    if vol_offer_id:
                        vol_size = vol_cfg.get("size_gb", 40)
                        # Vast.ai labels must be alphanumeric (no hyphens/underscores)
                        label_prefix = vol_cfg.get("label_prefix", "st-cache").replace("-", "")
                        label = f"{label_prefix}{machine_id}"
                        logger.info(
                            "vastai_volume_creating",
                            machine_id=machine_id,
                            vol_offer_id=vol_offer_id,
                            size_gb=vol_size,
                            label=label,
                        )
                        volume_info = {
                            "mount_path": mount_path,
                            "create_new": True,
                            "volume_id": vol_offer_id,
                            "size": vol_size,
                            "name": label,
                        }
                    else:
                        logger.info(
                            "vastai_volume_no_offer",
                            machine_id=machine_id,
                            msg="No volume storage available on this machine, skipping cache",
                        )

        # Use direct REST API for instance creation — the SDK silently
        # swallows volume-related errors and returns empty strings.
        result = await self._create_instance_api(
            offer_id=int(offer.offer_id),
            docker_image=docker_image,
            disk_gb=disk_gb,
            onstart_cmd=startup_cmd,
            env_str=env_str,
            volume_info=volume_info,
        )

        logger.debug("vastai_create_instance_response", result=repr(result)[:500])

        # SDK create_instance often returns None even on success.
        # Detect the new instance by diffing show_instances before/after.
        instance_id: str | None = None

        if isinstance(result, dict):
            instance_id = str(result.get("new_contract", result.get("id", "")))
        elif result is not None and str(result).strip():
            instance_id = str(result).strip()

        if not instance_id:
            # Poll show_instances to find the newly created instance
            for _poll in range(6):
                await asyncio.sleep(2)
                current = await asyncio.to_thread(self._client.show_instances)
                if isinstance(current, list):
                    new_instances = [
                        inst for inst in current
                        if inst.get("id") not in existing_ids
                    ]
                    if new_instances:
                        instance_id = str(new_instances[0]["id"])
                        logger.info(
                            "vastai_instance_discovered",
                            instance_id=instance_id,
                            method="show_instances_diff",
                        )
                        break

        if not instance_id:
            raise RuntimeError(
                f"Vast.ai create_instance returned no ID and could not find "
                f"new instance via show_instances (offer_id={offer.offer_id})"
            )

        logger.info("vastai_instance_created", instance_id=instance_id, offer_id=offer.offer_id)

        # Poll until ssh_host is available
        ssh_timeout_s = cfg.get("orchestrator", {}).get("ssh_ready_timeout_s", 600)
        max_attempts = ssh_timeout_s // 5
        for attempt in range(max_attempts):
            await asyncio.sleep(5)
            info = await self.get_instance(instance_id)
            ssh_host = info.get("ssh_host", "")
            ssh_port = int(info.get("ssh_port", 22))
            status = info.get("actual_status", "unknown")

            # Log progress every 30s (6 attempts)
            if attempt % 6 == 5:
                logger.info(
                    "vastai_waiting_for_ssh",
                    instance_id=instance_id,
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                    elapsed_s=(attempt + 1) * 5,
                    status=status,
                    ssh_host=ssh_host or "(not ready)",
                )

            if ssh_host and status == "running":
                logger.info(
                    "vastai_instance_ready",
                    instance_id=instance_id,
                    ssh_host=ssh_host,
                    ssh_port=ssh_port,
                    status=status,
                    wait_s=(attempt + 1) * 5,
                )
                return ProvisionedInstance(
                    instance_id=instance_id,
                    provider="vastai",
                    gpu_model=info.get("gpu_name", offer.gpu_name),
                    ssh_host=ssh_host,
                    ssh_port=ssh_port,
                    http_port=ssh_port + 1,  # Vast.ai tunnel convention
                    status=status,
                )

        raise TimeoutError(
            f"Vast.ai instance {instance_id} did not expose ssh_host within 600 s"
        )

    # -- Volume management ---------------------------------------------------

    async def list_volumes(self) -> list[dict]:
        """List all owned volumes."""
        result = await asyncio.to_thread(self._client.show_volumes)
        if isinstance(result, list):
            return result
        return []

    async def _create_instance_api(
        self,
        offer_id: int,
        docker_image: str,
        disk_gb: int,
        onstart_cmd: str,
        env_str: str,
        volume_info: dict[str, Any] | None = None,
    ) -> dict | None:
        """Create an instance via the Vast.ai REST API directly.

        The SDK's ``create_instance`` silently swallows errors when volume
        params are invalid, so we use the REST API for reliability.
        """
        import requests as _requests

        # Parse env string "-e K=V -e K2=V2" into dict
        env_dict: dict[str, str] = {}
        parts = env_str.split("-e ")
        for part in parts:
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                env_dict[k.strip()] = v.strip()

        json_blob: dict[str, Any] = {
            "client_id": "me",
            "image": docker_image,
            "env": env_dict,
            "disk": disk_gb,
            "onstart": onstart_cmd,
            "runtype": "ssh_direc ssh_proxy",
        }
        if volume_info:
            json_blob["volume_info"] = volume_info

        url = f"https://console.vast.ai/api/v0/asks/{offer_id}/"
        headers = {
            "Authorization": f"Bearer {self._client.api_key}",
            "Content-Type": "application/json",
        }

        def _do_request():
            r = _requests.put(url, headers=headers, json=json_blob, timeout=30)
            r.raise_for_status()
            return r.json()

        result = await asyncio.to_thread(_do_request)
        if isinstance(result, dict) and result.get("success"):
            return result
        logger.error("vastai_create_instance_api_failed", result=result)
        return None

    async def _find_volume_offer_for_machine(self, machine_id: int) -> int | None:
        """Find a volume offer (storage capacity) on a specific machine.

        Returns the volume offer ID suitable for ``create_instance --create-volume``,
        or None if the machine has no available volume storage.
        """
        try:
            results = await asyncio.to_thread(
                self._client.search_volumes,
                query=f"machine_id={machine_id}",
            )
            if isinstance(results, list):
                for r in results:
                    if r.get("machine_id") == machine_id:
                        return int(r["id"])
        except Exception as exc:
            logger.warning("vastai_volume_offer_search_failed", machine_id=machine_id, error=str(exc))
        return None

    async def find_volume_for_machine(self, machine_id: int) -> dict | None:
        """Find an existing shittytoken cache volume on a specific machine."""
        vol_cfg = cfg.get("orchestrator", {}).get("volume_cache", {})
        label_prefix = vol_cfg.get("label_prefix", "st-cache")

        volumes = await self.list_volumes()
        for vol in volumes:
            if vol.get("machine_id") == machine_id:
                label = vol.get("label", "") or ""
                if label.startswith(label_prefix):
                    return vol
        return None

    async def create_volume_on_machine(self, offer_id: int, size_gb: int, label: str) -> dict | None:
        """Create a persistent volume from an offer (machine).

        Uses the Vast.ai volume API: PUT /api/v0/volumes/ with {id, size}.
        """
        try:
            result = await asyncio.to_thread(
                self._client.create_volume,
                id=offer_id,
                size=size_gb,
            )
            logger.info("vastai_volume_created", offer_id=offer_id, size_gb=size_gb, label=label, result=repr(result)[:200])
            return result if isinstance(result, dict) else {"success": True}
        except Exception as exc:
            logger.warning("vastai_volume_create_failed", offer_id=offer_id, error=str(exc))
            return None

    async def delete_volume(self, volume_id: int) -> None:
        """Delete a persistent volume."""
        try:
            await asyncio.to_thread(self._client.delete_volume, id=volume_id)
            logger.info("vastai_volume_deleted", volume_id=volume_id)
        except Exception as exc:
            logger.warning("vastai_volume_delete_failed", volume_id=volume_id, error=str(exc))

    async def evict_stale_volumes(self, max_age_days: float) -> int:
        """Delete volumes that haven't been used in max_age_days. Returns count deleted."""
        import time as _time

        vol_cfg = cfg.get("orchestrator", {}).get("volume_cache", {})
        label_prefix = vol_cfg.get("label_prefix", "st-cache")
        volumes = await self.list_volumes()
        now = _time.time()
        evicted = 0

        for vol in volumes:
            label = vol.get("label", "") or ""
            if not label.startswith(label_prefix):
                continue

            # Use end_date (last use) or start_date as reference
            # Vast.ai timestamps are epoch floats
            last_used = vol.get("end_date") or vol.get("start_date") or 0
            if isinstance(last_used, str):
                continue  # skip unparseable
            age_days = (now - last_used) / 86400.0
            if age_days > max_age_days:
                instances = vol.get("instances", [])
                if instances:
                    logger.debug("vastai_volume_evict_skip_attached", volume_id=vol["id"], instances=instances)
                    continue
                logger.info("vastai_volume_evicting", volume_id=vol["id"], age_days=round(age_days, 1), machine_id=vol.get("machine_id"))
                await self.delete_volume(vol["id"])
                evicted += 1

        if evicted:
            logger.info("vastai_volumes_evicted", count=evicted)
        return evicted

    async def destroy_instance(self, instance_id: str) -> None:
        """Destroy an instance via the SDK."""
        await asyncio.to_thread(self._client.destroy_instance, id=int(instance_id))
        logger.info("vastai_instance_destroyed", instance_id=instance_id)

    async def get_instance(self, instance_id: str) -> dict:
        """Get instance details via the SDK."""
        # show_instance(id=) can return None with raw=True, so fall back
        # to filtering show_instances() by ID.
        result = await asyncio.to_thread(
            self._client.show_instance, id=int(instance_id)
        )
        if isinstance(result, dict):
            return result
        # Fallback: search in show_instances list
        instances = await asyncio.to_thread(self._client.show_instances)
        if isinstance(instances, list):
            for inst in instances:
                if str(inst.get("id")) == str(instance_id):
                    return inst
        return {}

    async def list_all_instances(self) -> list[dict]:
        """List all instances currently rented on this Vast.ai account."""
        result = await asyncio.to_thread(self._client.show_instances)
        if isinstance(result, list):
            return result
        return []

    @staticmethod
    def _to_gpu_offer(raw: dict) -> GPUOffer:
        """Convert a Vast.ai SDK offer dict to a GPUOffer."""
        return GPUOffer(
            offer_id=str(raw["id"]),
            provider="vastai",
            gpu_name=raw.get("gpu_name", "unknown"),
            num_gpus=int(raw.get("num_gpus", 1)),
            cost_per_hour_usd=raw.get("dph_total"),
            reliability=raw.get("reliability2"),
            inet_up_mbps=raw.get("inet_up"),
            inet_down_mbps=raw.get("inet_down"),
            inet_up_cost_per_gb=raw.get("inet_up_cost"),
            inet_down_cost_per_gb=raw.get("inet_down_cost"),
            pcie_bw_gbps=raw.get("pcie_bw"),
            bw_nvlink_gbps=raw.get("bw_nvlink"),
            dlperf=raw.get("dlperf"),
            raw=raw,
        )


# ---------------------------------------------------------------------------
# RunPod provider (spot-only via GraphQL)
# ---------------------------------------------------------------------------


class RunPodProvider(GPUProvider):
    """Provisions and manages spot GPU pods on RunPod.

    Uses the ``runpod`` SDK for queries and direct GraphQL mutations for
    spot-specific operations (``podRentInterruptable``, ``podBidResume``)
    which are not exposed by the SDK's convenience functions.

    All pods are created as **spot (interruptable)** instances.  Spot
    eviction delivers SIGTERM with a 5-second grace period and no advance
    warning — the caller must poll pod status externally to detect eviction.
    """

    def __init__(self, api_key: str, session: aiohttp.ClientSession | None = None) -> None:
        import runpod

        self._api_key = api_key
        runpod.api_key = api_key
        self._runpod = runpod

    @property
    def name(self) -> str:
        return "runpod"

    # -- GraphQL helpers -----------------------------------------------------

    def _graphql_query(self, query: str) -> dict[str, Any]:
        """Execute a raw GraphQL query/mutation against the RunPod API."""
        import requests

        resp = requests.post(
            "https://api.runpod.io/graphql",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"query": query},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if "errors" in data:
            raise RuntimeError(f"RunPod GraphQL error: {data['errors']}")
        return data.get("data", {})

    async def _async_graphql(self, query: str) -> dict[str, Any]:
        """Run a GraphQL query in a thread (the SDK is synchronous)."""
        return await asyncio.to_thread(self._graphql_query, query)

    # -- GPUProvider interface -----------------------------------------------

    async def find_offers(
        self,
        gpu_names: list[str],
        min_gpus: int = 2,
    ) -> list[GPUOffer]:
        """Query RunPod GPU types and return spot offers sorted by score."""
        _gpu_cfg = cfg["gpus"]
        runpod_cfg = cfg.get("orchestrator", {}).get("runpod", {})
        cloud_type = runpod_cfg.get("cloud_type", "COMMUNITY")

        all_gpus = await asyncio.to_thread(self._runpod.get_gpus)

        gpu_names_lower = {n.lower() for n in gpu_names}
        offers: list[GPUOffer] = []

        for gpu in all_gpus:
            display_name = gpu.get("displayName", "")
            if display_name.lower() not in gpu_names_lower:
                continue

            gpu_id = gpu.get("id", display_name)
            detail = await asyncio.to_thread(
                self._runpod.get_gpu, gpu_id, min_gpus
            )

            spot_price = self._extract_spot_price(detail, cloud_type)
            if spot_price is None or spot_price <= 0:
                continue

            min_vram = _gpu_cfg.get("min_gpu_total_ram_gb", 0)
            mem_gb = detail.get("memoryInGb", 0)
            if mem_gb * min_gpus < min_vram:
                continue

            if detail.get("maxGpuCount", 1) < min_gpus:
                continue

            offers.append(GPUOffer(
                offer_id=gpu_id,
                provider="runpod",
                gpu_name=display_name,
                num_gpus=min_gpus,
                cost_per_hour_usd=spot_price * min_gpus,
                reliability=None,
                inet_up_cost_per_gb=0.0,
                inet_down_cost_per_gb=0.0,
                raw={
                    **detail,
                    "spot_price_per_gpu": spot_price,
                    "cloud_type": cloud_type,
                },
            ))

        offers.sort(key=lambda o: o.score)

        logger.info(
            "runpod_offers_found",
            gpu_names=gpu_names,
            count=len(offers),
            top_score=round(offers[0].score, 4) if offers else None,
        )
        return offers

    async def create_instance(
        self,
        offer: GPUOffer,
        vllm_command: str,
        hf_token: str,
        disk_gb: int | None = None,
    ) -> ProvisionedInstance:
        """Create a spot pod via ``podRentInterruptable`` and wait for SSH."""
        _vllm_cfg = cfg.get("vllm", {})
        docker_image = _vllm_cfg.get("docker_image", "vllm/vllm-openai:latest")
        if disk_gb is None:
            disk_gb = _vllm_cfg.get("disk_gb", 50)

        cloud_type = offer.raw.get("cloud_type", "COMMUNITY")
        spot_price = offer.raw.get("spot_price_per_gpu") or 0

        # The vllm/vllm-openai image has ENTRYPOINT ["vllm", "serve"], so
        # dockerArgs (which overrides CMD) must only contain the arguments
        # after "vllm serve" to avoid doubling the command.
        docker_args = vllm_command.removeprefix("vllm serve ")

        # GraphQL env expects [{key, value}] objects, not a JSON string.
        mutation = f"""
        mutation {{
          podRentInterruptable(input: {{
            bidPerGpu: {spot_price:.4f}
            cloudType: {cloud_type}
            gpuCount: {offer.num_gpus}
            gpuTypeId: "{offer.offer_id}"
            name: "shittytoken-worker"
            imageName: "{docker_image}"
            containerDiskInGb: {disk_gb}
            volumeInGb: 0
            ports: "8080/http,22/tcp"
            startSsh: true
            supportPublicIp: true
            dockerArgs: "{docker_args}"
            env: [
              {{key: "HF_TOKEN", value: "{hf_token}"}}{self._build_env_entries(_vllm_cfg)}
            ]
          }}) {{
            id
            imageName
            machineId
          }}
        }}
        """

        logger.info(
            "runpod_creating_spot_pod",
            gpu_type=offer.offer_id,
            gpu_count=offer.num_gpus,
            spot_price=round(spot_price, 4),
            cloud_type=cloud_type,
        )

        result = await self._async_graphql(mutation)
        pod_data = result.get("podRentInterruptable", {})
        pod_id = pod_data.get("id")
        if not pod_id:
            raise RuntimeError(
                f"RunPod podRentInterruptable returned no pod ID: {result}"
            )

        logger.info("runpod_pod_created", pod_id=pod_id, offer_id=offer.offer_id)

        # Poll until pod is running with ports
        ssh_timeout_s = cfg.get("orchestrator", {}).get("ssh_ready_timeout_s", 600)
        max_attempts = ssh_timeout_s // 5
        for attempt in range(max_attempts):
            await asyncio.sleep(5)
            info = await self.get_instance(pod_id)
            status = info.get("desiredStatus", "")
            runtime = info.get("runtime") or {}
            ports = runtime.get("ports") or []

            # SSH goes through RunPod's proxy: {podHostId}@ssh.runpod.io
            machine = info.get("machine") or {}
            pod_host_id = machine.get("podHostId")

            if attempt % 6 == 5:
                logger.info(
                    "runpod_waiting_for_ready",
                    pod_id=pod_id,
                    attempt=attempt + 1,
                    elapsed_s=(attempt + 1) * 5,
                    status=status,
                    pod_host_id=pod_host_id or "(not ready)",
                )

            if pod_host_id and ports and status == "RUNNING":
                http_port = self._extract_http_port(ports)
                logger.info(
                    "runpod_pod_ready",
                    pod_id=pod_id,
                    ssh_user=pod_host_id,
                    ssh_host="ssh.runpod.io",
                    http_port=http_port,
                    wait_s=(attempt + 1) * 5,
                )
                return ProvisionedInstance(
                    instance_id=pod_id,
                    provider="runpod",
                    gpu_model=offer.gpu_name,
                    ssh_host="ssh.runpod.io",
                    ssh_port=22,
                    ssh_user=pod_host_id,
                    http_port=http_port,
                    status="running",
                )

        raise TimeoutError(
            f"RunPod pod {pod_id} did not become ready within 600 s"
        )

    async def destroy_instance(self, instance_id: str) -> None:
        """Terminate a pod permanently."""
        await asyncio.to_thread(self._runpod.terminate_pod, instance_id)
        logger.info("runpod_pod_destroyed", pod_id=instance_id)

    async def get_instance(self, instance_id: str) -> dict:
        """Get pod details via GraphQL (includes machine.podHostId for SSH).

        Returns empty dict on failure.
        """
        query = f"""
        query {{
          pod(input: {{podId: "{instance_id}"}}) {{
            id
            desiredStatus
            podType
            costPerHr
            gpuCount
            machineId
            imageName
            env
            machine {{
              podHostId
              gpuDisplayName
            }}
            runtime {{
              uptimeInSeconds
              ports {{
                ip
                isIpPublic
                privatePort
                publicPort
                type
              }}
            }}
          }}
        }}
        """
        try:
            data = await self._async_graphql(query)
            pod = data.get("pod")
            if isinstance(pod, dict):
                return pod
        except Exception as exc:
            logger.warning("runpod_get_pod_failed", pod_id=instance_id, error=str(exc))
        return {}

    async def list_all_instances(self) -> list[dict]:
        """List all pods on this RunPod account."""
        query = """
        query {
          myself {
            pods {
              id desiredStatus gpuCount costPerHr
              machine { gpuDisplayName }
            }
          }
        }
        """
        try:
            data = await self._async_graphql(query)
            pods = data.get("myself", {}).get("pods", [])
            return pods if isinstance(pods, list) else []
        except Exception as exc:
            logger.warning("runpod_list_pods_failed", error=str(exc))
            return []

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _build_env_entries(vllm_cfg: dict) -> str:
        """Build GraphQL env entries from config vllm.env dict."""
        entries = ""
        for k, v in vllm_cfg.get("env", {}).items():
            entries += f', {{key: "{k}", value: "{v}"}}'
        return entries

    @staticmethod
    def _extract_spot_price(detail: dict, cloud_type: str) -> float | None:
        """Extract the spot price from a GPU detail dict."""
        if cloud_type == "SECURE":
            return detail.get("secureSpotPrice")
        if cloud_type == "COMMUNITY":
            return detail.get("communitySpotPrice")
        # ALL — prefer community, fall back to secure
        return detail.get("communitySpotPrice") or detail.get("secureSpotPrice")

    @staticmethod
    def _extract_http_port(ports: list[dict]) -> int | None:
        """Extract the public port mapped to container port 8080 (vLLM)."""
        for port_entry in ports:
            if port_entry.get("privatePort") == 8080:
                return int(port_entry.get("publicPort", 8080))
        return None

    @staticmethod
    def is_spot_eviction(pod_info: dict) -> bool:
        """Return True if the pod status indicates a spot eviction.

        A spot eviction is detected when the pod is EXITED but was
        interruptable — distinguishing it from a manual stop.
        """
        status = pod_info.get("desiredStatus", "")
        pod_type = pod_info.get("podType", "")
        return status == "EXITED" and pod_type == "INTERRUPTABLE"


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


def get_provider(
    provider_name: str,
    vastai_api_key: str = "",
    runpod_api_key: str = "",
    session: aiohttp.ClientSession | None = None,
) -> GPUProvider:
    """Instantiate a provider by name.

    Args:
        provider_name: One of ``"vastai"``, ``"runpod"``.
        vastai_api_key: API key for Vast.ai.
        runpod_api_key: API key for RunPod.
        session: Optional shared aiohttp session.

    Raises:
        ValueError: If the provider name is unknown or the API key is missing.
    """
    if provider_name == "vastai":
        if not vastai_api_key:
            raise ValueError("vastai_api_key is required for the vastai provider")
        return VastAIProvider(api_key=vastai_api_key, session=session)
    if provider_name == "runpod":
        if not runpod_api_key:
            raise ValueError("runpod_api_key is required for the runpod provider")
        return RunPodProvider(api_key=runpod_api_key, session=session)
    raise ValueError(f"Unknown provider: {provider_name!r} (expected 'vastai' or 'runpod')")


# ---------------------------------------------------------------------------
# Provider-agnostic entry points
# ---------------------------------------------------------------------------


def build_worker_url(record) -> str:
    """Build the HTTP URL to reach the vLLM server on a provisioned instance.

    Provider-specific logic:
    - RunPod: HTTPS proxy URL based on pod ID and worker port
    - Vast.ai / default: direct HTTP to ssh_host on the http_port
    """
    worker_port = cfg.get("vllm", {}).get("worker_port", 8080)
    if record.provider == "runpod":
        return f"https://{record.instance_id}-{worker_port}.proxy.runpod.net"
    http_port = record.http_port or worker_port
    return f"http://{record.ssh_host}:{http_port}"


async def build_deployment_plan(
    provider: GPUProvider,
    config: Configuration,
    model_id: str,
    gpu_names: list[str],
    config_source: str = "knowledge_graph",
) -> DeploymentPlan:
    """
    Find the best offer and build a DeploymentPlan WITHOUT executing anything.
    Raises RuntimeError if no offers are found.
    """
    vllm_command = build_vllm_command(config, model_id)

    offers = await provider.find_offers(
        gpu_names=gpu_names, min_gpus=config.tensor_parallel_size
    )
    if not offers:
        raise RuntimeError(
            f"No GPU offers found on {provider.name} for {gpu_names}"
        )

    best = offers[0]
    return DeploymentPlan(
        config_source=config_source,
        tensor_parallel_size=config.tensor_parallel_size,
        max_model_len=config.max_model_len,
        gpu_memory_utilization=config.gpu_memory_utilization,
        quantization=config.quantization,
        kv_cache_dtype=config.kv_cache_dtype,
        max_num_seqs=config.max_num_seqs,
        enable_prefix_caching=config.enable_prefix_caching,
        enforce_eager=config.enforce_eager,
        vllm_command=vllm_command,
        offer=best,
        model_id=model_id,
    )


async def execute_deployment(
    plan: DeploymentPlan,
    provider: GPUProvider,
    hf_token: str,
) -> ProvisionedInstance:
    """
    Execute a previously approved DeploymentPlan.
    Rents the instance on the chosen provider and waits for SSH readiness.
    """
    instance = await provider.create_instance(
        offer=plan.offer,
        vllm_command=plan.vllm_command,
        hf_token=hf_token,
    )

    logger.info(
        "instance_provisioned",
        provider=plan.offer.provider,
        gpu_model=instance.gpu_model,
        instance_id=instance.instance_id,
    )
    return instance
