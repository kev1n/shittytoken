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
- --ipc=host  (required for tensor parallelism)

Provider abstraction:
- GPUProvider is the abstract interface
- VastAIProvider is the sole implementation for now
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

    Always injects --enable-prefix-caching and --ipc=host.
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

    parts: list[str] = [
        "vllm",
        "serve",
        model_id,
        "--tensor-parallel-size", str(config.tensor_parallel_size),
        "--max-model-len", str(config.max_model_len),
        "--gpu-memory-utilization", str(config.gpu_memory_utilization),
        "--max-num-seqs", str(config.max_num_seqs),
        "--enable-prefix-caching",
        "--ipc=host",
    ]

    if config.quantization is not None:
        parts += ["--quantization", config.quantization]

    if config.kv_cache_dtype != "auto":
        parts += ["--kv-cache-dtype", config.kv_cache_dtype]

    if config.enforce_eager:
        parts.append("--enforce-eager")

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
    """A running instance with SSH access."""

    instance_id: str
    provider: str
    gpu_model: str
    ssh_host: str
    ssh_port: int
    status: str


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
        disk_gb: int = 50,
    ) -> ProvisionedInstance:
        """
        Rent an instance via the SDK and poll until SSH is ready.
        """
        env_str = f"-e HF_TOKEN={hf_token}"

        result = await asyncio.to_thread(
            self._client.create_instance,
            id=int(offer.offer_id),
            image="vllm/vllm-openai:latest",
            disk=disk_gb,
            ssh=True,
            direct=True,
            onstart_cmd=vllm_command,
            env=env_str,
        )

        logger.debug("vastai_create_instance_response", result=repr(result)[:500])

        # SDK may return dict, string, int, or None depending on version
        if isinstance(result, dict):
            instance_id = str(result.get("new_contract", result.get("id", "")))
        elif result is not None:
            # Sometimes returns just the instance ID as a string or int
            instance_id = str(result)
        else:
            # None response — the instance was likely created but SDK didn't
            # return the ID. Fall back to the offer ID and hope for the best.
            logger.warning(
                "vastai_create_returned_none",
                offer_id=offer.offer_id,
                msg="SDK returned None — using offer_id as instance_id fallback",
            )
            instance_id = str(offer.offer_id)
        logger.info("vastai_instance_created", instance_id=instance_id, offer_id=offer.offer_id)

        # Poll until ssh_host is available (up to 10 minutes)
        for attempt in range(120):
            await asyncio.sleep(5)
            info = await self.get_instance(instance_id)
            ssh_host = info.get("ssh_host", "")
            ssh_port = int(info.get("ssh_port", 22))
            status = info.get("actual_status", "unknown")
            if ssh_host:
                logger.info(
                    "vastai_instance_ready",
                    instance_id=instance_id,
                    ssh_host=ssh_host,
                    ssh_port=ssh_port,
                    attempts=attempt + 1,
                )
                return ProvisionedInstance(
                    instance_id=instance_id,
                    provider="vastai",
                    gpu_model=info.get("gpu_name", offer.gpu_name),
                    ssh_host=ssh_host,
                    ssh_port=ssh_port,
                    status=status,
                )

        raise TimeoutError(
            f"Vast.ai instance {instance_id} did not expose ssh_host within 600 s"
        )

    async def destroy_instance(self, instance_id: str) -> None:
        """Destroy an instance via the SDK."""
        await asyncio.to_thread(self._client.destroy_instance, id=int(instance_id))
        logger.info("vastai_instance_destroyed", instance_id=instance_id)

    async def get_instance(self, instance_id: str) -> dict:
        """Get instance details via the SDK."""
        result = await asyncio.to_thread(
            self._client.show_instance, id=int(instance_id)
        )
        if isinstance(result, dict):
            return result
        return {}

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
# Provider-agnostic entry points
# ---------------------------------------------------------------------------


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
