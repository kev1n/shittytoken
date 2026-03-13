"""
Provisioner — GPU instance lifecycle management for Vast.ai and RunPod.

build_vllm_command() converts a Configuration dataclass into a vllm serve
command string.  All vLLM parameters come from Configuration; nothing is
hard-coded here.

INVARIANTS (enforced by build_vllm_command):
- config.gpu_memory_utilization < 1.0
- config.enable_prefix_caching is True

Always injected:
- --enable-prefix-caching
- --ipc=host  (required for tensor parallelism)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import aiohttp
import structlog

from ..config import cfg
from ..knowledge.schema import Configuration

logger = structlog.get_logger()

# Vast.ai REST API base URL
_VASTAI_API_BASE = "https://console.vast.ai/api/v0"
# RunPod GraphQL endpoint
_RUNPOD_API_BASE = "https://api.runpod.io/graphql"


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

    Maps Configuration fields:
    - tensor_parallel_size  → --tensor-parallel-size
    - max_model_len         → --max-model-len
    - gpu_memory_utilization→ --gpu-memory-utilization
    - quantization          → --quantization  (omitted if None)
    - kv_cache_dtype        → --kv-cache-dtype (omitted if "auto")
    - max_num_seqs          → --max-num-seqs
    - enforce_eager         → --enforce-eager (flag only, no value)
    - model_id              → positional argument after "serve"
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
# Shared result type
# ---------------------------------------------------------------------------


@dataclass
class ProvisionedInstance:
    instance_id: str
    provider: str      # "vastai" | "runpod"
    gpu_model: str
    ssh_host: str
    ssh_port: int
    status: str


@dataclass
class DeploymentPlan:
    """All details about a planned deployment, presented for human approval."""

    # Config source
    config_source: str  # "knowledge_graph" | "llm_proposal"

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

    # Provider & offer details
    provider: str       # "vastai" | "runpod"
    gpu_name: str
    num_gpus: int
    cost_per_hour_usd: float | None
    reliability: float | None
    offer_id: str

    # Networking & interconnect
    inet_up_mbps: float | None       # internet upload speed (MB/s)
    inet_down_mbps: float | None     # internet download speed (MB/s)
    inet_up_cost_per_gb: float | None   # $/GB upload
    inet_down_cost_per_gb: float | None # $/GB download
    pcie_bw_gbps: float | None       # PCIe bandwidth (GB/s)
    bw_nvlink_gbps: float | None     # NVLink bandwidth (GB/s), None if no NVLink
    dlperf: float | None             # Vast.ai DL performance score

    # Model
    model_id: str

    def estimated_bandwidth_cost_per_mtok(self) -> float | None:
        """
        Rough estimate of bandwidth cost per million output tokens.

        Assumes ~4 bytes per token streamed as SSE JSON (~20 bytes overhead).
        Upload dominates (server → client = upload from the instance's perspective).
        """
        if self.inet_up_cost_per_gb is None:
            return None
        bytes_per_token = 24  # ~4 bytes token + SSE framing
        gb_per_mtok = (bytes_per_token * 1_000_000) / (1024 ** 3)
        return gb_per_mtok * self.inet_up_cost_per_gb

    def display(self) -> str:
        """Human-readable summary for approval."""
        lines = [
            "",
            "=" * 64,
            "  DEPLOYMENT PLAN",
            "=" * 64,
            "",
            f"  Provider:       {self.provider}",
            f"  Offer ID:       {self.offer_id}",
            f"  GPU:            {self.num_gpus}× {self.gpu_name}",
        ]
        if self.cost_per_hour_usd is not None:
            lines.append(f"  Compute cost:   ${self.cost_per_hour_usd:.4f}/hr")
        if self.reliability is not None:
            lines.append(f"  Reliability:    {self.reliability:.1%}")
        if self.dlperf is not None:
            lines.append(f"  DL perf score:  {self.dlperf:.1f}")

        # Networking section
        lines.append("")
        lines.append("  Networking:")
        if self.inet_up_mbps is not None:
            lines.append(f"    Upload:       {self.inet_up_mbps:.0f} MB/s")
        if self.inet_down_mbps is not None:
            lines.append(f"    Download:     {self.inet_down_mbps:.0f} MB/s")
        if self.inet_up_cost_per_gb is not None:
            lines.append(f"    Upload cost:  ${self.inet_up_cost_per_gb:.4f}/GB")
        if self.inet_down_cost_per_gb is not None:
            lines.append(f"    Download cost: ${self.inet_down_cost_per_gb:.4f}/GB")
        bw_cost = self.estimated_bandwidth_cost_per_mtok()
        if bw_cost is not None:
            lines.append(f"    Est. BW cost: ${bw_cost:.6f}/Mtok (upload)")

        # Interconnect section
        lines.append("")
        lines.append("  Interconnect:")
        if self.pcie_bw_gbps is not None:
            lines.append(f"    PCIe BW:      {self.pcie_bw_gbps:.1f} GB/s")
        if self.bw_nvlink_gbps is not None:
            nvlink_str = f"{self.bw_nvlink_gbps:.1f} GB/s" if self.bw_nvlink_gbps > 0 else "none"
            lines.append(f"    NVLink BW:    {nvlink_str}")
        elif self.num_gpus > 1:
            lines.append(f"    NVLink BW:    none (PCIe-only multi-GPU)")

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
# Vast.ai provider
# ---------------------------------------------------------------------------


def _score_offer(offer: dict) -> float:
    """
    Composite score for ranking Vast.ai offers. Lower is better.

    Components (all normalized to $/hr-equivalent scale):
      1. Compute cost:   dph_total (dominant factor)
      2. Bandwidth cost: inet_up_cost × estimated GB/hr of token output
      3. Upload penalty:  penalize slow upload (tokens must reach clients)
      4. NVLink bonus:    discount for NVLink (faster tensor parallelism)

    For a bulk-inference platform, we want the cheapest instance that can
    actually deliver tokens to clients without a network bottleneck.
    """
    cost = offer.get("dph_total", float("inf"))

    # Estimated bandwidth cost per hour:
    # At ~20 TPS peak, ~24 bytes/tok SSE, that's ~1.7 MB/hr = negligible.
    # At scale (500 TPS), ~43 MB/hr ≈ 0.04 GB/hr.
    # We use 0.1 GB/hr as a conservative estimate.
    inet_up_cost = offer.get("inet_up_cost", 0) or 0
    bw_cost_per_hr = inet_up_cost * 0.1  # $/hr from bandwidth

    # Upload speed penalty: below 200 MB/s adds a penalty.
    # A 100 MB/s link can still stream ~4000 concurrent 25-byte/tok streams,
    # but we prefer faster links for burst handling.
    inet_up = offer.get("inet_up", 0) or 0
    upload_penalty = max(0, (200 - inet_up) / 200) * 0.05  # up to $0.05/hr penalty

    # NVLink bonus: multi-GPU without NVLink is PCIe-bottlenecked for
    # tensor parallelism. NVLink offers get a discount.
    nvlink = offer.get("bw_nvlink", 0) or 0
    num_gpus = offer.get("num_gpus", 1) or 1
    nvlink_bonus = 0.0
    if num_gpus > 1 and nvlink > 0:
        nvlink_bonus = -0.03  # $0.03/hr discount for NVLink

    return cost + bw_cost_per_hr + upload_penalty + nvlink_bonus


class VastAIProvisioner:
    """Provisions and manages GPU instances on Vast.ai."""

    def __init__(self, api_key: str, session: aiohttp.ClientSession) -> None:
        self._api_key = api_key
        self._session = session

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def find_offers(
        self,
        gpu_names: list[str],
        min_gpus: int | None = None,
        min_reliability: float | None = None,
        min_inet_up_mbps: float | None = None,
        min_pcie_bw_gbps: float | None = None,
    ) -> list[dict]:
        """
        POST /api/v0/bundles/ with filters for GPU, reliability, and network.

        Returns offers scored by a composite metric that balances cost,
        network quality, and interconnect speed — not just cheapest.
        """
        _gpu_cfg = cfg["gpus"]
        if min_gpus is None:
            min_gpus = _gpu_cfg["min_gpus"]
        if min_reliability is None:
            min_reliability = _gpu_cfg["min_reliability"]
        if min_inet_up_mbps is None:
            min_inet_up_mbps = _gpu_cfg["min_inet_up_mbps"]
        if min_pcie_bw_gbps is None:
            min_pcie_bw_gbps = _gpu_cfg["min_pcie_bw_gbps"]

        filters = {
            "verified": {"eq": True},
            "num_gpus": {"gte": min_gpus},
            "reliability2": {"gte": min_reliability},
            "rentable": {"eq": True},
            "gpu_name": {"in": gpu_names},
            "inet_up": {"gte": min_inet_up_mbps},
            "pcie_bw": {"gte": min_pcie_bw_gbps},
        }
        url = f"{_VASTAI_API_BASE}/bundles/"

        async with self._session.post(
            url,
            headers=self._headers(),
            json={"q": filters},
            timeout=aiohttp.ClientTimeout(total=30.0),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        offers = data.get("offers", [])

        # Score offers: lower is better. Composite of cost + bandwidth penalty.
        # We want cheap instances with good upload speed and PCIe/NVLink.
        for offer in offers:
            offer["_score"] = _score_offer(offer)
        offers.sort(key=lambda o: o["_score"])

        logger.info(
            "vastai_offers_found",
            gpu_names=gpu_names,
            min_gpus=min_gpus,
            count=len(offers),
            top_score=round(offers[0]["_score"], 4) if offers else None,
        )
        return offers

    async def create_instance(
        self,
        offer_id: str,
        vllm_command: str,
        hf_token: str,
        disk_gb: int = 50,
    ) -> ProvisionedInstance:
        """
        PUT /api/v0/asks/{offer_id}/ to rent an instance, then poll
        GET /api/v0/instances/{id}/ until ssh_host is populated.
        """
        url = f"{_VASTAI_API_BASE}/asks/{offer_id}/"
        body = {
            "client_id": "me",
            "image": "vllm/vllm-openai:latest",
            "env": {
                "HF_TOKEN": hf_token,
            },
            "disk": disk_gb,
            "onstart": vllm_command,
            "runtype": "ssh",
        }

        async with self._session.put(
            url,
            headers=self._headers(),
            json=body,
            timeout=aiohttp.ClientTimeout(total=60.0),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        instance_id = str(data.get("new_contract", data.get("id", "")))
        logger.info("vastai_instance_created", instance_id=instance_id, offer_id=offer_id)

        # Poll until ssh_host is available (up to 10 minutes)
        for attempt in range(120):
            await asyncio.sleep(5)
            info = await self.get_instance(instance_id)
            ssh_host = info.get("ssh_host") or info.get("public_ipaddr", "")
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
                    gpu_model=info.get("gpu_name", "unknown"),
                    ssh_host=ssh_host,
                    ssh_port=ssh_port,
                    status=status,
                )

        raise TimeoutError(
            f"Vast.ai instance {instance_id} did not expose ssh_host within 600 s"
        )

    async def destroy_instance(self, instance_id: str) -> None:
        """DELETE /api/v0/instances/{instance_id}/"""
        url = f"{_VASTAI_API_BASE}/instances/{instance_id}/"
        async with self._session.delete(
            url,
            headers=self._headers(),
            timeout=aiohttp.ClientTimeout(total=30.0),
        ) as resp:
            resp.raise_for_status()
        logger.info("vastai_instance_destroyed", instance_id=instance_id)

    async def get_instance(self, instance_id: str) -> dict:
        """GET /api/v0/instances/{instance_id}/"""
        url = f"{_VASTAI_API_BASE}/instances/{instance_id}/"
        async with self._session.get(
            url,
            headers=self._headers(),
            timeout=aiohttp.ClientTimeout(total=15.0),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
        # Vast.ai wraps single-instance responses in an "instances" list
        instances = data.get("instances")
        if instances and isinstance(instances, list):
            return instances[0]
        return data


# ---------------------------------------------------------------------------
# RunPod provider (fallback)
# ---------------------------------------------------------------------------


class RunPodProvisioner:
    """Fallback GPU provider when Vast.ai is unavailable or has no suitable offers."""

    def __init__(self, api_key: str, session: aiohttp.ClientSession) -> None:
        self._api_key = api_key
        self._session = session

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    async def find_offers(self, gpu_types: list[str]) -> list[dict]:
        """
        Query the RunPod GraphQL API for available GPU pod types.
        Returns a list of offer dicts with at least: id, gpu_type, cost_per_hr.
        """
        query = """
        query GpuTypes {
            gpuTypes {
                id
                displayName
                memoryInGb
                secureCloud
                communityCloud
                lowestPrice {
                    minimumBidPrice
                    uninterruptablePrice
                }
            }
        }
        """
        async with self._session.post(
            f"{_RUNPOD_API_BASE}?api_key={self._api_key}",
            headers=self._headers(),
            json={"query": query},
            timeout=aiohttp.ClientTimeout(total=30.0),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        all_types = data.get("data", {}).get("gpuTypes", [])
        matching = [
            g for g in all_types
            if any(gt.lower() in g.get("displayName", "").lower() for gt in gpu_types)
        ]
        logger.info("runpod_offers_found", gpu_types=gpu_types, count=len(matching))
        return matching

    async def create_instance(
        self,
        gpu_type: str,
        vllm_command: str,
        hf_token: str,
    ) -> ProvisionedInstance:
        """
        Create a RunPod pod via GraphQL mutation.
        Polls until the pod status is RUNNING and an IP is available.
        """
        mutation = """
        mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                desiredStatus
                imageName
            }
        }
        """
        variables = {
            "input": {
                "gpuTypeId": gpu_type,
                "name": "shittytoken-worker",
                "imageName": "vllm/vllm-openai:latest",
                "startCommand": vllm_command,
                "cloudType": "SECURE",
                "gpuCount": 2,
                "volumeInGb": 50,
                "containerDiskInGb": 20,
                "env": [{"key": "HF_TOKEN", "value": hf_token}],
            }
        }

        async with self._session.post(
            f"{_RUNPOD_API_BASE}?api_key={self._api_key}",
            headers=self._headers(),
            json={"query": mutation, "variables": variables},
            timeout=aiohttp.ClientTimeout(total=60.0),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        pod = data["data"]["podFindAndDeployOnDemand"]
        pod_id = pod["id"]
        logger.info("runpod_pod_created", pod_id=pod_id, gpu_type=gpu_type)

        # Poll for RUNNING status and SSH info
        status_query = """
        query Pod($podId: String!) {
            pod(input: {podId: $podId}) {
                id
                desiredStatus
                runtime {
                    gpus { id gpuUtilPercent }
                    ports { ip isIpPublic privatePort publicPort type }
                }
            }
        }
        """
        for attempt in range(120):
            await asyncio.sleep(5)
            async with self._session.post(
                f"{_RUNPOD_API_BASE}?api_key={self._api_key}",
                headers=self._headers(),
                json={"query": status_query, "variables": {"podId": pod_id}},
                timeout=aiohttp.ClientTimeout(total=15.0),
            ) as resp:
                resp.raise_for_status()
                status_data = await resp.json()

            pod_info = status_data.get("data", {}).get("pod", {})
            runtime = pod_info.get("runtime") or {}
            ports = runtime.get("ports") or []

            ssh_port_info = next(
                (p for p in ports if p.get("privatePort") == 22 and p.get("isIpPublic")),
                None,
            )
            if ssh_port_info:
                ssh_host = ssh_port_info["ip"]
                ssh_port = int(ssh_port_info["publicPort"])
                logger.info(
                    "runpod_pod_ready",
                    pod_id=pod_id,
                    ssh_host=ssh_host,
                    ssh_port=ssh_port,
                    attempts=attempt + 1,
                )
                return ProvisionedInstance(
                    instance_id=pod_id,
                    provider="runpod",
                    gpu_model=gpu_type,
                    ssh_host=ssh_host,
                    ssh_port=ssh_port,
                    status="running",
                )

        raise TimeoutError(
            f"RunPod pod {pod_id} did not become reachable within 600 s"
        )

    async def destroy_instance(self, instance_id: str) -> None:
        """Terminate a RunPod pod via GraphQL mutation."""
        mutation = """
        mutation TerminatePod($input: PodTerminateInput!) {
            podTerminate(input: $input)
        }
        """
        async with self._session.post(
            f"{_RUNPOD_API_BASE}?api_key={self._api_key}",
            headers=self._headers(),
            json={"query": mutation, "variables": {"input": {"podId": instance_id}}},
            timeout=aiohttp.ClientTimeout(total=30.0),
        ) as resp:
            resp.raise_for_status()
        logger.info("runpod_pod_destroyed", instance_id=instance_id)


# ---------------------------------------------------------------------------
# Provider-agnostic provisioning entry point
# ---------------------------------------------------------------------------


async def build_deployment_plan(
    vastai: VastAIProvisioner,
    runpod: RunPodProvisioner,
    config: Configuration,
    model_id: str,
    gpu_names: list[str],
    config_source: str = "knowledge_graph",
) -> DeploymentPlan:
    """
    Find the best offer and build a DeploymentPlan WITHOUT executing anything.

    Tries Vast.ai first, falls back to RunPod. Raises if no offers from either.
    """
    vllm_command = build_vllm_command(config, model_id)

    # --- Try Vast.ai ---
    vastai_offer = None
    try:
        offers = await vastai.find_offers(
            gpu_names=gpu_names, min_gpus=config.tensor_parallel_size
        )
        if offers:
            vastai_offer = offers[0]
    except (aiohttp.ClientError, TimeoutError) as exc:
        logger.warning("vastai_offer_search_failed", error=str(exc))

    _config_fields = dict(
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
        model_id=model_id,
    )

    if vastai_offer is not None:
        return DeploymentPlan(
            **_config_fields,
            provider="vastai",
            gpu_name=vastai_offer.get("gpu_name", gpu_names[0]),
            num_gpus=int(vastai_offer.get("num_gpus", config.tensor_parallel_size)),
            cost_per_hour_usd=vastai_offer.get("dph_total"),
            reliability=vastai_offer.get("reliability2"),
            offer_id=str(vastai_offer["id"]),
            inet_up_mbps=vastai_offer.get("inet_up"),
            inet_down_mbps=vastai_offer.get("inet_down"),
            inet_up_cost_per_gb=vastai_offer.get("inet_up_cost"),
            inet_down_cost_per_gb=vastai_offer.get("inet_down_cost"),
            pcie_bw_gbps=vastai_offer.get("pcie_bw"),
            bw_nvlink_gbps=vastai_offer.get("bw_nvlink"),
            dlperf=vastai_offer.get("dlperf"),
        )

    # --- Fall back to RunPod ---
    logger.info("vastai_no_offers", gpu_names=gpu_names, falling_back_to="runpod")
    runpod_offers = await runpod.find_offers(gpu_types=gpu_names)
    if not runpod_offers:
        raise RuntimeError(
            f"No GPU offers found on Vast.ai or RunPod for {gpu_names}"
        )

    best_runpod = runpod_offers[0]
    lowest = best_runpod.get("lowestPrice") or {}
    return DeploymentPlan(
        **_config_fields,
        provider="runpod",
        gpu_name=best_runpod.get("displayName", gpu_names[0]),
        num_gpus=config.tensor_parallel_size,
        cost_per_hour_usd=lowest.get("uninterruptablePrice"),
        reliability=None,
        offer_id=best_runpod.get("id", ""),
        # RunPod doesn't expose these granular network metrics
        inet_up_mbps=None,
        inet_down_mbps=None,
        inet_up_cost_per_gb=None,
        inet_down_cost_per_gb=None,
        pcie_bw_gbps=None,
        bw_nvlink_gbps=None,
        dlperf=None,
    )


async def execute_deployment(
    plan: DeploymentPlan,
    vastai: VastAIProvisioner,
    runpod: RunPodProvisioner,
    hf_token: str,
) -> ProvisionedInstance:
    """
    Execute a previously approved DeploymentPlan.
    Rents the instance on the chosen provider and waits for SSH readiness.
    """
    if plan.provider == "vastai":
        instance = await vastai.create_instance(
            offer_id=plan.offer_id,
            vllm_command=plan.vllm_command,
            hf_token=hf_token,
        )
    else:
        instance = await runpod.create_instance(
            gpu_type=plan.offer_id,
            vllm_command=plan.vllm_command,
            hf_token=hf_token,
        )

    logger.info(
        "instance_provisioned",
        provider=plan.provider,
        gpu_model=instance.gpu_model,
        instance_id=instance.instance_id,
    )
    return instance


async def provision_instance(
    vastai: VastAIProvisioner,
    runpod: RunPodProvisioner,
    config: Configuration,
    model_id: str,
    hf_token: str,
    gpu_names: list[str],
    config_source: str = "knowledge_graph",
) -> ProvisionedInstance:
    """
    Build + execute provisioning in one step (no HITL).

    Kept for backward compatibility. The orchestrator uses
    build_deployment_plan() + execute_deployment() with approval in between.
    """
    plan = await build_deployment_plan(
        vastai=vastai,
        runpod=runpod,
        config=config,
        model_id=model_id,
        gpu_names=gpu_names,
        config_source=config_source,
    )
    return await execute_deployment(plan, vastai, runpod, hf_token)
