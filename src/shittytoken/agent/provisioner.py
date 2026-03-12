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


# ---------------------------------------------------------------------------
# Vast.ai provider
# ---------------------------------------------------------------------------


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
        min_gpus: int = 2,
        min_reliability: float = 0.99,
    ) -> list[dict]:
        """
        POST /api/v0/bundles/ with filters.

        Returns a list of offers sorted by dph_total ascending (cheapest first).
        """
        filters = {
            "verified": {"eq": True},
            "num_gpus": {"gte": min_gpus},
            "reliability2": {"gte": min_reliability},
            "rentable": {"eq": True},
            "gpu_name": {"in": gpu_names},
        }
        url = f"{_VASTAI_API_BASE}/bundles/"
        params = {"q": str(filters)}

        async with self._session.post(
            url,
            headers=self._headers(),
            json={"q": filters},
            timeout=aiohttp.ClientTimeout(total=30.0),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        offers = data.get("offers", [])
        offers.sort(key=lambda o: o.get("dph_total", float("inf")))
        logger.info(
            "vastai_offers_found",
            gpu_names=gpu_names,
            min_gpus=min_gpus,
            count=len(offers),
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


async def provision_instance(
    vastai: VastAIProvisioner,
    runpod: RunPodProvisioner,
    config: Configuration,
    model_id: str,
    hf_token: str,
    gpu_names: list[str],
) -> ProvisionedInstance:
    """
    Build the vllm command from *config*, then try Vast.ai first.
    Falls back to RunPod if Vast.ai has no matching offers or creation fails.

    Logs {"event": "instance_provisioned", "provider": ..., "gpu_model": ...}
    on success.
    """
    vllm_command = build_vllm_command(config, model_id)

    # --- Try Vast.ai ---
    try:
        offers = await vastai.find_offers(gpu_names=gpu_names, min_gpus=config.tensor_parallel_size)
        if offers:
            best_offer = offers[0]
            instance = await vastai.create_instance(
                offer_id=str(best_offer["id"]),
                vllm_command=vllm_command,
                hf_token=hf_token,
            )
            logger.info(
                "instance_provisioned",
                provider="vastai",
                gpu_model=instance.gpu_model,
                instance_id=instance.instance_id,
            )
            return instance
        logger.info("vastai_no_offers", gpu_names=gpu_names, falling_back_to="runpod")
    except (aiohttp.ClientError, TimeoutError) as exc:
        logger.warning(
            "vastai_provision_failed",
            error=str(exc),
            falling_back_to="runpod",
        )

    # --- Fall back to RunPod ---
    gpu_type = gpu_names[0] if gpu_names else "NVIDIA GeForce RTX 4090"
    instance = await runpod.create_instance(
        gpu_type=gpu_type,
        vllm_command=vllm_command,
        hf_token=hf_token,
    )
    logger.info(
        "instance_provisioned",
        provider="runpod",
        gpu_model=instance.gpu_model,
        instance_id=instance.instance_id,
    )
    return instance
