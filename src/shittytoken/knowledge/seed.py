"""
Idempotent database seeder.

Merges foundational GPUModel, LLMModel, and initial Configuration nodes.
Safe to run multiple times — uses MERGE throughout.
"""

from __future__ import annotations

import structlog
from typing import TYPE_CHECKING

from . import queries
from .schema import Configuration

if TYPE_CHECKING:
    from neo4j import AsyncDriver

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Static reference data
# ---------------------------------------------------------------------------

GPU_MODELS = [
    {"name": "RTX 3090",       "vram_gb": 24, "memory_bandwidth_gbs": 936},
    {"name": "RTX 4090",       "vram_gb": 24, "memory_bandwidth_gbs": 1008},
    {"name": "A100 SXM4 80GB", "vram_gb": 80, "memory_bandwidth_gbs": 2000},
    {"name": "H100 SXM5 80GB", "vram_gb": 80, "memory_bandwidth_gbs": 3350},
]

LLM_MODELS = [
    {
        "model_id":       "Qwen/Qwen3.5-35B-A3B",
        "params_b":       35,
        "active_params_b": 3,
        "quantization":   "awq",
        "dtype":          "fp16",
    },
]

# Initial configurations for RTX 3090 and RTX 4090 share the same vLLM params.
# Each GPU gets its own Configuration node (separate config_ids).
_INITIAL_CONFIG_PARAMS = dict(
    tensor_parallel_size=2,
    max_model_len=8192,
    gpu_memory_utilization=0.90,
    quantization="awq",
    kv_cache_dtype="auto",
    max_num_seqs=64,
    enable_prefix_caching=True,
    enforce_eager=False,
)

# Stable UUIDs so seeding is idempotent across restarts.
_SEED_CONFIG_IDS = {
    "RTX 3090": "00000000-seed-0001-0000-000000000000",
    "RTX 4090": "00000000-seed-0002-0000-000000000000",
}

_SEED_MODEL_ID = "Qwen/Qwen3.5-35B-A3B"


# ---------------------------------------------------------------------------
# Cypher helpers (all Cypher lives in queries.py for production paths;
# seed-only MERGE helpers are below and only called by this module)
# ---------------------------------------------------------------------------

_MERGE_GPU = """
    MERGE (g:GPUModel {name: $name})
    ON CREATE SET
        g.vram_gb               = $vram_gb,
        g.memory_bandwidth_gbs  = $memory_bandwidth_gbs
    RETURN g.name AS name
"""

_MERGE_LLM = """
    MERGE (m:LLMModel {model_id: $model_id})
    ON CREATE SET
        m.params_b          = $params_b,
        m.active_params_b   = $active_params_b,
        m.quantization      = $quantization,
        m.dtype             = $dtype
    RETURN m.model_id AS model_id
"""

_LINK_CONFIG_TO_GPU_AND_MODEL = """
    MATCH (c:Configuration {config_id: $config_id})
    MATCH (g:GPUModel      {name: $gpu_name})
    MATCH (m:LLMModel      {model_id: $model_id})
    MERGE (c)-[:RUNS_ON]->(g)
    MERGE (c)-[:SERVES]->(m)
"""


async def seed(driver: "AsyncDriver") -> None:
    """
    Idempotent seed.  Safe to call multiple times.
    All GPU models and the initial Configurations are MERGEd, never
    overwritten.
    """
    log = logger.bind(phase="seed")

    # 1. GPU models
    async with driver.session() as session:
        for gpu in GPU_MODELS:
            await session.run(_MERGE_GPU, **gpu)
            log.info("seed.gpu", name=gpu["name"])

    # 2. LLM models
    async with driver.session() as session:
        for llm in LLM_MODELS:
            await session.run(_MERGE_LLM, **llm)
            log.info("seed.llm", model_id=llm["model_id"])

    # 3. Initial configurations — one per target GPU
    for gpu_name, config_id in _SEED_CONFIG_IDS.items():
        config = Configuration(
            config_id=config_id,
            **_INITIAL_CONFIG_PARAMS,  # type: ignore[arg-type]
        )
        await queries.write_configuration(driver, config)
        log.info("seed.config", config_id=config_id, gpu=gpu_name)

        async with driver.session() as session:
            await session.run(
                _LINK_CONFIG_TO_GPU_AND_MODEL,
                config_id=config_id,
                gpu_name=gpu_name,
                model_id=_SEED_MODEL_ID,
            )
            log.info("seed.linked", config_id=config_id, gpu=gpu_name)

    log.info("seed.complete")
