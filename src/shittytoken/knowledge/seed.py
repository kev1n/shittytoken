"""
Idempotent database seeder.

Merges foundational GPUModel, LLMModel, and initial Configuration nodes.
Safe to run multiple times — uses MERGE throughout.

All reference data comes from config.yml — nothing is hardcoded here.
"""

from __future__ import annotations

import structlog
from typing import TYPE_CHECKING

from . import queries
from .schema import Configuration
from ..config import cfg, gpu_catalog, serving_models, vllm_defaults, preferred_gpus

if TYPE_CHECKING:
    from neo4j import AsyncDriver

logger = structlog.get_logger()

# Stable UUIDs so seeding is idempotent across restarts.
# One per preferred GPU — generated deterministically from GPU index.
_SEED_CONFIG_ID_TEMPLATE = "00000000-seed-{idx:04d}-0000-000000000000"


# ---------------------------------------------------------------------------
# Cypher helpers (seed-only MERGE helpers, only called by this module)
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

    # 1. GPU models (full catalog)
    async with driver.session() as session:
        for gpu in gpu_catalog():
            await session.run(_MERGE_GPU, **gpu)
            log.info("seed.gpu", name=gpu["name"])

    # 2. LLM models
    async with driver.session() as session:
        for llm in serving_models():
            await session.run(_MERGE_LLM, **llm)
            log.info("seed.llm", model_id=llm["model_id"])

    # 3. Initial configurations — one per preferred GPU, per serving model
    defaults = vllm_defaults()
    for idx, gpu_name in enumerate(preferred_gpus(), start=1):
        config_id = _SEED_CONFIG_ID_TEMPLATE.format(idx=idx)
        config = Configuration(
            config_id=config_id,
            **defaults,  # type: ignore[arg-type]
        )
        await queries.write_configuration(driver, config)
        log.info("seed.config", config_id=config_id, gpu=gpu_name)

        for llm in serving_models():
            async with driver.session() as session:
                await session.run(
                    _LINK_CONFIG_TO_GPU_AND_MODEL,
                    config_id=config_id,
                    gpu_name=gpu_name,
                    model_id=llm["model_id"],
                )
                log.info("seed.linked", config_id=config_id, gpu=gpu_name)

    log.info("seed.complete")
