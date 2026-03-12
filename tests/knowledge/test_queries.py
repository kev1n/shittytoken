"""
Integration tests for the knowledge graph layer.

Requires Docker. Uses testcontainers[neo4j] to spin up a real Neo4j instance.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from datetime import datetime, timezone
from typing import AsyncGenerator

from testcontainers.neo4j import Neo4jContainer

from shittytoken.knowledge.schema import Configuration
from shittytoken.knowledge import queries, seed as seed_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def neo4j_container():
    """Start a Neo4j container for the entire test session."""
    with Neo4jContainer("neo4j:5") as container:
        yield container


@pytest_asyncio.fixture(scope="function")
async def driver(neo4j_container):
    """Provide a fresh AsyncDriver for each test, then close it."""
    from neo4j import AsyncGraphDatabase

    bolt_url = neo4j_container.get_connection_url()
    # testcontainers exposes bolt; credentials default to neo4j/password
    drv = AsyncGraphDatabase.driver(bolt_url, auth=("neo4j", "password"))
    yield drv
    await drv.close()


@pytest_asyncio.fixture(scope="function")
async def clean_driver(driver):
    """Wipe all nodes/rels before each test for isolation."""
    async with driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
    yield driver


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> Configuration:
    defaults = dict(
        tensor_parallel_size=2,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        quantization="awq",
        kv_cache_dtype="auto",
        max_num_seqs=64,
        enable_prefix_caching=True,
        enforce_eager=False,
    )
    defaults.update(overrides)
    return Configuration(**defaults)  # type: ignore[arg-type]


async def _seed_gpu(driver, name: str = "RTX 3090") -> None:
    async with driver.session() as session:
        await session.run(
            "MERGE (:GPUModel {name: $name, vram_gb: 24, memory_bandwidth_gbs: 936})",
            name=name,
        )


async def _seed_llm(driver, model_id: str = "Qwen/Qwen3.5-35B-A3B") -> None:
    async with driver.session() as session:
        await session.run(
            """
            MERGE (:LLMModel {
                model_id: $model_id,
                params_b: 35,
                active_params_b: 3,
                quantization: 'awq',
                dtype: 'fp16'
            })
            """,
            model_id=model_id,
        )


async def _link_config(driver, config_id: str, gpu_name: str, model_id: str) -> None:
    async with driver.session() as session:
        await session.run(
            """
            MATCH (c:Configuration {config_id: $config_id})
            MATCH (g:GPUModel {name: $gpu_name})
            MATCH (m:LLMModel {model_id: $model_id})
            MERGE (c)-[:RUNS_ON]->(g)
            MERGE (c)-[:SERVES]->(m)
            """,
            config_id=config_id,
            gpu_name=gpu_name,
            model_id=model_id,
        )


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Test 1 — seed() is idempotent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_seed_idempotent(clean_driver):
    """Calling seed() twice must not create duplicate nodes."""
    await seed_module.seed(clean_driver)
    await seed_module.seed(clean_driver)

    async with clean_driver.session() as session:
        gpu_result = await session.run("MATCH (g:GPUModel) RETURN count(g) AS n")
        gpu_count = (await gpu_result.single())["n"]

        llm_result = await session.run("MATCH (m:LLMModel) RETURN count(m) AS n")
        llm_count = (await llm_result.single())["n"]

        cfg_result = await session.run("MATCH (c:Configuration) RETURN count(c) AS n")
        cfg_count = (await cfg_result.single())["n"]

    assert gpu_count == 4, f"Expected 4 GPUModel nodes, got {gpu_count}"
    assert llm_count == 1, f"Expected 1 LLMModel node, got {llm_count}"
    assert cfg_count == 2, f"Expected 2 Configuration nodes, got {cfg_count}"


# ---------------------------------------------------------------------------
# Test 2 — best_config_for() returns highest-throughput PASSING config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_best_config_for_returns_highest_throughput_pass(clean_driver):
    """Must return the config linked to the passing result with highest tps."""
    gpu = "RTX 3090"
    model = "Qwen/Qwen3.5-35B-A3B"
    await _seed_gpu(clean_driver, gpu)
    await _seed_llm(clean_driver, model)

    # Config A — passing, tps 100
    cfg_a = _make_config()
    await queries.write_configuration(clean_driver, cfg_a)
    await _link_config(clean_driver, cfg_a.config_id, gpu, model)
    await queries.write_benchmark_result(
        clean_driver,
        result_id="result-a",
        config_id=cfg_a.config_id,
        gpu_model_name=gpu,
        verdict="pass",
        cold_ttft_p95_s=1.0,
        warm_ttft_p95_s_at_c1=0.5,
        peak_throughput_tps=100.0,
        prefix_cache_hit_rate_phase3=0.8,
        failed_request_rate=0.0,
        deltanet_cache_suspect=False,
        started_at=_now_iso(),
        completed_at=_now_iso(),
    )

    # Config B — passing, tps 200 (higher — should be returned)
    cfg_b = _make_config()
    await queries.write_configuration(clean_driver, cfg_b)
    await _link_config(clean_driver, cfg_b.config_id, gpu, model)
    await queries.write_benchmark_result(
        clean_driver,
        result_id="result-b",
        config_id=cfg_b.config_id,
        gpu_model_name=gpu,
        verdict="pass",
        cold_ttft_p95_s=1.0,
        warm_ttft_p95_s_at_c1=0.5,
        peak_throughput_tps=200.0,
        prefix_cache_hit_rate_phase3=0.9,
        failed_request_rate=0.0,
        deltanet_cache_suspect=False,
        started_at=_now_iso(),
        completed_at=_now_iso(),
    )

    result = await queries.best_config_for(clean_driver, gpu, model)
    assert result is not None
    assert result.config_id == cfg_b.config_id


# ---------------------------------------------------------------------------
# Test 3 — best_config_for() returns None when only failing benchmarks exist
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_best_config_for_returns_none_when_only_failing(clean_driver):
    """Must return None if no passing BenchmarkResult exists."""
    gpu = "RTX 3090"
    model = "Qwen/Qwen3.5-35B-A3B"
    await _seed_gpu(clean_driver, gpu)
    await _seed_llm(clean_driver, model)

    cfg = _make_config()
    await queries.write_configuration(clean_driver, cfg)
    await _link_config(clean_driver, cfg.config_id, gpu, model)
    await queries.write_benchmark_result(
        clean_driver,
        result_id="result-fail",
        config_id=cfg.config_id,
        gpu_model_name=gpu,
        verdict="fail",
        cold_ttft_p95_s=9.9,
        warm_ttft_p95_s_at_c1=9.9,
        peak_throughput_tps=0.0,
        prefix_cache_hit_rate_phase3=0.0,
        failed_request_rate=1.0,
        deltanet_cache_suspect=False,
        started_at=_now_iso(),
        completed_at=_now_iso(),
    )

    result = await queries.best_config_for(clean_driver, gpu, model)
    assert result is None


# ---------------------------------------------------------------------------
# Test 4 — write_oom_event() returns a non-empty string event_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_oom_event_returns_event_id(clean_driver):
    """write_oom_event must return a non-empty string immediately."""
    gpu = "RTX 3090"
    await _seed_gpu(clean_driver, gpu)

    cfg = _make_config()
    await queries.write_configuration(clean_driver, cfg)

    event_id = await queries.write_oom_event(
        clean_driver,
        config_id=cfg.config_id,
        gpu_model_name=gpu,
        error_type="loading",
        error_message="torch.OutOfMemoryError: CUDA out of memory",
        error_phase="loading",
        gpu_memory_free_gb=0.5,
        gpu_memory_total_gb=24.0,
    )

    assert isinstance(event_id, str)
    assert len(event_id) > 0


# ---------------------------------------------------------------------------
# Test 5 — update_oom_outcome(succeeded=True) creates RESOLVED_BY relationship
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_oom_outcome_success_creates_resolved_by(clean_driver):
    """When succeeded=True, a RESOLVED_BY edge must be created."""
    gpu = "RTX 3090"
    await _seed_gpu(clean_driver, gpu)

    orig_cfg = _make_config()
    await queries.write_configuration(clean_driver, orig_cfg)

    resolution_cfg = _make_config(max_model_len=4096)
    await queries.write_configuration(clean_driver, resolution_cfg)

    event_id = await queries.write_oom_event(
        clean_driver,
        config_id=orig_cfg.config_id,
        gpu_model_name=gpu,
        error_type="loading",
        error_message="OOM",
        error_phase="loading",
        gpu_memory_free_gb=0.2,
        gpu_memory_total_gb=24.0,
    )

    await queries.update_oom_outcome(
        clean_driver,
        event_id=event_id,
        succeeded=True,
        resolution_config_id=resolution_cfg.config_id,
    )

    async with clean_driver.session() as session:
        result = await session.run(
            """
            MATCH (o:OOMEvent {event_id: $event_id})-[:RESOLVED_BY]->(c:Configuration)
            RETURN c.config_id AS config_id, o.succeeded AS succeeded
            """,
            event_id=event_id,
        )
        record = await result.single()

    assert record is not None, "RESOLVED_BY relationship not found"
    assert record["config_id"] == resolution_cfg.config_id
    assert record["succeeded"] is True


# ---------------------------------------------------------------------------
# Test 6 — update_oom_outcome(succeeded=False) does NOT create RESOLVED_BY
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_oom_outcome_failure_no_resolved_by(clean_driver):
    """When succeeded=False, no RESOLVED_BY edge must be created."""
    gpu = "RTX 3090"
    await _seed_gpu(clean_driver, gpu)

    cfg = _make_config()
    await queries.write_configuration(clean_driver, cfg)

    event_id = await queries.write_oom_event(
        clean_driver,
        config_id=cfg.config_id,
        gpu_model_name=gpu,
        error_type="runtime",
        error_message="OOM during runtime",
        error_phase="runtime",
        gpu_memory_free_gb=0.1,
        gpu_memory_total_gb=24.0,
    )

    await queries.update_oom_outcome(
        clean_driver,
        event_id=event_id,
        succeeded=False,
    )

    async with clean_driver.session() as session:
        result = await session.run(
            """
            MATCH (o:OOMEvent {event_id: $event_id})
            OPTIONAL MATCH (o)-[:RESOLVED_BY]->(c)
            RETURN o.succeeded AS succeeded, c IS NULL AS no_resolution
            """,
            event_id=event_id,
        )
        record = await result.single()

    assert record["succeeded"] is False
    assert record["no_resolution"] is True, "RESOLVED_BY should not exist on failure"


# ---------------------------------------------------------------------------
# Test 7 — Configuration(gpu_memory_utilization=1.0) raises ValueError
# ---------------------------------------------------------------------------


def test_configuration_rejects_full_gpu_memory_utilization():
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        Configuration(
            tensor_parallel_size=2,
            max_model_len=8192,
            gpu_memory_utilization=1.0,
            quantization="awq",
            kv_cache_dtype="auto",
            max_num_seqs=64,
            enable_prefix_caching=True,
            enforce_eager=False,
        )


# ---------------------------------------------------------------------------
# Test 8 — Configuration(enable_prefix_caching=False) raises ValueError
# ---------------------------------------------------------------------------


def test_configuration_rejects_disabled_prefix_caching():
    with pytest.raises(ValueError, match="enable_prefix_caching"):
        Configuration(
            tensor_parallel_size=2,
            max_model_len=8192,
            gpu_memory_utilization=0.90,
            quantization="awq",
            kv_cache_dtype="auto",
            max_num_seqs=64,
            enable_prefix_caching=False,
            enforce_eager=False,
        )
