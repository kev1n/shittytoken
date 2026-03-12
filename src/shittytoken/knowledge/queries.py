"""
ALL Cypher for the knowledge graph lives here.

No other module may contain Cypher strings.
"""

from __future__ import annotations

import structlog
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from neo4j import AsyncDriver

    from .schema import Configuration

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Read queries
# ---------------------------------------------------------------------------


async def best_config_for(
    driver: "AsyncDriver",
    gpu_model_name: str,
    llm_model_id: str,
) -> "Configuration | None":
    """
    Returns the Configuration with highest peak_throughput_tps from a PASSING
    BenchmarkResult for the given GPU+model pair.
    Returns None if no proven config exists.
    """
    query = """
        MATCH (g:GPUModel {name: $gpu_model_name})
        MATCH (m:LLMModel {model_id: $llm_model_id})
        MATCH (c:Configuration)-[:RUNS_ON]->(g)
        MATCH (c)-[:SERVES]->(m)
        MATCH (c)-[:BENCHMARKED_AS]->(b:BenchmarkResult {verdict: 'pass'})
        MATCH (b)-[:BENCHMARKED_ON]->(g)
        RETURN c, b.peak_throughput_tps AS tps
        ORDER BY tps DESC
        LIMIT 1
    """
    async with driver.session() as session:
        result = await session.run(
            query,
            gpu_model_name=gpu_model_name,
            llm_model_id=llm_model_id,
        )
        record = await result.single()

    if record is None:
        logger.info(
            "best_config_for.no_result",
            gpu_model_name=gpu_model_name,
            llm_model_id=llm_model_id,
        )
        return None

    # Import here to avoid circular at module level
    from .schema import Configuration

    node = record["c"]
    props = dict(node)
    logger.info(
        "best_config_for.found",
        config_id=props.get("config_id"),
        tps=record["tps"],
    )
    return Configuration(
        config_id=props["config_id"],
        tensor_parallel_size=props["tensor_parallel_size"],
        max_model_len=props["max_model_len"],
        gpu_memory_utilization=props["gpu_memory_utilization"],
        quantization=props.get("quantization"),
        kv_cache_dtype=props["kv_cache_dtype"],
        max_num_seqs=props["max_num_seqs"],
        enable_prefix_caching=props["enable_prefix_caching"],
        enforce_eager=props["enforce_eager"],
        created_at=props.get("created_at", datetime.now(tz=timezone.utc)),
    )


async def prior_oom_resolutions(
    driver: "AsyncDriver",
    gpu_model_name: str,
    error_type: str,
    limit: int = 10,
) -> list[dict]:
    """
    Returns up to `limit` Configurations that successfully resolved an OOMEvent
    on this GPU type with this error_type, ordered by BenchmarkResult throughput desc.
    """
    query = """
        MATCH (o:OOMEvent {error_type: $error_type, succeeded: true})
        MATCH (o)-[:OCCURRED_ON]->(g:GPUModel {name: $gpu_model_name})
        MATCH (o)-[:RESOLVED_BY]->(c:Configuration)
        OPTIONAL MATCH (c)-[:BENCHMARKED_AS]->(b:BenchmarkResult {verdict: 'pass'})
        RETURN c, b.peak_throughput_tps AS tps
        ORDER BY tps DESC
        LIMIT $limit
    """
    async with driver.session() as session:
        result = await session.run(
            query,
            gpu_model_name=gpu_model_name,
            error_type=error_type,
            limit=limit,
        )
        records = await result.data()

    logger.info(
        "prior_oom_resolutions.found",
        gpu_model_name=gpu_model_name,
        error_type=error_type,
        count=len(records),
    )
    return [dict(r["c"]) | {"peak_throughput_tps": r["tps"]} for r in records]


# ---------------------------------------------------------------------------
# Write queries
# ---------------------------------------------------------------------------


async def write_configuration(driver: "AsyncDriver", config: "Configuration") -> str:
    """MERGE on config_id, SET on CREATE only (immutable). Returns config_id."""
    query = """
        MERGE (c:Configuration {config_id: $config_id})
        ON CREATE SET
            c.tensor_parallel_size      = $tensor_parallel_size,
            c.max_model_len             = $max_model_len,
            c.gpu_memory_utilization    = $gpu_memory_utilization,
            c.quantization              = $quantization,
            c.kv_cache_dtype            = $kv_cache_dtype,
            c.max_num_seqs              = $max_num_seqs,
            c.enable_prefix_caching     = $enable_prefix_caching,
            c.enforce_eager             = $enforce_eager,
            c.created_at                = $created_at
        RETURN c.config_id AS config_id
    """
    params = {
        "config_id": config.config_id,
        "tensor_parallel_size": config.tensor_parallel_size,
        "max_model_len": config.max_model_len,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "quantization": config.quantization,
        "kv_cache_dtype": config.kv_cache_dtype,
        "max_num_seqs": config.max_num_seqs,
        "enable_prefix_caching": config.enable_prefix_caching,
        "enforce_eager": config.enforce_eager,
        "created_at": config.created_at.isoformat(),
    }
    async with driver.session() as session:
        result = await session.run(query, **params)
        record = await result.single()

    config_id: str = record["config_id"]
    logger.info("write_configuration.done", config_id=config_id)
    return config_id


async def write_benchmark_result(
    driver: "AsyncDriver",
    result_id: str,
    config_id: str,
    gpu_model_name: str,
    **fields,
) -> str:
    """Creates BenchmarkResult node, links to Configuration and GPUModel."""
    query = """
        MATCH (c:Configuration {config_id: $config_id})
        MATCH (g:GPUModel      {name: $gpu_model_name})
        CREATE (b:BenchmarkResult {
            result_id:                      $result_id,
            verdict:                        $verdict,
            cold_ttft_p95_s:                $cold_ttft_p95_s,
            warm_ttft_p95_s_at_c1:          $warm_ttft_p95_s_at_c1,
            peak_throughput_tps:            $peak_throughput_tps,
            prefix_cache_hit_rate_phase3:   $prefix_cache_hit_rate_phase3,
            failed_request_rate:            $failed_request_rate,
            deltanet_cache_suspect:         $deltanet_cache_suspect,
            started_at:                     $started_at,
            completed_at:                   $completed_at
        })
        MERGE (c)-[:BENCHMARKED_AS]->(b)
        MERGE (b)-[:BENCHMARKED_ON]->(g)
        RETURN b.result_id AS result_id
    """
    params = {
        "result_id": result_id,
        "config_id": config_id,
        "gpu_model_name": gpu_model_name,
        **fields,
    }
    async with driver.session() as session:
        result = await session.run(query, **params)
        record = await result.single()

    rid: str = record["result_id"]
    logger.info("write_benchmark_result.done", result_id=rid, config_id=config_id)
    return rid


async def write_oom_event(
    driver: "AsyncDriver",
    config_id: str,
    gpu_model_name: str,
    **fields,
) -> str:
    """
    Creates OOMEvent node, links to Configuration and GPUModel.
    Returns event_id. MUST be called before any recovery action.
    """
    import uuid as _uuid_mod

    event_id: str = fields.pop("event_id", str(_uuid_mod.uuid4()))
    query = """
        MATCH (c:Configuration {config_id: $config_id})
        MATCH (g:GPUModel      {name: $gpu_model_name})
        CREATE (o:OOMEvent {
            event_id:               $event_id,
            error_type:             $error_type,
            error_message:          $error_message,
            error_phase:            $error_phase,
            gpu_memory_free_gb:     $gpu_memory_free_gb,
            gpu_memory_total_gb:    $gpu_memory_total_gb,
            occurred_at:            $occurred_at,
            succeeded:              null
        })
        MERGE (o)-[:OCCURRED_WITH]->(c)
        MERGE (o)-[:OCCURRED_ON]->(g)
        RETURN o.event_id AS event_id
    """
    params = {
        "event_id": event_id,
        "config_id": config_id,
        "gpu_model_name": gpu_model_name,
        "occurred_at": fields.pop(
            "occurred_at", datetime.now(tz=timezone.utc).isoformat()
        ),
        **fields,
    }
    async with driver.session() as session:
        result = await session.run(query, **params)
        record = await result.single()

    eid: str = record["event_id"]
    logger.info("write_oom_event.done", event_id=eid, config_id=config_id)
    return eid


async def update_oom_outcome(
    driver: "AsyncDriver",
    event_id: str,
    succeeded: bool,
    resolution_config_id: "str | None" = None,
) -> None:
    """
    Sets OOMEvent.succeeded and creates RESOLVED_BY relationship if succeeded.
    Called after recovery attempt completes (success or failure).
    """
    if succeeded and resolution_config_id is not None:
        query = """
            MATCH (o:OOMEvent {event_id: $event_id})
            MATCH (c:Configuration {config_id: $resolution_config_id})
            SET o.succeeded = true
            MERGE (o)-[:RESOLVED_BY]->(c)
        """
        params: dict = {
            "event_id": event_id,
            "resolution_config_id": resolution_config_id,
        }
    else:
        query = """
            MATCH (o:OOMEvent {event_id: $event_id})
            SET o.succeeded = $succeeded
        """
        params = {"event_id": event_id, "succeeded": succeeded}

    async with driver.session() as session:
        await session.run(query, **params)

    logger.info(
        "update_oom_outcome.done",
        event_id=event_id,
        succeeded=succeeded,
        resolution_config_id=resolution_config_id,
    )


async def write_final_instance_metrics(
    driver: "AsyncDriver",
    config_id: str,
    runtime_hours: float,
    total_tokens: int,
    cost_usd: float,
) -> None:
    """Records decommission metrics on the Configuration node."""
    query = """
        MATCH (c:Configuration {config_id: $config_id})
        SET
            c.runtime_hours  = $runtime_hours,
            c.total_tokens   = $total_tokens,
            c.cost_usd       = $cost_usd,
            c.decommissioned_at = $decommissioned_at
    """
    async with driver.session() as session:
        await session.run(
            query,
            config_id=config_id,
            runtime_hours=runtime_hours,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            decommissioned_at=datetime.now(tz=timezone.utc).isoformat(),
        )

    logger.info(
        "write_final_instance_metrics.done",
        config_id=config_id,
        runtime_hours=runtime_hours,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
    )
