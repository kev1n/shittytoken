"""
Async Neo4j driver wrapper for the ShittyToken knowledge graph.

This is the sole entry point for all knowledge-graph operations in the
application. Delegates all Cypher execution to queries.py.
"""

from __future__ import annotations

import structlog
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncGenerator, Optional

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession

from . import queries
from .schema import Configuration

if TYPE_CHECKING:
    pass

logger = structlog.get_logger()


class KnowledgeGraph:
    """Async context-manager wrapper around the Neo4j async driver."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._uri = uri
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            uri, auth=(user, password)
        )
        logger.info("knowledge_graph.init", uri=uri, user=user)

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------

    async def verify_connectivity(self) -> None:
        """Raises if the database is unreachable."""
        await self._driver.verify_connectivity()
        logger.info("knowledge_graph.connectivity_ok", uri=self._uri)

    # ------------------------------------------------------------------
    # Session context manager
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Yields a Neo4j AsyncSession."""
        async with self._driver.session() as s:
            yield s

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying driver."""
        await self._driver.close()
        logger.info("knowledge_graph.closed")

    # ------------------------------------------------------------------
    # Convenience delegates — all Cypher lives in queries.py
    # ------------------------------------------------------------------

    async def best_config_for(
        self,
        gpu_model_name: str,
        llm_model_id: str,
    ) -> "Configuration | None":
        return await queries.best_config_for(
            self._driver, gpu_model_name, llm_model_id
        )

    async def prior_oom_resolutions(
        self,
        gpu_model_name: str,
        error_type: str,
        limit: int = 10,
    ) -> list[dict]:
        return await queries.prior_oom_resolutions(
            self._driver, gpu_model_name, error_type, limit
        )

    async def write_configuration(self, config: Configuration) -> str:
        return await queries.write_configuration(self._driver, config)

    async def write_benchmark_result(
        self,
        result_id: str,
        config_id: str,
        gpu_model_name: str,
        **fields,
    ) -> str:
        return await queries.write_benchmark_result(
            self._driver, result_id, config_id, gpu_model_name, **fields
        )

    async def write_oom_event(
        self,
        config_id: str,
        gpu_model_name: str,
        **fields,
    ) -> str:
        return await queries.write_oom_event(
            self._driver, config_id, gpu_model_name, **fields
        )

    async def update_oom_outcome(
        self,
        event_id: str,
        succeeded: bool,
        resolution_config_id: "str | None" = None,
    ) -> None:
        return await queries.update_oom_outcome(
            self._driver, event_id, succeeded, resolution_config_id
        )

    async def write_final_instance_metrics(
        self,
        config_id: str,
        runtime_hours: float,
        total_tokens: int,
        cost_usd: float,
    ) -> None:
        return await queries.write_final_instance_metrics(
            self._driver, config_id, runtime_hours, total_tokens, cost_usd
        )
