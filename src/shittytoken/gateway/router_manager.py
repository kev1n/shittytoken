"""
RouterManager — lifecycle management for the custom Python router.

Launches the router as a subprocess (``python -m shittytoken.gateway.router``)
and manages workers via the admin HTTP API for hot-reload — no process
restarts needed when workers are added or removed.
"""

import asyncio
import sys
import time

import aiohttp
import structlog

from ..config import cfg

logger = structlog.get_logger()

_router_cfg = cfg["gateway"]["router"]
ROUTER_PORT = _router_cfg["port"]
HEALTH_CHECK_TIMEOUT_SEC = _router_cfg["health_check_timeout_sec"]
_ADMIN_BASE = f"http://localhost:{ROUTER_PORT}/admin/workers"
_HEALTH_URL = f"http://localhost:{ROUTER_PORT}/health"


class RouterManager:
    """
    Manages the custom router subprocess and its worker pool via admin API.

    Workers are added/removed via HTTP — no process restart needed.
    """

    def __init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None
        self._admin_token: str | None = _router_cfg.get("admin_token")
        self._known_workers: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self, worker_urls: list[str]) -> None:
        """Launch the router subprocess, wait for health, register initial workers."""
        cmd = [sys.executable, "-m", "shittytoken.gateway.router"]
        logger.info("router_starting", cmd=cmd)

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        if not await self._wait_for_health():
            stderr_bytes = b""
            if self._process.stderr:
                try:
                    stderr_bytes = await asyncio.wait_for(
                        self._process.stderr.read(4096), timeout=2.0
                    )
                except asyncio.TimeoutError:
                    pass
            stderr_text = stderr_bytes.decode(errors="replace").strip()
            raise RuntimeError(
                f"Router did not become healthy within "
                f"{HEALTH_CHECK_TIMEOUT_SEC}s. stderr: {stderr_text[:500]}"
            )

        logger.info("router_started", pid=self._process.pid)

        # Register initial workers
        for url in worker_urls:
            await self._add_worker(url)

    async def reload(self, worker_urls: list[str]) -> None:
        """Diff current vs desired workers and add/remove via admin API."""
        if not self._is_running():
            await self.start(worker_urls)
            return

        desired = set(worker_urls)
        to_add = desired - self._known_workers
        to_remove = self._known_workers - desired

        if not to_add and not to_remove:
            logger.debug("router_reload_noop", worker_count=len(worker_urls))
            return

        logger.info(
            "router_reload",
            adding=len(to_add),
            removing=len(to_remove),
        )

        for url in to_remove:
            await self._remove_worker(url)
        for url in to_add:
            await self._add_worker(url)

    async def stop(self) -> None:
        """Terminate the router process gracefully."""
        if self._process is None:
            return
        logger.info("router_stopping", pid=self._process.pid)
        await self._terminate_process(self._process)
        self._process = None
        self._known_workers.clear()
        logger.info("router_stopped")

    async def is_healthy(self, session: aiohttp.ClientSession | None = None) -> bool:
        """Probe GET /health. Returns False on any error."""
        try:
            if session:
                async with session.get(_HEALTH_URL, timeout=aiohttp.ClientTimeout(total=5.0)) as resp:
                    return resp.status == 200
            else:
                async with aiohttp.ClientSession() as temp:
                    async with temp.get(_HEALTH_URL, timeout=aiohttp.ClientTimeout(total=5.0)) as resp:
                        return resp.status == 200
        except Exception:  # noqa: BLE001
            return False

    @property
    def current_workers(self) -> list[str]:
        return sorted(self._known_workers)

    # ------------------------------------------------------------------
    # Admin API calls
    # ------------------------------------------------------------------

    def _admin_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._admin_token:
            headers["X-Admin-Token"] = self._admin_token
        return headers

    async def _add_worker(self, url: str) -> None:
        """POST /admin/workers to register a worker."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                _ADMIN_BASE,
                json={"url": url},
                headers=self._admin_headers(),
                timeout=aiohttp.ClientTimeout(total=10.0),
            ) as resp:
                if resp.status == 201:
                    self._known_workers.add(url)
                    logger.info("router_manager.worker_added", url=url)
                elif resp.status == 409:
                    # Already present — sync our state
                    self._known_workers.add(url)
                    logger.debug("router_manager.worker_already_present", url=url)
                else:
                    body = await resp.text()
                    logger.error(
                        "router_manager.add_worker_failed",
                        url=url,
                        status=resp.status,
                        body=body[:200],
                    )

    async def _remove_worker(self, url: str) -> None:
        """DELETE /admin/workers to deregister a worker."""
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                _ADMIN_BASE,
                json={"url": url},
                headers=self._admin_headers(),
                timeout=aiohttp.ClientTimeout(total=10.0),
            ) as resp:
                if resp.status == 200:
                    self._known_workers.discard(url)
                    logger.info("router_manager.worker_removed", url=url)
                elif resp.status == 404:
                    self._known_workers.discard(url)
                    logger.debug("router_manager.worker_already_gone", url=url)
                else:
                    body = await resp.text()
                    logger.error(
                        "router_manager.remove_worker_failed",
                        url=url,
                        status=resp.status,
                        body=body[:200],
                    )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def _wait_for_health(self) -> bool:
        deadline = time.monotonic() + HEALTH_CHECK_TIMEOUT_SEC
        async with aiohttp.ClientSession() as session:
            while time.monotonic() < deadline:
                if self._process and self._process.returncode is not None:
                    logger.error(
                        "router_process_exited_during_startup",
                        returncode=self._process.returncode,
                    )
                    return False
                if await self.is_healthy(session):
                    return True
                await asyncio.sleep(0.5)
        return False

    async def _terminate_process(
        self,
        process: asyncio.subprocess.Process,
        sigkill_after_sec: float = 5.0,
    ) -> None:
        try:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=sigkill_after_sec)
            except asyncio.TimeoutError:
                logger.warning("router_process_sigkill", pid=process.pid)
                process.kill()
                await process.wait()
        except ProcessLookupError:
            pass
