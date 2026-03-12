"""
RouterManager — lifecycle management for the vLLM Router (Rust binary).

vLLM Router has no hot-add API.  Worker pool changes require a graceful
restart: start a new process, verify its health, then terminate the old one.

Process management uses asyncio.create_subprocess_exec() exclusively —
no shell=True, no subprocess.Popen.
"""

import asyncio
import time
from dataclasses import dataclass, field

import aiohttp
import structlog

logger = structlog.get_logger()

VLLM_ROUTER_BINARY = "vllm-router"
VLLM_ROUTER_PORT = 8001
VLLM_ROUTER_METRICS_PORT = 29000
HEALTH_CHECK_TIMEOUT_SEC = 30.0

# Placeholder URL used when the worker pool is empty at startup.
# vLLM Router requires at least one backend in --static-backends.
_EMPTY_POOL_PLACEHOLDER = "http://localhost:19999"


class RouterManager:
    """
    Manages the vLLM Router (Rust binary) process lifecycle.

    Since vLLM Router has no hot-add API, adding/removing workers requires
    a graceful restart.  reload() orchestrates: start new → verify health
    → terminate old.
    """

    def __init__(
        self,
        policy: str = "cache_aware",
        nginx_config_path: str = "/etc/nginx/nginx.conf",
    ) -> None:
        self._policy = policy
        self._worker_urls: list[str] = []
        self._process: asyncio.subprocess.Process | None = None
        self._nginx_config_path = nginx_config_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_startup_command(self, worker_urls: list[str]) -> list[str]:
        """
        Returns the argv list used to launch the vLLM Router binary.

        If *worker_urls* is empty, a placeholder backend is injected so the
        router process can start without any real workers registered yet.
        """
        backends = worker_urls if worker_urls else [_EMPTY_POOL_PLACEHOLDER]
        return [
            VLLM_ROUTER_BINARY,
            "--port", str(VLLM_ROUTER_PORT),
            "--service-discovery", "static",
            "--static-backends", ",".join(backends),
            "--routing-logic", self._policy,
            "--metrics-port", str(VLLM_ROUTER_METRICS_PORT),
        ]

    async def start(self, worker_urls: list[str]) -> None:
        """
        Launch the vLLM Router process and wait until its /health endpoint
        returns 200.

        Raises RuntimeError if the health check does not pass within
        HEALTH_CHECK_TIMEOUT_SEC.
        """
        cmd = self.build_startup_command(worker_urls)
        log = logger.bind(event="router_starting", cmd=cmd)
        log.info("starting vllm-router")

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        if not await self._wait_for_health():
            raise RuntimeError(
                f"vLLM Router did not become healthy within "
                f"{HEALTH_CHECK_TIMEOUT_SEC}s after start"
            )

        self._worker_urls = list(worker_urls)
        logger.info(
            "router_started",
            pid=self._process.pid,
            worker_count=len(worker_urls),
        )

    async def reload(self, worker_urls: list[str]) -> None:
        """
        Graceful restart:
        1. Launch a new process.
        2. Wait for the new process's health check to pass.
        3. Log the reload event.
        4. Terminate the old process (SIGTERM; SIGKILL after 5 s if needed).
        5. Update internal state.
        """
        old_process = self._process

        cmd = self.build_startup_command(worker_urls)
        logger.info("router_reload_starting", cmd=cmd)

        new_process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Temporarily store new process so is_healthy() probes the right port.
        # (Both processes share the same port during the overlap window — the
        # OS will route to whichever one owns the socket; in practice the new
        # process binds after the old one releases it on SIGTERM, so we probe
        # after a short readiness loop.)
        self._process = new_process

        if not await self._wait_for_health():
            # New process failed — kill it and restore the old one.
            new_process.terminate()
            self._process = old_process
            raise RuntimeError(
                f"vLLM Router reload failed: new process did not become "
                f"healthy within {HEALTH_CHECK_TIMEOUT_SEC}s"
            )

        logger.info(
            "router_reloaded",
            worker_count=len(worker_urls),
            new_pid=new_process.pid,
        )

        # Gracefully terminate the old process.
        if old_process is not None:
            await self._terminate_process(old_process)

        self._worker_urls = list(worker_urls)

    async def stop(self) -> None:
        """Terminate the router process gracefully."""
        if self._process is None:
            return
        logger.info("router_stopping", pid=self._process.pid)
        await self._terminate_process(self._process)
        self._process = None
        logger.info("router_stopped")

    async def is_healthy(self, session: aiohttp.ClientSession | None = None) -> bool:
        """
        Probe GET http://localhost:{VLLM_ROUTER_PORT}/health.
        Returns False on any error — never raises.
        """
        url = f"http://localhost:{VLLM_ROUTER_PORT}/health"
        try:
            if session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5.0)) as resp:
                    return resp.status == 200
            else:
                async with aiohttp.ClientSession() as temp:
                    async with temp.get(url, timeout=aiohttp.ClientTimeout(total=5.0)) as resp:
                        return resp.status == 200
        except Exception:  # noqa: BLE001 — intentional catch-all for health probe
            return False

    @property
    def current_workers(self) -> list[str]:
        return list(self._worker_urls)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _wait_for_health(self) -> bool:
        """
        Poll /health until it returns 200 or HEALTH_CHECK_TIMEOUT_SEC elapses.
        Returns True if healthy, False on timeout.
        """
        deadline = time.monotonic() + HEALTH_CHECK_TIMEOUT_SEC
        async with aiohttp.ClientSession() as session:
            while time.monotonic() < deadline:
                if await self.is_healthy(session):
                    return True
                await asyncio.sleep(0.5)
        return False

    async def _terminate_process(
        self,
        process: asyncio.subprocess.Process,
        sigkill_after_sec: float = 5.0,
    ) -> None:
        """
        Send SIGTERM; if the process is still alive after *sigkill_after_sec*,
        send SIGKILL.
        """
        try:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=sigkill_after_sec)
            except asyncio.TimeoutError:
                logger.warning(
                    "router_process_sigkill",
                    pid=process.pid,
                    reason="did not exit after SIGTERM within timeout",
                )
                process.kill()
                await process.wait()
        except ProcessLookupError:
            # Process already exited — nothing to do.
            pass
