"""
RouterManager — lifecycle management for the vLLM Router (Rust binary).

Install: uv pip install vllm-router
Repo:    https://github.com/vllm-project/router

Backend changes use --dynamic-config-json: the router polls a JSON config
file every ~10 seconds and hot-reloads backends WITHOUT restarting the
process. No connections are dropped during scaling events.

Process management uses asyncio.create_subprocess_exec() exclusively —
no shell=True, no subprocess.Popen.
"""

import asyncio
import json
import time
from pathlib import Path

import aiohttp
import structlog

from ..config import cfg

logger = structlog.get_logger()

VLLM_ROUTER_BINARY = "vllm-router"

# All config from config.yml
_router_cfg = cfg["gateway"]["router"]
VLLM_ROUTER_PORT = _router_cfg["port"]
VLLM_ROUTER_METRICS_PORT = _router_cfg["metrics_port"]
HEALTH_CHECK_TIMEOUT_SEC = _router_cfg["health_check_timeout_sec"]
_DEFAULT_CONFIG_PATH = _router_cfg["config_path"]


class RouterManager:
    """
    Manages the vLLM Router (Rust binary) process lifecycle.

    Worker pool changes are handled by rewriting the dynamic config JSON
    file. The router polls this file and hot-reloads backends — no process
    restart, no dropped connections.
    """

    def __init__(
        self,
        policy: str | None = None,
        static_models: list[str] | None = None,
        config_path: str | None = None,
    ) -> None:
        self._policy = policy or _router_cfg["policy"]
        self._static_models = static_models or []
        self._worker_urls: list[str] = []
        self._config_path = Path(config_path or _DEFAULT_CONFIG_PATH)
        self._process: asyncio.subprocess.Process | None = None

    # ------------------------------------------------------------------
    # Dynamic config file
    # ------------------------------------------------------------------

    def _write_config(self, worker_urls: list[str]) -> None:
        """
        Write the dynamic config JSON file that the router polls.

        Format per vLLM Router docs:
        https://docs.vllm.ai/projects/production-stack/en/latest/user_manual/router/json.html
        """
        config = {
            "service_discovery": "static",
            "routing_logic": self._policy,
            "static_backends": ",".join(worker_urls) if worker_urls else "",
        }
        if self._static_models:
            config["static_models"] = ",".join(self._static_models)

        self._config_path.write_text(json.dumps(config, indent=2))
        logger.info(
            "router_config_written",
            path=str(self._config_path),
            worker_count=len(worker_urls),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_startup_command(self) -> list[str]:
        """
        Returns the argv list used to launch the vLLM Router binary.

        Uses --dynamic-config-json so the router watches the config file
        for backend changes without needing a restart.
        """
        return [
            VLLM_ROUTER_BINARY,
            "--port", str(VLLM_ROUTER_PORT),
            "--dynamic-config-json", str(self._config_path),
            "--prometheus-port", str(VLLM_ROUTER_METRICS_PORT),
            "--engine-stats-interval", str(_router_cfg["engine_stats_interval"]),
            "--log-stats",
        ]

    async def start(self, worker_urls: list[str]) -> None:
        """
        Write the initial config file, launch the vLLM Router process,
        and wait until its /health endpoint returns 200.

        Raises RuntimeError if the health check does not pass within
        HEALTH_CHECK_TIMEOUT_SEC.
        """
        self._write_config(worker_urls)
        cmd = self.build_startup_command()
        logger.info("router_starting", cmd=cmd)

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
        Update the backend pool by rewriting the dynamic config file.

        The router polls the file every ~10 seconds and hot-reloads.
        No process restart. No dropped connections.

        If the router process is not running, starts it.
        """
        self._write_config(worker_urls)
        self._worker_urls = list(worker_urls)

        if self._process is None or self._process.returncode is not None:
            logger.warning("router_not_running_starting")
            await self.start(worker_urls)
            return

        logger.info(
            "router_config_updated",
            worker_count=len(worker_urls),
            pid=self._process.pid,
            note="router will pick up changes within ~10s",
        )

    async def stop(self) -> None:
        """Terminate the router process gracefully."""
        if self._process is None:
            return
        logger.info("router_stopping", pid=self._process.pid)
        await self._terminate_process(self._process)
        self._process = None

        # Clean up config file
        if self._config_path.exists():
            self._config_path.unlink()

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
