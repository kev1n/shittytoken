"""
SSHManager — persistent AsyncSSH connections for orchestrated GPU workers.

Uses asyncssh with keepalive to detect dead connections early.
stream_logs() tails the primary process stdout (PID 1) for startup monitoring.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import asyncssh
import structlog

logger = structlog.get_logger()

# VRAM tolerance: GPU must report at least 95% of expected VRAM.
_VRAM_TOLERANCE = 0.95


@dataclass
class SSHSession:
    host: str
    port: int
    _conn: asyncssh.SSHClientConnection | None = field(default=None, repr=False)


class SSHManager:
    """
    Manages persistent AsyncSSH connections per worker.

    keepalive_interval and keepalive_count_max are configured at connection
    time to detect dead connections without a full timeout.
    """

    def __init__(
        self,
        private_key_path: str,
        keepalive_interval: int = 30,
    ) -> None:
        self._private_key_path = private_key_path
        self._keepalive_interval = keepalive_interval
        # asyncssh drops the connection after keepalive_count_max missed probes
        self._keepalive_count_max = 3

    async def connect(
        self,
        host: str,
        port: int,
        username: str = "root",
    ) -> SSHSession:
        """
        Establish a persistent SSH connection to *host*:*port*.

        Returns an SSHSession containing the live connection.
        Raises asyncssh.Error on connection failure.
        """
        conn = await asyncssh.connect(
            host=host,
            port=port,
            username=username,
            client_keys=[self._private_key_path],
            known_hosts=None,           # spot instances have ephemeral host keys
            keepalive_interval=self._keepalive_interval,
            keepalive_count_max=self._keepalive_count_max,
        )
        session = SSHSession(host=host, port=port, _conn=conn)
        logger.info("ssh_connected", host=host, port=port, username=username)
        return session

    async def run_command(
        self,
        session: SSHSession,
        command: str,
    ) -> tuple[str, str]:
        """
        Execute *command* on the remote host.

        Returns (stdout, stderr).
        Raises asyncssh.ProcessError on non-zero exit code.
        Raises RuntimeError if the session has no active connection.
        """
        if session._conn is None:
            raise RuntimeError(
                f"SSHSession for {session.host}:{session.port} has no active connection"
            )

        result = await session._conn.run(command, check=True)
        return result.stdout or "", result.stderr or ""

    async def verify_gpu(
        self,
        session: SSHSession,
        expected_gpu_name: str,
        expected_vram_gb: int,
    ) -> bool:
        """
        Verify that the remote machine has the expected GPU.

        Runs: nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

        Returns True if:
        - GPU name contains *expected_gpu_name* (case-insensitive)
        - Reported VRAM >= expected_vram_gb * 0.95

        Logs mismatch details and returns False on any failure.
        """
        command = "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
        try:
            stdout, _ = await self.run_command(session, command)
        except asyncssh.ProcessError as exc:
            logger.warning(
                "gpu_verify_command_failed",
                host=session.host,
                error=str(exc),
            )
            return False

        for line in stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) < 2:
                continue

            gpu_name = parts[0].strip()
            vram_raw = parts[1].strip()  # e.g. "24576 MiB"

            # Parse VRAM value in MiB → convert to GiB
            try:
                vram_mib = float(vram_raw.split()[0])
                vram_gib = vram_mib / 1024.0
            except (ValueError, IndexError):
                logger.warning(
                    "gpu_verify_vram_parse_failed",
                    host=session.host,
                    raw_vram=vram_raw,
                )
                return False

            name_ok = expected_gpu_name.lower() in gpu_name.lower()
            vram_ok = vram_gib >= expected_vram_gb * _VRAM_TOLERANCE

            if not name_ok or not vram_ok:
                logger.warning(
                    "gpu_verify_mismatch",
                    host=session.host,
                    expected_gpu_name=expected_gpu_name,
                    actual_gpu_name=gpu_name,
                    expected_vram_gb=expected_vram_gb,
                    actual_vram_gib=round(vram_gib, 1),
                    name_ok=name_ok,
                    vram_ok=vram_ok,
                )
                return False

            logger.info(
                "gpu_verified",
                host=session.host,
                gpu_name=gpu_name,
                vram_gib=round(vram_gib, 1),
            )
            return True

        # No lines parsed
        logger.warning("gpu_verify_no_output", host=session.host, stdout=stdout)
        return False

    async def stream_logs(
        self,
        session: SSHSession,
        line_callback,  # async callable(str) -> None
    ) -> None:
        """
        Stream the primary process stdout (via /proc/1/fd/1) to *line_callback*.

        Calls line_callback for each line received.
        Stops when:
        - The SSH connection drops (asyncssh.Error)
        - line_callback raises StopIteration
        - The remote command exits

        Never raises; connection errors are logged at WARNING level.
        """
        if session._conn is None:
            raise RuntimeError(
                f"SSHSession for {session.host}:{session.port} has no active connection"
            )

        command = "tail -f /proc/1/fd/1"
        try:
            async with session._conn.create_process(command) as proc:
                async for line in proc.stdout:
                    try:
                        await line_callback(line.rstrip("\n"))
                    except StopIteration:
                        return
        except asyncssh.Error as exc:
            logger.warning(
                "ssh_stream_logs_connection_error",
                host=session.host,
                error=str(exc),
            )

    async def close(self, session: SSHSession) -> None:
        """Close the SSH connection for *session*."""
        if session._conn is not None:
            session._conn.close()
            await session._conn.wait_closed()
            session._conn = None
            logger.info("ssh_disconnected", host=session.host, port=session.port)
