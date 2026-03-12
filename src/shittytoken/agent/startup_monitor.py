"""
StartupMonitor — reads vLLM log lines from a stream and classifies the
startup outcome as READY, OOM, CUDA_ERROR, or TIMEOUT.

Designed to be fed by SSHManager.stream_logs() via an async generator.
"""

from __future__ import annotations

import asyncio
import re

import structlog

logger = structlog.get_logger()

READINESS_PATTERN = re.compile(r"Application startup complete|Uvicorn running")
OOM_PATTERN = re.compile(r"torch\.cuda\.OutOfMemoryError|CUDA out of memory")
CUDA_ERROR_PATTERN = re.compile(r"CUDA error: device-side assert triggered")


class StartupResult:
    READY = "ready"
    OOM = "oom"
    CUDA_ERROR = "cuda_error"
    TIMEOUT = "timeout"


async def monitor_startup(
    line_generator,  # async generator yielding log lines (str)
    timeout_sec: float,
) -> tuple[str, str]:
    """
    Consume *line_generator* until a known terminal pattern is found or
    *timeout_sec* elapses.

    Returns a (result, line) tuple where result is one of StartupResult.*:
    - READY      — vLLM printed a readiness banner
    - OOM        — torch OOM or CUDA OOM detected
    - CUDA_ERROR — device-side assert triggered
    - TIMEOUT    — timeout_sec elapsed with no match

    On non-READY outcomes the full line buffer is logged at WARNING level for
    post-mortem debugging.
    """
    buffer: list[str] = []

    async def _read_lines() -> tuple[str, str]:
        async for line in line_generator:
            buffer.append(line)
            if READINESS_PATTERN.search(line):
                return (StartupResult.READY, line)
            if OOM_PATTERN.search(line):
                return (StartupResult.OOM, line)
            if CUDA_ERROR_PATTERN.search(line):
                return (StartupResult.CUDA_ERROR, line)
        # Generator exhausted without a match
        return (StartupResult.TIMEOUT, "")

    try:
        result, matched_line = await asyncio.wait_for(
            _read_lines(),
            timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        result = StartupResult.TIMEOUT
        matched_line = ""

    if result != StartupResult.READY:
        logger.warning(
            "startup_monitor_failed",
            result=result,
            matched_line=matched_line,
            log_lines_collected=len(buffer),
            tail=buffer[-50:] if len(buffer) > 50 else buffer,
        )
    else:
        logger.info(
            "startup_monitor_ready",
            matched_line=matched_line,
            log_lines_collected=len(buffer),
        )

    return (result, matched_line)
