import asyncio
import json
import time
from dataclasses import dataclass

import aiohttp
import structlog

logger = structlog.get_logger()


@dataclass
class RequestResult:
    session_id: str
    success: bool
    ttft_sec: float | None  # None on failure; callers use float('inf') for stats
    total_duration_sec: float
    output_text: str
    tokens_generated: int  # count of non-empty delta chunks received
    error: str | None


async def send_chat_completion(
    session: aiohttp.ClientSession,
    base_url: str,
    messages: list[dict],
    max_tokens: int,
    session_id: str,
    request_timeout: float,
    first_token_timeout: float,
) -> RequestResult:
    """
    Sends POST {base_url}/v1/chat/completions with stream=True.

    TTFT measurement:
    - Records send_time = time.monotonic() before the request.
    - On first non-empty delta.content, records ttft = time.monotonic() - send_time.

    SSE parsing:
    - Each line starting with "data: " is a payload line.
    - "data: [DONE]" = end of stream, exits loop.
    - Parses JSON from "data: {json}" lines.
    - Extracts choices[0].delta.content.
    - Skips empty content for TTFT measurement.
    - If JSON parse fails on a chunk: logs the error and skips the chunk (does not abort).

    On ANY failure (HTTP error, timeout, connection error, JSON decode of outer body):
    - Returns RequestResult(success=False, ttft_sec=None, output_text="", ...).
    - NEVER raises exceptions from this function.

    aiohttp timeout handling:
    - Uses aiohttp.ClientTimeout(total=request_timeout, connect=10).
    - Catches aiohttp.ClientError, asyncio.TimeoutError — returns failed result.
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    # Sanitize messages: strip control characters that break JSON parsing
    # on the vLLM side (model output can contain arbitrary bytes).
    clean_messages = []
    for msg in messages:
        content = msg.get("content", "")
        # Remove ASCII control chars except \n and \t
        clean_content = "".join(
            c if c >= " " or c in "\n\t" else " " for c in content
        )
        clean_messages.append({**msg, "content": clean_content})

    payload = {
        "messages": clean_messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    timeout = aiohttp.ClientTimeout(total=request_timeout, connect=10)
    send_time = time.monotonic()

    try:
        async with session.post(url, json=payload, timeout=timeout) as response:
            if response.status != 200:
                body = await response.text()
                elapsed = time.monotonic() - send_time
                logger.warning(
                    "sse_client.http_error",
                    session_id=session_id,
                    status=response.status,
                    body=body[:200],
                )
                return RequestResult(
                    session_id=session_id,
                    success=False,
                    ttft_sec=None,
                    total_duration_sec=elapsed,
                    output_text="",
                    tokens_generated=0,
                    error=f"HTTP {response.status}: {body[:200]}",
                )

            ttft_sec: float | None = None
            output_parts: list[str] = []
            tokens_generated: int = 0

            async for raw_line in response.content:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")

                if not line.startswith("data: "):
                    continue

                data_str = line[len("data: "):]

                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "sse_client.chunk_json_decode_error",
                        session_id=session_id,
                        raw=data_str[:200],
                        error=str(exc),
                    )
                    continue

                try:
                    content = chunk["choices"][0]["delta"].get("content", "")
                except (KeyError, IndexError, TypeError) as exc:
                    logger.warning(
                        "sse_client.chunk_structure_error",
                        session_id=session_id,
                        error=str(exc),
                    )
                    continue

                if content:
                    if ttft_sec is None:
                        ttft_sec = time.monotonic() - send_time
                    output_parts.append(content)
                    tokens_generated += 1

            total_duration_sec = time.monotonic() - send_time
            return RequestResult(
                session_id=session_id,
                success=True,
                ttft_sec=ttft_sec,
                total_duration_sec=total_duration_sec,
                output_text="".join(output_parts),
                tokens_generated=tokens_generated,
                error=None,
            )

    except asyncio.TimeoutError as exc:
        elapsed = time.monotonic() - send_time
        logger.warning(
            "sse_client.timeout",
            session_id=session_id,
            elapsed_sec=elapsed,
            error=str(exc),
        )
        return RequestResult(
            session_id=session_id,
            success=False,
            ttft_sec=None,
            total_duration_sec=elapsed,
            output_text="",
            tokens_generated=0,
            error=f"TimeoutError: {exc}",
        )
    except aiohttp.ClientError as exc:
        elapsed = time.monotonic() - send_time
        logger.warning(
            "sse_client.client_error",
            session_id=session_id,
            elapsed_sec=elapsed,
            error=str(exc),
        )
        return RequestResult(
            session_id=session_id,
            success=False,
            ttft_sec=None,
            total_duration_sec=elapsed,
            output_text="",
            tokens_generated=0,
            error=f"ClientError: {exc}",
        )
