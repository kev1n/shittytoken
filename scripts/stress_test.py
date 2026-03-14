#!/usr/bin/env python3
"""
ShittyToken Stress Test — Simulate realistic multi-user load against the gateway.

Downloads the Nemotron-3-Nano RL Training Blend from HuggingFace,
creates N virtual users (each with their own API key and balance),
then ramps up concurrent requests to stress the autoscaler.

Usage:
    uv run python scripts/stress_test.py [OPTIONS]

Options:
    --gateway-url       Gateway endpoint (default: http://localhost:8001)
    --num-users         Number of simulated users (default: 10)
    --max-concurrency   Peak concurrent requests (default: 30)
    --ramp-steps        Number of ramp-up stages (default: 5)
    --step-duration     Seconds per ramp stage (default: 60)
    --max-tokens        Max output tokens per request (default: 256)
    --request-timeout   Per-request timeout in seconds (default: 120)
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import asyncpg
import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=True),
    ]
)
logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

POSTGRES_DSN = "postgresql://shittytoken:shittytoken_dev@localhost:5432/shittytoken"
REDIS_URL = "redis://localhost:6379/0"
BALANCE_CENTS = 100_000  # $1,000 per user


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_nemotron_prompts(max_rows: int = 500) -> list[list[dict[str, str]]]:
    """Stream & extract chat messages from the Nemotron RL blend JSONL.

    The dataset is a 6.9GB JSONL file — we stream it and stop after max_rows
    to avoid downloading the full thing.
    """
    from huggingface_hub import hf_hub_url, get_token
    import requests as req

    logger.info("dataset.streaming", name="nvidia/Nemotron-3-Nano-RL-Training-Blend", max_rows=max_rows)

    url = hf_hub_url(
        repo_id="nvidia/Nemotron-3-Nano-RL-Training-Blend",
        filename="train.jsonl",
        repo_type="dataset",
    )
    headers = {}
    token = get_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    prompts: list[list[dict[str, str]]] = []
    lines_read = 0

    with req.get(url, headers=headers, stream=True, timeout=30) as resp:
        resp.raise_for_status()
        buf = b""
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                lines_read += 1

                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                params = row.get("responses_create_params")
                if not params:
                    continue
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except json.JSONDecodeError:
                        continue

                messages_raw = params.get("input", [])
                if not messages_raw:
                    continue

                messages: list[dict[str, str]] = []
                for m in messages_raw:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    if not content:
                        continue
                    messages.append({"role": role, "content": content[:4000]})

                if messages:
                    prompts.append(messages)

                if len(prompts) >= max_rows:
                    break
            if len(prompts) >= max_rows:
                break

    logger.info("dataset.prompts_extracted", count=len(prompts), lines_read=lines_read)
    return prompts


# ---------------------------------------------------------------------------
# User provisioning (direct DB — bypasses web UI)
# ---------------------------------------------------------------------------

@dataclass
class VirtualUser:
    email: str
    user_id: str
    api_key: str  # plaintext sk-st-...
    key_hash: str


async def provision_users(num_users: int) -> list[VirtualUser]:
    """Create test users with API keys and balance directly in Postgres + Redis."""
    import redis.asyncio as aioredis

    pool = await asyncpg.create_pool(POSTGRES_DSN)
    r = aioredis.from_url(REDIS_URL)

    users: list[VirtualUser] = []
    for i in range(num_users):
        email = f"stresstest-{i:04d}@shittytoken.test"

        # Check if already exists
        row = await pool.fetchrow("SELECT id FROM users WHERE email = $1", email)
        if row:
            user_id = str(row["id"])
        else:
            row = await pool.fetchrow(
                "INSERT INTO users (email) VALUES ($1) RETURNING id", email
            )
            user_id = str(row["id"])

        # API key
        plaintext = f"sk-st-stress-{secrets.token_hex(16)}"
        key_hash = hashlib.sha256(plaintext.encode()).hexdigest()

        await pool.execute(
            "INSERT INTO api_keys (key_hash, user_id, name) VALUES ($1, $2, $3) "
            "ON CONFLICT (key_hash) DO NOTHING",
            key_hash,
            row["id"],
            f"stress-test-{i}",
        )

        # Ensure balance
        await r.set(f"balance:{user_id}", str(BALANCE_CENTS))

        users.append(VirtualUser(
            email=email, user_id=user_id, api_key=plaintext, key_hash=key_hash,
        ))

    await pool.close()
    await r.aclose()
    logger.info("users.provisioned", count=len(users))
    return users


# ---------------------------------------------------------------------------
# Request worker
# ---------------------------------------------------------------------------

@dataclass
class RequestStats:
    started_at: float = 0.0
    total_requests: int = 0
    success: int = 0
    failed: int = 0
    timeouts: int = 0
    rate_limited: int = 0
    total_ttft_sec: float = 0.0
    total_duration_sec: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    latencies: list[float] = field(default_factory=list)
    ttfts: list[float] = field(default_factory=list)

    def record(
        self,
        success: bool,
        duration: float,
        ttft: float | None,
        prompt_tokens: int,
        completion_tokens: int,
        status_code: int | None = None,
    ) -> None:
        self.total_requests += 1
        self.latencies.append(duration)
        if success:
            self.success += 1
            self.total_duration_sec += duration
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            if ttft is not None:
                self.ttfts.append(ttft)
                self.total_ttft_sec += ttft
        elif status_code == 429:
            self.rate_limited += 1
            self.failed += 1
        else:
            self.failed += 1

    def summary(self) -> dict[str, Any]:
        elapsed = time.monotonic() - self.started_at if self.started_at else 0
        rps = self.total_requests / elapsed if elapsed > 0 else 0
        p50 = sorted(self.latencies)[len(self.latencies) // 2] if self.latencies else 0
        p95_idx = int(len(self.latencies) * 0.95)
        p95 = sorted(self.latencies)[p95_idx] if self.latencies else 0
        ttft_p50 = sorted(self.ttfts)[len(self.ttfts) // 2] if self.ttfts else 0
        ttft_p95 = sorted(self.ttfts)[int(len(self.ttfts) * 0.95)] if self.ttfts else 0
        return {
            "elapsed_s": round(elapsed, 1),
            "total": self.total_requests,
            "ok": self.success,
            "fail": self.failed,
            "rate_limited": self.rate_limited,
            "rps": round(rps, 2),
            "latency_p50": round(p50, 2),
            "latency_p95": round(p95, 2),
            "ttft_p50": round(ttft_p50, 2),
            "ttft_p95": round(ttft_p95, 2),
            "prompt_tok": self.total_prompt_tokens,
            "completion_tok": self.total_completion_tokens,
        }


async def send_request(
    session: aiohttp.ClientSession,
    gateway_url: str,
    user: VirtualUser,
    messages: list[dict[str, str]],
    max_tokens: int,
    request_timeout: float,
    stats: RequestStats,
) -> None:
    """Send a single streaming chat completion and record stats."""
    url = f"{gateway_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {user.api_key}"}
    timeout = aiohttp.ClientTimeout(total=request_timeout, connect=10)
    send_time = time.monotonic()

    ttft: float | None = None
    tokens = 0
    prompt_tokens = 0
    completion_tokens = 0
    status_code: int | None = None

    try:
        async with session.post(url, json=payload, headers=headers, timeout=timeout) as resp:
            status_code = resp.status
            if resp.status != 200:
                duration = time.monotonic() - send_time
                stats.record(False, duration, None, 0, 0, status_code=status_code)
                return

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str == "[DONE]":
                    # Try to get usage from the last chunk before [DONE]
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Extract usage if present (final chunk)
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)

                try:
                    delta = chunk["choices"][0]["delta"]
                    content = delta.get("content", "")
                    reasoning = delta.get("reasoning_content", "")
                except (KeyError, IndexError, TypeError):
                    continue

                if content or reasoning:
                    if ttft is None:
                        ttft = time.monotonic() - send_time
                    tokens += 1

            duration = time.monotonic() - send_time
            stats.record(True, duration, ttft, prompt_tokens, completion_tokens)

    except asyncio.TimeoutError:
        duration = time.monotonic() - send_time
        stats.timeouts += 1
        stats.record(False, duration, None, 0, 0)
    except (aiohttp.ClientError, ConnectionError):
        duration = time.monotonic() - send_time
        stats.record(False, duration, None, 0, 0)


# ---------------------------------------------------------------------------
# Load driver
# ---------------------------------------------------------------------------

async def run_stress_test(
    gateway_url: str,
    users: list[VirtualUser],
    prompts: list[list[dict[str, str]]],
    max_concurrency: int,
    ramp_steps: int,
    step_duration: float,
    max_tokens: int,
    request_timeout: float,
) -> None:
    """Ramp up load in stages, logging stats after each stage."""
    stats = RequestStats()
    stats.started_at = time.monotonic()

    connector = aiohttp.TCPConnector(limit=max_concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Calculate concurrency levels for each step
        levels = [
            max(1, int(max_concurrency * (i + 1) / ramp_steps))
            for i in range(ramp_steps)
        ]

        for step_idx, concurrency in enumerate(levels):
            step_start = time.monotonic()
            logger.info(
                "ramp.step_start",
                step=f"{step_idx + 1}/{ramp_steps}",
                concurrency=concurrency,
                elapsed_s=round(time.monotonic() - stats.started_at, 1),
            )

            # Semaphore controls concurrency within this step
            sem = asyncio.Semaphore(concurrency)
            tasks: list[asyncio.Task] = []
            stop_event = asyncio.Event()

            async def _worker(worker_id: int) -> None:
                while not stop_event.is_set():
                    async with sem:
                        if stop_event.is_set():
                            break
                        user = random.choice(users)
                        msgs = random.choice(prompts)
                        await send_request(
                            session, gateway_url, user, msgs,
                            max_tokens, request_timeout, stats,
                        )

            # Spawn workers
            for i in range(concurrency):
                tasks.append(asyncio.create_task(_worker(i)))

            # Let the step run for step_duration, logging stats periodically
            step_elapsed = 0.0
            log_interval = 10.0
            while step_elapsed < step_duration:
                wait_time = min(log_interval, step_duration - step_elapsed)
                await asyncio.sleep(wait_time)
                step_elapsed = time.monotonic() - step_start
                logger.info("stats.snapshot", **stats.summary())

            # Signal workers to stop and wait for in-flight requests
            stop_event.set()
            await asyncio.gather(*tasks, return_exceptions=True)

            logger.info(
                "ramp.step_complete",
                step=f"{step_idx + 1}/{ramp_steps}",
                concurrency=concurrency,
                **stats.summary(),
            )

    logger.info("stress_test.complete", **stats.summary())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="ShittyToken Stress Test")
    parser.add_argument("--gateway-url", default="http://localhost:8001")
    parser.add_argument("--num-users", type=int, default=25)
    parser.add_argument("--max-concurrency", type=int, default=80)
    parser.add_argument("--ramp-steps", type=int, default=6)
    parser.add_argument("--step-duration", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    args = parser.parse_args()

    logger.info(
        "stress_test.config",
        gateway_url=args.gateway_url,
        num_users=args.num_users,
        max_concurrency=args.max_concurrency,
        ramp_steps=args.ramp_steps,
        step_duration_s=args.step_duration,
        max_tokens=args.max_tokens,
    )

    # 1. Load dataset prompts
    prompts = load_nemotron_prompts()
    if not prompts:
        logger.error("No prompts loaded from dataset!")
        return

    # 2. Provision virtual users with API keys and balance
    users = await provision_users(args.num_users)

    # 3. Run the stress test
    await run_stress_test(
        gateway_url=args.gateway_url,
        users=users,
        prompts=prompts,
        max_concurrency=args.max_concurrency,
        ramp_steps=args.ramp_steps,
        step_duration=args.step_duration,
        max_tokens=args.max_tokens,
        request_timeout=args.request_timeout,
    )


if __name__ == "__main__":
    asyncio.run(main())
