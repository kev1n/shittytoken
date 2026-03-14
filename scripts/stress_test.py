#!/usr/bin/env python3
"""
ShittyToken Stress Test — Realistic multi-user conversational load.

Simulates real API usage patterns:
- Users have ongoing conversations (multi-turn with growing context)
- ~75% of prompt tokens are cached (prior conversation turns)
- Output is short relative to input (~150-200 tokens per ~1000 input)
- Mix of conversation lengths and topics from Nemotron dataset

Usage:
    uv run python scripts/stress_test.py [OPTIONS]
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
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
    """Stream & extract chat messages from the Nemotron RL blend JSONL."""
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
# Conversation simulator
# ---------------------------------------------------------------------------

@dataclass
class Conversation:
    """Simulates a multi-turn conversation with growing context.

    Each turn appends the prior assistant response + a new user message,
    so the prefix (all prior turns) is cacheable by vLLM's prefix caching.
    """
    system_prompt: str
    turns: list[dict[str, str]] = field(default_factory=list)
    turn_count: int = 0
    max_turns: int = 8

    @property
    def is_finished(self) -> bool:
        return self.turn_count >= self.max_turns

    def build_messages(self, user_message: str) -> list[dict[str, str]]:
        """Build the full message array for the next API call.

        The prefix (system + all prior turns) will be cached by vLLM.
        Only the new user message requires fresh computation.
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.turns)
        messages.append({"role": "user", "content": user_message})
        return messages

    def record_response(self, user_message: str, assistant_response: str) -> None:
        """Record a completed turn to grow the conversation context."""
        self.turns.append({"role": "user", "content": user_message})
        self.turns.append({"role": "assistant", "content": assistant_response})
        self.turn_count += 1


# System prompts that create realistic conversational contexts
SYSTEM_PROMPTS = [
    "You are a helpful coding assistant. Give concise, working code examples. Keep explanations brief.",
    "You are a data analysis expert. When asked about data, provide clear summaries and insights. Be concise.",
    "You are a technical writer helping document software APIs. Write clear, structured documentation.",
    "You are a DevOps engineer helping with infrastructure. Give practical, specific advice.",
    "You are a product manager helping plan features. Think about user impact and feasibility.",
    "You are a security researcher. Analyze code and configurations for vulnerabilities. Be specific.",
    "You are a database expert. Help with query optimization, schema design, and migrations.",
    "You are a machine learning engineer. Help with model selection, training, and deployment.",
]

# Follow-up questions that build on prior conversation context
FOLLOWUPS = [
    "Can you expand on that last point?",
    "How would I test this?",
    "What are the edge cases I should handle?",
    "Can you refactor that to be more efficient?",
    "What would the error handling look like?",
    "How would this work at scale?",
    "Can you add type annotations to that?",
    "What are the security implications?",
    "How would I monitor this in production?",
    "Can you write a unit test for that?",
    "What's the performance impact?",
    "How does this compare to the alternative approach?",
    "Can you add logging to that?",
    "What configuration would you recommend?",
    "How would I deploy this?",
]


def create_conversation(prompts: list[list[dict[str, str]]]) -> Conversation:
    """Create a new conversation with a random system prompt and 3-8 max turns."""
    return Conversation(
        system_prompt=random.choice(SYSTEM_PROMPTS),
        max_turns=random.randint(3, 8),
    )


def get_user_message(conv: Conversation, prompts: list[list[dict[str, str]]]) -> str:
    """Get the next user message — initial prompt from dataset, then followups."""
    if conv.turn_count == 0:
        # First turn: use a real prompt from the dataset
        prompt_msgs = random.choice(prompts)
        user_parts = [m["content"] for m in prompt_msgs if m["role"] == "user"]
        return " ".join(user_parts)[:2000] if user_parts else "Help me with a coding task."
    else:
        # Subsequent turns: short follow-up that references prior context
        return random.choice(FOLLOWUPS)


# ---------------------------------------------------------------------------
# User provisioning
# ---------------------------------------------------------------------------

@dataclass
class VirtualUser:
    email: str
    user_id: str
    api_key: str
    key_hash: str


async def provision_users(num_users: int) -> list[VirtualUser]:
    """Create test users with API keys and balance."""
    import redis.asyncio as aioredis

    pool = await asyncpg.create_pool(POSTGRES_DSN)
    r = aioredis.from_url(REDIS_URL)

    users: list[VirtualUser] = []
    for i in range(num_users):
        email = f"stresstest-{i:04d}@shittytoken.test"

        row = await pool.fetchrow("SELECT id FROM users WHERE email = $1", email)
        if row:
            user_id = str(row["id"])
        else:
            row = await pool.fetchrow(
                "INSERT INTO users (email) VALUES ($1) RETURNING id", email
            )
            user_id = str(row["id"])

        plaintext = f"sk-st-stress-{secrets.token_hex(16)}"
        key_hash = hashlib.sha256(plaintext.encode()).hexdigest()

        await pool.execute(
            "INSERT INTO api_keys (key_hash, user_id, name) VALUES ($1, $2, $3) "
            "ON CONFLICT (key_hash) DO NOTHING",
            key_hash,
            row["id"],
            f"stress-test-{i}",
        )
        await r.set(f"balance:{user_id}", str(BALANCE_CENTS))

        users.append(VirtualUser(
            email=email, user_id=user_id, api_key=plaintext, key_hash=key_hash,
        ))

    await pool.close()
    await r.aclose()
    logger.info("users.provisioned", count=len(users))
    return users


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class RequestStats:
    started_at: float = 0.0
    total_requests: int = 0
    success: int = 0
    failed: int = 0
    timeouts: int = 0
    rate_limited: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int = 0
    latencies: list[float] = field(default_factory=list)
    ttfts: list[float] = field(default_factory=list)

    def record(
        self,
        success: bool,
        duration: float,
        ttft: float | None,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        status_code: int | None = None,
    ) -> None:
        self.total_requests += 1
        self.latencies.append(duration)
        if success:
            self.success += 1
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_cached_tokens += cached_tokens
            if ttft is not None:
                self.ttfts.append(ttft)
        elif status_code == 429:
            self.rate_limited += 1
            self.failed += 1
        else:
            self.failed += 1

    def summary(self) -> dict[str, Any]:
        elapsed = time.monotonic() - self.started_at if self.started_at else 0
        rps = self.total_requests / elapsed if elapsed > 0 else 0

        sorted_lat = sorted(self.latencies) if self.latencies else [0]
        sorted_ttft = sorted(self.ttfts) if self.ttfts else [0]

        total_input = self.total_prompt_tokens or 1
        cache_pct = (self.total_cached_tokens / total_input * 100) if total_input > 1 else 0

        return {
            "elapsed_s": round(elapsed, 1),
            "total": self.total_requests,
            "ok": self.success,
            "fail": self.failed,
            "rate_limited": self.rate_limited,
            "rps": round(rps, 2),
            "lat_p50": round(sorted_lat[len(sorted_lat) // 2], 2),
            "lat_p95": round(sorted_lat[int(len(sorted_lat) * 0.95)], 2),
            "ttft_p50": round(sorted_ttft[len(sorted_ttft) // 2], 2),
            "ttft_p95": round(sorted_ttft[int(len(sorted_ttft) * 0.95)], 2),
            "in_tok": self.total_prompt_tokens,
            "out_tok": self.total_completion_tokens,
            "cached_tok": self.total_cached_tokens,
            "cache%": round(cache_pct, 1),
        }


# ---------------------------------------------------------------------------
# Request sender
# ---------------------------------------------------------------------------

async def send_request(
    session: aiohttp.ClientSession,
    gateway_url: str,
    user: VirtualUser,
    messages: list[dict[str, str]],
    max_tokens: int,
    request_timeout: float,
    stats: RequestStats,
) -> str:
    """Send a streaming chat completion. Returns the assistant's response text."""
    url = f"{gateway_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Authorization": f"Bearer {user.api_key}"}
    timeout = aiohttp.ClientTimeout(total=request_timeout, connect=10)
    send_time = time.monotonic()

    ttft: float | None = None
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    output_parts: list[str] = []
    status_code: int | None = None

    try:
        async with session.post(url, json=payload, headers=headers, timeout=timeout) as resp:
            status_code = resp.status
            if resp.status != 200:
                duration = time.monotonic() - send_time
                stats.record(False, duration, None, 0, 0, status_code=status_code)
                return ""

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Extract usage from final chunk
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    ptd = usage.get("prompt_tokens_details") or {}
                    cached_tokens = ptd.get("cached_tokens", 0)

                try:
                    delta = chunk["choices"][0]["delta"]
                    content = delta.get("content", "")
                    reasoning = delta.get("reasoning_content", "")
                except (KeyError, IndexError, TypeError):
                    continue

                if content or reasoning:
                    if ttft is None:
                        ttft = time.monotonic() - send_time
                if content:
                    output_parts.append(content)

            duration = time.monotonic() - send_time
            stats.record(True, duration, ttft, prompt_tokens, completion_tokens, cached_tokens)
            return "".join(output_parts)

    except asyncio.TimeoutError:
        duration = time.monotonic() - send_time
        stats.timeouts += 1
        stats.record(False, duration, None, 0, 0)
        return ""
    except (aiohttp.ClientError, ConnectionError):
        duration = time.monotonic() - send_time
        stats.record(False, duration, None, 0, 0)
        return ""


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
    """Ramp up load with realistic conversation patterns."""
    stats = RequestStats()
    stats.started_at = time.monotonic()

    # Each user has an active conversation pool
    user_conversations: dict[str, list[Conversation]] = {
        u.user_id: [] for u in users
    }

    connector = aiohttp.TCPConnector(limit=max_concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
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

            sem = asyncio.Semaphore(concurrency)
            tasks: list[asyncio.Task] = []
            stop_event = asyncio.Event()

            async def _worker(worker_id: int) -> None:
                while not stop_event.is_set():
                    async with sem:
                        if stop_event.is_set():
                            break

                        user = random.choice(users)

                        # Get or create a conversation for this user
                        convs = user_conversations[user.user_id]
                        # Remove finished conversations
                        convs[:] = [c for c in convs if not c.is_finished]

                        # 70% chance to continue an existing conversation (cache hit)
                        # 30% chance to start a new one (cache miss)
                        conv: Conversation
                        if convs and random.random() < 0.7:
                            conv = random.choice(convs)
                        else:
                            conv = create_conversation(prompts)
                            convs.append(conv)

                        user_msg = get_user_message(conv, prompts)
                        messages = conv.build_messages(user_msg)

                        response = await send_request(
                            session, gateway_url, user, messages,
                            max_tokens, request_timeout, stats,
                        )

                        # Record the turn so next request has growing context
                        if response:
                            conv.record_response(user_msg, response[:500])

            for i in range(concurrency):
                tasks.append(asyncio.create_task(_worker(i)))

            step_elapsed = 0.0
            log_interval = 10.0
            while step_elapsed < step_duration:
                wait_time = min(log_interval, step_duration - step_elapsed)
                await asyncio.sleep(wait_time)
                step_elapsed = time.monotonic() - step_start
                logger.info("stats.snapshot", **stats.summary())

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
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Max output tokens — kept low for realistic ratio")
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

    # 1. Load dataset prompts (used as initial conversation starters)
    prompts = load_nemotron_prompts()
    if not prompts:
        logger.error("No prompts loaded from dataset!")
        return

    # 2. Provision virtual users
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
