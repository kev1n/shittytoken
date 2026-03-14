#!/usr/bin/env python3
"""
ShittyToken Stress Test — Realistic multi-user load with varied context sizes.

Simulates real API usage patterns:
- Distribution of context sizes from quick questions to massive docs
- Multi-turn conversations with growing, cacheable prefixes
- ~75% of prompt tokens are cached (prior conversation turns)
- Output is short relative to input (~150-200 tokens)
- Mix of user personas: casual users, power users, batch processors

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
import string
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
BALANCE_CENTS = 1_000_000  # $10,000 per user (large context = expensive)


# ---------------------------------------------------------------------------
# Synthetic context generators
# ---------------------------------------------------------------------------

# ~4 chars per token on average for English text
CHARS_PER_TOKEN = 4

# Realistic code snippets used to build large contexts
CODE_TEMPLATES = [
    '''def {name}({args}) -> {ret}:
    """{doc}"""
    result = []
    for item in {arg0}:
        if item.{attr} > threshold:
            processed = transform(item, mode="{mode}")
            result.append(processed)
    return {ret_expr}
''',
    '''class {name}:
    """{doc}"""

    def __init__(self, {args}):
        self.{arg0} = {arg0}
        self._cache = {{}}
        self._lock = asyncio.Lock()

    async def process(self, data: list[dict]) -> dict:
        async with self._lock:
            results = await asyncio.gather(*[
                self._handle_item(item) for item in data
            ])
        return {{"items": results, "count": len(results)}}

    async def _handle_item(self, item: dict) -> dict:
        key = item.get("id", "")
        if key in self._cache:
            return self._cache[key]
        result = await self._transform(item)
        self._cache[key] = result
        return result
''',
    '''async def {name}_handler(request: Request) -> Response:
    """{doc}"""
    body = await request.json()
    validate_schema(body, {name}_schema)

    async with get_db_session() as db:
        existing = await db.fetch_one(
            "SELECT * FROM {table} WHERE id = :id",
            {{"id": body["id"]}}
        )
        if existing:
            await db.execute(
                "UPDATE {table} SET {col} = :val WHERE id = :id",
                {{"val": body["{col}"], "id": body["id"]}}
            )
        else:
            await db.execute(
                "INSERT INTO {table} ({col}, created_at) VALUES (:val, NOW())",
                {{"val": body["{col}"]}}
            )

    return Response(status_code=200, body={{"status": "ok"}})
''',
]

NAMES = ["process_batch", "DataPipeline", "UserService", "MetricsAggregator",
         "handle_webhook", "sync_records", "validate_config", "transform_output",
         "CacheManager", "EventProcessor", "QueryOptimizer", "load_balancer"]
ATTRS = ["score", "priority", "timestamp", "status", "weight", "confidence"]
MODES = ["strict", "relaxed", "batch", "streaming", "incremental"]
TABLES = ["users", "events", "metrics", "configs", "sessions", "audit_log"]
COLS = ["data", "payload", "metadata", "state", "result", "content"]


def _gen_code_block() -> str:
    """Generate a realistic-looking code block (~200-500 chars)."""
    tmpl = random.choice(CODE_TEMPLATES)
    name = random.choice(NAMES)
    return tmpl.format(
        name=name,
        args="self, data, threshold=0.5",
        arg0="data",
        ret="list",
        doc=f"Process {name} with validation and error handling.",
        attr=random.choice(ATTRS),
        mode=random.choice(MODES),
        ret_expr="result",
        table=random.choice(TABLES),
        col=random.choice(COLS),
    )


def _gen_prose_paragraph() -> str:
    """Generate realistic technical prose (~100-300 chars)."""
    topics = [
        "The system architecture follows a microservices pattern with event-driven communication between components. Each service maintains its own data store and communicates through an async message bus. This ensures loose coupling and independent deployability.",
        "Performance testing revealed that the primary bottleneck occurs during concurrent database writes when the connection pool is exhausted. We implemented connection pooling with PgBouncer and added retry logic with exponential backoff to handle transient failures.",
        "The deployment pipeline consists of three stages: build, test, and deploy. Each stage runs in an isolated container to ensure reproducibility. The test stage includes unit tests, integration tests, and a synthetic load test that must pass before promotion to production.",
        "Error handling follows a hierarchical pattern where domain-specific exceptions are caught at the service layer, logged with full context, and translated to appropriate HTTP status codes at the API boundary. All errors include correlation IDs for distributed tracing.",
        "The caching strategy uses a two-tier approach: an in-process LRU cache for frequently accessed small objects, and a distributed Redis cache for larger computed results. Cache invalidation is event-driven through the message bus to maintain consistency across instances.",
        "Authentication uses JWT tokens with short expiry times and refresh token rotation. Each API request is validated against the token's scope and rate-limited per user. Service-to-service communication uses mutual TLS with certificate rotation managed by the platform.",
        "The monitoring stack includes Prometheus for metrics collection, Grafana for visualization, and structured logging with correlation IDs. Alerts are configured for SLO violations with a 5-minute evaluation window and PagerDuty integration for critical incidents.",
        "Database migrations are managed through versioned scripts that support both forward and backward migrations. Each migration is tested against a snapshot of production data before deployment. Schema changes that require table locks are executed during maintenance windows.",
    ]
    return random.choice(topics)


def generate_context_padding(target_tokens: int) -> str:
    """Generate realistic-looking context to reach a target token count.

    Mixes code blocks and prose to simulate a user pasting a codebase
    or document into a conversation.
    """
    target_chars = target_tokens * CHARS_PER_TOKEN
    parts: list[str] = []
    current_chars = 0

    while current_chars < target_chars:
        # 60% code, 40% prose
        if random.random() < 0.6:
            block = _gen_code_block()
        else:
            block = _gen_prose_paragraph()
        parts.append(block)
        current_chars += len(block)

    text = "\n\n".join(parts)
    # Trim to approximate target
    return text[:target_chars]


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_nemotron_prompts(max_rows: int = 500) -> list[str]:
    """Stream & extract user prompts from the Nemotron RL blend JSONL."""
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

    prompts: list[str] = []
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

                user_parts = [
                    m.get("content", "")
                    for m in messages_raw
                    if m.get("role") == "user" and m.get("content")
                ]
                if user_parts:
                    prompts.append(" ".join(user_parts)[:4000])

                if len(prompts) >= max_rows:
                    break
            if len(prompts) >= max_rows:
                break

    logger.info("dataset.prompts_extracted", count=len(prompts), lines_read=lines_read)
    return prompts


# ---------------------------------------------------------------------------
# User personas — different usage patterns
# ---------------------------------------------------------------------------

@dataclass
class UserPersona:
    """Defines how a user interacts with the API."""
    name: str
    # Token count range for context (min, max)
    context_range: tuple[int, int]
    # Probability weights for context size buckets
    max_turns: tuple[int, int]
    # How often they continue vs start new conversations
    continue_prob: float
    # Max output tokens
    max_tokens: int
    # Weight for how often this persona appears
    weight: float


PERSONAS = [
    UserPersona(
        name="casual",
        context_range=(500, 5_000),
        max_turns=(2, 4),
        continue_prob=0.5,
        max_tokens=200,
        weight=0.35,
    ),
    UserPersona(
        name="developer",
        context_range=(2_000, 30_000),
        max_turns=(3, 8),
        continue_prob=0.7,
        max_tokens=300,
        weight=0.30,
    ),
    UserPersona(
        name="power_user",
        context_range=(10_000, 80_000),
        max_turns=(4, 10),
        continue_prob=0.8,
        max_tokens=200,
        weight=0.20,
    ),
    UserPersona(
        name="batch_processor",
        context_range=(50_000, 131_000),
        max_turns=(1, 3),
        continue_prob=0.3,
        max_tokens=500,
        weight=0.15,
    ),
]


def pick_persona() -> UserPersona:
    weights = [p.weight for p in PERSONAS]
    return random.choices(PERSONAS, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Conversation simulator
# ---------------------------------------------------------------------------

@dataclass
class Conversation:
    """Multi-turn conversation with growing, cacheable context."""
    system_prompt: str
    persona: UserPersona
    turns: list[dict[str, str]] = field(default_factory=list)
    turn_count: int = 0
    max_turns: int = 5
    # The initial context document (stays constant = cacheable prefix)
    context_doc: str = ""

    @property
    def is_finished(self) -> bool:
        return self.turn_count >= self.max_turns

    def build_messages(self, user_message: str) -> list[dict[str, str]]:
        """Build message array. System + context_doc + prior turns = cacheable prefix."""
        messages = [{"role": "system", "content": self.system_prompt}]
        if self.context_doc:
            messages.append({
                "role": "user",
                "content": f"Here is the context I'm working with:\n\n{self.context_doc}",
            })
            messages.append({
                "role": "assistant",
                "content": "I've reviewed the context. What would you like to know?",
            })
        messages.extend(self.turns)
        messages.append({"role": "user", "content": user_message})
        return messages

    def record_response(self, user_message: str, assistant_response: str) -> None:
        self.turns.append({"role": "user", "content": user_message})
        self.turns.append({"role": "assistant", "content": assistant_response})
        self.turn_count += 1


SYSTEM_PROMPTS = [
    "You are a helpful coding assistant. Give concise, working code examples.",
    "You are a data analysis expert. Provide clear summaries and insights.",
    "You are a technical writer helping document software APIs.",
    "You are a DevOps engineer helping with infrastructure.",
    "You are a security researcher. Analyze code for vulnerabilities.",
    "You are a database expert. Help with query optimization and schema design.",
    "You are a machine learning engineer helping with model deployment.",
    "You are a software architect reviewing system designs.",
]

FOLLOWUPS = [
    "Can you expand on that last point?",
    "How would I test this?",
    "What are the edge cases I should handle?",
    "Can you refactor that to be more efficient?",
    "What would the error handling look like?",
    "How would this work at scale?",
    "What are the security implications?",
    "How would I monitor this in production?",
    "Can you write a unit test for that?",
    "What's the performance impact of this approach?",
    "How does this compare to the alternative?",
    "What configuration would you recommend for production?",
    "Can you explain the tradeoffs here?",
    "What would a migration plan look like?",
    "How should I handle backwards compatibility?",
]


def create_conversation(prompts: list[str], persona: UserPersona) -> Conversation:
    """Create a conversation with context sized to the persona."""
    target_tokens = random.randint(*persona.context_range)
    context_doc = ""
    if target_tokens > 1000:
        context_doc = generate_context_padding(target_tokens)

    return Conversation(
        system_prompt=random.choice(SYSTEM_PROMPTS),
        persona=persona,
        max_turns=random.randint(*persona.max_turns),
        context_doc=context_doc,
    )


def get_user_message(conv: Conversation, prompts: list[str]) -> str:
    if conv.turn_count == 0:
        prompt = random.choice(prompts)
        if conv.context_doc:
            questions = [
                "Can you review this code and suggest improvements?",
                "What are the main issues with this implementation?",
                "How would you refactor this for better performance?",
                "Can you explain how this system works?",
                "What tests should I write for this?",
                "Are there any security vulnerabilities here?",
                "How would you add error handling to this?",
                "What's the time complexity of the main operations?",
            ]
            return random.choice(questions)
        return prompt[:2000]
    else:
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
    persona: UserPersona


async def provision_users(num_users: int) -> list[VirtualUser]:
    import redis.asyncio as aioredis

    pool = await asyncpg.create_pool(POSTGRES_DSN)
    r = aioredis.from_url(REDIS_URL)

    users: list[VirtualUser] = []
    for i in range(num_users):
        email = f"stresstest-{i:04d}@shittytoken.test"
        persona = pick_persona()

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
            email=email, user_id=user_id, api_key=plaintext,
            key_hash=key_hash, persona=persona,
        ))

    await pool.close()
    await r.aclose()

    persona_counts = {}
    for u in users:
        persona_counts[u.persona.name] = persona_counts.get(u.persona.name, 0) + 1
    logger.info("users.provisioned", count=len(users), personas=persona_counts)
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
    latencies: list[float] = field(default_factory=list)
    ttfts: list[float] = field(default_factory=list)
    # Track context size distribution
    context_buckets: dict[str, int] = field(default_factory=lambda: {
        "tiny(<1k)": 0, "small(1-5k)": 0, "med(5-30k)": 0,
        "large(30-80k)": 0, "huge(80k+)": 0,
    })

    def _bucket(self, tokens: int) -> str:
        if tokens < 1000:
            return "tiny(<1k)"
        if tokens < 5000:
            return "small(1-5k)"
        if tokens < 30000:
            return "med(5-30k)"
        if tokens < 80000:
            return "large(30-80k)"
        return "huge(80k+)"

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
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            if ttft is not None:
                self.ttfts.append(ttft)
            bucket = self._bucket(prompt_tokens)
            self.context_buckets[bucket] = self.context_buckets.get(bucket, 0) + 1
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

        tps = (self.total_prompt_tokens + self.total_completion_tokens) / elapsed if elapsed > 0 else 0

        return {
            "elapsed_s": round(elapsed, 1),
            "total": self.total_requests,
            "ok": self.success,
            "fail": self.failed,
            "rate_limited": self.rate_limited,
            "rps": round(rps, 2),
            "tps": round(tps, 0),
            "lat_p50": round(sorted_lat[len(sorted_lat) // 2], 2),
            "lat_p95": round(sorted_lat[int(len(sorted_lat) * 0.95)], 2),
            "ttft_p50": round(sorted_ttft[len(sorted_ttft) // 2], 2),
            "ttft_p95": round(sorted_ttft[int(len(sorted_ttft) * 0.95)], 2),
            "in_tok": self.total_prompt_tokens,
            "out_tok": self.total_completion_tokens,
            "buckets": dict(self.context_buckets),
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
                if content:
                    output_parts.append(content)

            duration = time.monotonic() - send_time
            stats.record(True, duration, ttft, prompt_tokens, completion_tokens)
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
    prompts: list[str],
    max_concurrency: int,
    ramp_steps: int,
    step_duration: float,
    request_timeout: float,
) -> None:
    """Ramp up load with realistic conversation patterns and context sizes."""
    stats = RequestStats()
    stats.started_at = time.monotonic()

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
                        persona = user.persona

                        convs = user_conversations[user.user_id]
                        convs[:] = [c for c in convs if not c.is_finished]

                        conv: Conversation
                        if convs and random.random() < persona.continue_prob:
                            conv = random.choice(convs)
                        else:
                            conv = create_conversation(prompts, persona)
                            convs.append(conv)

                        user_msg = get_user_message(conv, prompts)
                        messages = conv.build_messages(user_msg)

                        response = await send_request(
                            session, gateway_url, user, messages,
                            persona.max_tokens, request_timeout, stats,
                        )

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
    parser.add_argument("--num-users", type=int, default=50)
    parser.add_argument("--max-concurrency", type=int, default=100)
    parser.add_argument("--ramp-steps", type=int, default=6)
    parser.add_argument("--step-duration", type=float, default=90.0)
    parser.add_argument("--request-timeout", type=float, default=300.0,
                        help="Timeout per request (large contexts need more time)")
    args = parser.parse_args()

    logger.info(
        "stress_test.config",
        gateway_url=args.gateway_url,
        num_users=args.num_users,
        max_concurrency=args.max_concurrency,
        ramp_steps=args.ramp_steps,
        step_duration_s=args.step_duration,
        personas={p.name: f"{p.context_range[0]}-{p.context_range[1]}tok, weight={p.weight}" for p in PERSONAS},
    )

    prompts = load_nemotron_prompts()
    if not prompts:
        logger.error("No prompts loaded from dataset!")
        return

    users = await provision_users(args.num_users)

    await run_stress_test(
        gateway_url=args.gateway_url,
        users=users,
        prompts=prompts,
        max_concurrency=args.max_concurrency,
        ramp_steps=args.ramp_steps,
        step_duration=args.step_duration,
        request_timeout=args.request_timeout,
    )


if __name__ == "__main__":
    asyncio.run(main())
