#!/usr/bin/env python3
"""
ShittyToken Stress Test — Agentic cache workload through the gateway.

Based on entropy-script3.py pattern:
- Fetches real text from Project Gutenberg as document corpus
- Each user has a session with a FIXED document (cacheable prefix)
- Multi-turn agentic pipeline: extract → analyze → cross-ref → synthesize → plan
- Context grows each turn (history appended), prefix stays cached
- 150:1 encode:decode ratio (realistic for long-context analysis)
- Cache hit rate ≈ (avg_turns-1)/avg_turns per session

Routes through the ShittyToken gateway with auth, billing, rate limiting.
Uses Prometheus /metrics for cache hit rate (prompt_tokens_details is broken in vLLM V1).

Usage:
    uv run python scripts/stress_test.py [OPTIONS]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import secrets
import threading
import time
from dataclasses import dataclass

import asyncpg
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

POSTGRES_DSN = "postgresql://shittytoken:shittytoken_dev@localhost:5432/shittytoken"
REDIS_URL = "redis://localhost:6379/0"
BALANCE_CENTS = 10_000_000  # $100,000 per user (long context is expensive)

# Gutenberg books — fetched until we have enough text
BOOK_URLS = [
    "https://www.gutenberg.org/files/1661/1661-0.txt",   # Sherlock Holmes
    "https://www.gutenberg.org/files/1342/1342-0.txt",   # Pride and Prejudice
    "https://www.gutenberg.org/files/84/84-0.txt",       # Frankenstein
    "https://www.gutenberg.org/files/2701/2701-0.txt",   # Moby Dick
]

# Agentic task pipeline — cycles through these
AGENT_TASKS = [
    (
        "Extract",
        "List the 3 most important named entities (people, places, or organizations) "
        "from the NEW section of the document added this round. "
        "Output only the names, one per line, nothing else."
    ),
    (
        "Analyze",
        "Using the entities just extracted, identify which one plays the most pivotal "
        "role in this section and explain why in exactly one sentence."
    ),
    (
        "CrossRef",
        "Find one direct quote from the document that best illustrates the main "
        "conflict or tension in this section. Quote it exactly, under 20 words."
    ),
    (
        "Synthesize",
        "In exactly two sentences: summarize what occurred in this section and how "
        "it connects to the key entity identified above."
    ),
    (
        "Plan",
        "Given everything analyzed so far, state in one sentence the single most "
        "important question an investigator should pursue next."
    ),
]


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def fetch_corpus(target_chars: int) -> str:
    """Download Gutenberg books until we have enough text."""
    corpus = ""
    for url in BOOK_URLS:
        if len(corpus) >= target_chars:
            break
        try:
            print(f"  Fetching {url} ...")
            r = requests.get(url, timeout=30)
            text = r.text
            start = text.find("*** START OF")
            if start != -1:
                text = text[text.find("\n", start)+1:]
            end = text.find("*** END OF")
            if end != -1:
                text = text[:end]
            text = re.sub(r'\n{3,}', '\n\n', text).strip()
            corpus += "\n\n" + text
            print(f"    {len(text):,} chars (~{len(text)//4:,} tokens)")
        except Exception as e:
            print(f"    Failed: {e}")
    return corpus[:target_chars]


# ---------------------------------------------------------------------------
# User provisioning (sync version for simplicity)
# ---------------------------------------------------------------------------

@dataclass
class VirtualUser:
    email: str
    user_id: str
    api_key: str


def provision_users_sync(num_users: int) -> list[VirtualUser]:
    """Create test users with API keys and balance using sync pg."""
    import asyncio
    return asyncio.run(_provision_users_async(num_users))


async def _provision_users_async(num_users: int) -> list[VirtualUser]:
    import redis.asyncio as aioredis
    pool = await asyncpg.create_pool(POSTGRES_DSN)
    r = aioredis.from_url(REDIS_URL)
    users = []

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
            key_hash, row["id"], f"stress-test-{i}",
        )
        await r.set(f"balance:{user_id}", str(BALANCE_CENTS))
        users.append(VirtualUser(email=email, user_id=user_id, api_key=plaintext))

    await pool.close()
    await r.aclose()
    return users


# ---------------------------------------------------------------------------
# Chat completion (streaming, through gateway)
# ---------------------------------------------------------------------------

def chat_completion(
    gateway_url: str,
    api_key: str,
    messages: list[dict],
    max_tokens: int = 64,
) -> dict:
    """Send a streaming chat completion through the gateway."""
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    t0 = time.time()
    ttft_ms = None
    content = ""
    usage = {}

    with requests.post(
        f"{gateway_url.rstrip('/')}/v1/chat/completions",
        json=payload, headers=headers, stream=True, timeout=300,
    ) as resp:
        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith(b"data: "):
                line = line[6:]
            if line == b"[DONE]":
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in chunk:
                return chunk
            delta = ((chunk.get("choices") or [{}])[0]).get("delta", {})
            token = delta.get("content") or delta.get("reasoning_content") or ""
            if token and ttft_ms is None:
                ttft_ms = (time.time() - t0) * 1000
            content += delta.get("content") or ""
            if chunk.get("usage"):
                usage = chunk["usage"]

    total_ms = (time.time() - t0) * 1000
    return {
        "choices": [{"message": {"content": content}}],
        "usage": usage,
        "_elapsed_ms": total_ms,
        "_ttft_ms": ttft_ms if ttft_ms is not None else total_ms,
    }


# ---------------------------------------------------------------------------
# Prometheus cache counters
# ---------------------------------------------------------------------------

def get_prometheus_cache(worker_urls: list[str]) -> tuple[float, float]:
    """Scrape prefix cache hits/queries from all workers."""
    total_hits = 0.0
    total_queries = 0.0
    for url in worker_urls:
        try:
            text = requests.get(f"{url}/metrics", timeout=5).text
            hits = re.search(r'vllm:prefix_cache_hits_total\{[^}]*\}\s+([\d.e+]+)', text)
            queries = re.search(r'vllm:prefix_cache_queries_total\{[^}]*\}\s+([\d.e+]+)', text)
            if hits:
                total_hits += float(hits.group(1))
            if queries:
                total_queries += float(queries.group(1))
        except Exception:
            pass
    return total_hits, total_queries


def get_worker_urls(gateway_url: str) -> list[str]:
    """Get worker URLs from gateway admin API."""
    try:
        resp = requests.get(f"{gateway_url.rstrip('/')}/admin/workers", timeout=5)
        workers = resp.json()
        return [w["url"] for w in workers if isinstance(w, dict)]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class Result:
    user_num: int
    session_num: int
    session_req: int
    task: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    elapsed_ms: float
    output: str


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    parser = argparse.ArgumentParser(description="ShittyToken Stress Test")
    parser.add_argument("--gateway-url", default="http://localhost:8001")
    parser.add_argument("--users", type=int, default=10)
    parser.add_argument("--duration", type=int, default=300, help="Run duration in seconds")
    parser.add_argument("--doc-tokens-min", type=int, default=500)
    parser.add_argument("--doc-tokens-max", type=int, default=95000)
    parser.add_argument("--session-min-turns", type=int, default=3)
    parser.add_argument("--session-max-turns", type=int, default=12)
    parser.add_argument("--session-max-tokens", type=int, default=100000)
    parser.add_argument("--encode-decode-ratio", type=int, default=150)
    parser.add_argument("--start-jitter", type=int, default=15)
    args = parser.parse_args()

    GATEWAY_URL = args.gateway_url
    USERS = args.users
    RUN_DURATION_S = args.duration
    DOC_TOKENS_MIN = args.doc_tokens_min
    DOC_TOKENS_MAX = args.doc_tokens_max
    SESSION_MIN_TURNS = args.session_min_turns
    SESSION_MAX_TURNS = args.session_max_turns
    SESSION_MAX_TOKENS = args.session_max_tokens
    ENCODE_DECODE_RATIO = args.encode_decode_ratio
    START_JITTER_S = args.start_jitter

    print("=" * 80)
    print("SHITTYTOKEN AGENTIC CACHE WORKLOAD TEST")
    print(f"  Gateway:           {GATEWAY_URL}")
    print(f"  Users:             {USERS}")
    print(f"  Duration:          {RUN_DURATION_S}s")
    print(f"  Doc size/session:  {DOC_TOKENS_MIN:,}–{DOC_TOKENS_MAX:,} tokens")
    print(f"  Session turns:     {SESSION_MIN_TURNS}–{SESSION_MAX_TURNS}")
    print(f"  Encode:decode:     {ENCODE_DECODE_RATIO}:1")
    print(f"  Start jitter:      0–{START_JITTER_S}s")
    print("=" * 80 + "\n")

    # Provision users
    print("Provisioning virtual users...")
    users = provision_users_sync(USERS)
    print(f"  {len(users)} users ready\n")

    # Fetch corpus
    target_chars = DOC_TOKENS_MAX * 8  # 2x headroom
    print(f"Fetching corpus (~{target_chars//4:,} tokens needed)...")
    corpus = fetch_corpus(target_chars)
    print(f"Corpus: {len(corpus):,} chars (~{len(corpus)//4:,} tokens)\n")

    # Get worker URLs for Prometheus scraping
    worker_urls = get_worker_urls(GATEWAY_URL)
    print(f"Workers: {worker_urls}\n")
    prom_start = get_prometheus_cache(worker_urls)

    system_header = (
        "You are a document analysis agent working through a long document. "
        "Each request, new document sections are appended below. "
        "Respond concisely — your output will become context for your next request.\n\n"
        "=== DOCUMENT ===\n"
    )

    results: list[Result] = []
    results_lock = threading.Lock()
    stop_event = threading.Event()
    stop_reason = "Time limit reached"

    print(f"{'U':>2} {'Ses':>3} {'Req':>4}  {'Task':<10} {'Prompt':>8} {'Out':>4} {'TTFT':>7} {'Total':>7}  {'tok/s':>6}")
    print("-" * 75)
    run_start = time.time()

    timer = threading.Timer(RUN_DURATION_S, stop_event.set)
    timer.daemon = True
    timer.start()

    def user_workflow(user_idx: int):
        nonlocal stop_reason
        user = users[user_idx]
        time.sleep(random.uniform(0, START_JITTER_S))

        session_num = 0
        while not stop_event.is_set():
            session_num += 1
            doc_tokens = random.randint(DOC_TOKENS_MIN, DOC_TOKENS_MAX)
            doc_chars = doc_tokens * 4
            # Random position in corpus for this session's document
            doc_start = random.randint(0, max(0, len(corpus) - doc_chars))
            current_doc = corpus[doc_start:doc_start + doc_chars]
            session_len = random.randint(SESSION_MIN_TURNS, SESSION_MAX_TURNS)
            system_prompt = system_header + current_doc
            history: list[dict] = []
            task_idx = random.randint(0, len(AGENT_TASKS) - 1)
            session_req = 0

            for _ in range(session_len):
                if stop_event.is_set():
                    break

                approx_prompt = (len(system_prompt) + sum(len(m["content"]) for m in history)) // 4
                max_gen = max(1, approx_prompt // ENCODE_DECODE_RATIO)

                task_label, task_prompt = AGENT_TASKS[task_idx % len(AGENT_TASKS)]
                task_idx += 1
                session_req += 1

                msgs = [{"role": "system", "content": system_prompt}]
                msgs += history
                msgs.append({"role": "user", "content": task_prompt})

                try:
                    resp = chat_completion(GATEWAY_URL, user.api_key, msgs, max_tokens=max_gen)
                except Exception as e:
                    stop_event.set()
                    with results_lock:
                        if stop_reason == "Time limit reached":
                            stop_reason = f"Request error (user {user_idx}): {e}"
                    return

                if "error" in resp:
                    # Non-fatal: log and continue (might be rate limit, etc.)
                    with results_lock:
                        print(f"  [user {user_idx}] Error: {str(resp['error'])[:100]}")
                    continue

                usage = resp.get("usage") or {}
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                elapsed_ms = resp.get("_elapsed_ms", 0.0)
                ttft_ms = resp.get("_ttft_ms", 0.0)
                msg = ((resp.get("choices") or [{}])[0].get("message") or {})
                output = msg.get("content") or ""

                decode_ms = elapsed_ms - ttft_ms
                toks_per_s = (completion_tokens / decode_ms * 1000) if decode_ms > 0 else 0.0

                result = Result(
                    user_num=user_idx, session_num=session_num,
                    session_req=session_req, task=task_label,
                    prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                    ttft_ms=ttft_ms, elapsed_ms=elapsed_ms, output=output,
                )
                with results_lock:
                    results.append(result)
                    print(
                        f"{user_idx:>2} {session_num:>3} {session_req:>4}  {task_label:<10} "
                        f"{prompt_tokens:>8,} {completion_tokens:>4} {ttft_ms:>6.0f}ms {elapsed_ms:>6.0f}ms"
                        f"  {toks_per_s:>5.1f}/s"
                        + ("  ← new session" if session_req == 1 else "")
                    )

                history.append({"role": "user", "content": task_prompt})
                history.append({"role": "assistant", "content": output})

                if prompt_tokens >= SESSION_MAX_TOKENS:
                    break

    threads = [threading.Thread(target=user_workflow, args=(i,), daemon=True)
               for i in range(USERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    timer.cancel()

    # ── Summary ──────────────────────────────────────────────────────────────
    wall_ms = (time.time() - run_start) * 1000
    total_prompt = sum(r.prompt_tokens for r in results)
    total_completion = sum(r.completion_tokens for r in results)
    avg_ttft_ms = sum(r.ttft_ms for r in results) / len(results) if results else 0
    avg_elapsed_ms = sum(r.elapsed_ms for r in results) / len(results) if results else 0

    # Cache stats from Prometheus delta
    prom_end = get_prometheus_cache(worker_urls)
    prom_queries = prom_end[1] - prom_start[1]
    prom_hits = prom_end[0] - prom_start[0]
    hit_rate = (prom_hits / prom_queries) if prom_queries > 0 else 0.0
    total_cached = int(total_prompt * hit_rate)
    total_uncached = total_prompt - total_cached

    avg_prompt = total_prompt / len(results) if results else 0
    avg_completion = total_completion / len(results) if results else 0
    actual_ratio = (avg_prompt / avg_completion) if avg_completion > 0 else 0

    # Billing estimate
    pricing_input = 2.5   # cents per 1M tokens
    pricing_output = 15.0  # cents per 1M tokens
    input_cost = total_prompt / 1_000_000 * pricing_input
    output_cost = total_completion / 1_000_000 * pricing_output
    total_cost = input_cost + output_cost

    print("\n" + "=" * 80)
    print(f"STOP: {stop_reason}\n")
    print(f"  Total requests:              {len(results)}")
    print(f"  Wall time:                   {wall_ms/1000:.1f}s\n")
    print(f"  ── Token accounting ──")
    print(f"  Total prompt tokens:         {total_prompt:>12,}")
    print(f"    Cached (KV hit):           {total_cached:>12,}  ({hit_rate*100:.1f}%)")
    print(f"    Uncached (prefilled):      {total_uncached:>12,}  ({(1-hit_rate)*100:.1f}%)")
    print(f"  Completion tokens:           {total_completion:>12,}")
    print(f"  Avg encode:decode ratio:     {actual_ratio:>11.0f}:1  (target: {args.encode_decode_ratio}:1)")
    print(f"  Avg prompt tokens/req:       {avg_prompt:>11,.0f}")
    print(f"  Avg completion tokens/req:   {avg_completion:>11,.0f}")
    print(f"\n  ── Latency ──")
    print(f"  Avg TTFT:                    {avg_ttft_ms:>8.0f} ms")
    print(f"  Avg total latency/req:       {avg_elapsed_ms:>8.0f} ms")
    avg_decode_ms = avg_elapsed_ms - avg_ttft_ms
    avg_toks_per_s = (avg_completion / avg_decode_ms * 1000) if avg_decode_ms > 0 else 0
    print(f"  Avg decode throughput:       {avg_toks_per_s:>8.1f} tok/s")
    print(f"\n  ── Billing ──")
    print(f"  Input cost:                  ${input_cost/100:.4f}")
    print(f"  Output cost:                 ${output_cost/100:.4f}")
    print(f"  Total cost:                  ${total_cost/100:.4f}")
    print("=" * 80)

    # Save results
    out = {
        "config": {
            "gateway_url": GATEWAY_URL, "users": USERS,
            "doc_tokens_min": DOC_TOKENS_MIN, "doc_tokens_max": DOC_TOKENS_MAX,
            "session_min_turns": SESSION_MIN_TURNS, "session_max_turns": SESSION_MAX_TURNS,
            "session_max_tokens": SESSION_MAX_TOKENS,
            "encode_decode_ratio": ENCODE_DECODE_RATIO,
            "run_duration_s": RUN_DURATION_S,
        },
        "stop_reason": stop_reason,
        "summary": {
            "requests": len(results),
            "total_prompt_tokens": total_prompt,
            "total_cached_tokens": total_cached,
            "total_uncached_tokens": total_uncached,
            "total_completion_tokens": total_completion,
            "wall_ms": wall_ms,
            "cache_hit_rate": hit_rate,
            "actual_encode_decode_ratio": actual_ratio,
            "cost_cents": total_cost,
        },
        "requests": [
            {
                "user": r.user_num, "session": r.session_num,
                "session_req": r.session_req, "task": r.task,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "ttft_ms": round(r.ttft_ms, 1),
                "elapsed_ms": round(r.elapsed_ms, 1),
                "output_preview": r.output[:120],
            }
            for r in results
        ],
    }
    with open("workload_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nResults saved to workload_results.json")


if __name__ == "__main__":
    run()
