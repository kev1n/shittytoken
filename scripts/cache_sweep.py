#!/usr/bin/env python3
"""Binary search for optimal user count before cache hit rate drops off.

Uses identical doc sizes for all tests to keep it apples-to-apples.
KV cache is ~227K tokens. We fix doc size so each user's prefix is
a consistent size, then vary only the number of concurrent users.
"""
import subprocess
import json
import sys

DURATION = 90
DOC_MIN = 5000
DOC_MAX = 20000  # fixed for all tests — avg ~12K prefix per user
CACHE_THRESHOLD = 0.50

results = {}

def test_users(n):
    if n in results:
        return results[n]
    print(f"\n  Testing {n} users ({DURATION}s, docs {DOC_MIN}-{DOC_MAX} tokens)...", flush=True)
    subprocess.run(
        [sys.executable, "scripts/stress_test.py",
         "--users", str(n), "--duration", str(DURATION),
         "--ramp-users", "0", "--start-jitter", "1",
         "--doc-tokens-min", str(DOC_MIN), "--doc-tokens-max", str(DOC_MAX)],
        capture_output=True, text=True, timeout=DURATION + 120,
    )
    with open("workload_results.json") as f:
        s = json.load(f)["summary"]
    rate = s["cache_hit_rate"]
    reqs = s["requests"]
    avg_prompt = s["total_prompt_tokens"] / max(reqs, 1)
    results[n] = rate
    print(f"  → {n} users: {rate*100:.1f}% cache, {reqs} reqs, avg {avg_prompt:.0f} prompt tok", flush=True)
    return rate

print(f"Binary search: threshold={CACHE_THRESHOLD*100:.0f}%, docs {DOC_MIN}-{DOC_MAX} tok")
print(f"KV cache ~227K tokens. Avg prefix ~{(DOC_MIN+DOC_MAX)//2} tokens/user.\n")

lo, hi = 3, 20

# Test both endpoints fresh with same params
test_users(lo)
test_users(hi)

while hi - lo > 1:
    mid = (lo + hi) // 2
    rate = test_users(mid)
    if rate >= CACHE_THRESHOLD:
        lo = mid
    else:
        hi = mid
    print(f"  bounds: lo={lo} ({results[lo]*100:.1f}%), hi={hi} ({results[hi]*100:.1f}%)")

print(f"\n{'='*50}")
print(f"RESULT: sweet spot is {lo} users")
print(f"\nAll results:")
for n in sorted(results):
    print(f"  {n:>3} users: {results[n]*100:.1f}% cache hit rate")
