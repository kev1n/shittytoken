"""
CacheAwarePolicy — consistent hashing with least-loaded fallback.

Routes requests to workers that are most likely to have the relevant
KV-cache prefix already resident, falling back to the least-loaded
healthy worker when the target is overloaded or unhealthy.
"""

from __future__ import annotations

import bisect
import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shittytoken.gateway.worker_pool import WorkerState


class CacheAwarePolicy:
    """Cache-aware consistent hashing with least-loaded fallback."""

    def __init__(
        self,
        vnodes: int = 150,
        epsilon: float = 0.25,
    ) -> None:
        self.vnodes = vnodes
        self.epsilon = epsilon

        # Consistent hash ring: sorted list of (hash_int, worker_url).
        self._ring: list[tuple[int, str]] = []
        self._ring_keys: list[int] = []  # parallel sorted key list for bisect
        self._ring_worker_urls: frozenset[str] = frozenset()  # tracks when to rebuild

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_prefix_key(
        messages: list[dict],
        key_hash: str | None = None,
    ) -> str:
        """SHA-256 of api_key_hash + system_prompt content.

        Hashes only the system/developer messages (the stable, cacheable
        prefix).  When *key_hash* is provided it is prepended so that the
        same system prompt from different API keys routes to the same
        worker per-key, maximising vLLM prefix-cache hits.

        Returns the hex-digest string.
        """
        system_parts: list[str] = []
        for msg in messages:
            if msg.get("role") in ("system", "developer"):
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        block.get("text", "") for block in content if isinstance(block, dict)
                    )
                system_parts.append(str(content))
        combined = "\n".join(system_parts)
        if key_hash:
            combined = key_hash + "\n" + combined
        return hashlib.sha256(combined.encode()).hexdigest()

    def select(
        self,
        prefix_key: str,
        workers: list[WorkerState],
    ) -> WorkerState:
        """Pick the best worker for *prefix_key* using CHWBL.

        Consistent Hashing with Bounded Loads: walk the ring from the
        hash point and pick the first healthy worker whose load is below
        ``mean_load * (1 + epsilon) + 1``.  Falls back to the least-loaded
        healthy worker, then least-loaded overall.
        """
        if len(workers) == 1:
            return workers[0]

        # Only rebuild ring when the worker set changes.
        current_urls = frozenset(w.url for w in workers)
        if current_urls != self._ring_worker_urls:
            self._build_ring(workers)
            self._ring_worker_urls = current_urls

        url_to_worker = {w.url: w for w in workers}

        def _load(w):
            """Effective load: max of scraped metrics and local tracking."""
            return max(w.requests_running, getattr(w, 'local_in_flight', 0))

        mean_load = sum(_load(w) for w in workers) / len(workers)
        load_bound = mean_load * (1 + self.epsilon) + 1  # +1 avoids 0-mean edge

        # Primary target from consistent hash
        target = url_to_worker[self._ring_lookup(prefix_key)]
        if target.healthy and _load(target) <= load_bound:
            return target

        # Walk ring to find next worker under bound
        h = self._hash(prefix_key)
        idx = bisect.bisect_left(self._ring_keys, h)
        seen: set[str] = set()
        for _ in range(len(self._ring)):
            if idx >= len(self._ring):
                idx = 0
            url = self._ring[idx][1]
            if url not in seen:
                seen.add(url)
                candidate = url_to_worker[url]
                if candidate.healthy and _load(candidate) <= load_bound:
                    return candidate
            idx += 1

        # All above bound — least loaded healthy, then least loaded overall
        healthy = [w for w in workers if w.healthy]
        pool = healthy if healthy else workers
        return min(pool, key=_load)

    # ------------------------------------------------------------------
    # Ring internals
    # ------------------------------------------------------------------

    def _build_ring(self, workers: list[WorkerState]) -> None:
        """Rebuild the consistent hash ring from the current worker list."""
        ring: list[tuple[int, str]] = []
        for w in workers:
            for i in range(self.vnodes):
                h = self._hash(f"{w.url}:{i}")
                ring.append((h, w.url))
        ring.sort(key=lambda pair: pair[0])
        self._ring = ring
        self._ring_keys = [pair[0] for pair in ring]

    def _ring_lookup(self, prefix_key: str) -> str:
        """Find the nearest worker URL on the ring for *prefix_key*."""
        h = self._hash(prefix_key)
        idx = bisect.bisect_left(self._ring_keys, h)
        if idx >= len(self._ring):
            idx = 0  # wrap around
        return self._ring[idx][1]

    @staticmethod
    def _hash(key: str) -> int:
        """Return an integer hash for *key* using SHA-256."""
        return int(hashlib.sha256(key.encode()).hexdigest(), 16)
