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
        overload_running: int = 32,
        overload_cache_pct: float = 0.9,
        vnodes: int = 150,
    ) -> None:
        self.overload_running = overload_running
        self.overload_cache_pct = overload_cache_pct
        self.vnodes = vnodes

        # Consistent hash ring: sorted list of (hash_int, worker_url).
        self._ring: list[tuple[int, str]] = []
        self._ring_keys: list[int] = []  # parallel sorted key list for bisect

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_prefix_key(messages: list[dict]) -> str:
        """SHA-256 of system_prompt + first_user_message.

        ``messages[0]`` is typically the system prompt and ``messages[1]``
        the first user message.  If fewer than two messages are provided
        we hash whatever is available.

        Returns the hex-digest string.
        """
        parts: list[str] = []
        for msg in messages[:2]:
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle structured content blocks (e.g. [{"type":"text","text":"…"}])
                content = " ".join(
                    block.get("text", "") for block in content if isinstance(block, dict)
                )
            parts.append(str(content))
        combined = "\n".join(parts)
        return hashlib.sha256(combined.encode()).hexdigest()

    def select(
        self,
        prefix_key: str,
        workers: list[WorkerState],
    ) -> WorkerState:
        """Pick the best worker for *prefix_key*.

        1. Single worker — return it immediately.
        2. Consistent-hash to the nearest ring node.
        3. If the target is overloaded or unhealthy, fall back to the
           least-loaded healthy worker.
        4. If ALL workers are unhealthy, return the one with the fewest
           running requests anyway.
        """
        if len(workers) == 1:
            return workers[0]

        # Rebuild ring (cheap for typical pool sizes).
        self._build_ring(workers)

        url_to_worker = {w.url: w for w in workers}
        target = url_to_worker[self._ring_lookup(prefix_key)]

        if target.healthy and not self._is_overloaded(target):
            return target

        # Fallback: least-loaded healthy worker.
        healthy = [w for w in workers if w.healthy]
        if healthy:
            return min(healthy, key=lambda w: w.requests_running)

        # All unhealthy — pick fewest requests.
        return min(workers, key=lambda w: w.requests_running)

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

    def _is_overloaded(self, worker: WorkerState) -> bool:
        return (
            worker.requests_running > self.overload_running
            or worker.kv_cache_pct > self.overload_cache_pct
        )

    @staticmethod
    def _hash(key: str) -> int:
        """Return an integer hash for *key* using SHA-256."""
        return int(hashlib.sha256(key.encode()).hexdigest(), 16)
