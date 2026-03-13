"""
CostTracker — tracks per-instance and aggregate cloud compute costs.

Logs a cost summary periodically and exposes Prometheus-compatible gauges.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


@dataclass
class _InstanceCost:
    cost_per_hour_usd: float
    started_at: float  # wall-clock time.time()


class CostTracker:
    """Track cost for active GPU instances."""

    def __init__(self) -> None:
        self._instances: dict[str, _InstanceCost] = {}
        self._cumulative_usd: float = 0.0
        self._last_log_time: float = 0.0

    def register(self, instance_id: str, cost_per_hour_usd: float) -> None:
        """Start tracking cost for an instance."""
        self._instances[instance_id] = _InstanceCost(
            cost_per_hour_usd=cost_per_hour_usd,
            started_at=time.time(),
        )
        logger.info(
            "cost_tracker.registered",
            instance_id=instance_id,
            cost_per_hour_usd=round(cost_per_hour_usd, 4),
        )

    def deregister(self, instance_id: str) -> None:
        """Stop tracking and add final cost to cumulative total."""
        entry = self._instances.pop(instance_id, None)
        if entry:
            runtime_hours = (time.time() - entry.started_at) / 3600.0
            cost = runtime_hours * entry.cost_per_hour_usd
            self._cumulative_usd += cost
            logger.info(
                "cost_tracker.deregistered",
                instance_id=instance_id,
                runtime_hours=round(runtime_hours, 3),
                cost_usd=round(cost, 4),
                cumulative_usd=round(self._cumulative_usd, 4),
            )

    @property
    def hourly_burn_usd(self) -> float:
        """Current $/hr burn rate across all tracked instances."""
        return sum(e.cost_per_hour_usd for e in self._instances.values())

    @property
    def cumulative_cost_usd(self) -> float:
        """Total spend: completed instances + accrued on running instances."""
        total = self._cumulative_usd
        now = time.time()
        for entry in self._instances.values():
            runtime_hours = (now - entry.started_at) / 3600.0
            total += runtime_hours * entry.cost_per_hour_usd
        return total

    @property
    def active_instances(self) -> int:
        return len(self._instances)

    def maybe_log_summary(self, interval_s: float = 60.0) -> None:
        """Log cost summary if at least interval_s has passed since last log."""
        now = time.time()
        if now - self._last_log_time < interval_s:
            return
        self._last_log_time = now
        logger.info(
            "cost_summary",
            active_instances=self.active_instances,
            hourly_burn_usd=round(self.hourly_burn_usd, 4),
            cumulative_cost_usd=round(self.cumulative_cost_usd, 4),
        )
