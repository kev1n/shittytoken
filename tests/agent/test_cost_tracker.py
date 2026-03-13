"""
Tests for CostTracker.

Covers:
- register + deregister: cumulative cost accumulation
- hourly_burn_usd: sum of rates across multiple instances
- cumulative_cost_usd: completed + running instance costs
- maybe_log_summary: logs at intervals
- deregister unknown instance: no-op
- register with 0 cost: works without error
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from structlog.testing import capture_logs

from shittytoken.agent.cost_tracker import CostTracker

MODULE = "shittytoken.agent.cost_tracker"


class TestRegisterDeregister:
    """register + deregister accumulates cumulative cost correctly."""

    def test_cumulative_cost_after_deregister(self):
        tracker = CostTracker()
        with patch(f"{MODULE}.time") as mock_time:
            # Register at t=0
            mock_time.time.return_value = 1000.0
            tracker.register("inst-1", cost_per_hour_usd=1.0)

            # Deregister at t=3600 (1 hour later) -> $1.00
            mock_time.time.return_value = 4600.0
            tracker.deregister("inst-1")

        assert tracker.active_instances == 0
        assert tracker._cumulative_usd == pytest.approx(1.0, abs=0.01)

    def test_multiple_register_deregister(self):
        tracker = CostTracker()
        with patch(f"{MODULE}.time") as mock_time:
            mock_time.time.return_value = 0.0
            tracker.register("inst-1", cost_per_hour_usd=2.0)

            mock_time.time.return_value = 1800.0  # 0.5 hours -> $1.00
            tracker.deregister("inst-1")

            mock_time.time.return_value = 2000.0
            tracker.register("inst-2", cost_per_hour_usd=3.0)

            mock_time.time.return_value = 5600.0  # 1 hour -> $3.00
            tracker.deregister("inst-2")

        assert tracker._cumulative_usd == pytest.approx(4.0, abs=0.01)


class TestHourlyBurn:
    """hourly_burn_usd is the sum of cost_per_hour across active instances."""

    def test_single_instance(self):
        tracker = CostTracker()
        with patch(f"{MODULE}.time") as mock_time:
            mock_time.time.return_value = 0.0
            tracker.register("inst-1", cost_per_hour_usd=1.50)

        assert tracker.hourly_burn_usd == pytest.approx(1.50)

    def test_multiple_instances(self):
        tracker = CostTracker()
        with patch(f"{MODULE}.time") as mock_time:
            mock_time.time.return_value = 0.0
            tracker.register("inst-1", cost_per_hour_usd=1.00)
            tracker.register("inst-2", cost_per_hour_usd=2.50)
            tracker.register("inst-3", cost_per_hour_usd=0.75)

        assert tracker.hourly_burn_usd == pytest.approx(4.25)

    def test_empty_tracker(self):
        tracker = CostTracker()
        assert tracker.hourly_burn_usd == 0.0


class TestCumulativeCost:
    """cumulative_cost_usd includes both completed and running instance costs."""

    def test_includes_running_instances(self):
        tracker = CostTracker()
        with patch(f"{MODULE}.time") as mock_time:
            # Register at t=0
            mock_time.time.return_value = 0.0
            tracker.register("inst-1", cost_per_hour_usd=2.0)

            # Check cumulative at t=1800 (0.5 hours) -> $1.00 accrued
            mock_time.time.return_value = 1800.0
            assert tracker.cumulative_cost_usd == pytest.approx(1.0, abs=0.01)

    def test_includes_completed_and_running(self):
        tracker = CostTracker()
        with patch(f"{MODULE}.time") as mock_time:
            # First instance: register and deregister
            mock_time.time.return_value = 0.0
            tracker.register("inst-1", cost_per_hour_usd=1.0)
            mock_time.time.return_value = 3600.0  # 1 hour -> $1.00
            tracker.deregister("inst-1")

            # Second instance: still running
            mock_time.time.return_value = 3600.0
            tracker.register("inst-2", cost_per_hour_usd=2.0)

            # Check at t=5400 (inst-2 running 0.5 hours -> $1.00)
            mock_time.time.return_value = 5400.0
            # Total: $1.00 (completed) + $1.00 (running) = $2.00
            assert tracker.cumulative_cost_usd == pytest.approx(2.0, abs=0.01)


class TestMaybeLogSummary:
    """maybe_log_summary logs at intervals."""

    def test_logs_on_first_call(self):
        tracker = CostTracker()
        with capture_logs() as logs:
            tracker.maybe_log_summary(interval_s=60.0)

        cost_logs = [l for l in logs if l.get("event") == "cost_summary"]
        assert len(cost_logs) == 1

    def test_skips_within_interval(self):
        tracker = CostTracker()
        with patch(f"{MODULE}.time") as mock_time:
            mock_time.time.return_value = 100.0
            tracker.maybe_log_summary(interval_s=60.0)

            with capture_logs() as logs:
                mock_time.time.return_value = 130.0  # only 30s later
                tracker.maybe_log_summary(interval_s=60.0)

            cost_logs = [l for l in logs if l.get("event") == "cost_summary"]
            assert len(cost_logs) == 0

    def test_logs_after_interval(self):
        tracker = CostTracker()
        with patch(f"{MODULE}.time") as mock_time:
            mock_time.time.return_value = 100.0
            tracker.maybe_log_summary(interval_s=60.0)

            with capture_logs() as logs:
                mock_time.time.return_value = 200.0  # 100s later
                tracker.maybe_log_summary(interval_s=60.0)

            cost_logs = [l for l in logs if l.get("event") == "cost_summary"]
            assert len(cost_logs) == 1


class TestEdgeCases:
    """Edge cases: unknown instance deregister, zero cost."""

    def test_deregister_unknown_is_noop(self):
        tracker = CostTracker()
        # Should not raise
        tracker.deregister("nonexistent-id")
        assert tracker._cumulative_usd == 0.0
        assert tracker.active_instances == 0

    def test_register_zero_cost(self):
        tracker = CostTracker()
        with patch(f"{MODULE}.time") as mock_time:
            mock_time.time.return_value = 0.0
            tracker.register("inst-free", cost_per_hour_usd=0.0)

        assert tracker.active_instances == 1
        assert tracker.hourly_burn_usd == 0.0

        with patch(f"{MODULE}.time") as mock_time:
            mock_time.time.return_value = 7200.0
            tracker.deregister("inst-free")

        assert tracker._cumulative_usd == 0.0
