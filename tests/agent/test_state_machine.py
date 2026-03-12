"""
Tests for InstanceStateMachine.

Covers:
- Valid transitions succeed
- Invalid transitions raise ValueError
- Every transition emits a structlog log entry with event=instance_state_transition
- Terminal states (FAILED, TERMINATED) reject all further transitions
"""

from __future__ import annotations

import pytest
import structlog
from structlog.testing import capture_logs

from shittytoken.agent.state_machine import (
    InstanceRecord,
    InstanceState,
    InstanceStateMachine,
)


def _make_sm(state: InstanceState = InstanceState.PROVISIONING) -> InstanceStateMachine:
    """Helper: create a state machine starting from the given state."""
    record = InstanceRecord(
        instance_id="test-instance-001",
        provider="vastai",
        gpu_model="RTX 4090",
    )
    sm = InstanceStateMachine(record)
    # Force the starting state without going through transition() so we can
    # test transitions from any state directly.
    sm._record.state = state
    return sm


# ---------------------------------------------------------------------------
# Valid transitions
# ---------------------------------------------------------------------------


def test_provisioning_to_benchmarking_succeeds():
    sm = _make_sm(InstanceState.PROVISIONING)
    sm.transition(InstanceState.BENCHMARKING, reason="gpu_verified")
    assert sm.state == InstanceState.BENCHMARKING


def test_benchmarking_to_serving_succeeds():
    sm = _make_sm(InstanceState.BENCHMARKING)
    sm.transition(InstanceState.SERVING, reason="benchmark_passed")
    assert sm.state == InstanceState.SERVING


def test_serving_to_draining_succeeds():
    sm = _make_sm(InstanceState.SERVING)
    sm.transition(InstanceState.DRAINING, reason="scale_down")
    assert sm.state == InstanceState.DRAINING


def test_draining_to_terminated_succeeds():
    sm = _make_sm(InstanceState.DRAINING)
    sm.transition(InstanceState.TERMINATED, reason="drain_complete")
    assert sm.state == InstanceState.TERMINATED


def test_any_state_to_failed_succeeds():
    """Every non-terminal state can transition to FAILED."""
    for start_state in (
        InstanceState.PROVISIONING,
        InstanceState.BENCHMARKING,
        InstanceState.SERVING,
        InstanceState.DRAINING,
    ):
        sm = _make_sm(start_state)
        sm.transition(InstanceState.FAILED, reason="error_occurred")
        assert sm.state == InstanceState.FAILED


# ---------------------------------------------------------------------------
# Invalid transitions
# ---------------------------------------------------------------------------


def test_provisioning_to_serving_raises():
    """PROVISIONING → SERVING skips BENCHMARKING; must be rejected."""
    sm = _make_sm(InstanceState.PROVISIONING)
    with pytest.raises(ValueError, match="provisioning"):
        sm.transition(InstanceState.SERVING, reason="skip_benchmark")


def test_provisioning_to_terminated_raises():
    """PROVISIONING → TERMINATED is not a valid transition."""
    sm = _make_sm(InstanceState.PROVISIONING)
    with pytest.raises(ValueError):
        sm.transition(InstanceState.TERMINATED, reason="direct_terminate")


def test_serving_to_benchmarking_raises():
    """Backwards transitions are invalid."""
    sm = _make_sm(InstanceState.SERVING)
    with pytest.raises(ValueError):
        sm.transition(InstanceState.BENCHMARKING, reason="re_benchmark")


# ---------------------------------------------------------------------------
# Terminal states
# ---------------------------------------------------------------------------


def test_terminated_to_any_raises():
    """TERMINATED is terminal — no further transitions allowed."""
    sm = _make_sm(InstanceState.TERMINATED)
    for target in InstanceState:
        with pytest.raises(ValueError, match="terminal"):
            sm.transition(target, reason="should_fail")


def test_failed_to_any_raises():
    """FAILED is terminal — no further transitions allowed."""
    sm = _make_sm(InstanceState.FAILED)
    for target in InstanceState:
        with pytest.raises(ValueError, match="terminal"):
            sm.transition(target, reason="should_fail")


# ---------------------------------------------------------------------------
# Structured log emission
# ---------------------------------------------------------------------------


def test_transition_emits_structured_log():
    """Every successful transition must emit a log entry with the correct fields."""
    sm = _make_sm(InstanceState.PROVISIONING)

    with capture_logs() as logs:
        sm.transition(InstanceState.BENCHMARKING, reason="test_reason")

    assert len(logs) == 1, f"Expected 1 log entry, got {len(logs)}: {logs}"
    entry = logs[0]

    assert entry.get("event") == "instance_state_transition"
    assert entry.get("instance_id") == "test-instance-001"
    assert entry.get("from_state") == InstanceState.PROVISIONING.value
    assert entry.get("to_state") == InstanceState.BENCHMARKING.value
    assert entry.get("reason") == "test_reason"


def test_multiple_transitions_each_emit_log():
    """Each individual transition emits exactly one log entry."""
    sm = _make_sm(InstanceState.PROVISIONING)

    with capture_logs() as logs:
        sm.transition(InstanceState.BENCHMARKING, reason="step1")
        sm.transition(InstanceState.SERVING, reason="step2")
        sm.transition(InstanceState.DRAINING, reason="step3")
        sm.transition(InstanceState.TERMINATED, reason="step4")

    assert len(logs) == 4
    events = [log["event"] for log in logs]
    assert all(e == "instance_state_transition" for e in events)


def test_failed_transition_does_not_emit_log():
    """A rejected transition must NOT emit a log entry."""
    sm = _make_sm(InstanceState.PROVISIONING)

    with capture_logs() as logs:
        with pytest.raises(ValueError):
            sm.transition(InstanceState.SERVING, reason="invalid")

    assert len(logs) == 0, f"Should emit no logs on rejected transition, got: {logs}"


# ---------------------------------------------------------------------------
# Record access
# ---------------------------------------------------------------------------


def test_record_property_returns_same_object():
    record = InstanceRecord(
        instance_id="abc",
        provider="runpod",
        gpu_model="A100",
    )
    sm = InstanceStateMachine(record)
    assert sm.record is record


def test_state_changes_reflected_in_record():
    sm = _make_sm(InstanceState.PROVISIONING)
    sm.transition(InstanceState.BENCHMARKING, reason="ok")
    assert sm.record.state == InstanceState.BENCHMARKING
