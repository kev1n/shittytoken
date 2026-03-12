"""
Tests for src/shittytoken/benchmark/request_generator.py.

All tests are pure unit tests — no I/O.
"""

import pytest

from shittytoken.benchmark.constants import RETURNING_USER_FRACTION
from shittytoken.benchmark.request_generator import VirtualUserPool
from shittytoken.benchmark.workloads import WorkloadProfile


# ---------------------------------------------------------------------------
# Returning-user fraction
# ---------------------------------------------------------------------------


def test_returning_user_fraction_within_tolerance():
    """
    After 1000 next_request() calls the returning fraction must be within
    ±5 percentage points of RETURNING_USER_FRACTION (0.80).
    """
    pool = VirtualUserPool(max_sessions=200)

    # Warm the pool with at least a few sessions so returning-user logic
    # has sessions to return to from the start.
    for _ in range(20):
        req = pool.next_request()
        pool.record_reply(req.session_id, "warmup reply")

    total = 1000
    returning = 0
    for _ in range(total):
        req = pool.next_request()
        if not req.is_new_session:
            returning += 1
        pool.record_reply(req.session_id, "reply text")

    fraction = returning / total
    tolerance = 0.05
    assert abs(fraction - RETURNING_USER_FRACTION) <= tolerance, (
        f"Returning fraction {fraction:.3f} is more than {tolerance} "
        f"away from expected {RETURNING_USER_FRACTION}"
    )


# ---------------------------------------------------------------------------
# record_reply appends assistant message
# ---------------------------------------------------------------------------


def test_record_reply_appends_assistant_message():
    """record_reply causes the session's messages to include the assistant turn."""
    pool = VirtualUserPool(max_sessions=50)
    req = pool.next_request()
    session_id = req.session_id

    pool.record_reply(session_id, "hello")

    # Get a new request from the same session to inspect its messages.
    # We need to force a returning request by replacing the pool's sessions
    # to only contain our target session and ensure returning path.
    # Simpler: access internal state for assertion.
    session = pool._sessions[session_id]  # type: ignore[attr-defined]
    assert {"role": "assistant", "content": "hello"} in session.messages


def test_record_reply_unknown_session_is_noop():
    """record_reply on an unknown session_id does not raise."""
    pool = VirtualUserPool(max_sessions=50)
    pool.record_reply("non-existent-session-id", "hello")  # must not raise


# ---------------------------------------------------------------------------
# Session distribution across profiles
# ---------------------------------------------------------------------------


def test_sessions_distributed_across_all_profiles():
    """
    After generating enough new sessions, all three WorkloadProfile values
    appear at least once.
    """
    pool = VirtualUserPool(max_sessions=300)
    profiles_seen: set[WorkloadProfile] = set()

    # Generate enough new sessions to see all 3 profiles with high probability.
    # With equal probability (1/3 each) and 50 tries, P(missing any) < 0.001.
    attempts = 0
    while len(profiles_seen) < len(WorkloadProfile) and attempts < 500:
        req = pool.next_request()
        if req.is_new_session:
            profiles_seen.add(req.profile)
        attempts += 1

    assert profiles_seen == set(WorkloadProfile), (
        f"Not all profiles seen after {attempts} attempts. "
        f"Seen: {profiles_seen}"
    )


# ---------------------------------------------------------------------------
# messages is a copy, not a reference
# ---------------------------------------------------------------------------


def test_generated_request_messages_is_copy():
    """
    Mutating GeneratedRequest.messages must not affect the session's internal
    message list.
    """
    pool = VirtualUserPool(max_sessions=50)
    req = pool.next_request()

    # Mutate the copy returned in the GeneratedRequest.
    original_len = len(req.messages)
    req.messages.append({"role": "user", "content": "injected"})

    # The session's internal messages must be unaffected.
    session = pool._sessions.get(req.session_id)  # type: ignore[attr-defined]
    if session is not None:
        assert len(session.messages) == original_len, (
            "Mutating GeneratedRequest.messages affected the internal session messages."
        )
