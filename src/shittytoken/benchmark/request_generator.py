import random
import uuid
from dataclasses import dataclass, field

import structlog

from .constants import MAX_SESSIONS, RETURNING_USER_FRACTION
from .workloads import (
    WORKLOAD_SPECS,
    WorkloadProfile,
    make_query,
    make_system_prompt,
)

logger = structlog.get_logger()

# Distribute sessions across all three profiles with equal probability.
_ALL_PROFILES: list[WorkloadProfile] = list(WorkloadProfile)


@dataclass
class VirtualSession:
    session_id: str
    profile: WorkloadProfile
    messages: list[dict]
    turn_count: int = 0


@dataclass
class GeneratedRequest:
    session_id: str
    profile: WorkloadProfile
    messages: list[dict]  # snapshot at generation time (copy, not reference)
    is_new_session: bool


class VirtualUserPool:
    """
    Simulates a realistic mix of returning and new users to drive prefix cache hits.

    80% chance next_request() extends an existing session (genuine prefix cache hit).
    20% chance next_request() starts a new session.
    Sessions are distributed across 3 workload profiles with equal probability.
    record_reply() appends real assistant text to the session so subsequent requests
    share the actual prefix (not synthetic), generating real cache hits.
    """

    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._max_sessions = max_sessions
        self._sessions: dict[str, VirtualSession] = {}

    def next_request(self) -> GeneratedRequest:
        """Return the next request to send, creating a new session or extending an existing one."""
        use_existing = (
            bool(self._sessions)
            and random.random() < RETURNING_USER_FRACTION
        )

        if use_existing:
            session = random.choice(list(self._sessions.values()))
            spec = WORKLOAD_SPECS[session.profile]
            query_text = make_query(session.profile, spec.query_tokens)
            # Extend session with a new user turn.
            session.messages.append({"role": "user", "content": query_text})
            session.turn_count += 1
            logger.debug(
                "request_generator.returning_user",
                session_id=session.session_id,
                turn=session.turn_count,
            )
            return GeneratedRequest(
                session_id=session.session_id,
                profile=session.profile,
                messages=list(session.messages),  # copy, not reference
                is_new_session=False,
            )
        else:
            return self._create_new_session()

    def record_reply(self, session_id: str, reply: str) -> None:
        """
        Append the model's reply to the session so subsequent turns share the real prefix.
        No-op if session_id is unknown (session may have been evicted).
        """
        session = self._sessions.get(session_id)
        if session is None:
            logger.debug(
                "request_generator.record_reply_unknown_session",
                session_id=session_id,
            )
            return
        session.messages.append({"role": "assistant", "content": reply})

    def session_count(self) -> int:
        return len(self._sessions)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_new_session(self) -> GeneratedRequest:
        """Create a brand-new session, evicting the oldest if at capacity."""
        profile = random.choice(_ALL_PROFILES)
        spec = WORKLOAD_SPECS[profile]
        session_id = str(uuid.uuid4())

        system_text = make_system_prompt(profile, spec.system_prompt_tokens)
        query_text = make_query(profile, spec.query_tokens)

        messages: list[dict] = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": query_text},
        ]

        session = VirtualSession(
            session_id=session_id,
            profile=profile,
            messages=messages,
            turn_count=1,
        )

        # Evict a random session if at capacity to bound memory usage.
        if len(self._sessions) >= self._max_sessions:
            evict_id = random.choice(list(self._sessions.keys()))
            del self._sessions[evict_id]
            logger.debug(
                "request_generator.session_evicted",
                evicted_session_id=evict_id,
                new_session_id=session_id,
            )

        self._sessions[session_id] = session

        logger.debug(
            "request_generator.new_session",
            session_id=session_id,
            profile=profile,
        )
        return GeneratedRequest(
            session_id=session_id,
            profile=profile,
            messages=list(messages),  # copy, not reference
            is_new_session=True,
        )
