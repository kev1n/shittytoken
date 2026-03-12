"""
InstanceStateMachine — tracks lifecycle state for a single provisioned GPU instance.

Valid state graph:
  PROVISIONING → BENCHMARKING | FAILED
  BENCHMARKING → SERVING | FAILED
  SERVING      → DRAINING | FAILED
  DRAINING     → TERMINATED | FAILED
  TERMINATED   → (terminal)
  FAILED       → (terminal)

Every transition emits a structured JSON log entry.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import structlog

logger = structlog.get_logger()


class InstanceState(str, Enum):
    PROVISIONING = "provisioning"
    BENCHMARKING = "benchmarking"
    SERVING = "serving"
    DRAINING = "draining"
    TERMINATED = "terminated"
    FAILED = "failed"


VALID_TRANSITIONS: dict[InstanceState, set[InstanceState]] = {
    InstanceState.PROVISIONING: {InstanceState.BENCHMARKING, InstanceState.FAILED},
    InstanceState.BENCHMARKING: {InstanceState.SERVING, InstanceState.FAILED},
    InstanceState.SERVING:      {InstanceState.DRAINING, InstanceState.FAILED},
    InstanceState.DRAINING:     {InstanceState.TERMINATED, InstanceState.FAILED},
    InstanceState.TERMINATED:   set(),
    InstanceState.FAILED:       set(),
}


@dataclass
class InstanceRecord:
    """Mutable runtime record for one provisioned GPU instance."""

    instance_id: str
    provider: str          # "vastai" | "runpod"
    gpu_model: str
    ssh_host: str = ""
    ssh_port: int = 22
    worker_url: str = ""
    state: InstanceState = InstanceState.PROVISIONING
    config_id: str = ""    # KG Configuration.config_id
    created_at: float = field(default_factory=time.time)
    state_changed_at: float = field(default_factory=time.time)


class InstanceStateMachine:
    """
    Manages state transitions for one InstanceRecord.

    transition() raises ValueError for invalid transitions.
    Every transition logs:
        {"event": "instance_state_transition",
         "instance_id": ..., "from": ..., "to": ..., "reason": ...}
    """

    def __init__(self, record: InstanceRecord) -> None:
        self._record = record

    def transition(self, to: InstanceState, reason: str) -> None:
        """
        Move the instance to state *to*.

        Raises ValueError if the transition is not in VALID_TRANSITIONS.
        """
        from_state = self._record.state
        allowed = VALID_TRANSITIONS[from_state]

        if to not in allowed:
            raise ValueError(
                f"Invalid state transition for instance {self._record.instance_id}: "
                f"{from_state.value} → {to.value}. "
                f"Allowed targets from {from_state.value}: "
                f"{[s.value for s in allowed] if allowed else '(none — terminal state)'}"
            )

        self._record.state = to
        self._record.state_changed_at = time.time()

        logger.info(
            "instance_state_transition",
            instance_id=self._record.instance_id,
            from_state=from_state.value,
            to_state=to.value,
            reason=reason,
        )

    @property
    def state(self) -> InstanceState:
        return self._record.state

    @property
    def record(self) -> InstanceRecord:
        return self._record
