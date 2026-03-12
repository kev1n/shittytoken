from dataclasses import dataclass
from enum import Enum

import structlog

logger = structlog.get_logger()

# Approximate characters per token — used for synthetic text generation.
_CHARS_PER_TOKEN: int = 4


class WorkloadProfile(str, Enum):
    CODING = "coding"
    CHAT = "chat"
    DOCUMENT_QA = "document_qa"


@dataclass
class WorkloadSpec:
    profile: WorkloadProfile
    system_prompt_tokens: int
    query_tokens: int
    expected_output_tokens: int


WORKLOAD_SPECS: dict[WorkloadProfile, WorkloadSpec] = {
    WorkloadProfile.CODING: WorkloadSpec(WorkloadProfile.CODING, 2200, 350, 750),
    WorkloadProfile.CHAT: WorkloadSpec(WorkloadProfile.CHAT, 200, 60, 125),
    WorkloadProfile.DOCUMENT_QA: WorkloadSpec(
        WorkloadProfile.DOCUMENT_QA, 4500, 125, 150
    ),
}

# ---------------------------------------------------------------------------
# Source fragments — repeated to fill the target token count.
# Each fragment is intentionally verbose to keep the token estimate stable.
# ---------------------------------------------------------------------------

_CODING_SYSTEM_FRAGMENT: str = (
    "def process_batch(items: list[dict], max_workers: int = 4) -> list[dict]:\n"
    '    """Process a batch of items concurrently.\n\n'
    "    Args:\n"
    "        items: A list of dictionaries, each representing a work item\n"
    "            with keys 'id', 'payload', and 'priority'.\n"
    "        max_workers: Maximum number of concurrent worker threads.\n\n"
    "    Returns:\n"
    "        A list of result dictionaries with keys 'id', 'result',\n"
    "        'elapsed_ms', and 'error' (None on success).\n\n"
    "    Raises:\n"
    "        ValueError: If items is empty or max_workers < 1.\n"
    '    """\n'
)

_CODING_QUERY_FRAGMENT: str = (
    "Refactor the following function to use asyncio instead of threading, "
    "add type annotations, and improve the docstring with concrete examples. "
    "Ensure backward compatibility with the existing call signature. "
    "Explain each change in a short inline comment. "
)

_CHAT_SYSTEM_FRAGMENT: str = (
    "You are a helpful, concise assistant. "
    "Respond conversationally and keep answers brief. "
    "When you are unsure, say so rather than guessing. "
    "Always be polite and supportive. "
)

_CHAT_QUERY_FRAGMENT: str = (
    "Can you help me understand this better? "
    "I am trying to learn and would appreciate a clear explanation. "
    "Please keep it simple and easy to follow. "
)

_DOCUMENT_QA_SYSTEM_FRAGMENT: str = (
    "The following is an excerpt from a technical report on distributed systems "
    "performance. Section 3.2 discusses replication strategies:\n\n"
    "Synchronous replication guarantees that every write is acknowledged by all "
    "replicas before the client receives confirmation. This approach eliminates "
    "data loss on leader failure but increases write latency by the round-trip "
    "time to the slowest replica. In a geo-distributed deployment with replicas "
    "in three AWS regions (us-east-1, eu-west-1, ap-southeast-1), measured P99 "
    "write latency increased from 4 ms (single-region) to 187 ms (synchronous "
    "three-region replication) under a 10 KB median payload workload.\n\n"
    "Asynchronous replication keeps write latency low by acknowledging writes "
    "at the leader before propagating to followers. Under the same geo-distributed "
    "setup, P99 write latency remained at 5 ms. However, a leader failure before "
    "replication completes results in data loss bounded by the replication lag, "
    "which averaged 23 ms and peaked at 410 ms during cross-region congestion "
    "events observed over a 90-day measurement window.\n\n"
)

_DOCUMENT_QA_QUERY_FRAGMENT: str = (
    "Based on the passage above, what is the trade-off between synchronous and "
    "asynchronous replication in terms of latency and data durability? "
    "Cite specific numbers from the text. "
)


def _repeat_to_token_count(fragment: str, target_tokens: int) -> str:
    """Returns a string of approximately `target_tokens` tokens by repeating `fragment`."""
    target_chars = target_tokens * _CHARS_PER_TOKEN
    if len(fragment) == 0:
        return ""
    repeats = max(1, -(-target_chars // len(fragment)))  # ceiling division
    text = (fragment * repeats)[:target_chars]
    return text


def make_system_prompt(profile: WorkloadProfile, target_tokens: int) -> str:
    """Generates a synthetic system prompt of approximately `target_tokens` tokens."""
    fragments: dict[WorkloadProfile, str] = {
        WorkloadProfile.CODING: _CODING_SYSTEM_FRAGMENT,
        WorkloadProfile.CHAT: _CHAT_SYSTEM_FRAGMENT,
        WorkloadProfile.DOCUMENT_QA: _DOCUMENT_QA_SYSTEM_FRAGMENT,
    }
    fragment = fragments[profile]
    result = _repeat_to_token_count(fragment, target_tokens)
    logger.debug(
        "make_system_prompt",
        profile=profile,
        target_tokens=target_tokens,
        actual_chars=len(result),
    )
    return result


def make_query(profile: WorkloadProfile, target_tokens: int) -> str:
    """Generates a synthetic user query of approximately `target_tokens` tokens."""
    fragments: dict[WorkloadProfile, str] = {
        WorkloadProfile.CODING: _CODING_QUERY_FRAGMENT,
        WorkloadProfile.CHAT: _CHAT_QUERY_FRAGMENT,
        WorkloadProfile.DOCUMENT_QA: _DOCUMENT_QA_QUERY_FRAGMENT,
    }
    fragment = fragments[profile]
    result = _repeat_to_token_count(fragment, target_tokens)
    logger.debug(
        "make_query",
        profile=profile,
        target_tokens=target_tokens,
        actual_chars=len(result),
    )
    return result
