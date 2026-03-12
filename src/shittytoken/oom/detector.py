"""
OOM classifier: distinguishes loading-phase OOM from runtime KV-cache OOM.

Loading OOM  — model weights don't fit in VRAM.
Runtime OOM  — KV cache is exhausted during inference.

Fix strategies are completely different; never confuse them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class OOMType(str, Enum):
    LOADING = "loading"   # weights don't fit
    RUNTIME = "runtime"   # KV cache exhausted during inference


# Patterns that indicate weight loading phase
LOADING_PATTERNS = [
    re.compile(r"Loading model weights"),
    re.compile(r"Initializing the engine"),
    re.compile(r"model\.safetensors"),
    re.compile(r"loading checkpoint"),
]

# Patterns that indicate runtime inference phase
RUNTIME_PATTERNS = [
    re.compile(r"num_requests_running"),
    re.compile(r"KV cache"),
    re.compile(r"expand_buffer"),
    re.compile(r"swap.*blocks"),
]


@dataclass
class OOMClassification:
    oom_type: OOMType
    confidence: str        # "high" | "medium" | "low"
    matched_pattern: str   # which pattern triggered the classification
    raw_error: str


def classify_oom(error_message: str, log_context: str = "") -> OOMClassification:
    """
    Classifies an OOM error as LOADING or RUNTIME based on:
    1. Patterns in the error_message itself
    2. Patterns in the surrounding log_context (lines before the OOM)

    Default: if ambiguous, classify as LOADING (safer — avoids wrong fixes).

    The error_message is the torch.cuda.OutOfMemoryError line.
    The log_context is the last N lines of container logs before the error.
    """
    combined = error_message + "\n" + log_context

    # Search error_message first (higher confidence)
    for pattern in RUNTIME_PATTERNS:
        m = pattern.search(error_message)
        if m:
            return OOMClassification(
                oom_type=OOMType.RUNTIME,
                confidence="high",
                matched_pattern=pattern.pattern,
                raw_error=error_message,
            )

    for pattern in LOADING_PATTERNS:
        m = pattern.search(error_message)
        if m:
            return OOMClassification(
                oom_type=OOMType.LOADING,
                confidence="high",
                matched_pattern=pattern.pattern,
                raw_error=error_message,
            )

    # Fall back to log_context (medium confidence)
    for pattern in RUNTIME_PATTERNS:
        m = pattern.search(log_context)
        if m:
            return OOMClassification(
                oom_type=OOMType.RUNTIME,
                confidence="medium",
                matched_pattern=pattern.pattern,
                raw_error=error_message,
            )

    for pattern in LOADING_PATTERNS:
        m = pattern.search(log_context)
        if m:
            return OOMClassification(
                oom_type=OOMType.LOADING,
                confidence="medium",
                matched_pattern=pattern.pattern,
                raw_error=error_message,
            )

    # Ambiguous — default to LOADING (safer: avoids wrong KV-cache fix)
    return OOMClassification(
        oom_type=OOMType.LOADING,
        confidence="low",
        matched_pattern="<default>",
        raw_error=error_message,
    )
