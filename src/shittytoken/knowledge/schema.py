"""
Knowledge graph node dataclasses.

All nodes are append-only; Configuration is immutable once written.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


@dataclass
class GPUModel:
    name: str
    vram_gb: float
    memory_bandwidth_gbs: float


@dataclass
class LLMModel:
    model_id: str
    params_b: float
    active_params_b: float
    quantization: Optional[str]  # "awq", "fp8", "fp16", or None
    dtype: str


@dataclass
class Configuration:
    tensor_parallel_size: int
    max_model_len: int
    gpu_memory_utilization: float
    quantization: Optional[str]       # "awq", "fp8", "fp16", or None
    kv_cache_dtype: str               # "auto" or "fp8"
    max_num_seqs: int
    enable_prefix_caching: bool
    enforce_eager: bool
    config_id: str = field(default_factory=_uuid)
    created_at: datetime = field(default_factory=_now)

    def __post_init__(self) -> None:
        if self.gpu_memory_utilization >= 1.0:
            raise ValueError(
                f"gpu_memory_utilization must be < 1.0, got {self.gpu_memory_utilization}"
            )
        if not self.enable_prefix_caching:
            raise ValueError(
                "enable_prefix_caching must be True — prefix caching is always required"
            )


@dataclass
class BenchmarkResult:
    config_id: str
    gpu_model_name: str
    verdict: str                         # "pass" | "fail"
    cold_ttft_p95_s: float
    warm_ttft_p95_s_at_c1: float
    peak_throughput_tps: float
    prefix_cache_hit_rate_phase3: float
    failed_request_rate: float
    deltanet_cache_suspect: bool
    started_at: datetime
    completed_at: datetime
    result_id: str = field(default_factory=_uuid)


@dataclass
class OOMEvent:
    config_id: str
    gpu_model_name: str
    error_type: str         # "loading" | "runtime"
    error_message: str      # raw text
    error_phase: str        # "loading" | "runtime"
    gpu_memory_free_gb: float
    gpu_memory_total_gb: float
    occurred_at: datetime = field(default_factory=_now)
    event_id: str = field(default_factory=_uuid)
    succeeded: Optional[bool] = None   # None until resolved
