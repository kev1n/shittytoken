"""
Tests for build_vllm_command() in agent/provisioner.py.

Covers all architecture invariants and flag-mapping rules.
"""

from __future__ import annotations

import pytest

from shittytoken.agent.provisioner import build_vllm_command
from shittytoken.knowledge.schema import Configuration


def _make_config(**overrides) -> Configuration:
    """
    Return a valid Configuration with sensible defaults.
    Override any field by passing it as a keyword argument.
    """
    defaults = dict(
        tensor_parallel_size=2,
        max_model_len=32768,
        gpu_memory_utilization=0.90,
        quantization=None,
        kv_cache_dtype="auto",
        max_num_seqs=256,
        enable_prefix_caching=True,
        enforce_eager=False,
    )
    defaults.update(overrides)
    return Configuration(**defaults)


MODEL_ID = "Qwen/Qwen3.5-35B-A3B"


# ---------------------------------------------------------------------------
# Always-present flags
# ---------------------------------------------------------------------------


def test_includes_enable_prefix_caching():
    cmd = build_vllm_command(_make_config(), MODEL_ID)
    assert "--enable-prefix-caching" in cmd


def test_includes_ipc_host():
    cmd = build_vllm_command(_make_config(), MODEL_ID)
    assert "--ipc=host" in cmd


# ---------------------------------------------------------------------------
# Optional flags — present when non-default
# ---------------------------------------------------------------------------


def test_quantization_flag_included_when_set():
    cmd = build_vllm_command(_make_config(quantization="awq"), MODEL_ID)
    assert "--quantization awq" in cmd


def test_quantization_flag_absent_when_none():
    cmd = build_vllm_command(_make_config(quantization=None), MODEL_ID)
    assert "--quantization" not in cmd


def test_kv_cache_dtype_flag_included_when_not_auto():
    cmd = build_vllm_command(_make_config(kv_cache_dtype="fp8"), MODEL_ID)
    assert "--kv-cache-dtype fp8" in cmd


def test_kv_cache_dtype_flag_absent_when_auto():
    cmd = build_vllm_command(_make_config(kv_cache_dtype="auto"), MODEL_ID)
    assert "--kv-cache-dtype" not in cmd


def test_enforce_eager_flag_included_when_true():
    cmd = build_vllm_command(_make_config(enforce_eager=True), MODEL_ID)
    assert "--enforce-eager" in cmd


def test_enforce_eager_flag_absent_when_false():
    cmd = build_vllm_command(_make_config(enforce_eager=False), MODEL_ID)
    assert "--enforce-eager" not in cmd


# ---------------------------------------------------------------------------
# Required flags with values
# ---------------------------------------------------------------------------


def test_tensor_parallel_size_flag():
    cmd = build_vllm_command(_make_config(tensor_parallel_size=4), MODEL_ID)
    assert "--tensor-parallel-size 4" in cmd


def test_max_model_len_flag():
    cmd = build_vllm_command(_make_config(max_model_len=16384), MODEL_ID)
    assert "--max-model-len 16384" in cmd


def test_gpu_memory_utilization_flag():
    cmd = build_vllm_command(_make_config(gpu_memory_utilization=0.85), MODEL_ID)
    assert "--gpu-memory-utilization 0.85" in cmd


def test_max_num_seqs_flag():
    cmd = build_vllm_command(_make_config(max_num_seqs=512), MODEL_ID)
    assert "--max-num-seqs 512" in cmd


# ---------------------------------------------------------------------------
# model_id positioning
# ---------------------------------------------------------------------------


def test_model_id_is_first_positional_after_serve():
    """
    The model_id must appear immediately after "vllm serve" as the first
    positional argument.
    """
    cmd = build_vllm_command(_make_config(), MODEL_ID)
    tokens = cmd.split()
    assert tokens[0] == "vllm"
    assert tokens[1] == "serve"
    assert tokens[2] == MODEL_ID, (
        f"Expected model_id '{MODEL_ID}' as the third token, got '{tokens[2]}'"
    )


def test_model_id_present_in_output():
    cmd = build_vllm_command(_make_config(), MODEL_ID)
    assert MODEL_ID in cmd


# ---------------------------------------------------------------------------
# Invariant violations — CRITICAL
# ---------------------------------------------------------------------------


def test_gpu_memory_utilization_exactly_1_raises():
    """
    CRITICAL invariant: gpu_memory_utilization must be < 1.0.
    Configuration.__post_init__ already guards this, so we test that the
    Configuration constructor raises ValueError (not just build_vllm_command).
    """
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        _make_config(gpu_memory_utilization=1.0)


def test_gpu_memory_utilization_above_1_raises():
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        _make_config(gpu_memory_utilization=1.1)


def test_build_vllm_command_raises_on_utilization_1():
    """
    Even if somehow a Configuration with utilization=1.0 is bypassed,
    build_vllm_command must independently raise ValueError.

    We bypass Configuration.__post_init__ to test the guard inside
    build_vllm_command itself.
    """
    import datetime

    # Construct a Configuration without going through __post_init__
    config = Configuration.__new__(Configuration)
    config.tensor_parallel_size = 2
    config.max_model_len = 32768
    config.gpu_memory_utilization = 1.0   # violates invariant
    config.quantization = None
    config.kv_cache_dtype = "auto"
    config.max_num_seqs = 256
    config.enable_prefix_caching = True
    config.enforce_eager = False
    config.config_id = "bypass-test"
    config.created_at = datetime.datetime.now(tz=datetime.timezone.utc)

    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        build_vllm_command(config, MODEL_ID)


def test_build_vllm_command_raises_when_prefix_caching_false():
    """
    Even if a Configuration has enable_prefix_caching=False,
    build_vllm_command must raise ValueError.
    """
    import datetime

    config = Configuration.__new__(Configuration)
    config.tensor_parallel_size = 2
    config.max_model_len = 32768
    config.gpu_memory_utilization = 0.90
    config.quantization = None
    config.kv_cache_dtype = "auto"
    config.max_num_seqs = 256
    config.enable_prefix_caching = False   # violates invariant
    config.enforce_eager = False
    config.config_id = "bypass-prefix-test"
    config.created_at = datetime.datetime.now(tz=datetime.timezone.utc)

    with pytest.raises(ValueError, match="enable_prefix_caching"):
        build_vllm_command(config, MODEL_ID)


# ---------------------------------------------------------------------------
# Output is a single string (not a list)
# ---------------------------------------------------------------------------


def test_output_is_string():
    result = build_vllm_command(_make_config(), MODEL_ID)
    assert isinstance(result, str)


def test_command_starts_with_vllm_serve():
    cmd = build_vllm_command(_make_config(), MODEL_ID)
    assert cmd.startswith("vllm serve "), f"Command does not start with 'vllm serve ': {cmd}"
