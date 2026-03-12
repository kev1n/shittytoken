"""
Integration tests for the full run_benchmark() pipeline against the mock server.

Each test starts a real aiohttp TestServer backed by the mock_openai_server app
and runs the full benchmark pipeline against it with a short level_duration_sec
so the suite completes in a reasonable time (~14-20 seconds total for phase 3).
"""
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer, TestClient

import tests.benchmark.mock_openai_server as mock_server
from shittytoken.benchmark.runner import run_benchmark
from shittytoken.benchmark.schema import BenchmarkVerdict, FailReason


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
async def mock_app_server():
    """
    Yields a running TestServer backed by the mock OpenAI app.
    Resets all global counters before and after each test.

    Uses minimal token output and fast token rate to keep tests quick even at
    high concurrency levels (e.g. concurrency=64 with level_duration_sec=2.0).
    """
    mock_server.reset_counters()
    # Reset configurable globals to safe defaults.
    mock_server.MOCK_TTFT_MS = 10        # 10ms TTFT — fast first token
    mock_server.MOCK_TOKEN_RATE_PER_SEC = 2000  # fast streaming
    mock_server.MOCK_CACHE_HIT_RATE = 0.65
    mock_server.MOCK_FAIL_RATE = 0.0  # default: no failures
    mock_server.MOCK_NUM_OUTPUT_TOKENS = 5  # minimal output tokens

    app = mock_server.create_app()
    server = TestServer(app)
    await server.start_server()

    yield server

    await server.close()
    mock_server.reset_counters()


def _base_url(server: TestServer) -> str:
    return str(server.make_url("/")).rstrip("/")


# ---------------------------------------------------------------------------
# Test 1: Full pipeline passes against default mock server
# ---------------------------------------------------------------------------


async def test_full_pipeline_passes(mock_app_server):
    """
    Full benchmark against default mock server (65% cache hits, 0% fail rate)
    should return verdict=PASS.
    """
    url = _base_url(mock_app_server)
    result = await run_benchmark(
        worker_url=url,
        model_id="mock-model",
        gpu_model="mock-gpu",
        raw_config={},
        level_duration_sec=2.0,
    )
    assert result.verdict == BenchmarkVerdict.PASS, (
        f"Expected PASS but got FAIL. fail_reasons={result.fail_reasons}"
    )
    assert result.fail_reasons == []
    # Sanity checks on populated fields.
    assert len(result.phase_metrics) == 3
    assert len(result.concurrency_sweep) == 7


# ---------------------------------------------------------------------------
# Test 2: deltanet_cache_suspect is set when cache hit rate is near zero
# ---------------------------------------------------------------------------


async def test_deltanet_cache_suspect_flag(mock_app_server, monkeypatch):
    """
    Patching MOCK_CACHE_HIT_RATE to 0.0 means the /metrics endpoint will report
    0 cache hits, so compute_delta_hit_rate returns 0.0, triggering the flag.
    """
    monkeypatch.setattr(mock_server, "MOCK_CACHE_HIT_RATE", 0.0)

    url = _base_url(mock_app_server)
    result = await run_benchmark(
        worker_url=url,
        model_id="mock-model",
        gpu_model="mock-gpu",
        raw_config={},
        level_duration_sec=2.0,
    )
    assert result.deltanet_cache_suspect is True


# ---------------------------------------------------------------------------
# Test 3: Failed requests counted → verdict=FAIL with HIGH_ERROR_RATE
# ---------------------------------------------------------------------------


async def test_all_requests_fail_high_error_rate(mock_app_server, monkeypatch):
    """
    Setting MOCK_FAIL_RATE=1.0 makes all requests return HTTP 500.
    Expect verdict=FAIL and HIGH_ERROR_RATE in fail_reasons.
    """
    monkeypatch.setattr(mock_server, "MOCK_FAIL_RATE", 1.0)

    url = _base_url(mock_app_server)
    result = await run_benchmark(
        worker_url=url,
        model_id="mock-model",
        gpu_model="mock-gpu",
        raw_config={},
        level_duration_sec=2.0,
    )
    assert result.verdict == BenchmarkVerdict.FAIL
    assert FailReason.HIGH_ERROR_RATE in result.fail_reasons


# ---------------------------------------------------------------------------
# Test 4: run_benchmark() never raises — returns BenchmarkResult(verdict=FAIL)
# ---------------------------------------------------------------------------


async def test_run_benchmark_never_raises_on_server_error(mock_app_server, monkeypatch):
    """
    Point run_benchmark at a non-existent URL so every connection fails.
    Must return BenchmarkResult(verdict=FAIL) without raising.
    """
    # Use a port that nothing is listening on.
    bad_url = "http://127.0.0.1:19999"

    result = await run_benchmark(
        worker_url=bad_url,
        model_id="mock-model",
        gpu_model="mock-gpu",
        raw_config={},
        level_duration_sec=2.0,
    )
    # Must not raise — just return a FAIL result.
    assert result is not None
    assert result.verdict == BenchmarkVerdict.FAIL
    # started_at and completed_at must be populated.
    assert result.started_at > 0
    assert result.completed_at >= result.started_at


# ---------------------------------------------------------------------------
# Test 5: Benchmark result fields are well-formed
# ---------------------------------------------------------------------------


async def test_result_fields_populated(mock_app_server):
    """
    Verify that all BenchmarkResult fields are populated with reasonable values
    on a successful run.
    """
    url = _base_url(mock_app_server)
    result = await run_benchmark(
        worker_url=url,
        model_id="mock-model",
        gpu_model="mock-gpu",
        raw_config={"test": True},
        level_duration_sec=2.0,
    )
    assert result.worker_url == url
    assert result.model_id == "mock-model"
    assert result.gpu_model == "mock-gpu"
    assert result.raw_config == {"test": True}
    assert result.cold_cache_baseline_ttft_p95 >= 0
    assert result.peak_throughput_tokens_per_sec >= 0
    assert result.warm_cache_ttft_p95_at_concurrency_1 >= 0
    # phase_metrics should have exactly 3 phases.
    assert len(result.phase_metrics) == 3
    assert result.phase_metrics[0].phase_number == 1
    assert result.phase_metrics[1].phase_number == 2
    assert result.phase_metrics[2].phase_number == 3
    # concurrency_sweep should have 7 levels.
    assert len(result.concurrency_sweep) == 7
    sweep_concurrencies = [p.concurrency for p in result.concurrency_sweep]
    assert sweep_concurrencies == [1, 2, 4, 8, 16, 32, 64]
