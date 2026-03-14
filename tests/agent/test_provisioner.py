"""
Tests for the provisioner module.

Covers:
- build_vllm_command() invariants and flag mapping
- VastAIProvider.find_offers() with live API (requires VASTAI_API_KEY)
- GPUOffer scoring and filtering
- DeploymentPlan construction
- Query string building
- Client-side GPU name filtering
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shittytoken.agent.provisioner import (
    GPUOffer,
    GPUProvider,
    VastAIProvider,
    DeploymentPlan,
    ProvisionedInstance,
    build_deployment_plan,
    build_vllm_command,
)
from shittytoken.config import cfg, preferred_gpus
from shittytoken.knowledge.schema import Configuration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> Configuration:
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


MODEL_ID = "cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit"

# Check if VASTAI_API_KEY is set for live tests
_has_vastai_key = bool(os.environ.get("VASTAI_API_KEY") or "")

# Try loading from .env if not in environment
if not _has_vastai_key:
    try:
        from shittytoken.config import Settings
        _s = Settings()
        _has_vastai_key = bool(_s.vastai_api_key)
    except Exception:
        pass

requires_vastai = pytest.mark.skipif(
    not _has_vastai_key,
    reason="VASTAI_API_KEY not set — skipping live API tests",
)


def _make_offer(**overrides) -> GPUOffer:
    defaults = dict(
        offer_id="12345",
        provider="vastai",
        gpu_name="RTX 3090",
        num_gpus=2,
        cost_per_hour_usd=0.50,
        reliability=0.995,
        inet_up_mbps=300.0,
        inet_down_mbps=500.0,
        inet_up_cost_per_gb=0.02,
        inet_down_cost_per_gb=0.02,
        pcie_bw_gbps=10.0,
        bw_nvlink_gbps=0.0,
        dlperf=15.0,
    )
    defaults.update(overrides)
    return GPUOffer(**defaults)


# ===================================================================
# build_vllm_command — invariants and flag mapping
# ===================================================================


class TestBuildVllmCommand:
    """build_vllm_command() produces correct vllm serve strings."""

    def test_always_has_prefix_caching(self):
        cmd = build_vllm_command(_make_config(), MODEL_ID)
        assert "--enable-prefix-caching" in cmd

    def test_has_host_and_port(self):
        cmd = build_vllm_command(_make_config(), MODEL_ID)
        assert "--host" in cmd
        assert "--port" in cmd
        assert "0.0.0.0" in cmd
        assert "8080" in cmd

    def test_model_id_is_first_positional(self):
        cmd = build_vllm_command(_make_config(), MODEL_ID)
        tokens = cmd.split()
        assert tokens[:3] == ["vllm", "serve", MODEL_ID]

    def test_quantization_awq(self):
        cmd = build_vllm_command(_make_config(quantization="awq"), MODEL_ID)
        assert "--quantization awq" in cmd

    def test_quantization_none_omitted(self):
        cmd = build_vllm_command(_make_config(quantization=None), MODEL_ID)
        assert "--quantization" not in cmd

    def test_kv_cache_dtype_fp8(self):
        cmd = build_vllm_command(_make_config(kv_cache_dtype="fp8"), MODEL_ID)
        assert "--kv-cache-dtype fp8" in cmd

    def test_kv_cache_dtype_auto_omitted(self):
        cmd = build_vllm_command(_make_config(kv_cache_dtype="auto"), MODEL_ID)
        assert "--kv-cache-dtype" not in cmd

    def test_enforce_eager_true(self):
        cmd = build_vllm_command(_make_config(enforce_eager=True), MODEL_ID)
        assert "--enforce-eager" in cmd

    def test_enforce_eager_false_omitted(self):
        cmd = build_vllm_command(_make_config(enforce_eager=False), MODEL_ID)
        assert "--enforce-eager" not in cmd

    def test_all_numeric_params(self):
        cmd = build_vllm_command(
            _make_config(
                tensor_parallel_size=4,
                max_model_len=16384,
                gpu_memory_utilization=0.85,
                max_num_seqs=128,
            ),
            MODEL_ID,
        )
        assert "--tensor-parallel-size 4" in cmd
        assert "--max-model-len 16384" in cmd
        assert "--gpu-memory-utilization 0.85" in cmd
        assert "--max-num-seqs 128" in cmd

    def test_lmcache_kv_transfer_config_present(self):
        """When lmcache is enabled in config, --kv-transfer-config must appear."""
        cmd = build_vllm_command(_make_config(), MODEL_ID)
        lmcache_cfg = cfg.get("vllm", {}).get("lmcache", {})
        if lmcache_cfg.get("enabled", False):
            assert "--kv-transfer-config" in cmd
            assert "LMCacheConnectorV1" in cmd
        else:
            assert "--kv-transfer-config" not in cmd

    def test_output_is_string_starting_with_vllm_serve(self):
        cmd = build_vllm_command(_make_config(), MODEL_ID)
        assert isinstance(cmd, str)
        assert cmd.startswith("vllm serve ")


class TestBuildVllmCommandInvariants:
    """Critical invariant violations must raise ValueError."""

    def test_utilization_1_raises_at_config_level(self):
        with pytest.raises(ValueError, match="gpu_memory_utilization"):
            _make_config(gpu_memory_utilization=1.0)

    def test_utilization_above_1_raises(self):
        with pytest.raises(ValueError, match="gpu_memory_utilization"):
            _make_config(gpu_memory_utilization=1.1)

    def test_utilization_1_raises_in_build_command(self):
        """Guard inside build_vllm_command catches bypassed config."""
        import datetime

        config = Configuration.__new__(Configuration)
        config.tensor_parallel_size = 2
        config.max_model_len = 32768
        config.gpu_memory_utilization = 1.0
        config.quantization = None
        config.kv_cache_dtype = "auto"
        config.max_num_seqs = 256
        config.enable_prefix_caching = True
        config.enforce_eager = False
        config.config_id = "bypass-test"
        config.created_at = datetime.datetime.now(tz=datetime.timezone.utc)

        with pytest.raises(ValueError, match="gpu_memory_utilization"):
            build_vllm_command(config, MODEL_ID)

    def test_prefix_caching_false_raises(self):
        import datetime

        config = Configuration.__new__(Configuration)
        config.tensor_parallel_size = 2
        config.max_model_len = 32768
        config.gpu_memory_utilization = 0.90
        config.quantization = None
        config.kv_cache_dtype = "auto"
        config.max_num_seqs = 256
        config.enable_prefix_caching = False
        config.enforce_eager = False
        config.config_id = "bypass-prefix-test"
        config.created_at = datetime.datetime.now(tz=datetime.timezone.utc)

        with pytest.raises(ValueError, match="enable_prefix_caching"):
            build_vllm_command(config, MODEL_ID)


# ===================================================================
# GPUOffer — scoring and data integrity
# ===================================================================


class TestGPUOfferScore:
    """Composite scoring must rank offers correctly."""

    def test_cheaper_offer_scores_lower(self):
        cheap = _make_offer(cost_per_hour_usd=0.30)
        expensive = _make_offer(cost_per_hour_usd=1.20)
        assert cheap.score < expensive.score

    def test_nvlink_bonus_applied_for_multi_gpu(self):
        with_nvlink = _make_offer(num_gpus=2, bw_nvlink_gbps=100.0)
        without_nvlink = _make_offer(num_gpus=2, bw_nvlink_gbps=0.0)
        assert with_nvlink.score < without_nvlink.score

    def test_nvlink_bonus_not_applied_for_single_gpu(self):
        with_nvlink = _make_offer(num_gpus=1, bw_nvlink_gbps=100.0)
        without_nvlink = _make_offer(num_gpus=1, bw_nvlink_gbps=0.0)
        assert with_nvlink.score == without_nvlink.score

    def test_slow_upload_penalized(self):
        fast = _make_offer(inet_up_mbps=500.0)
        slow = _make_offer(inet_up_mbps=50.0)
        assert fast.score < slow.score

    def test_none_cost_gives_infinite_score(self):
        offer = _make_offer(cost_per_hour_usd=None)
        assert offer.score == float("inf")

    def test_score_is_deterministic(self):
        offer = _make_offer()
        assert offer.score == offer.score


class TestGPUOfferFromRaw:
    """_to_gpu_offer must correctly map Vast.ai SDK fields."""

    def test_maps_all_fields(self):
        raw = {
            "id": 99999,
            "gpu_name": "RTX 3090",
            "num_gpus": 2,
            "dph_total": 0.42,
            "reliability2": 0.997,
            "inet_up": 300.5,
            "inet_down": 800.0,
            "inet_up_cost": 0.01,
            "inet_down_cost": 0.02,
            "pcie_bw": 12.5,
            "bw_nvlink": 0.0,
            "dlperf": 18.3,
        }
        offer = VastAIProvider._to_gpu_offer(raw)

        assert offer.offer_id == "99999"
        assert offer.provider == "vastai"
        assert offer.gpu_name == "RTX 3090"
        assert offer.num_gpus == 2
        assert offer.cost_per_hour_usd == 0.42
        assert offer.reliability == 0.997
        assert offer.inet_up_mbps == 300.5
        assert offer.inet_down_mbps == 800.0
        assert offer.pcie_bw_gbps == 12.5
        assert offer.bw_nvlink_gbps == 0.0
        assert offer.dlperf == 18.3
        assert offer.raw is raw

    def test_missing_fields_default_to_none(self):
        raw = {"id": 1, "gpu_name": "RTX 4090"}
        offer = VastAIProvider._to_gpu_offer(raw)
        assert offer.num_gpus == 1
        assert offer.cost_per_hour_usd is None
        assert offer.inet_up_mbps is None
        assert offer.pcie_bw_gbps is None


# ===================================================================
# Client-side GPU name filtering
# ===================================================================


class TestClientSideGPUFilter:
    """
    The SDK's 'in' operator can be fuzzy (e.g. matching 'RTX 5060 Ti'
    when searching for 'RTX 3090'). The client-side filter must reject
    non-exact matches.
    """

    def _filter(self, raw_offers: list[dict], gpu_names: list[str], min_gpus: int = 2) -> list[dict]:
        """Replicate the client-side filter from VastAIProvider.find_offers."""
        gpu_names_lower = {n.lower() for n in gpu_names}
        return [
            o for o in raw_offers
            if o.get("gpu_name", "").lower() in gpu_names_lower
            and int(o.get("num_gpus", 0)) >= min_gpus
        ]

    def test_exact_match_passes(self):
        offers = [{"gpu_name": "RTX 3090", "id": 1, "num_gpus": 2}]
        assert len(self._filter(offers, ["RTX 3090"])) == 1

    def test_wrong_gpu_rejected(self):
        offers = [{"gpu_name": "RTX 5060 Ti", "id": 1, "num_gpus": 2}]
        assert len(self._filter(offers, ["RTX 3090", "RTX 4090"])) == 0

    def test_case_insensitive(self):
        offers = [{"gpu_name": "rtx 3090", "id": 1, "num_gpus": 2}]
        assert len(self._filter(offers, ["RTX 3090"])) == 1

    def test_mixed_offers_filtered(self):
        offers = [
            {"gpu_name": "RTX 3090", "id": 1, "num_gpus": 2},
            {"gpu_name": "RTX 5060 Ti", "id": 2, "num_gpus": 2},
            {"gpu_name": "RTX 4090", "id": 3, "num_gpus": 2},
            {"gpu_name": "A100 SXM4 80GB", "id": 4, "num_gpus": 2},
        ]
        result = self._filter(offers, ["RTX 3090", "RTX 4090"])
        names = {o["gpu_name"] for o in result}
        assert names == {"RTX 3090", "RTX 4090"}

    def test_missing_gpu_name_rejected(self):
        offers = [{"id": 1, "num_gpus": 2}]
        assert len(self._filter(offers, ["RTX 3090"])) == 0

    def test_empty_offers_returns_empty(self):
        assert self._filter([], ["RTX 3090"]) == []

    def test_single_gpu_rejected_when_min_is_2(self):
        """1x RTX 3090 must be rejected when we need 2+ GPUs."""
        offers = [{"gpu_name": "RTX 3090", "id": 1, "num_gpus": 1}]
        assert len(self._filter(offers, ["RTX 3090"], min_gpus=2)) == 0

    def test_two_gpus_accepted_when_min_is_2(self):
        offers = [{"gpu_name": "RTX 3090", "id": 1, "num_gpus": 2}]
        assert len(self._filter(offers, ["RTX 3090"], min_gpus=2)) == 1

    def test_missing_num_gpus_rejected(self):
        """Offers without num_gpus field default to 0, which fails min_gpus check."""
        offers = [{"gpu_name": "RTX 3090", "id": 1}]
        assert len(self._filter(offers, ["RTX 3090"], min_gpus=2)) == 0


# ===================================================================
# Query string building
# ===================================================================


class TestQueryStringBuilding:
    """Verify the SDK query string is constructed correctly."""

    def _build_query(self, gpu_names: list[str], min_gpus: int = 2) -> str:
        """Replicate query building logic from VastAIProvider.find_offers."""
        _gpu_cfg = cfg["gpus"]

        if len(gpu_names) == 1:
            gpu_filter = f"gpu_name={gpu_names[0].replace(' ', '_')}"
        else:
            gpu_values = "[" + ",".join(n.replace(" ", "_") for n in gpu_names) + "]"
            gpu_filter = f"gpu_name in {gpu_values}"

        query_parts = [
            gpu_filter,
            f"num_gpus>={min_gpus}",
            f"gpu_total_ram>={_gpu_cfg['min_gpu_total_ram_gb']}",
            f"reliability>={_gpu_cfg['min_reliability']}",
            f"inet_up>={_gpu_cfg['min_inet_up_mbps']}",
            f"pcie_bw>={_gpu_cfg['min_pcie_bw_gbps']}",
            "rentable=True",
        ]
        return " ".join(query_parts)

    def test_single_gpu_exact_match(self):
        q = self._build_query(["RTX 3090"])
        assert "gpu_name=RTX_3090" in q
        assert "gpu_name in" not in q

    def test_multiple_gpus_uses_in(self):
        q = self._build_query(["RTX 3090", "RTX 4090"])
        assert "gpu_name in [RTX_3090,RTX_4090]" in q

    def test_spaces_become_underscores(self):
        q = self._build_query(["A100 SXM4 80GB"])
        assert "gpu_name=A100_SXM4_80GB" in q

    def test_min_gpus_included(self):
        q = self._build_query(["RTX 3090"], min_gpus=4)
        assert "num_gpus>=4" in q

    def test_min_vram_from_config(self):
        q = self._build_query(["RTX 3090"])
        expected = f"gpu_total_ram>={cfg['gpus']['min_gpu_total_ram_gb']}"
        assert expected in q

    def test_rentable_always_present(self):
        q = self._build_query(["RTX 3090"])
        assert "rentable=True" in q

    def test_no_double_equals(self):
        """SDK uses single = for exact match, NOT ==."""
        q = self._build_query(["RTX 3090"])
        assert "==" not in q

    def test_config_preferred_gpus_produce_valid_query(self):
        """The actual preferred GPUs from config.yml must produce a valid query."""
        gpus = preferred_gpus()
        q = self._build_query(gpus)
        # Must contain either exact match or 'in' syntax
        if len(gpus) == 1:
            assert f"gpu_name={gpus[0].replace(' ', '_')}" in q
        else:
            assert "gpu_name in [" in q


# ===================================================================
# VastAIProvider.find_offers — live API tests
# ===================================================================


class TestVastAIFindOffersLive:
    """
    Live API tests against Vast.ai. Requires VASTAI_API_KEY.
    These tests make real API calls — they validate that our query
    syntax actually works and returns the GPUs we expect.
    """

    def _get_provider(self) -> VastAIProvider:
        key = os.environ.get("VASTAI_API_KEY", "")
        if not key:
            from shittytoken.config import Settings
            key = Settings().vastai_api_key
        return VastAIProvider(api_key=key)

    @requires_vastai
    @pytest.mark.asyncio
    async def test_find_3090_offers_returns_only_3090s(self):
        """Search for RTX 3090 offers — every result must be an RTX 3090."""
        provider = self._get_provider()
        offers = await provider.find_offers(gpu_names=["RTX 3090"], min_gpus=1)
        for offer in offers:
            assert offer.gpu_name.lower() == "rtx 3090", (
                f"Expected 'RTX 3090' but got '{offer.gpu_name}'"
            )

    @requires_vastai
    @pytest.mark.asyncio
    async def test_find_4090_offers_returns_only_4090s(self):
        """Search for RTX 4090 offers — every result must be an RTX 4090."""
        provider = self._get_provider()
        offers = await provider.find_offers(gpu_names=["RTX 4090"], min_gpus=1)
        for offer in offers:
            assert offer.gpu_name.lower() == "rtx 4090", (
                f"Expected 'RTX 4090' but got '{offer.gpu_name}'"
            )

    @requires_vastai
    @pytest.mark.asyncio
    async def test_find_preferred_gpus_no_wrong_models(self):
        """
        Search using config.yml preferred GPUs — no unexpected GPU models.
        This is THE test that would have caught the RTX 5060 Ti bug.
        """
        provider = self._get_provider()
        gpus = preferred_gpus()
        offers = await provider.find_offers(gpu_names=gpus, min_gpus=2)

        allowed = {g.lower() for g in gpus}
        for offer in offers:
            assert offer.gpu_name.lower() in allowed, (
                f"Got unexpected GPU '{offer.gpu_name}' — "
                f"expected one of {gpus}"
            )

    @requires_vastai
    @pytest.mark.asyncio
    async def test_offers_meet_minimum_vram(self):
        """Every offer must have at least min_gpu_total_ram_gb total VRAM."""
        provider = self._get_provider()
        gpus = preferred_gpus()
        min_vram = cfg["gpus"]["min_gpu_total_ram_gb"]
        offers = await provider.find_offers(gpu_names=gpus, min_gpus=2)

        for offer in offers:
            # num_gpus * per-gpu VRAM should meet the minimum
            # We can check via the raw dict if available
            total_ram = offer.raw.get("gpu_total_ram")
            if total_ram is not None:
                assert total_ram >= min_vram, (
                    f"Offer {offer.offer_id} has {total_ram}GB total VRAM, "
                    f"minimum is {min_vram}GB"
                )

    @requires_vastai
    @pytest.mark.asyncio
    async def test_offers_have_required_fields(self):
        """Every offer must have basic fields populated."""
        provider = self._get_provider()
        offers = await provider.find_offers(gpu_names=["RTX 3090"], min_gpus=1)

        if not offers:
            pytest.skip("No RTX 3090 offers available right now")

        offer = offers[0]
        assert offer.offer_id
        assert offer.provider == "vastai"
        assert offer.gpu_name
        assert offer.num_gpus >= 1
        assert offer.cost_per_hour_usd is not None
        assert offer.cost_per_hour_usd > 0

    @requires_vastai
    @pytest.mark.asyncio
    async def test_offers_sorted_by_score(self):
        """Returned offers must be sorted by composite score (ascending)."""
        provider = self._get_provider()
        offers = await provider.find_offers(gpu_names=preferred_gpus(), min_gpus=2)

        if len(offers) < 2:
            pytest.skip("Need at least 2 offers to test sorting")

        for i in range(len(offers) - 1):
            assert offers[i].score <= offers[i + 1].score, (
                f"Offer {i} (score={offers[i].score:.4f}) > "
                f"offer {i+1} (score={offers[i+1].score:.4f})"
            )

    @requires_vastai
    @pytest.mark.asyncio
    async def test_all_offers_have_min_gpus(self):
        """Every offer must have at least min_gpus GPUs (the 1x RTX 3090 bug)."""
        provider = self._get_provider()
        gpus = preferred_gpus()
        min_gpus = cfg["gpus"]["min_gpus"]
        offers = await provider.find_offers(gpu_names=gpus, min_gpus=min_gpus)

        for offer in offers:
            assert offer.num_gpus >= min_gpus, (
                f"Offer {offer.offer_id} has {offer.num_gpus}x {offer.gpu_name}, "
                f"expected >= {min_gpus}"
            )

    @requires_vastai
    @pytest.mark.asyncio
    async def test_no_offers_for_nonexistent_gpu(self):
        """Searching for a GPU that doesn't exist should return empty list."""
        provider = self._get_provider()
        offers = await provider.find_offers(
            gpu_names=["NVIDIA FakeGPU 9999"],
            min_gpus=1,
        )
        assert offers == []


# ===================================================================
# VastAIProvider.find_offers — mocked unit tests
# ===================================================================


class TestVastAIFindOffersMocked:
    """Unit tests using mocked SDK to test filtering logic in isolation."""

    def _make_provider_with_mock(self, raw_results: list[dict]) -> VastAIProvider:
        """Create a VastAIProvider with a mocked SDK client."""
        provider = VastAIProvider.__new__(VastAIProvider)
        provider._session = None

        mock_client = MagicMock()
        mock_client.search_offers.return_value = raw_results
        provider._client = mock_client
        return provider

    @pytest.mark.asyncio
    async def test_filters_out_wrong_gpu_names(self):
        """Client-side filter must reject non-matching GPU names."""
        raw = [
            {"id": 1, "gpu_name": "RTX 3090", "num_gpus": 2, "dph_total": 0.5},
            {"id": 2, "gpu_name": "RTX 5060 Ti", "num_gpus": 2, "dph_total": 0.3},
            {"id": 3, "gpu_name": "RTX 4090", "num_gpus": 2, "dph_total": 0.7},
        ]
        provider = self._make_provider_with_mock(raw)
        offers = await provider.find_offers(["RTX 3090", "RTX 4090"], min_gpus=1)

        names = {o.gpu_name for o in offers}
        assert "RTX 5060 Ti" not in names
        assert names == {"RTX 3090", "RTX 4090"}

    @pytest.mark.asyncio
    async def test_filters_out_single_gpu_when_min_is_2(self):
        """1x GPU offers must be rejected when min_gpus=2."""
        raw = [
            {"id": 1, "gpu_name": "RTX 3090", "num_gpus": 1, "dph_total": 0.3},
            {"id": 2, "gpu_name": "RTX 3090", "num_gpus": 2, "dph_total": 0.5},
            {"id": 3, "gpu_name": "RTX 3090", "num_gpus": 4, "dph_total": 0.9},
        ]
        provider = self._make_provider_with_mock(raw)
        offers = await provider.find_offers(["RTX 3090"], min_gpus=2)

        for o in offers:
            assert o.num_gpus >= 2, f"Got {o.num_gpus}x GPU, expected >= 2"
        assert len(offers) == 2

    @pytest.mark.asyncio
    async def test_handles_non_list_response(self):
        """SDK sometimes returns error strings instead of lists."""
        provider = self._make_provider_with_mock("error: invalid query")
        offers = await provider.find_offers(["RTX 3090"])
        assert offers == []

    @pytest.mark.asyncio
    async def test_empty_results_returns_empty(self):
        provider = self._make_provider_with_mock([])
        offers = await provider.find_offers(["RTX 3090"])
        assert offers == []

    @pytest.mark.asyncio
    async def test_results_sorted_by_score(self):
        raw = [
            {"id": 1, "gpu_name": "RTX 3090", "num_gpus": 2, "dph_total": 1.00},
            {"id": 2, "gpu_name": "RTX 3090", "num_gpus": 2, "dph_total": 0.30},
            {"id": 3, "gpu_name": "RTX 3090", "num_gpus": 2, "dph_total": 0.60},
        ]
        provider = self._make_provider_with_mock(raw)
        offers = await provider.find_offers(["RTX 3090"])

        assert offers[0].cost_per_hour_usd == 0.30
        assert offers[-1].cost_per_hour_usd == 1.00


# ===================================================================
# DeploymentPlan
# ===================================================================


class TestDeploymentPlan:
    """DeploymentPlan display and bandwidth estimation."""

    def _make_plan(self, **offer_overrides) -> DeploymentPlan:
        config = _make_config(quantization="awq")
        offer = _make_offer(**offer_overrides)
        return DeploymentPlan(
            config_source="knowledge_graph",
            tensor_parallel_size=config.tensor_parallel_size,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            quantization=config.quantization,
            kv_cache_dtype=config.kv_cache_dtype,
            max_num_seqs=config.max_num_seqs,
            enable_prefix_caching=config.enable_prefix_caching,
            enforce_eager=config.enforce_eager,
            vllm_command=build_vllm_command(config, MODEL_ID),
            offer=offer,
            model_id=MODEL_ID,
        )

    def test_display_shows_gpu_info(self):
        plan = self._make_plan()
        text = plan.display()
        assert "RTX 3090" in text
        assert "2x" in text

    def test_display_shows_cost(self):
        plan = self._make_plan(cost_per_hour_usd=0.42)
        text = plan.display()
        assert "$0.42" in text

    def test_display_shows_model_id(self):
        plan = self._make_plan()
        text = plan.display()
        assert MODEL_ID in text

    def test_display_shows_vllm_command(self):
        plan = self._make_plan()
        text = plan.display()
        assert "vllm serve" in text

    def test_bandwidth_cost_estimate(self):
        plan = self._make_plan(inet_up_cost_per_gb=0.05)
        bw_cost = plan.estimated_bandwidth_cost_per_mtok()
        assert bw_cost is not None
        assert bw_cost > 0

    def test_bandwidth_cost_none_when_no_cost_data(self):
        plan = self._make_plan(inet_up_cost_per_gb=None)
        assert plan.estimated_bandwidth_cost_per_mtok() is None


# ===================================================================
# build_deployment_plan — integration
# ===================================================================


class TestBuildDeploymentPlan:
    """build_deployment_plan() correctly wires provider → plan."""

    @pytest.mark.asyncio
    async def test_raises_when_no_offers(self):
        mock_provider = AsyncMock(spec=GPUProvider)
        mock_provider.find_offers.return_value = []
        mock_provider.name = "mock"

        config = _make_config(quantization="awq")
        with pytest.raises(RuntimeError, match="No GPU offers found"):
            await build_deployment_plan(
                provider=mock_provider,
                config=config,
                model_id=MODEL_ID,
                gpu_names=["RTX 3090"],
            )

    @pytest.mark.asyncio
    async def test_selects_best_offer(self):
        cheap = _make_offer(offer_id="cheap", cost_per_hour_usd=0.20)
        expensive = _make_offer(offer_id="expensive", cost_per_hour_usd=2.00)

        mock_provider = AsyncMock(spec=GPUProvider)
        mock_provider.find_offers.return_value = [cheap, expensive]
        mock_provider.name = "mock"

        config = _make_config(quantization="awq")
        plan = await build_deployment_plan(
            provider=mock_provider,
            config=config,
            model_id=MODEL_ID,
            gpu_names=["RTX 3090"],
        )

        assert plan.offer.offer_id == "cheap"

    @pytest.mark.asyncio
    async def test_plan_contains_vllm_command(self):
        offer = _make_offer()
        mock_provider = AsyncMock(spec=GPUProvider)
        mock_provider.find_offers.return_value = [offer]
        mock_provider.name = "mock"

        config = _make_config(quantization="awq")
        plan = await build_deployment_plan(
            provider=mock_provider,
            config=config,
            model_id=MODEL_ID,
            gpu_names=["RTX 3090"],
        )

        assert plan.vllm_command == build_vllm_command(config, MODEL_ID)
        assert plan.model_id == MODEL_ID
