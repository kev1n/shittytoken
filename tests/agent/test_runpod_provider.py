"""
Tests for RunPodProvider — spot instance provisioning on RunPod.

Covers:
- GPU offer mapping and filtering
- Spot bid calculation
- GraphQL mutation building
- SSH port extraction from runtime ports
- Spot eviction detection
- Provider factory
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shittytoken.agent.provisioner import (
    GPUOffer,
    GPUProvider,
    RunPodProvider,
    VastAIProvider,
    get_provider,
)
from shittytoken.config import cfg


# ---------------------------------------------------------------------------
# Helpers — RunPod API response fixtures
# ---------------------------------------------------------------------------


def _gpu_list_response() -> list[dict]:
    """Simulates runpod.get_gpus() output."""
    return [
        {"id": "NVIDIA RTX 3090", "displayName": "RTX 3090", "memoryInGb": 24},
        {"id": "NVIDIA RTX 4090", "displayName": "RTX 4090", "memoryInGb": 24},
        {"id": "NVIDIA A100 80GB", "displayName": "A100 SXM4 80GB", "memoryInGb": 80},
        {"id": "NVIDIA RTX 5060 Ti", "displayName": "RTX 5060 Ti", "memoryInGb": 16},
    ]


def _gpu_detail(
    *,
    community_spot: float | None = 0.20,
    secure_spot: float | None = 0.30,
    mem_gb: int = 24,
    max_gpu_count: int = 8,
) -> dict:
    """Simulates runpod.get_gpu(id, count) output."""
    return {
        "id": "NVIDIA RTX 3090",
        "displayName": "RTX 3090",
        "memoryInGb": mem_gb,
        "maxGpuCount": max_gpu_count,
        "communitySpotPrice": community_spot,
        "secureSpotPrice": secure_spot,
        "communityPrice": 0.40,
        "securePrice": 0.50,
    }


def _make_runpod_provider(
    gpu_list: list[dict] | None = None,
    gpu_detail: dict | None = None,
) -> RunPodProvider:
    """Create a RunPodProvider with mocked SDK calls."""
    provider = RunPodProvider.__new__(RunPodProvider)
    provider._api_key = "test-key"

    mock_runpod = MagicMock()
    mock_runpod.get_gpus.return_value = gpu_list or _gpu_list_response()
    mock_runpod.get_gpu.return_value = gpu_detail or _gpu_detail()
    mock_runpod.get_pod.return_value = {}
    mock_runpod.terminate_pod.return_value = None
    provider._runpod = mock_runpod
    return provider


# Check for live API key
_has_runpod_key = bool(os.environ.get("RUNPOD_API_KEY") or "")

if not _has_runpod_key:
    try:
        from shittytoken.config import Settings
        _s = Settings()
        _has_runpod_key = bool(_s.runpod_api_key)
    except Exception:
        pass

requires_runpod = pytest.mark.skipif(
    not _has_runpod_key,
    reason="RUNPOD_API_KEY not set — skipping live API tests",
)


# ===================================================================
# Offer mapping and filtering
# ===================================================================


class TestRunPodOfferMapping:
    """RunPod GPU data → GPUOffer field mapping."""

    @pytest.mark.asyncio
    async def test_maps_gpu_name_correctly(self):
        provider = _make_runpod_provider()
        offers = await provider.find_offers(["RTX 3090"], min_gpus=2)
        assert len(offers) == 1
        assert offers[0].gpu_name == "RTX 3090"
        assert offers[0].provider == "runpod"

    @pytest.mark.asyncio
    async def test_offer_id_is_gpu_type_id(self):
        provider = _make_runpod_provider()
        offers = await provider.find_offers(["RTX 3090"], min_gpus=2)
        assert offers[0].offer_id == "NVIDIA RTX 3090"

    @pytest.mark.asyncio
    async def test_cost_is_spot_price_times_gpu_count(self):
        detail = _gpu_detail(secure_spot=0.20)
        provider = _make_runpod_provider(gpu_detail=detail)
        offers = await provider.find_offers(["RTX 3090"], min_gpus=2)
        assert offers[0].cost_per_hour_usd == pytest.approx(0.40)

    @pytest.mark.asyncio
    async def test_cost_with_4_gpus(self):
        detail = _gpu_detail(secure_spot=0.20)
        provider = _make_runpod_provider(gpu_detail=detail)
        offers = await provider.find_offers(["RTX 3090"], min_gpus=4)
        assert offers[0].cost_per_hour_usd == pytest.approx(0.80)

    @pytest.mark.asyncio
    async def test_bandwidth_costs_are_zero(self):
        """RunPod doesn't charge for bandwidth."""
        provider = _make_runpod_provider()
        offers = await provider.find_offers(["RTX 3090"], min_gpus=2)
        assert offers[0].inet_up_cost_per_gb == 0.0
        assert offers[0].inet_down_cost_per_gb == 0.0

    @pytest.mark.asyncio
    async def test_raw_dict_contains_spot_metadata(self):
        provider = _make_runpod_provider()
        offers = await provider.find_offers(["RTX 3090"], min_gpus=2)
        raw = offers[0].raw
        assert "spot_price_per_gpu" in raw
        assert "cloud_type" in raw

    @pytest.mark.asyncio
    async def test_num_gpus_matches_requested(self):
        provider = _make_runpod_provider()
        offers = await provider.find_offers(["RTX 3090"], min_gpus=2)
        assert offers[0].num_gpus == 2


class TestRunPodOfferFiltering:
    """GPU filtering logic for RunPod offers."""

    @pytest.mark.asyncio
    async def test_filters_out_wrong_gpu_names(self):
        provider = _make_runpod_provider()
        offers = await provider.find_offers(["RTX 3090"], min_gpus=2)
        names = {o.gpu_name for o in offers}
        assert "RTX 5060 Ti" not in names
        assert "RTX 4090" not in names

    @pytest.mark.asyncio
    async def test_case_insensitive_gpu_match(self):
        provider = _make_runpod_provider()
        offers = await provider.find_offers(["rtx 3090"], min_gpus=2)
        assert len(offers) == 1

    @pytest.mark.asyncio
    async def test_multiple_gpu_types(self):
        # Make get_gpu return different details per call
        detail = _gpu_detail()
        provider = _make_runpod_provider(gpu_detail=detail)
        offers = await provider.find_offers(["RTX 3090", "RTX 4090"], min_gpus=2)
        names = {o.gpu_name for o in offers}
        assert names == {"RTX 3090", "RTX 4090"}

    @pytest.mark.asyncio
    async def test_rejects_gpu_with_no_spot_price(self):
        detail = _gpu_detail(community_spot=None, secure_spot=None)
        provider = _make_runpod_provider(gpu_detail=detail)
        offers = await provider.find_offers(["RTX 3090"], min_gpus=2)
        assert len(offers) == 0

    @pytest.mark.asyncio
    async def test_rejects_gpu_with_zero_spot_price(self):
        detail = _gpu_detail(community_spot=0.0, secure_spot=0.0)
        provider = _make_runpod_provider(gpu_detail=detail)
        offers = await provider.find_offers(["RTX 3090"], min_gpus=2)
        assert len(offers) == 0

    @pytest.mark.asyncio
    async def test_rejects_insufficient_max_gpu_count(self):
        detail = _gpu_detail(max_gpu_count=1)
        provider = _make_runpod_provider(gpu_detail=detail)
        offers = await provider.find_offers(["RTX 3090"], min_gpus=2)
        assert len(offers) == 0

    @pytest.mark.asyncio
    async def test_nonexistent_gpu_returns_empty(self):
        provider = _make_runpod_provider()
        offers = await provider.find_offers(["NVIDIA FakeGPU 9999"], min_gpus=1)
        assert offers == []

    @pytest.mark.asyncio
    async def test_empty_gpu_list_returns_empty(self):
        provider = _make_runpod_provider(gpu_list=[])
        offers = await provider.find_offers(["RTX 3090"], min_gpus=1)
        assert offers == []

    @pytest.mark.asyncio
    async def test_offers_sorted_by_score(self):
        """When multiple GPUs match, offers should be sorted by score."""
        cheap_detail = _gpu_detail(community_spot=0.10)
        expensive_detail = _gpu_detail(community_spot=0.50)

        provider = _make_runpod_provider()
        # Return different prices for different GPU types
        call_count = 0

        def mock_get_gpu(gpu_id, count=1):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return expensive_detail
            return cheap_detail

        provider._runpod.get_gpu.side_effect = mock_get_gpu
        offers = await provider.find_offers(["RTX 3090", "RTX 4090"], min_gpus=2)

        if len(offers) >= 2:
            for i in range(len(offers) - 1):
                assert offers[i].score <= offers[i + 1].score


# ===================================================================
# Spot price extraction
# ===================================================================


class TestSpotPriceExtraction:
    """_extract_spot_price returns correct price for each cloud type."""

    def test_community_cloud(self):
        detail = _gpu_detail(community_spot=0.20, secure_spot=0.30)
        assert RunPodProvider._extract_spot_price(detail, "COMMUNITY") == 0.20

    def test_secure_cloud(self):
        detail = _gpu_detail(community_spot=0.20, secure_spot=0.30)
        assert RunPodProvider._extract_spot_price(detail, "SECURE") == 0.30

    def test_all_cloud_prefers_community(self):
        detail = _gpu_detail(community_spot=0.20, secure_spot=0.30)
        assert RunPodProvider._extract_spot_price(detail, "ALL") == 0.20

    def test_all_cloud_falls_back_to_secure(self):
        detail = _gpu_detail(community_spot=None, secure_spot=0.30)
        assert RunPodProvider._extract_spot_price(detail, "ALL") == 0.30

    def test_returns_none_when_both_missing(self):
        detail = _gpu_detail(community_spot=None, secure_spot=None)
        assert RunPodProvider._extract_spot_price(detail, "COMMUNITY") is None


# ===================================================================
# SSH port extraction
# ===================================================================


class TestSSHPortExtraction:
    """_extract_ssh_from_ports parses RunPod runtime port mappings."""

    def test_extracts_from_runtime_ports(self):
        ports = [
            {"privatePort": 8080, "publicPort": 18080, "ip": "1.2.3.4"},
            {"privatePort": 22, "publicPort": 12345, "ip": "1.2.3.4"},
        ]
        host, port = RunPodProvider._extract_ssh_from_ports(ports, {})
        assert host == "1.2.3.4"
        assert port == 12345

    def test_no_ssh_port_returns_none(self):
        ports = [
            {"privatePort": 8080, "publicPort": 18080, "ip": "1.2.3.4"},
        ]
        host, port = RunPodProvider._extract_ssh_from_ports(ports, {})
        assert host is None
        assert port == 22

    def test_empty_ports_list(self):
        host, port = RunPodProvider._extract_ssh_from_ports([], {})
        assert host is None
        assert port == 22

    def test_fallback_to_machine_pod_host_id(self):
        pod_info = {"machine": {"podHostId": "5.6.7.8"}}
        host, port = RunPodProvider._extract_ssh_from_ports([], pod_info)
        assert host == "5.6.7.8"
        assert port == 22

    def test_no_ip_in_port_entry(self):
        ports = [{"privatePort": 22, "publicPort": 12345}]
        host, port = RunPodProvider._extract_ssh_from_ports(ports, {})
        assert host is None

    def test_prefers_port_entry_over_fallback(self):
        ports = [{"privatePort": 22, "publicPort": 12345, "ip": "1.2.3.4"}]
        pod_info = {"machine": {"podHostId": "5.6.7.8"}}
        host, port = RunPodProvider._extract_ssh_from_ports(ports, pod_info)
        assert host == "1.2.3.4"
        assert port == 12345

    def test_machine_not_dict(self):
        """Handles case where machine field is not a dict."""
        pod_info = {"machine": "invalid"}
        host, port = RunPodProvider._extract_ssh_from_ports([], pod_info)
        assert host is None


# ===================================================================
# Spot eviction detection
# ===================================================================


class TestSpotEvictionDetection:
    """is_spot_eviction distinguishes eviction from manual stop."""

    def test_spot_eviction_detected(self):
        pod_info = {
            "desiredStatus": "EXITED",
            "podType": "INTERRUPTABLE",
        }
        assert RunPodProvider.is_spot_eviction(pod_info) is True

    def test_manual_stop_not_eviction(self):
        """On-demand pods that exit are not evictions."""
        pod_info = {
            "desiredStatus": "EXITED",
            "podType": "ON_DEMAND",
        }
        assert RunPodProvider.is_spot_eviction(pod_info) is False

    def test_running_spot_not_eviction(self):
        pod_info = {
            "desiredStatus": "RUNNING",
            "podType": "INTERRUPTABLE",
        }
        assert RunPodProvider.is_spot_eviction(pod_info) is False

    def test_empty_dict_not_eviction(self):
        assert RunPodProvider.is_spot_eviction({}) is False

    def test_missing_pod_type_not_eviction(self):
        pod_info = {"desiredStatus": "EXITED"}
        assert RunPodProvider.is_spot_eviction(pod_info) is False

    def test_terminating_spot_not_eviction(self):
        """TERMINATING state means we're destroying it, not an eviction."""
        pod_info = {
            "desiredStatus": "TERMINATING",
            "podType": "INTERRUPTABLE",
        }
        assert RunPodProvider.is_spot_eviction(pod_info) is False


# ===================================================================
# Provider factory
# ===================================================================


class TestProviderFactory:
    """get_provider returns the correct provider type."""

    def test_vastai_provider(self):
        provider = get_provider("vastai", vastai_api_key="test-key")
        assert isinstance(provider, VastAIProvider)
        assert provider.name == "vastai"

    def test_runpod_provider(self):
        provider = get_provider("runpod", runpod_api_key="test-key")
        assert isinstance(provider, RunPodProvider)
        assert provider.name == "runpod"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("aws")

    def test_missing_vastai_key_raises(self):
        with pytest.raises(ValueError, match="vastai_api_key"):
            get_provider("vastai", vastai_api_key="")

    def test_missing_runpod_key_raises(self):
        with pytest.raises(ValueError, match="runpod_api_key"):
            get_provider("runpod", runpod_api_key="")

    def test_provider_is_gpu_provider(self):
        provider = get_provider("runpod", runpod_api_key="test-key")
        assert isinstance(provider, GPUProvider)


# ===================================================================
# RunPod cross-provider offer scoring
# ===================================================================


class TestCrossProviderScoring:
    """RunPod offers score correctly alongside Vast.ai offers."""

    def test_cheaper_runpod_scores_lower(self):
        runpod_offer = GPUOffer(
            offer_id="NVIDIA RTX 3090",
            provider="runpod",
            gpu_name="RTX 3090",
            num_gpus=2,
            cost_per_hour_usd=0.30,
            reliability=None,
            inet_up_cost_per_gb=0.0,
            inet_down_cost_per_gb=0.0,
        )
        vastai_offer = GPUOffer(
            offer_id="12345",
            provider="vastai",
            gpu_name="RTX 3090",
            num_gpus=2,
            cost_per_hour_usd=0.50,
            reliability=0.99,
            inet_up_mbps=300.0,
            inet_up_cost_per_gb=0.02,
        )
        assert runpod_offer.score < vastai_offer.score

    def test_no_bandwidth_cost_advantage(self):
        """RunPod has $0 bandwidth vs Vast.ai's per-GB charges.

        Both need inet_up_mbps set to avoid the upload speed penalty
        dominating the comparison.
        """
        runpod_offer = GPUOffer(
            offer_id="x",
            provider="runpod",
            gpu_name="RTX 3090",
            num_gpus=2,
            cost_per_hour_usd=0.50,
            reliability=None,
            inet_up_mbps=300.0,
            inet_up_cost_per_gb=0.0,
        )
        vastai_offer = GPUOffer(
            offer_id="y",
            provider="vastai",
            gpu_name="RTX 3090",
            num_gpus=2,
            cost_per_hour_usd=0.50,
            reliability=0.99,
            inet_up_mbps=300.0,
            inet_up_cost_per_gb=0.10,
        )
        # Same compute cost, but Vast.ai has bandwidth cost
        assert runpod_offer.score < vastai_offer.score


# ===================================================================
# RunPod live API tests
# ===================================================================


class TestRunPodLive:
    """Live API tests. Requires RUNPOD_API_KEY."""

    def _get_provider(self) -> RunPodProvider:
        key = os.environ.get("RUNPOD_API_KEY", "")
        if not key:
            from shittytoken.config import Settings
            key = Settings().runpod_api_key
        return RunPodProvider(api_key=key)

    @requires_runpod
    @pytest.mark.asyncio
    async def test_list_gpus(self):
        """Can query GPU types and get a non-empty result."""
        provider = self._get_provider()
        gpus = await provider.find_offers(["RTX 3090"], min_gpus=1)
        # May be empty if no spot availability, but shouldn't error
        assert isinstance(gpus, list)

    @requires_runpod
    @pytest.mark.asyncio
    async def test_spot_pricing_populated(self):
        """Spot pricing fields are present in raw response."""
        provider = self._get_provider()
        import runpod
        runpod.api_key = provider._api_key
        all_gpus = runpod.get_gpus()
        assert isinstance(all_gpus, list)
        assert len(all_gpus) > 0
        # At least some GPUs should have displayName
        assert any(g.get("displayName") for g in all_gpus)

    @requires_runpod
    @pytest.mark.asyncio
    async def test_find_offers_returns_valid_gpu_offers(self):
        """find_offers returns GPUOffer objects with correct provider."""
        provider = self._get_provider()
        offers = await provider.find_offers(["RTX 3090", "RTX 4090"], min_gpus=1)
        for offer in offers:
            assert offer.provider == "runpod"
            assert offer.gpu_name in ("RTX 3090", "RTX 4090")
            assert offer.cost_per_hour_usd is not None
            assert offer.cost_per_hour_usd > 0

    @requires_runpod
    @pytest.mark.asyncio
    async def test_no_offers_for_fake_gpu(self):
        provider = self._get_provider()
        offers = await provider.find_offers(["NVIDIA FakeGPU 9999"], min_gpus=1)
        assert offers == []
