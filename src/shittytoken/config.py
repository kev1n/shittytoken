"""
Settings — layered configuration for the orchestration agent.

config.yml holds all tunable parameters (models, GPUs, thresholds, ports).
.env holds secrets (API keys, passwords).

Usage:
    from shittytoken.config import settings, cfg

    settings.vastai_api_key        # from .env
    cfg["models"]["serving"][0]    # from config.yml
    cfg["gateway"]["router"]["port"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings


# ── config.yml loader ───────────────────────────────────────────────────────

def _find_config_yml() -> Path:
    """Walk up from this file to find config.yml in the project root."""
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        candidate = parent / "config.yml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "config.yml not found — expected in the project root"
    )


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load config.yml and return it as a dict."""
    if path is None:
        path = _find_config_yml()
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# Singleton — loaded once at import time.
cfg: dict[str, Any] = load_config()


# ── Helper accessors (avoid deep dict lookups everywhere) ───────────────────

def serving_models() -> list[dict[str, Any]]:
    """Return the list of model dicts from config."""
    return cfg["models"]["serving"]


def primary_model_id() -> str:
    """Return the model_id of the first serving model."""
    return serving_models()[0]["model_id"]


def gpu_catalog() -> list[dict[str, Any]]:
    """Return the list of GPU spec dicts."""
    return cfg["gpus"]["catalog"]


def preferred_gpus() -> list[str]:
    """Return the list of preferred GPU names for provisioning."""
    return cfg["gpus"]["preferred"]


def vllm_defaults() -> dict[str, Any]:
    """Return the default vLLM configuration dict."""
    return dict(cfg["vllm"]["defaults"])


def gateway_cfg() -> dict[str, Any]:
    """Return the gateway config section."""
    return cfg["gateway"]


def benchmark_cfg() -> dict[str, Any]:
    """Return the benchmark config section."""
    return cfg["benchmark"]


# ── Secrets (.env) ──────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """Secrets and connection strings — loaded from environment / .env file."""

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "shittytoken_dev"

    # GPU providers
    vastai_api_key: str = ""
    runpod_api_key: str = ""

    # Hugging Face
    huggingface_token: str = ""

    # Agent LLM
    anthropic_api_key: str = ""
    agent_model: str = "claude-opus-4-6"

    # Gateway
    gateway_admin_token: str = ""

    # SSH
    ssh_private_key_path: str = "~/.ssh/id_ed25519"

    class Config:
        env_file = ".env"
        extra = "ignore"
