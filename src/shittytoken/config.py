"""
Settings — environment-variable backed configuration for the orchestration agent.

All fields are readable from environment variables or a .env file.
pydantic-settings is a separate package from pydantic v2 and must be installed
independently: pip install pydantic-settings
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "shittytoken_dev"

    # GPU providers
    vastai_api_key: str = ""
    runpod_api_key: str = ""

    # Anthropic
    anthropic_api_key: str = ""
    oom_reasoning_model: str = "claude-opus-4-6"

    # Gateway
    gateway_router_url: str = "http://localhost:8001"
    gateway_admin_token: str = ""

    # SSH
    ssh_private_key_path: str = "~/.ssh/id_rsa"
    ssh_keepalive_interval: int = 30

    # Orchestrator scaling
    scale_up_waiting_threshold: int = 10
    scale_down_idle_seconds: int = 300
    scale_down_cache_max: float = 0.50

    # Benchmark
    benchmark_min_throughput_tps: float = 20.0

    # Monitoring intervals (seconds)
    metrics_poll_interval_s: int = 10
    health_check_interval_s: int = 30
    startup_monitor_timeout_s: int = 600

    class Config:
        env_file = ".env"
