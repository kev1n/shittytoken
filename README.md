# ShittyToken

**Dirt-cheap LLM inference via autonomous spot GPU orchestration**

ShittyToken provisions spot GPUs from Vast.ai and RunPod, deploys vLLM workers over SSH, and routes inference requests through a custom async proxy with cache-aware routing. Users are billed per-token using a credit block system backed by a PostgreSQL ledger and Redis hot-path for sub-millisecond balance checks.

## Architecture

```
Client (Bearer API key)
  |
  v
Custom Router (:8001) --- aiohttp async server
  |  |-- Auth middleware (Redis: key cache, rate limits, balance)
  |  |-- Cache-aware consistent hash routing
  |  |-- SSE proxy (zero-copy streaming, usage extraction)
  |  '-- Usage pipeline -> Postgres ledger + Redis balance
  |
  v
vLLM Workers (spot GPUs via SSH tunnels)
  |
  ^
Orchestrator (PydanticAI agent)
  |-- Provision spot instances (Vast.ai / RunPod)
  |-- Deploy vLLM via SSH
  |-- Benchmark (throughput, TTFT, cache hit rate)
  |-- Register workers via admin API
  '-- OOM recovery + spot eviction handling
```

## Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Orchestrator | `src/shittytoken/agent/` | Autonomous GPU provisioning, deployment, monitoring |
| Custom Router | `src/shittytoken/gateway/` | SSE proxy, cache-aware routing, admin API, metrics |
| Billing | `src/shittytoken/billing/` | Credit blocks, Redis hot-path, Postgres ledger, reconciliation |
| Benchmark | `src/shittytoken/benchmark/` | Throughput, TTFT, prefix cache, long-context testing |
| Knowledge Graph | `src/shittytoken/knowledge/` | Neo4j -- GPU configs, benchmark results, OOM events |

## Billing Architecture

- **Credit blocks**: Each top-up creates a discrete block. Blocks are deducted FIFO (oldest first), support expiration dates, and handle promotional credits.
- **Redis hot-path**: Sub-millisecond balance checks via atomic Lua scripts, API key caching, and per-key rate limiting (RPM and TPM).
- **PostgreSQL**: Append-only ledger serving as the system of record. Credit blocks use row-level locking for safe concurrent deduction.
- **Usage pipeline**: Pluggable publisher/consumer architecture. Defaults to an in-process queue; swap in Kafka by setting `kafka_bootstrap_servers` in config. The consumer deducts cost from credit blocks FIFO and syncs the new balance to Redis.
- **Reconciler**: Runs every 60s (configurable) to detect and correct drift between the Postgres ledger and Redis balance cache.
- **Stripe**: Payment collection only. A Stripe webhook creates a credit block in Postgres -- Stripe never touches balance logic directly.

## Routing

- **Cache-aware consistent hashing**: Routes based on SHA-256 of system prompt + first user message, sending repeat conversations to workers that likely have the KV-cache prefix resident.
- **Overload fallback**: When the target worker exceeds running request or cache utilization thresholds, requests fall back to the least-loaded healthy worker.
- **Hot-reload**: Workers are added and removed at runtime via the admin API (`/admin/workers`) with no process restart required.
- **SSE pass-through**: Streams Server-Sent Events from vLLM workers directly to the client without intermediate JSON parsing. Token usage is extracted from the final SSE chunk for billing.

## Quick Start

```bash
# Install
uv sync --extra dev

# Configure
cp .env.example .env
# Edit .env with your API keys

# Run tests
uv run pytest

# Start the router (dev mode, no auth)
uv run shittytoken-router

# Start the orchestrator
uv run shittytoken run
```

## Configuration

- **`config.yml`** -- All tunable parameters: model serving config, GPU catalog and requirements, vLLM defaults, gateway settings, orchestrator behavior, benchmark thresholds.
- **`.env`** -- Secrets only: API keys (Vast.ai, RunPod, Anthropic, Stripe), database credentials, SSH key path.

Auth, billing, and rate limiting are toggled via `gateway.auth.enabled` in `config.yml`. When disabled (the default), the router runs without authentication for local development.

## Infrastructure Requirements

When auth and billing are enabled, the following services are required:

- **PostgreSQL** -- Billing ledger, credit blocks, usage events, API key storage
- **Redis** -- Balance cache, API key cache, rate limiting

Optional:

- **Neo4j** -- Knowledge graph for GPU configurations, benchmark results, and OOM event history (used by the orchestrator)
- **Kafka** -- Usage event pipeline (defaults to in-process queue if not configured)
- **Stripe** -- Payment collection via webhooks

## Development

- **Package manager**: [uv](https://docs.astral.sh/uv/) -- do not use pip directly
- **Python**: 3.11+
- **Tests**: `uv run pytest`
- **Build system**: Hatchling

## License

Proprietary.
