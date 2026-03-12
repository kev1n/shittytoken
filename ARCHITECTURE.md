# ShittyToken — Architecture

> This document is the source of truth for the system's design.
> Implementation must honor this document. If implementation reveals
> the design was wrong, update this document deliberately — don't
> silently drift.
>
> Last updated: 2026-03-12
> Decision Register: See DECISION-REGISTER.md

## 1. Overview

ShittyToken delivers the cheapest possible LLM inference tokens by orchestrating spot GPU instances through an AI agent that handles the entire lifecycle — from procurement to OOM recovery — while routing traffic through a throughput-optimized gateway built on vLLM. The platform targets users who need bulk inference (coding assistants, chat, document QA) and are willing to accept degraded latency under high concurrency in exchange for prices that undercut every competitor. The system explicitly trades latency for cost: no SLAs, no TTFT guarantees under load, just maximum tokens per dollar.

## 2. North Star

**North Star (prose):**
ShittyToken exists to make LLM inference as cheap as physically possible by relentlessly optimizing tokens-per-dollar rather than tokens-per-second. A single AI agent eliminates all operational overhead — it procures spot GPUs, configures vLLM, qualifies instances via benchmark, routes traffic, recovers from failures, and scales down autonomously. The system succeeds when it produces more tokens per GPU-hour than any comparable managed service, and when a new GPU type or model can be onboarded with zero human intervention.

**North Star Assertions:**

| # | Assertion | User Approved | Status |
|---|-----------|--------------|--------|
| 1 | Every routing decision optimizes batch utilization (throughput), not TTFT | [ ] | Active |
| 2 | The agent can provision a new vLLM worker end-to-end without human input | [ ] | Active |
| 3 | No instance enters the gateway's worker pool without passing the qualification benchmark | [ ] | Active |
| 4 | All OOM recovery attempts are recorded in the knowledge graph regardless of outcome | [ ] | Active |
| 5 | The agent never hard-codes GPU/model configurations; all configs derive from the knowledge graph | [ ] | Active |

## 3. Quality Attribute Priorities

**qa_version:** 1

| Rank | Quality Attribute | Rationale |
|------|------------------|-----------|
| 1 | Scalability | The core value proposition depends on elastically adding/removing spot GPU workers as demand fluctuates |
| 2 | Maintainability | Rapid iteration requires that each subsystem (agent, gateway, benchmark, KG) is independently understandable and changeable |
| 3 | Rapid Iteration | The model selection, routing policy, and OOM recovery logic will change frequently as we learn from production data |
| 4 | Correctness | OOM classification must be accurate; misclassified errors waste instance-hours and corrupt the knowledge graph |
| 5 | Performance (throughput) | Throughput is the product; latency is explicitly de-prioritized |

## 4. Design Principles

1. **Throughput over latency, always.** The gateway routes to maximize batch size, not minimize TTFT. High `num_requests_waiting` is acceptable; low GPU utilization is not. *Because:* ShittyToken's price advantage disappears the moment we start protecting latency at the cost of GPU utilization.

2. **The knowledge graph is the source of configuration truth.** No vLLM startup command, GPU selection filter, or OOM fix is hard-coded. All configuration is derived by querying the knowledge graph for proven configurations for the detected (GPU model, LLM model) pair. *Because:* Hard-coded heuristics don't generalize to new hardware and don't improve over time.

3. **No instance goes live without passing its benchmark.** The qualification benchmark runs on every new instance before gateway registration. *Because:* Spot instances occasionally have silent hardware faults (throttling, bad VRAM sectors) that only show up under load; unqualified instances corrupt quality metrics and waste user requests.

4. **The agent records every failure before attempting recovery.** OOM events, benchmark failures, and SSH errors are written to the knowledge graph before any recovery action is taken. *Because:* If recovery itself fails, we need the original error preserved to learn from it.

5. **Each subsystem has a single owner.** The agent orchestrates; the gateway routes; the benchmark measures; the knowledge graph stores. No subsystem performs another's job. *Because:* Cross-cutting responsibilities are the primary source of unmaintainable code in long-running autonomous systems.

## 5. System Boundaries

### Orchestration Agent
- **Responsibility:** Full lifecycle management of vLLM worker instances — demand detection, GPU procurement, SSH verification, benchmark execution, gateway registration, monitoring, OOM recovery, and decommission.
- **Owns:** Instance state machine (provisioning → benchmarking → serving → draining → terminated), knowledge graph write access.
- **Exposes:** Instance status events consumed by the gateway; knowledge graph queries for configuration.
- **Consumes:** Gateway aggregate metrics (queue depth, cache utilization); provider APIs (Vast.ai, RunPod); knowledge graph read/write; SSH control plane to each worker.
- **Invariants:** One agent instance is the sole writer to the knowledge graph. Agent runs on the central server, never on GPU workers.

### Gateway (vLLM Router + Nginx)
- **Responsibility:** Route incoming inference requests to healthy, registered vLLM workers. Proxy SSE streams without buffering. Maintain per-worker health state.
- **Owns:** Active worker pool (URL list), per-worker circuit-breaker state.
- **Exposes:** OpenAI-compatible `/v1/chat/completions` and `/v1/models` endpoints to clients.
- **Consumes:** Worker `/health`, `/v1/models` (readiness), and `/metrics` (routing weight). vLLM Router for LB policy execution.
- **Invariants:** Only workers that have passed the qualification benchmark are in the pool. `proxy_buffering off` — SSE chunks are never buffered. Upstream connections are cancelled on client disconnect.

### vLLM Workers
- **Responsibility:** Serve LLM inference via OpenAI-compatible HTTP API. Report health and metrics. Nothing else.
- **Owns:** KV cache, model weights in VRAM.
- **Exposes:** `/v1/chat/completions`, `/v1/models`, `/health`, `/metrics` on port 8000.
- **Consumes:** Model weights from HuggingFace (at startup). Requests from gateway.
- **Invariants:** Workers are stateless from the system's perspective — they can be destroyed and replaced without data loss. `--enable-prefix-caching` is always enabled. `--ipc=host` is always set for tensor parallelism.

### Qualification Benchmark
- **Responsibility:** Measure a new worker's actual performance (cold/warm TTFT, throughput under concurrency) and verify that prefix caching is functioning. Record results in the knowledge graph.
- **Owns:** Nothing persistent; writes benchmark results to the knowledge graph via the agent.
- **Exposes:** Pass/fail signal to the agent, plus structured metrics (TTFT curves, cache hit rates, throughput at concurrency levels).
- **Consumes:** The worker under test's `/v1/chat/completions` and `/metrics` endpoints.
- **Invariants:** Benchmark always runs the cold-cache phase before the warm-cache phase. Server-side `/metrics` is scraped concurrently with request generation to correlate cache hits with TTFT. Failed requests are counted (not silently dropped).

### Knowledge Graph (Neo4j)
- **Responsibility:** Store institutional knowledge about GPU/model configurations, benchmark history, OOM events, and their resolutions. Enable the agent to query for proven configurations without re-learning.
- **Owns:** `GPUModel`, `LLMModel`, `Configuration`, `BenchmarkResult`, `OOMEvent` nodes and their relationships.
- **Exposes:** Cypher query interface to the agent.
- **Consumes:** Agent writes on every provisioning, benchmark, OOM, and decommission event.
- **Invariants:** OOM events are written before recovery is attempted. Every `Configuration` node has a link to the `BenchmarkResult` that validated it (or a failed attempt).

### Boundary Rules
- The gateway never writes to the knowledge graph; it only reads worker pool state from the agent.
- vLLM workers never connect to the knowledge graph or to each other.
- The benchmark runner is invoked by the agent only; it has no direct relationship with the gateway.
- The agent is the only process that calls provider APIs (Vast.ai, RunPod).

## 6. Data Flow

### Client Inference Request (Happy Path)
1. Client → Gateway: `POST /v1/chat/completions` with `stream: true`
2. Gateway → vLLM Router: selects worker by `consistent_hash` or `cache_aware` policy
3. Gateway → Worker: proxies request with `proxy_buffering off`
4. Worker → Gateway → Client: SSE token stream, chunk-by-chunk
5. Worker: on stream completion, updates `num_requests_running` metric
6. Gateway: scrapes `/metrics` every N seconds, updates routing weights

### Client Inference Request (Worker Fails Mid-Stream)
1. Worker returns HTTP 5xx or connection drops
2. Gateway circuit breaker trips (open state)
3. Gateway cancels upstream connection, returns error to client
4. Agent detects health failure via `/health` polling, initiates OOM recovery or reprovisioning

### Instance Provisioning (Happy Path)
1. Agent detects: `num_requests_waiting` exceeds threshold across workers
2. Agent queries KG: "best config for (target model, available GPU types) ranked by throughput"
3. Agent calls provider API: filter by GPU type, VRAM, reliability; select cheapest
4. Agent: creates instance with vLLM Docker image and KG-derived startup command
5. Agent: polls provider API until `ssh_host` populated and status = running
6. Agent → Worker via SSH: `nvidia-smi` to verify hardware matches expectation
7. Agent → Worker via SSH: streams container logs, watches for startup completion or failure
8. Agent: polls `GET /v1/models` until model is fully loaded
9. Agent → Benchmark: runs qualification suite against worker
10. Agent: on benchmark pass, registers worker URL with gateway; traffic begins
11. Agent → KG: writes `BenchmarkResult` node linked to `Configuration` and `GPUModel`

### OOM Recovery Flow
1. Agent detects `torch.OutOfMemoryError` in streamed container logs
2. Agent → KG: writes `OOMEvent` node (error message, phase, GPU state, linked configuration)
3. Agent → KG: queries prior OOM resolutions on same GPU type
4. Agent (LLM reasoning): classifies error as loading-OOM vs runtime-OOM
5. Agent: proposes smallest config change (prefer `max_model_len` reduction over quantization change)
6. Agent: destroys current instance, provisions new one with updated config
7. Agent → KG: writes resolution outcome (success or failure), linked to `OOMEvent`

### Scale-Down
1. Agent detects: `num_requests_waiting` = 0 across workers for sustained period AND `gpu_cache_perc` < 0.50
2. Agent: selects least-utilized worker
3. Agent → Gateway: removes worker from pool (no new requests routed)
4. Agent: waits for in-flight requests to drain (watches `num_requests_running` → 0)
5. Agent → Provider API: `DELETE /instances/{id}`
6. Agent → KG: writes final instance metrics (runtime hours, total tokens, cost)

## 7. Cross-Cutting Concerns

### Authentication & Authorization
- **Client auth:** TODO — API key validation at the gateway level. Keys map to rate limits and model tiers.
- **Worker auth:** Workers are on private network segments; no public exposure beyond the gateway. Provider SSH keys managed by the agent's secrets store.
- **KG auth:** Neo4j credentials held by the agent only; no other subsystem has direct DB access.

### Error Handling
- Errors are classified at the point of origin (SSH errors, provider API errors, vLLM errors, KG errors) before propagating.
- OOM errors trigger the dedicated recovery flow — they are never silently retried.
- Provider API failures (instance creation) are retried with exponential backoff, then escalated to the agent's alert channel.
- Client-facing errors are translated to OpenAI-compatible error responses at the gateway.

### Logging & Observability
- **Agent:** Structured JSON logs for every state transition (instance provisioning, benchmark start/end, OOM event, gateway registration/deregistration).
- **Workers:** Container logs streamed via SSH by the agent; specific error patterns trigger state machine transitions.
- **Gateway:** Access logs with `request_id` for distributed tracing. vLLM Router Prometheus metrics at port 29000.
- **Workers:** vLLM `/metrics` endpoint (Prometheus) scraped every 2 seconds by benchmark; scrape interval TBD for production routing.
- **KG:** All writes are idempotent with timestamps; the graph itself is the audit log of system history.

### Configuration
- vLLM startup parameters are generated from the knowledge graph, not from environment variables or config files.
- The agent's own config (provider API keys, Neo4j credentials, gateway URL, demand thresholds) is injected via environment variables.
- No magic numbers in code — all thresholds (OOM trigger patterns, scale-up/down thresholds, benchmark pass criteria) are named constants, configurable via the agent's config.

### Spot Instance Interruption
- Providers (Vast.ai, RunPod) can reclaim spot instances with little warning.
- The agent maintains a heartbeat health check on all workers. A worker that stops responding is treated identically to an OOM failure.
- TODO: Define grace period and drain behavior on interruption signals (if provider exposes them).

## 8. Key Decisions & Rationale

- **DR-1: Qwen 3.5 35B-A3B (MoE) as default model** — 3.3× higher decode throughput vs 27B dense on identical hardware due to MoE sparse activation, at acceptable quality cost for bulk inference workloads. See DECISION-REGISTER.md.
- **DR-2: vLLM Router with `consistent_hash` policy** — routes same-session requests to same worker, building prefix cache hits without per-prefix cache state tracking (which vLLM doesn't expose). See DECISION-REGISTER.md.
- **DR-3: Neo4j as knowledge graph** — native graph queries for OOM resolution lookup and config ranking are significantly cleaner than relational joins. See DECISION-REGISTER.md.
- **DR-4: AsyncSSH for agent-to-worker control plane** — supports concurrent multi-instance management with persistent keepalive connections. See DECISION-REGISTER.md.
- **DR-5: Custom benchmark over existing tools** — vLLM's `benchmark_serving.py` silently drops failed requests and cannot correlate server-side cache hit metrics with client-side TTFT. Custom tool is required for qualification gate. See DECISION-REGISTER.md.
- **DR-6: aiohttp for benchmark HTTP client** — ~7.5× faster than httpx for high-concurrency single requests per vLLM benchmark tooling precedent. See DECISION-REGISTER.md.

## 9. Known Risks

| Risk | Severity | Mitigation | Accepted Because |
|------|----------|------------|-----------------|
| vLLM prefix caching may not work with DeltaNet hybrid attention (both Qwen 3.5 models) | H | Benchmark explicitly measures actual `prefix_cache_hit_rate`; routing shifts to load-based if caching is ineffective | Both model variants are affected equally; benchmark detects it before traffic is routed |
| Vast.ai/RunPod spot instances can be reclaimed without warning | H | Agent heartbeat detects failure; rapid reprovisioning from KG config | Cost savings justify the interruption risk; no SLAs to violate |
| Neo4j single-node failure takes down OOM recovery reasoning | M | KG is read for config but agent can fall back to memory-calculation heuristics | KG is local or close-by; full HA adds complexity not warranted at this stage |
| Knowledge graph corrupt configs (bad benchmark result linked to wrong GPU) | M | Benchmark always runs on fresh instances; config nodes are immutable once written; new runs create new nodes | Mutable history would be worse; immutable append-only is safer |
| MoE expert routing overhead on TP=2 vs single-GPU | L | Benchmark measures actual throughput on target TP configuration | Expected <10% overhead based on community benchmarks; acceptable |

## 10. Review Criteria

- [ ] All vLLM startup parameter generation reads from the knowledge graph — no hard-coded values in agent code
- [ ] Gateway confirms `proxy_buffering off` and long idle timeouts on all upstream connections
- [ ] Benchmark records failed requests (not silently drops them) in throughput calculations
- [ ] Agent cancels upstream provider API requests before destroying an instance (avoid orphaned charges)
- [ ] OOM events are written to KG before recovery action starts (not after)
- [ ] Benchmark cold-cache phase runs before warm-cache phase (verified by phase sequencing, not just assertion)
- [ ] `--enable-prefix-caching` and `--ipc=host` flags are present in every generated vLLM startup command
- [ ] Worker is removed from gateway pool before SSH drain; no new requests routed during drain
- [ ] `gpu_memory_utilization` is never set to 1.0 in any generated configuration
- [ ] Agent logs structured JSON for all state transitions (not free-form strings)
