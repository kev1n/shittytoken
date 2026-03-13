# ShittyToken Production Architecture

## System Overview

```mermaid
graph TB
    subgraph Users["User-Facing Layer"]
        Client["Client SDKs<br/>(OpenAI-compatible)"]
        WebUI["Web UI<br/>:8080"]
        Stripe["Stripe<br/>(Payments)"]
    end

    subgraph Gateway["API Gateway (:8001)"]
        direction TB
        ReqID["Request ID<br/>Middleware"]
        Auth["Auth Middleware<br/>(Bearer Token)"]
        Proxy["SSE Proxy<br/>(streaming pass-through)"]
        AdminAPI["Admin API<br/>/admin/workers"]
        Metrics["/metrics<br/>(Prometheus)"]
        Models["/v1/models"]

        ReqID --> Auth --> Proxy
    end

    subgraph Routing["Routing Layer"]
        Policy["CacheAwarePolicy<br/>(consistent hashing)"]
        WorkerPool["WorkerPool<br/>(health + metrics scraper)"]
    end

    subgraph Workers["vLLM Workers (Spot GPU Instances)"]
        W1["vLLM Worker 1<br/>RTX 3090 x2"]
        W2["vLLM Worker 2<br/>RTX 3090 x2"]
        WN["vLLM Worker N<br/>(auto-scaled)"]
    end

    subgraph Orchestrator["Orchestrator (Control Plane)"]
        direction TB
        MainLoop["Main Loop<br/>(demand detect + scaling)"]
        Provision["Provisioner<br/>(Vast.ai / RunPod)"]
        Qualify["Qualification<br/>(benchmark + verify)"]
        Health["HeartbeatMonitor<br/>(consecutive failure threshold)"]
        DeathMon["Instance Death Monitor<br/>(provider polling)"]
        StuckSweep["Stuck Instance Sweep<br/>(timeout cleanup)"]
        CostTrack["CostTracker<br/>($/hr burn rate)"]
        StateMachine["State Machine<br/>(PROVISIONING → SERVING → DRAINING)"]

        MainLoop --> Provision
        MainLoop --> StuckSweep
        Provision --> Qualify
        Qualify --> Health
    end

    subgraph Billing["Billing Pipeline"]
        direction TB
        Pipeline["BillingPipeline<br/>(publish → consume → deduct)"]
        Reconciler["Reconciler<br/>(periodic Redis↔PG sync)"]
    end

    subgraph DataStores["Data Stores"]
        Redis[("Redis 7<br/>Balance cache<br/>Rate limits<br/>API key cache<br/>State store<br/>Sessions")]
        Postgres[("PostgreSQL 16<br/>Users, API keys<br/>Credit blocks (FIFO)<br/>Ledger (append-only)<br/>Usage events")]
        Neo4j[("Neo4j<br/>GPU catalog<br/>LLM configs<br/>Benchmark results<br/>OOM events")]
    end

    subgraph Monitoring["Observability"]
        Prometheus["Prometheus"]
        Grafana["Grafana Dashboard"]
    end

    %% User-facing connections
    Client -->|"POST /v1/chat/completions"| Gateway
    WebUI -->|"Signup, Top-up, API Keys"| Postgres
    WebUI -->|"Stripe Checkout"| Stripe
    Stripe -->|"Webhook"| WebUI

    %% Gateway internals
    Proxy --> Policy
    Policy --> WorkerPool
    WorkerPool -->|"route request"| Workers
    Proxy -->|"usage event"| Pipeline

    %% Auth hot path
    Auth -->|"key lookup + rate limit"| Redis
    Auth -->|"fallback key lookup"| Postgres

    %% Billing
    Pipeline -->|"FIFO deduct"| Postgres
    Pipeline -->|"update balance"| Redis
    Reconciler -->|"read balance"| Postgres
    Reconciler -->|"sync balance"| Redis

    %% Orchestrator connections
    MainLoop -->|"scrape /metrics"| Workers
    Health -->|"poll /health"| Workers
    DeathMon -->|"poll provider API"| Provision
    Provision -->|"rent spot GPU"| CloudAPIs["Vast.ai / RunPod API"]
    MainLoop -->|"register/deregister"| AdminAPI
    StateMachine -->|"persist state"| Redis
    CostTrack -->|"log $/hr burn"| Monitoring

    %% Knowledge graph
    Qualify -->|"read configs"| Neo4j
    Qualify -->|"write benchmarks"| Neo4j

    %% Monitoring
    Metrics --> Prometheus
    Prometheus --> Grafana
    WorkerPool -->|"scrape worker /metrics"| Workers

    %% Styling
    classDef store fill:#1a1a2e,stroke:#00ff88,color:#fff
    classDef service fill:#16213e,stroke:#4cc9f0,color:#fff
    classDef external fill:#0f3460,stroke:#e94560,color:#fff
    classDef user fill:#1a1a2e,stroke:#f72585,color:#fff

    class Redis,Postgres,Neo4j store
    class Gateway,Orchestrator,Billing,Routing service
    class Stripe,CloudAPIs external
    class Client,WebUI user
```

## Data Flow: Chat Completion Request

```mermaid
sequenceDiagram
    participant C as Client
    participant GW as Gateway
    participant R as Redis
    participant PG as Postgres
    participant W as vLLM Worker
    participant BP as Billing Pipeline

    C->>GW: POST /v1/chat/completions<br/>Authorization: Bearer sk-xxx

    Note over GW: Request ID Middleware
    Note over GW: Auth Middleware

    GW->>R: GET apikey:{hash}
    R-->>GW: {user_id, rate_limits}

    GW->>R: ZCARD ratelimit:rpm:{hash}
    R-->>GW: count < limit ✓

    GW->>R: GET balance:{user_id}
    R-->>GW: 5000 cents ✓

    GW->>R: ZADD ratelimit:rpm:{hash}

    Note over GW: CacheAwarePolicy<br/>SHA256(system_prompt + first_msg)<br/>→ consistent hash → worker

    GW->>W: POST /v1/chat/completions<br/>{stream: true, stream_options: {include_usage: true}}

    loop SSE Streaming
        W-->>GW: data: {"choices":[...]}
        GW-->>C: data: {"choices":[...]}
    end

    W-->>GW: data: {"usage":{"prompt_tokens":150,"completion_tokens":80}}
    GW-->>C: data: [DONE]

    GW->>BP: publish_usage(user_id, 150, 80)

    Note over BP: Async processing

    BP->>PG: INSERT usage_events
    BP->>PG: deduct_credits_fifo(user_id, cost)
    BP->>R: SET balance:{user_id} new_balance
```

## Instance Lifecycle State Machine

```mermaid
stateDiagram-v2
    [*] --> PROVISIONING: rent spot GPU

    PROVISIONING --> BENCHMARKING: SSH ready + vLLM loaded
    PROVISIONING --> FAILED: timeout / error

    BENCHMARKING --> SERVING: benchmark passed
    BENCHMARKING --> FAILED: benchmark failed

    SERVING --> DRAINING: scale down / health failure / shutdown
    SERVING --> FAILED: instance death detected

    DRAINING --> TERMINATED: drain complete + destroyed

    FAILED --> [*]: cleaned up
    TERMINATED --> [*]: cleaned up

    note right of PROVISIONING
        Stuck sweep: destroyed after
        stuck_instance_timeout_s (1320s)
    end note

    note right of SERVING
        HeartbeatMonitor: /health every 30s
        Death Monitor: provider API every 15s
        Cost Tracker: $/hr burn rate
    end note

    note right of DRAINING
        drain=True: wait for in-flight
        requests to complete
    end note
```

## Scaling Decision Flow

```mermaid
flowchart TD
    Tick["_tick() every 10s"] --> CheckServing{Any SERVING<br/>workers?}

    CheckServing -->|No| Cooldown{In cooldown?}
    Cooldown -->|Yes| Wait["Wait remaining_s"]
    Cooldown -->|No| ProvisionMin["Provision to<br/>min_workers"]

    CheckServing -->|Yes| Scrape["Scrape aggregate<br/>metrics"]

    Scrape --> ScaleUp{requests_waiting ><br/>threshold?}
    ScaleUp -->|Yes| Provision["_guarded_provision()<br/>(lock serialized)"]

    ScaleUp -->|No| ScaleDown{Worker idle ><br/>300s AND<br/>kv_cache < 50%?}
    ScaleDown -->|Yes| Drain["Drain + destroy<br/>least-utilized"]
    ScaleDown -->|No| Sweep

    Provision --> Sweep
    Drain --> Sweep

    Sweep["_sweep_stuck_instances()"] --> CostLog["cost_tracker<br/>.maybe_log_summary()"]
```

## Billing Architecture

```mermaid
flowchart LR
    subgraph HotPath["Hot Path (< 1ms)"]
        direction TB
        AuthMW["Auth Middleware"]
        RedisOps["Redis:<br/>• API key cache (5min TTL)<br/>• RPM sliding window<br/>• TPM bucket counter<br/>• Balance atomic read"]
    end

    subgraph ColdPath["Cold Path (async)"]
        direction TB
        UsagePub["Usage Publisher<br/>(in-process queue<br/>or Kafka)"]
        UsageCon["Usage Consumer"]
        FIFODeduct["FIFO Credit Deduction<br/>(SELECT ... FOR UPDATE<br/>oldest blocks first)"]
    end

    subgraph Reconciliation["Reconciliation (every 60s)"]
        direction TB
        Recon["Reconciler"]
        Expire["Expire credit blocks<br/>past expires_at"]
        Sync["Sync Redis balance<br/>to match Postgres"]
    end

    subgraph TopUp["Top-Up Flow"]
        direction TB
        WebForm["Web UI: $5/$10/$25/$50"]
        StripeCheckout["Stripe Checkout"]
        Webhook["Webhook handler"]
        CreditBlock["Create CreditBlock<br/>+ LedgerEvent"]
        CacheUpdate["INCRBY Redis balance"]
    end

    AuthMW --> RedisOps
    RedisOps -->|"cache miss"| PG[("Postgres")]

    UsagePub --> UsageCon
    UsageCon --> FIFODeduct
    FIFODeduct --> PG
    UsageCon -->|"update"| Redis[("Redis")]

    Recon --> PG
    Recon --> Redis

    WebForm --> StripeCheckout
    StripeCheckout --> Webhook
    Webhook --> CreditBlock
    CreditBlock --> PG
    Webhook --> CacheUpdate
    CacheUpdate --> Redis
```

## Infrastructure Topology (Production)

```mermaid
graph TB
    subgraph Internet
        Users["Users / Client Apps"]
    end

    subgraph ControlPlane["Control Plane (always-on VPS)"]
        WebApp["Web UI :8080"]
        GatewayProc["Gateway :8001"]
        OrchestratorProc["Orchestrator"]
        NginxLB["Nginx :80<br/>(TLS termination)"]
    end

    subgraph Persistence["Persistence Layer"]
        PG[("PostgreSQL 16<br/>Billing ledger")]
        Redis[("Redis 7<br/>Cache + state")]
        Neo4j[("Neo4j<br/>Knowledge graph")]
    end

    subgraph SpotGPUs["Spot GPU Instances (elastic)"]
        GPU1["Vast.ai<br/>RTX 3090 x2<br/>vLLM :8080"]
        GPU2["Vast.ai<br/>RTX 3090 x2<br/>vLLM :8080"]
        GPUN["...more as demand grows"]
    end

    subgraph External["External Services"]
        StripeAPI["Stripe API"]
        VastAPI["Vast.ai API"]
        RunPodAPI["RunPod API"]
        HF["HuggingFace<br/>(model weights)"]
    end

    subgraph Observability["Observability"]
        Prom["Prometheus"]
        Graf["Grafana"]
    end

    Users --> NginxLB
    NginxLB --> GatewayProc
    NginxLB --> WebApp

    GatewayProc --> GPU1
    GatewayProc --> GPU2
    GatewayProc --> GPUN

    GatewayProc --> Redis
    GatewayProc --> PG
    WebApp --> PG
    WebApp --> Redis
    WebApp --> StripeAPI

    OrchestratorProc --> VastAPI
    OrchestratorProc --> RunPodAPI
    OrchestratorProc --> Neo4j
    OrchestratorProc --> Redis
    OrchestratorProc --> GatewayProc

    GPU1 -->|"download weights"| HF

    GatewayProc -->|"/metrics"| Prom
    GPU1 -->|"/metrics"| Prom
    Prom --> Graf
```
