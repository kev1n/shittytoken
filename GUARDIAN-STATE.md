# Guardian State

> This file tracks the current state of the architecture-guardian
> workflow. It is the Orchestrator's recovery mechanism after context
> compaction. The Orchestrator MUST update this file after every phase
> transition, milestone completion, escalation resolution, and review
> disposition.
>
> After compaction, re-read this file immediately to restore state.

## Grounding Statement

> ShittyToken exists to deliver the cheapest possible LLM inference tokens
> by orchestrating spot GPU instances through an AI agent that handles the
> full lifecycle autonomously.
> It optimizes for tokens-per-dollar (throughput), rapid iteration, and
> scalability of the worker fleet.
> It explicitly does NOT optimize for TTFT, provide SLAs, or require human
> intervention in the provisioning/recovery loop.
> The user's core intent is: build a bulk inference platform that undercuts
> every competitor on price by treating latency as a non-goal.
> Success means: the agent can procure, qualify, serve, recover, and
> decommission GPU workers with zero human input, and the cost per million
> tokens is lower than any managed alternative.

**Established:** Phase 1a, 2026-03-12
**Last confirmed:** 2026-03-12

## Current Phase

**Phase:** 1a
**Started:** 2026-03-12

## Design Document Locations

- ARCHITECTURE.md: /Users/kdawg/Documents/shittytoken/ARCHITECTURE.md
- DECISION-REGISTER.md: (not yet created)
- RESEARCH-LOG.md: (not yet created)
- IMPLEMENTATION-PLAN.md: (not yet created)

## Milestones

| # | Name | Status | Branch/Commit | Notes |
|---|------|--------|---------------|-------|
| 1 | Architecture doc approved | In Progress | - | Awaiting user review of ARCHITECTURE.md |
| 2 | Knowledge graph schema designed | Not Started | - | |
| 3 | Orchestration agent MVP | Not Started | - | Core provisioning + SSH + health loop |
| 4 | Qualification benchmark | Not Started | - | aiohttp + metrics correlation |
| 5 | Gateway integration | Not Started | - | vLLM Router + Nginx config |
| 6 | OOM recovery flow | Not Started | - | Neo4j queries + LLM reasoning prompt |
| 7 | End-to-end demo | Not Started | - | Single model, single provider |

## Phase Transitions

| From | To | Reason | User Approved | Date |
|------|-----|--------|--------------|------|
| - | 1a | Session start / setup | - | 2026-03-12 |
| 1a | 1b | Tier classification confirmed | [ ] | |
| 1b | 2 | Requirements summary approved | [ ] | |
| 2 | 3 | Architecture chosen | [ ] | |
| 3 | 4 | Attack vectors dispositioned | [ ] | |
| 4 | 5 | Design docs approved | [ ] | |
| 5 | 6 | Implementation plan approved | [ ] | |

## Pending Escalations

| Escalation | From | Status | Assigned To |
|-----------|------|--------|-------------|
| None | - | - | - |

## Provisional Decisions

| Decision | Milestone | Status | Resolution Deadline |
|----------|-----------|--------|-------------------|
| None | - | - | - |

## Unresolved Review Findings

| Finding | Reviewer | Severity | Disposition | Status |
|---------|----------|----------|-------------|--------|
| None | - | - | - | - |

## Orchestrator Delegation Log

| Task | Delegated To | Rationale (if NOT delegated) |
|------|-------------|---------------------------|
| Architecture doc generation | Implementer (setup) | Initial scaffold from user plan |

## Scope Classification (Phase 1a)

**Confirmed Tier:** 3
**Tier Override:** None

| Signal | Value | Triggered? |
|--------|-------|------------|
| File count | 0 (greenfield) | No |
| External integrations/APIs | 4 (Vast.ai, RunPod, Neo4j, vLLM) | Yes |
| Explicit constraints mentioned | 3 (no SLAs, spot-only, throughput-first) | Yes |
| Existing systems to preserve | 0 | No |
| Multi-session task | Yes | Yes |

## Skip Audit

**Skip recorded:** No
**Skip-time date:** N/A
**Phases skipped:** None

| Signal | Value at Skip-Time |
|--------|-------------------|
| File count | N/A |
| External integrations/APIs | N/A |
| Explicit constraints | N/A |
| Existing systems | N/A |
| Multi-session | N/A |

**Skip count:** 0

## North Star Assertions

**North Star (prose):** ShittyToken delivers the cheapest possible LLM inference by autonomously orchestrating spot GPU workers end-to-end — procurement, configuration, qualification, serving, recovery, and decommission — optimizing every decision for tokens-per-dollar rather than latency.

| # | Assertion | User Approved | Status |
|---|-----------|--------------|--------|
| 1 | Every routing decision optimizes batch utilization (throughput), not TTFT | [ ] | Active |
| 2 | The agent can provision a new vLLM worker end-to-end without human input | [ ] | Active |
| 3 | No instance enters the gateway's worker pool without passing the qualification benchmark | [ ] | Active |
| 4 | All OOM recovery attempts are recorded in the knowledge graph regardless of outcome | [ ] | Active |
| 5 | The agent never hard-codes GPU/model configurations; all configs derive from the knowledge graph | [ ] | Active |

## QA Priority Tracking

**qa_version:** 1

| Rank | Quality Attribute | Original Rank |
|------|------------------|---------------|
| 1 | Scalability | 1 |
| 2 | Maintainability | 2 |
| 3 | Rapid Iteration | 3 |
| 4 | Correctness | 4 |
| 5 | Performance (throughput) | 5 |

## Trivial Change Accumulator

**Counter:** 0
**Last reset date:** 2026-03-12

## Convergence Tracker

| Round | Critical | High | Medium | Low | New | Resolved |
|-------|----------|------|--------|-----|-----|----------|

## Mode

**Operating Mode:** Interrogative
**Last User Gate:** Phase 1a, 2026-03-12
