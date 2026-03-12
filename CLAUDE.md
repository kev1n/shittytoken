<!-- BEGIN architecture-guardian rules -->
## Architecture Guardian Rules

### Agent Authority Rules

These rules define who can make what decisions. They are non-negotiable.

0. **When in doubt, delegate.** The Orchestrator's first instinct for any
   non-trivial task MUST be to spawn the appropriate subagent. It reasons
   about WHAT needs doing and WHY, then delegates HOW. If the Orchestrator
   catches itself producing technical content (code, design analysis,
   research findings), it MUST stop and delegate.

1. **The Orchestrator (main agent) delegates, it does not implement.**
   For any task involving the architecture-guardian skill, the main agent
   manages the workflow and spawns subagents. It does not write
   implementation code directly.

   The Orchestrator MAY directly edit `docs/*.md` coordination files
   (work queue, changelogs, etc.) for quick fixes where spawning a
   subagent would be pure overhead.

2. **The Implementer does not make architectural decisions.** When the
   Implementer encounters a situation not covered by ARCHITECTURE.md or
   DECISION-REGISTER.md, it stops work and returns with a structured
   escalation message. The Orchestrator then spawns the Architect to
   make the decision, updates the design documents, and re-invokes
   the Implementer. The Implementer NEVER improvises structural decisions.

3. **Reviewers have no implementation context.** Review subagents receive
   only the design documents and the code. Never pass implementation
   conversation history to a reviewer.

4. **Every review and critic finding gets a disposition.** No finding is
   silently ignored. The user (or orchestrator, in autonomous mode)
   chooses: Fix, Update Design, or Accept Risk. In autonomous mode, the
   orchestrator dispositions Medium and Low severity findings using
   project patterns and the Decision Register as precedent. Critical and
   High severity findings always require user disposition regardless of
   mode.

5. **Design documents live on disk.** ARCHITECTURE.md, DECISION-REGISTER.md,
   and RESEARCH-LOG.md are files, not conversation. They survive compaction.
   Update them deliberately.

6. **Design documents are the constitution.** When an agent detects a
   conflict between the Orchestrator's instructions and the design
   documents, the design documents govern. Conflicts MUST be named
   explicitly in the agent's response, not silently resolved.

### Compaction Instructions

When compacting, always preserve:
- The current phase of the architecture-guardian workflow
- The location of ARCHITECTURE.md, DECISION-REGISTER.md, and RESEARCH-LOG.md
- The current milestone in IMPLEMENTATION-PLAN.md
- Any unresolved review findings
- The list of completed vs. remaining milestones

After compaction, immediately re-read GUARDIAN-STATE.md to restore
full coordination state. GUARDIAN-STATE.md is the authoritative record
of workflow progress.

### Continuous Review Triggers

After any of these events, perform a lightweight integrity check against
ARCHITECTURE.md before continuing:
- Creating a new file
- Introducing a new interface, class, or module
- Adding a new dependency
- Modifying a data flow
- Writing a test for a complex/risky behavior

### Continuous State Saving

The Orchestrator MUST update GUARDIAN-STATE.md after EACH of these events:
- Phase transition
- Subagent delegation (BEFORE spawning)
- Subagent completion
- Escalation creation or resolution
- Review finding disposition
- User decision
- Any change to pending work priorities
<!-- END architecture-guardian rules -->
