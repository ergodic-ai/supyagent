# Supyagent Development Sprints

This folder contains detailed sprint plans for building supyagent incrementally.

## Sprint Overview

| Sprint | Title | Duration | Focus |
|--------|-------|----------|-------|
| [1](sprint_1_foundation.md) | Foundation | 3-4 days | Project setup, LiteLLM integration, basic agent loop |
| [2](sprint_2_sessions.md) | Sessions | 2-3 days | Persistent sessions, JSONL storage, resume/switch |
| [3](sprint_3_repl.md) | Interactive REPL | 3-4 days | Meta-commands, credential prompting, rich UI |
| [4](sprint_4_execution.md) | Execution Mode | 2-3 days | Non-interactive agents, batch processing, pipelines |
| [5](sprint_5_multiagent.md) | Multi-Agent | 4-5 days | Agent delegation, planning agent, orchestration |
| [6](sprint_6_polish.md) | Polish | 3-4 days | Streaming, error handling, docs, production features |
| [7](sprint_7_context_management.md) | Context Management | 3-4 days | Token counting, summarization, context window limits |
| [8](sprint_8_process_supervisor.md) | Process Supervisor | 5-6 days | Non-blocking execution, subprocess agents, process management |

**Total estimated time: ~26-33 days**

## Dependency Graph

```
Sprint 1 (Foundation)
    │
    ▼
Sprint 2 (Sessions)
    │
    ▼
Sprint 3 (REPL) ──────┐
    │                 │
    ▼                 ▼
Sprint 4 (Execution)  │
    │                 │
    └────────┬────────┘
             │
             ▼
      Sprint 5 (Multi-Agent)
             │
             ▼
      Sprint 6 (Polish)
             │
             ▼
      Sprint 7 (Context Management)
             │
             ▼
      Sprint 8 (Process Supervisor)
```

## How to Use These Sprints

1. **Read the plan first** (`../plans/initial_plan.md`) for overall architecture
2. **Start with Sprint 1** — get the foundation working end-to-end
3. **Each sprint builds on the previous** — complete acceptance criteria before moving on
4. **Test scenarios are included** — use them to validate each sprint
5. **Iterate** — these are guidelines, adapt as you learn

## Quick Links

- [Initial Architecture Plan](../plans/initial_plan.md)
- [supypowers AGENTS.md](https://github.com/ergodic-ai/supypowers/blob/main/AGENTS.md)
- [LiteLLM Documentation](https://docs.litellm.ai/docs/)
