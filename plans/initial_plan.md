# Supyagent — Initial Architecture Plan

## Vision

**Supyagent** is an LLM agent framework that uses [supypowers](https://github.com/ergodic-ai/supypowers) as its tool execution layer and [LiteLLM](https://docs.litellm.ai/docs/) for multi-provider LLM support.

The system supports two agent modes:
- **Interactive Agents**: Session-oriented REPL with rich user interaction
- **Execution Agents**: Stateless input→output pipelines for automation

Agents can create and invoke other agents, enabling a **multi-agent architecture** with a central planning agent that orchestrates specialized sub-agents.

---

## Core Principles

1. **CLI-first**: Mirror the simplicity of `supypowers` commands
2. **Persistent Sessions**: All conversations are saved and resumable
3. **Credential Prompting**: LLM can request API keys/secrets from users at runtime
4. **Composable Agents**: Agents can spawn and delegate to other agents
5. **Provider Agnostic**: Use any LLM via LiteLLM (OpenAI, Anthropic, Ollama, etc.)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         supyagent CLI                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │ Interactive │    │  Execution  │    │   Planning Agent    │ │
│  │    REPL     │    │    Mode     │    │   (Orchestrator)    │ │
│  └──────┬──────┘    └──────┬──────┘    └──────────┬──────────┘ │
│         │                  │                      │             │
│         └──────────────────┴──────────────────────┘             │
│                            │                                    │
│  ┌─────────────────────────┴─────────────────────────────────┐ │
│  │                    Agent Runtime                           │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌───────────────────┐  │ │
│  │  │   Session   │  │  Credential │  │   Agent Registry  │  │ │
│  │  │   Manager   │  │   Manager   │  │   (Multi-Agent)   │  │ │
│  │  └─────────────┘  └─────────────┘  └───────────────────┘  │ │
│  └───────────────────────────────────────────────────────────┘ │
│                            │                                    │
│  ┌─────────────────────────┴─────────────────────────────────┐ │
│  │                    Integration Layer                       │ │
│  │  ┌─────────────────────┐    ┌───────────────────────────┐ │ │
│  │  │      LiteLLM        │    │       supypowers          │ │ │
│  │  │  (LLM Provider)     │    │    (Tool Execution)       │ │ │
│  │  └─────────────────────┘    └───────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
project/
├── supypowers/                  # Tool definitions (Python functions)
│   ├── web_search.py
│   ├── file_ops.py
│   └── ...
│
├── agents/                      # Agent definitions (YAML manifests)
│   ├── planner.yaml            # Built-in orchestrator agent
│   ├── researcher.yaml
│   ├── coder.yaml
│   └── _template.yaml          # Template for `supyagent new`
│
├── .supyagent/                  # Runtime state (gitignored)
│   ├── sessions/               # Conversation history
│   │   └── <agent>/
│   │       ├── <session_id>.jsonl
│   │       └── current.json    # Pointer to active session
│   ├── credentials/            # Encrypted credential store
│   │   └── <agent>.enc
│   └── registry.json           # Active agent instances
│
└── supyagent.toml              # Global configuration (optional)
```

---

## Agent Definition Schema

```yaml
# agents/researcher.yaml
name: researcher
description: A research assistant that finds and synthesizes information
version: "1.0"

# Agent type: "interactive" or "execution"
type: interactive

# LLM configuration (LiteLLM format)
model:
  provider: anthropic/claude-3-5-sonnet-20241022
  temperature: 0.7
  max_tokens: 4096

# System prompt
system_prompt: |
  You are a meticulous research assistant. You verify facts,
  cite sources, and present balanced viewpoints.
  
  When you need API keys or credentials to use a tool, you can
  request them from the user using the `request_credential` function.

# Tool permissions
tools:
  allow:
    - web_search:*          # All functions in web_search.py
    - summarize:summarize   # Specific function
  deny:
    - dangerous:*           # Explicit blocklist

# Credentials this agent may request
credentials:
  - name: SEARCH_API_KEY
    description: "API key for web search service"
    required: false

# Sub-agents this agent can invoke
delegates:
  - summarizer
  - fact_checker

# Resource limits
limits:
  max_tool_calls_per_turn: 10
  max_total_tool_calls: 100
  max_tokens_per_session: 100000
```

---

## Session Format

Sessions are stored as JSONL (one JSON object per line):

```jsonl
{"type": "meta", "session_id": "abc123", "agent": "researcher", "created_at": "2024-01-15T10:00:00Z", "model": "anthropic/claude-3-5-sonnet-20241022"}
{"type": "user", "content": "Find papers on transformer architectures", "ts": "2024-01-15T10:00:05Z"}
{"type": "assistant", "content": "I'll search for recent papers...", "tool_calls": [{"id": "tc1", "name": "web_search:search", "arguments": {"query": "transformer architecture papers 2024"}}], "ts": "2024-01-15T10:00:07Z"}
{"type": "tool_result", "tool_call_id": "tc1", "name": "web_search:search", "result": {"ok": true, "data": [...]}, "ts": "2024-01-15T10:00:09Z"}
{"type": "credential_request", "name": "SEARCH_API_KEY", "description": "API key for web search", "ts": "2024-01-15T10:00:10Z"}
{"type": "credential_response", "name": "SEARCH_API_KEY", "provided": true, "ts": "2024-01-15T10:00:15Z"}
{"type": "assistant", "content": "I found 5 relevant papers...", "ts": "2024-01-15T10:00:12Z"}
```

---

## CLI Commands

### Agent Management
```bash
supyagent new <name>                    # Create new agent from template
supyagent list                          # List all defined agents
supyagent show <agent>                  # Show agent details + available tools
supyagent delete <agent>                # Delete an agent
```

### Interactive Mode
```bash
supyagent chat <agent>                  # Start/resume interactive session
supyagent chat <agent> --new            # Force new session
supyagent chat <agent> --session <id>   # Resume specific session
```

### Execution Mode
```bash
supyagent run <agent> "<task>"          # One-shot execution
supyagent run <agent> --input file.md   # Task from file
supyagent run <agent> --output json     # Output format: json|markdown|raw
supyagent run <agent> < input.txt       # Pipe input
```

### Session Management
```bash
supyagent sessions <agent>              # List sessions for agent
supyagent sessions <agent> --delete <id> # Delete a session
supyagent export <agent> <session_id>   # Export session as markdown
```

### REPL Meta-Commands
```
/help                    # Show available commands
/history [n]             # Show last n messages
/save [name]             # Save checkpoint
/load <name>             # Load checkpoint
/sessions                # List all sessions
/switch <session_id>     # Switch to different session
/new                     # Start fresh session
/tools                   # List available tools
/model [provider/model]  # Show or change model
/clear                   # Clear screen
/export [file]           # Export conversation
/quit                    # Exit (alias: /exit, /q)
```

---

## Credential Prompting System

When an LLM needs a credential it doesn't have:

1. **LLM calls `request_credential` tool** with name and description
2. **Runtime intercepts** and prompts user in REPL:
   ```
   🔑 The agent is requesting a credential:
      Name: SLACK_API_TOKEN
      Purpose: To send messages to your Slack workspace
      
   Enter value (or 'skip' to decline): ▊
   ```
3. **User provides value** (input hidden like password)
4. **Runtime stores encrypted** in `.supyagent/credentials/<agent>.enc`
5. **Value injected** as environment variable for tool execution
6. **Persisted for future sessions** (optional, user-controlled)

For execution mode, credentials must be pre-configured or passed via `--secrets`.

---

## Multi-Agent Architecture

### Agent Invocation

Agents can invoke other agents as tools:

```yaml
# agents/planner.yaml
delegates:
  - researcher      # Can invoke researcher agent
  - coder          # Can invoke coder agent
  - reviewer       # Can invoke reviewer agent
```

The planner sees these as tools:
```json
{
  "name": "delegate_to_researcher",
  "description": "Delegate a research task to the researcher agent",
  "parameters": {
    "task": "string - The research task to perform",
    "context": "string - Optional context from current conversation"
  }
}
```

### Planning Agent

A special orchestrator agent that:
1. Receives high-level tasks
2. Breaks them into subtasks
3. Delegates to specialized agents
4. Synthesizes results

```yaml
# agents/planner.yaml
name: planner
type: interactive
description: Orchestrates complex tasks by delegating to specialized agents

system_prompt: |
  You are a planning agent. Your job is to:
  1. Understand the user's goal
  2. Break it into subtasks
  3. Delegate to the right specialist agents
  4. Synthesize their outputs into a coherent result
  
  Available specialists:
  - researcher: Finding and summarizing information
  - coder: Writing and reviewing code
  - reviewer: Quality checking and feedback

delegates:
  - researcher
  - coder  
  - reviewer
```

### Execution Flow

```
User → Planner → [researcher, coder, reviewer] → Planner → User
         │              │                            │
         │              └── Independent execution ───┘
         │                   (parallel when possible)
         │
         └── Maintains overall context and goal
```

---

## Configuration

### Global Config (`supyagent.toml`)

```toml
[defaults]
model = "anthropic/claude-3-5-sonnet-20241022"
temperature = 0.7

[credentials]
# Global credentials available to all agents
encryption_key_env = "SUPYAGENT_KEY"  # Or generate on first run

[storage]
sessions_dir = ".supyagent/sessions"
credentials_dir = ".supyagent/credentials"

[limits]
max_tool_calls_per_turn = 20
max_session_tokens = 500000
```

### Environment Variables

```bash
# LLM providers (LiteLLM format)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_API_BASE=http://localhost:11434

# Supyagent
SUPYAGENT_KEY=...           # For credential encryption
SUPYAGENT_DEFAULT_MODEL=... # Override default model
```

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| CLI Framework | `click` or `typer` |
| LLM Integration | `litellm` |
| Tool Execution | `supypowers` (subprocess) |
| Data Validation | `pydantic` |
| Config Parsing | `pyyaml`, `tomli` |
| Credential Storage | `cryptography` (Fernet) |
| Rich Terminal | `rich` |

---

## Success Criteria

### Phase 1 (MVP)
- [ ] Basic agent loop with LiteLLM
- [ ] Tool execution via supypowers
- [ ] Session persistence (JSONL)
- [ ] Interactive REPL with basic commands

### Phase 2 (Core Features)
- [ ] Credential prompting system
- [ ] Execution mode (non-interactive)
- [ ] Session management commands
- [ ] Agent YAML schema validation

### Phase 3 (Multi-Agent)
- [ ] Agent-to-agent delegation
- [ ] Planning agent template
- [ ] Parallel sub-agent execution
- [ ] Context passing between agents

### Phase 4 (Polish)
- [ ] Rich terminal UI
- [ ] Streaming responses
- [ ] Memory/learning system
- [ ] Documentation & examples

---

## Open Questions

1. **Credential sharing**: Should credentials be per-agent or global?
2. **Session branching**: Support "what-if" conversation forks?
3. **Cost tracking**: Integrate LiteLLM's cost tracking?
4. **Plugins**: Allow custom Python extensions beyond supypowers?
5. **Web UI**: Future consideration for visual session management?

---

## References

- [supypowers AGENTS.md](https://github.com/ergodic-ai/supypowers/blob/main/AGENTS.md)
- [LiteLLM Documentation](https://docs.litellm.ai/docs/)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
