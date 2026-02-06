# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install with dev dependencies (uses uv + hatchling)
uv pip install -e ".[dev]"

# Install with optional browser tools
uv pip install -e ".[browser]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_agent.py

# Run a specific test
pytest tests/test_agent.py::TestAgent::test_send_message -v

# Lint
ruff check .

# Lint with auto-fix
ruff check . --fix

# Format check (ruff handles both)
ruff format --check .
```

## Architecture

Supyagent is an LLM agent framework that uses [supypowers](https://github.com/ergodic-ai/supypowers) for tool execution and [LiteLLM](https://github.com/BerriAI/litellm) for multi-provider LLM access. Agents are defined in YAML files in `agents/` and tools are Python scripts in `powers/`.

### Core Loop

The `Agent` class (`supyagent/core/agent.py`) drives the central conversation loop:
1. User message -> LLM call (via `LLMClient` wrapping LiteLLM)
2. If LLM returns tool calls, execute them and feed results back
3. Repeat until LLM returns a text-only response

Two execution modes:
- **Interactive** (`Agent`): Stateful chat with session persistence, credential prompting, streaming
- **Execution** (`ExecutionRunner` in `core/executor.py`): Stateless input->output pipeline for automation; no session storage, no credential prompting

### Tool System

Tools are external Python scripts managed by supypowers. The flow:
1. `discover_tools()` runs `supypowers docs --format json` to get available tools
2. Tool definitions are converted to OpenAI function-calling format with `script__function` naming (double underscore)
3. Tool execution runs `supypowers run script:function '{args}'` as a subprocess
4. The `ProcessSupervisor` (`core/supervisor.py`) wraps execution with timeouts, auto-backgrounding, and lifecycle management

The supervisor runs on a persistent background thread with its own asyncio event loop. Sync callers use `run_supervisor_coroutine()` to bridge into it.

Three special tool types are always injected (not from supypowers):
- `request_credential` - lets the LLM ask the user for API keys
- `delegate_to_<name>` / `spawn_agent` - multi-agent delegation tools
- `list_processes` / `get_process_output` / `kill_process` - background process management

### Multi-Agent Delegation

Agents can delegate to other agents listed in their `delegates` config field. The `DelegationManager` (`core/delegation.py`) handles this:
- **Subprocess mode** (default): Child runs via `supyagent exec <agent> --task "..."` with the `ProcessSupervisor`
- **In-process mode**: Child `Agent` or `ExecutionRunner` instantiated directly

The `AgentRegistry` (`core/registry.py`) tracks parent-child relationships and enforces a max delegation depth to prevent infinite loops. Registry state persists to `.supyagent/registry.json`.

### Context Management

The `ContextManager` (`core/context_manager.py`) handles long conversations:
- Full message history always persists to disk (JSONL via `SessionManager`)
- For LLM calls, it builds a trimmed message list: system prompt + optional summary + recent messages that fit in the context budget
- Auto-summarization triggers when message count or token count exceeds configurable thresholds

### Configuration Layers

- **Agent configs**: YAML files in `agents/` parsed into `AgentConfig` (Pydantic model in `models/agent_config.py`)
- **Global config** (API keys): Encrypted with Fernet, stored in `~/.supyagent/config/`, loaded into env vars at startup via `load_config()`
- **Per-agent credentials**: Encrypted with Fernet, stored in `.supyagent/credentials/<agent>.enc`
- **Sessions**: JSONL files in `.supyagent/sessions/<agent>/<session_id>.jsonl`

### CLI

Single entry point: `supyagent` (Click-based, `supyagent/cli/main.py`). Key commands: `init`, `new`, `chat`, `run`, `list`, `batch`, `plan`, `exec` (internal), `config`, `sessions`, `agents`, `cleanup`.

### Default Tools

Bundled in `supyagent/default_tools/` and copied to user's `powers/` dir on `supyagent init`. Includes: shell, files, edit, find, search, web, browser.

## Key Conventions

- Python 3.11+ required. Ruff for linting (line-length 100, rules: E, F, I, N, W; E501 ignored).
- All models use Pydantic v2 (`models/` package).
- Tests use pytest with `pytest-asyncio` (auto mode). Fixtures in `tests/conftest.py`. Tests mock LLM responses and supypowers calls rather than making real API calls.
- Tool results follow `{"ok": bool, "data": ..., "error": ...}` convention throughout.
- The `will_create_tools` flag on an agent config causes tool-creation instructions to be appended to its system prompt, enabling the agent to write new supypowers scripts at runtime.
