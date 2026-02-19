# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_service.py

# Lint
ruff check .

# Lint with auto-fix
ruff check . --fix
```

## Architecture

Supyagent is a cloud CLI that connects AI agents to third-party services (Gmail, Slack, GitHub, Calendar, etc.) via the supyagent service API. Users authenticate with `supyagent connect`, then use cloud tools from the CLI or generate skill files for AI coding assistants.

### Source Files

| File | Purpose |
|------|---------|
| `supyagent/cli/main.py` | CLI entry point — all commands (Click-based) |
| `supyagent/cli/skills.py` | SKILL.md file generation for AI coding assistants |
| `supyagent/core/config.py` | `ConfigManager` — Fernet-encrypted key storage in `~/.supyagent/config/` |
| `supyagent/core/service.py` | `ServiceClient` — HTTP client for supyagent service API, device auth flow |
| `supyagent/utils/binary.py` | Binary content materialization (PDFs, images from API responses) |

### CLI Commands

| Command | Purpose |
|---------|---------|
| `supyagent connect` | Device authorization flow to authenticate with the service |
| `supyagent disconnect` | Remove stored credentials |
| `supyagent status` | Show connection status and available tools |
| `supyagent inbox` | View/manage webhook events from connected integrations |
| `supyagent doctor` | Diagnose setup (connectivity, config, tools) |
| `supyagent config set/list/delete/import/export` | Manage encrypted API keys |
| `supyagent service tools` | List available cloud tools |
| `supyagent service run <tool> '<json>'` | Execute a cloud tool directly |
| `supyagent service inbox` | Detailed inbox management |
| `supyagent skills generate` | Generate SKILL.md files for AI coding assistants |

### How It Works

1. `supyagent connect` runs a device auth flow against the supyagent service (`https://app.supyagent.com`)
2. The API key is stored encrypted in `~/.supyagent/config/`
3. `ServiceClient` uses the key to call `GET /api/v1/tools` (discovery) and execute tools via HTTP
4. `skills generate` transforms discovered tools into SKILL.md files that Claude Code, Cursor, etc. can use

### Dependencies

Only 4 runtime dependencies: `click`, `rich`, `cryptography`, `httpx`.

## Key Conventions

- Python 3.11+ required. Ruff for linting (line-length 100, rules: E, F, I, N, W; E501 ignored).
- Tests use pytest with `pytest-asyncio` (auto mode). Tests mock HTTP calls rather than making real API calls.
- Tool results follow `{"ok": bool, "data": ..., "error": ...}` convention.
- The full agent framework (LLM loop, supypowers, multi-agent delegation, sessions, memory) lives on the `dev` branch.
