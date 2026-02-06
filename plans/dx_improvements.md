# Developer Experience (DX) Improvement Plan

> Concrete improvements to make supyagent easier to set up, use, debug, and extend.

---

## Sprint A: Fail-Fast Validation & Error Messages

**Goal:** Users find out what's wrong immediately, not 3 minutes into a chat session.

### A.1 Agent Config Validation Command

Add `supyagent validate <agent>` that checks:
- YAML parses correctly
- All required fields present with valid types
- Model provider string is recognized by litellm
- Delegate names reference agents that actually exist in agents/
- Tool permission patterns are valid globs
- Credential specs reference env vars that are set (or warn if not)

Show friendly errors instead of raw Pydantic tracebacks:

```
$ supyagent validate myagent
  ✗ model.provider: "claude-bad" is not a recognized model
    Did you mean: anthropic/claude-3-5-sonnet-20241022?
  ✗ delegates: "summarizer" not found in agents/
    Available agents: planner, researcher, worker, tester
  ✓ tools, credentials, context settings OK
```

**Files:** `cli/main.py` (new command), `models/agent_config.py` (add `validate()` method)

### A.2 Wrap Pydantic Validation Errors

When `load_agent_config()` fails, catch `ValidationError` and translate to human-readable messages. Currently users see raw Pydantic output like:

```
1 validation error for AgentConfig
model -> provider
  field required (type=value_error.missing)
```

Replace with:

```
Error loading agents/myagent.yaml:
  model.provider is required — set it to your LLM model, e.g. "anthropic/claude-3-5-sonnet-20241022"
```

**Files:** `models/agent_config.py`, `cli/main.py` (wrap load calls)

### A.3 Credential Error Messages with Examples

When execution mode fails because a credential is missing, show the exact command:

```
Error: Credential GITHUB_TOKEN is required but not set.

Fix: supyagent run myagent "task" --secrets GITHUB_TOKEN=ghp_xxx
 or: supyagent run myagent "task" --secrets .env
```

**Files:** `core/credentials.py`, `core/executor.py`

---

## Sprint B: Discoverability & Help

**Goal:** Users can discover features without reading source code.

### B.1 `supyagent doctor` Command

Diagnose the entire setup in one command:

```
$ supyagent doctor
  ✓ supypowers installed (v0.3.2)
  ✓ agents/ directory found (5 agents)
  ✓ API keys configured: OPENROUTER_API_KEY
  ✗ ANTHROPIC_API_KEY not set (needed by: planner, summarizer)
  ✓ Default tools installed (8 tools)
  ✓ Sessions directory writable
  ✓ Config encryption working
```

**Files:** `cli/main.py` (new command)

### B.2 `supyagent tools` Command

Show all available tools without starting a chat:

```
$ supyagent tools
  shell__run_command     Run a shell command
  files__read_file       Read file contents
  files__write_file      Write to a file
  web__fetch_url         Fetch a URL
  ...

$ supyagent tools --agent myagent
  (filtered by myagent's tool permissions)
```

**Files:** `cli/main.py` (new command group), reuse `discover_tools()` + `filter_tools()`

### B.3 Improve `supyagent new` Templates

- Add `--model` flag to specify provider at creation time (currently hardcoded to claude-3-5-sonnet)
- Add `--from <existing-agent>` to clone and customize an existing agent
- Add comments in generated YAML explaining each section

```
$ supyagent new mybot --model openrouter/google/gemini-2.5-flash
```

**Files:** `cli/main.py` (update `new` command)

### B.4 Agent Config Schema Reference

Auto-generate a schema reference from AgentConfig's Pydantic model. Either:
- `supyagent schema` command that prints all fields with types, defaults, and descriptions
- Or a generated `docs/schema.md`

Include the undocumented but important fields:
- `limits.max_tool_calls_per_turn`
- `limits.circuit_breaker_threshold`
- `supervisor.force_background_patterns`
- `context.max_messages_before_summary`

**Files:** `cli/main.py` (new command), or script to generate docs

---

## Sprint C: Debugging & Observability

**Goal:** When something goes wrong, users can figure out why.

### C.1 Global `--debug` Flag

Add a top-level flag that sets Python logging to DEBUG:

```
$ supyagent --debug chat myagent
[DEBUG] Loaded agent config: myagent (model=openrouter/google/gemini-2.5-flash)
[DEBUG] Discovered 12 tools (3 filtered by deny list)
[DEBUG] Context limit: 128000 tokens (via litellm)
[DEBUG] LLM request: 847 tokens (system=312, messages=423, tools=4521)
[DEBUG] LLM response: 156 tokens, 2 tool calls
[DEBUG] Executing shell__run_command (timeout=30s, mode=auto)
...
```

**Files:** `cli/main.py` (add `--debug` to `@cli.group()`), add `logging.basicConfig()` call

### C.2 Token Usage Display After Each Turn

Show token stats after each chat response (optional, toggleable with `/tokens` command):

```
You> Explain this codebase
Agent> [response...]

  tokens: 1,247 in / 523 out | context: 2,891 / 128,000 (2%) | tools: 4,521

```

This helps users understand costs and when summarization will trigger.

**Files:** `cli/main.py` (chat loop), `core/agent.py` (track token counts from LLM response)

### C.3 `/debug` Chat Command

Toggle verbose mode mid-session without restarting:

```
/debug on    — Show tool inputs/outputs, LLM token counts, context status
/debug off   — Normal mode
```

**Files:** `cli/main.py` (add to command handler)

### C.4 Log to File

Write structured logs to `~/.supyagent/logs/` so users can share them for debugging:

```
$ supyagent chat myagent --log-file debug.log
```

Or always log to `~/.supyagent/logs/<date>.log` when `--debug` is on.

**Files:** `cli/main.py`, logging config

---

## Sprint D: Session UX

**Goal:** Sessions are easy to find, manage, and understand.

### D.1 Short Session IDs

Current IDs are 8-char hex UUIDs (e.g., `a1b2c3d4`). This works but isn't memorable. Options:
- Show truncated IDs in list (first 4 chars) with disambiguation
- Allow matching by prefix: `supyagent chat agent --session a1b2`
- Or use human-readable IDs: `chat-2024-01-15-1`

**Files:** `models/session.py`, `core/session_manager.py`, `cli/main.py`

### D.2 Delete Individual Sessions from CLI

Currently only `--delete-all` exists. Add single-session deletion:

```
$ supyagent sessions myagent --delete a1b2c3d4
  ✓ Deleted session a1b2c3d4 ("How to cook pasta...")
```

**Files:** `cli/main.py` (add `--delete` option to sessions command)

### D.3 Show Message Count and Size in Session List

```
$ supyagent sessions myagent
  ID        Title                    Messages  Updated
  a1b2c3d4  How to cook pasta        42        2024-01-15 10:30  ← current
  e5f6g7h8  Python debugging tips    8         2024-01-14 16:22
```

**Files:** `core/session_manager.py` (add message_count to `list_sessions`), `cli/main.py`

### D.4 `/rename` Chat Command

Rename the current session:

```
/rename My debugging session
  ✓ Session renamed to "My debugging session"
```

**Files:** `cli/main.py` (add to command handler), `core/session_manager.py`

---

## Sprint E: Tool Authoring DX

**Goal:** Creating custom tools is obvious and well-guided.

### E.1 `supyagent tools new <name>` Scaffolding

Generate a tool template with the correct structure:

```
$ supyagent tools new github_api
  ✓ Created supypowers/github_api.py

  Next: Edit the file and add your functions.
  The tool will be available as github_api__<function_name>
```

**Files:** `cli/main.py` (new subcommand), template string

### E.2 `supyagent tools test <script> <function>`

Test a tool outside of an agent:

```
$ supyagent tools test shell run_command '{"command": "echo hello"}'
  ✓ ok: true
  data: "hello\n"
  duration: 42ms
```

**Files:** `cli/main.py` (new subcommand), reuse `execute_tool()`

### E.3 Tool Permission Documentation

Document the glob pattern matching clearly in help text and README:

```
tools:
  allow:
    - "*"              # All tools
    - "shell:*"        # All functions in shell.py
    - "files:read_*"   # Only read functions in files.py
  deny:
    - "shell:run_command"  # Block dangerous commands
```

**Files:** README, agent template comments

---

## Priority Order

| Sprint | Impact | Effort | Ship First? |
|--------|--------|--------|-------------|
| A: Validation | High (prevents frustration) | Medium | Yes |
| B: Discoverability | High (reduces learning curve) | Medium | Yes |
| C: Debugging | Medium (helps power users) | Low-Medium | Second |
| D: Session UX | Medium (quality of life) | Low | Second |
| E: Tool Authoring | Medium (helps extenders) | Low | Third |

**Suggested execution order:** A.1 → B.1 → B.2 → A.2 → C.1 → D.2 → B.3 → rest.

Start with `validate` and `doctor` — these are the two commands that help users the most when something isn't working.
