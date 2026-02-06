# Supyagent Improvement Plan

> Based on a comprehensive review of the entire codebase as of v0.2.7.

---

## 🔴 Priority 1: Architecture & Code Quality

### 1. Unify Agent & ExecutionRunner

`agent.py` (~630 lines) and `executor.py` (~435 lines) share ~60% of their logic — tool execution, streaming, tool-call collection, delegation setup — all copy-pasted with slight variations. This is the single biggest maintenance burden.

**What to do:**
- Extract a shared `BaseAgentEngine` class with the common loop (LLM call → collect tool calls → execute → loop)
- `Agent` adds: session persistence, credential prompting, `/commands`, context management
- `ExecutionRunner` adds: stateless pipeline mode, structured I/O
- The `ToolCallObj` inner class is redefined identically in 3 places (`agent.py` streaming, `executor.py` streaming, `executor.py` non-streaming). Extract to a shared utility.

**Impact:** Cuts ~400 lines of duplication, makes every future feature apply to both modes automatically.

---

### 2. Fix Cross-Process Depth Enforcement

Each `supyagent exec` subprocess creates a fresh `AgentRegistry()`, so `MAX_DEPTH=5` restarts at 0 for every new subprocess. An infinite delegation loop (A→B→A→B→...) is currently possible.

**What to do:**
- Pass `--depth N` flag through `supyagent exec` CLI command
- Increment depth from parent context in `_delegate_subprocess()`
- Reject execution in `exec_agent()` if depth >= MAX_DEPTH
- Store depth in `DelegationContext` so it's part of the serialized context

**Impact:** Prevents runaway delegation chains across process boundaries.

---

### 3. Remove Dual Tool Execution Paths

`execute_tool()` has two code paths: the old `subprocess.run()` path and the new supervisor-based async path. Both `agent.py` and `executor.py` duplicate the fnmatch pattern-matching logic for `force_background`/`force_sync`. 

**What to do:**
- Always route through the supervisor (the sync fallback is barely used now)
- Move the force_background/force_sync pattern matching into the supervisor itself (it already has this logic via `_should_force_background`)
- Remove the duplicated fnmatch blocks from `agent.py._execute_tool_call()` and `executor.py._execute_tool()`

**Impact:** One execution path, one place for pattern matching, fewer bugs.

---

## 🟠 Priority 2: Robustness & Reliability

### 4. Error Handling & Resilience

Currently, many error paths return generic `{"ok": False, "error": str(e)}` dicts that lose context.

**What to do:**
- Define a proper `ToolResult` Pydantic model (instead of raw dicts everywhere) with fields: `ok`, `data`, `error`, `error_type`, `duration_ms`, `process_id`
- Add retry logic for transient LLM failures in `LLMClient.chat()` (LiteLLM supports this but it's not configured)
- Add circuit-breaker for tools that fail repeatedly (e.g., after 3 consecutive failures of the same tool, warn the agent)
- Handle `KeyboardInterrupt` gracefully in `ExecutionRunner.run()` (currently uncaught)

---

### 5. Token Counting & Context Window Safety

`tokens.py` / `context_manager.py` work, but have gaps:

**What to do:**
- `get_context_limit()` has a hardcoded model→limit map that will always be stale. Use LiteLLM's `model_cost` data or fall back to a conservative default.
- Add token counting for tool definitions (currently not counted — a large toolset can silently exceed the context window)
- The summarization prompt itself consumes tokens but isn't accounted for in `generate_summary()`
- Add a "panic mode" — if messages still exceed context after summarization, truncate the oldest non-system messages instead of silently failing

---

### 6. Session Management Improvements

**What to do:**
- `_save_session()` rewrites the entire JSONL file on every title update. For long sessions this becomes expensive. Fix: only rewrite on explicit save, append-only during chat.
- Add session search/filter (by date, keyword, agent)
- Add `supyagent sessions --delete-all <agent>` for cleanup
- Add session export to sharable formats (HTML with rich rendering)

---

## 🟡 Priority 3: Developer Experience

### 7. Default Tools: Consistency & Quality

The default tools work but have inconsistencies:

**What to do:**
- `shell.py`: `run_command()` uses `subprocess.run()` directly (bypasses the supervisor). Should route through supervisor for timeout/background support, or at minimum respect the same timeout settings.
- `edit.py`: `multi_edit()` doesn't validate that edits don't overlap. Out-of-order line replacements can corrupt files.
- `files.py`: `read_file_lines()` and `read_file()` overlap. Consider merging into one function with optional line range.
- `search.py`: `search()` doesn't support file type filtering. Add `--include`/`--exclude` glob patterns.
- `find.py`: `directory_tree()` should respect `.gitignore` by default.
- **All tools:** Add a `@tool_metadata(category="files", safe=True)` decorator or similar for tool classification (helps with auto-documentation and permission management)

---

### 8. CLI Polish

**What to do:**
- Version string is hardcoded as `"0.2.6"` in `@click.version_option()` but `pyproject.toml` says `"0.2.7"`. Auto-read from `__init__.py` or `importlib.metadata`.
- `supyagent new` template hardcodes `anthropic/claude-3-5-sonnet-20241022`. Should read from `supyagent config` or `.envsupyagent` for the user's default provider.
- Add `supyagent doctor` command: check supypowers installed, check API keys configured, check tools directory exists, check Python version, verify a simple LLM call works.
- Add `supyagent upgrade` to check PyPI for newer version.
- `supyagent chat` doesn't have a `--model` override flag. Add it.
- Add tab-completion for agent names and `/commands`.

---

### 9. Configuration System

**What to do:**
- `config.py`'s `ConfigManager` stores API keys in plain JSON at `~/.supyagent/config.json`. The `credentials.py` `CredentialManager` uses Fernet encryption. These two systems should be unified — all secrets should be encrypted.
- Support `supyagent.toml` project-level config file (referenced in initial_plan.md but never implemented)
- Add per-agent environment variable overrides (e.g., `SUPYAGENT_MODEL=openai/gpt-4o supyagent chat myagent`)
- Support `.envsupyagent` file loading in `load_config()` (currently requires manual `source`)

---

## 🟢 Priority 4: Features

### 10. Streaming for Delegated Agents

Currently, delegation always returns the complete response at the end. For long-running delegated tasks, the parent agent (and user) sees nothing until completion.

**What to do:**
- Add a streaming delegation mode where the child process writes incremental output to its log file
- Parent can poll the log file for progress updates
- Show a progress indicator in the chat UI when delegation is in progress
- Alternative: use a Unix pipe or named pipe for real-time streaming between parent and child

---

### 11. Parallel Tool Execution

The agent currently executes tool calls sequentially. When the LLM returns multiple tool calls in one response, they could run in parallel.

**What to do:**
- Detect when multiple tool calls are returned in one turn
- If none depend on each other, execute them concurrently via the supervisor
- Add a `parallel: true/false` per-tool config option for tools that have side effects and must be sequential
- Show parallel execution status in the UI

---

### 12. Agent Memory / Knowledge Base

Agents lose all context between sessions (except raw conversation history). 

**What to do:**
- Add a per-agent persistent memory store (key-value or vector-based)
- Allow agents to explicitly store/retrieve memories across sessions
- Add `remember` and `recall` as built-in tools
- Store learned preferences (e.g., "user prefers TypeScript", "project uses uv")

---

### 13. Cost & Token Tracking

There's no visibility into how much each session/tool/delegation costs.

**What to do:**
- Track tokens used per LLM call (LiteLLM returns this in responses)
- Aggregate cost per session, per agent, per delegation chain
- Add `/cost` REPL command and `supyagent cost` CLI command
- Add budget limits per agent (hard stop after $X or N tokens)
- Show cost in the `/context` display

---

### 14. Tool Creation Workflow

`will_create_tools: true` injects instructions into the system prompt, but the workflow is manual (agent writes a .py file, hopes it works).

**What to do:**
- Add a `create_tool` built-in tool that creates, validates, and tests a new supypowers tool in one step
- Auto-run `supypowers docs` after creation to verify the tool is discoverable
- Add `supyagent tools test <script:function>` CLI command to test a tool with sample inputs
- Hot-reload tools during a chat session (currently requires restart)

---

### 15. Observability & Debugging

**What to do:**
- Add structured logging throughout (currently uses bare `logging.getLogger()` with no configuration)
- Add a `--debug` global flag that enables verbose logging to stderr
- Add `supyagent replay <session_id>` to replay a session step-by-step (useful for debugging agent behavior)
- Add OpenTelemetry traces for LLM calls, tool execution, and delegation
- Export agent execution traces as a timeline/flame chart

---

## 🔵 Priority 5: Ecosystem & Distribution

### 16. Plugin System

**What to do:**
- Define a `supyagent.plugins` entry point in pyproject.toml
- Allow third-party packages to register tools, agents, and middleware
- Support tool packs: `pip install supyagent-tools-aws` auto-registers AWS tools
- Add middleware hooks: `before_llm_call`, `after_tool_execution`, `on_delegation`

---

### 17. Testing & CI

**What to do:**
- Current tests are comprehensive for unit cases but lack integration tests that run actual LLM calls (even against a mock server)
- Add a mock LLM server for CI (avoid real API calls in CI)
- Add end-to-end tests: `supyagent init` → `supyagent new` → `supyagent run` → verify output
- Add `supyagent self-test` command for users to verify their installation
- Set up GitHub Actions CI/CD with automatic PyPI publishing on tag

---

### 18. Documentation

**What to do:**
- `README.md` is minimal. Add:
  - Quick start guide with working examples
  - Architecture overview diagram
  - Tool creation tutorial
  - Multi-agent orchestration guide
  - Configuration reference
- Add docsite (MkDocs or similar) deployed to GitHub Pages
- Add `supyagent docs` command to open documentation in browser
- Add inline `--help` examples for every CLI command

---

## 📋 Suggested Sprint Order

| Sprint | Items | Effort |
|--------|-------|--------|
| **S1: Foundation** | #1 (Unify Agent/Executor), #3 (Single execution path), #8 (CLI version fix) | 1 week |
| **S2: Safety** | #2 (Cross-process depth), #4 (Error handling), #5 (Token safety) | 1 week |
| **S3: DX** | #7 (Tool consistency), #9 (Config unification), #8 (CLI polish) | 1 week |
| **S4: Features** | #11 (Parallel tools), #13 (Cost tracking), #14 (Tool creation) | 1-2 weeks |
| **S5: Memory** | #12 (Agent memory), #10 (Streaming delegation) | 1-2 weeks |
| **S6: Polish** | #15 (Observability), #6 (Sessions), #18 (Documentation) | 1 week |
| **S7: Ecosystem** | #16 (Plugins), #17 (CI/CD) | 1-2 weeks |

---

## Quick Wins (< 1 day each)

1. Fix CLI version string (`@click.version_option` → read from package metadata)
2. Add `--depth` flag to `supyagent exec` (prevents infinite delegation)
3. Remove the `ToolCallObj` inner class duplication (extract to shared utility)
4. Add `.envsupyagent` auto-loading in `load_config()`
5. Make `directory_tree()` respect `.gitignore`
6. Add `--model` flag to `supyagent chat`
7. Log tool execution duration in the chat UI (already tracked by supervisor, just not displayed)
