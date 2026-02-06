# Sprint 9: Architecture Refactoring

**Goal**: Eliminate code duplication between Agent and ExecutionRunner, enforce cross-process delegation depth, and consolidate tool execution into a single path through the supervisor.

**Duration**: ~5-6 days

**Depends on**: Sprint 8 (Process Supervisor)

---

## Problem Statement

Three architectural issues have accumulated that increase maintenance burden and create correctness bugs:

1. **Agent/ExecutionRunner duplication** — `agent.py` (630 lines) and `executor.py` (435 lines) share ~60% of their logic. The streaming tool-call collector, tool dispatch logic, delegation setup, and the `ToolCallObj` inner class are all copy-pasted with slight variations. Every new feature must be implemented twice.

2. **Cross-process depth is not enforced** — Each `supyagent exec` subprocess creates a fresh `AgentRegistry()`, so `MAX_DEPTH=5` restarts at 0. An infinite delegation loop (A→B→A→B→…) across process boundaries is currently possible.

3. **Dual tool execution paths** — `execute_tool()` in `tools.py` still has the old `subprocess.run()` path alongside the new supervisor path. Both `agent.py` and `executor.py` duplicate `fnmatch` pattern-matching logic for `force_background`/`force_sync` that already exists inside the supervisor.

---

## Design Principles

- **Zero behavioral change** — All existing tests must pass. The refactor should be invisible to users and to the LLM.
- **Incremental extraction** — Extract shared code in small, testable steps. Don't rewrite from scratch.
- **Single responsibility** — The base engine handles the LLM loop. Subclasses add only their unique concerns.

---

## Deliverables

### 9.1 Extract Shared `ToolCallObj`

The `ToolCallObj` inner class is defined identically in 3 places:
- `agent.py` line 446-449 (inside `send_message_stream`)
- `executor.py` line 230-233 (inside streaming branch of `run`)
- Used implicitly in `executor.py` non-streaming branch via LiteLLM's response objects

**What to do:**

```python
# supyagent/core/models.py (new file, or add to existing models/)

class ToolCallObj:
    """
    Lightweight wrapper matching the LiteLLM tool_call interface.
    
    Used when reconstructing tool calls from streaming chunks,
    where we collect raw dicts instead of LiteLLM objects.
    """
    __slots__ = ("id", "function")
    
    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.function = _FunctionObj(name, arguments)


class _FunctionObj:
    __slots__ = ("name", "arguments")
    
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments
```

- [ ] Create the shared class in `supyagent/core/models.py`
- [ ] Replace all 3 inline definitions with imports
- [ ] Ensure tests still pass

**Lines saved:** ~15 lines of duplication removed

---

### 9.2 Extract `BaseAgentEngine`

The core agent loop is the same in both classes:

```
1. Build messages for LLM
2. Call LLM (streaming or non-streaming)
3. Collect tool calls from response/stream
4. If no tool calls → return content
5. For each tool call → dispatch to handler
6. Loop back to step 1
```

**What to do:**

```python
# supyagent/core/engine.py (new file)

from abc import ABC, abstractmethod

class BaseAgentEngine(ABC):
    """
    Shared agent loop for both interactive and execution modes.
    
    Subclasses implement:
    - _get_messages() → messages to send to LLM
    - _dispatch_tool_call(tool_call) → result dict
    - _on_assistant_message(content, tool_calls) → hook for persistence
    - _on_tool_result(tool_call_id, name, result) → hook for persistence
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = LLMClient(
            model=config.model.provider,
            temperature=config.model.temperature,
            max_tokens=config.model.max_tokens,
        )
        self.tools: list[dict] = []
        self.messages: list[dict] = []
        
        # Delegation support
        self.delegation_mgr: DelegationManager | None = None
        self.instance_id: str | None = None
    
    def _setup_delegation(self, registry=None, parent_instance_id=None):
        """Set up delegation if this agent has delegates."""
        if self.config.delegates:
            from supyagent.core.delegation import DelegationManager
            from supyagent.core.registry import AgentRegistry
            
            registry = registry or AgentRegistry()
            self.delegation_mgr = DelegationManager(
                registry, self, grandparent_instance_id=parent_instance_id
            )
            self.instance_id = self.delegation_mgr.parent_id
    
    def _load_base_tools(self) -> list[dict]:
        """Load supypowers tools filtered by config permissions."""
        tools = []
        sp_tools = discover_tools()
        openai_tools = supypowers_to_openai_tools(sp_tools)
        filtered = filter_tools(openai_tools, self.config.tools)
        tools.extend(filtered)
        
        # Delegation tools
        if self.delegation_mgr:
            tools.extend(self.delegation_mgr.get_delegation_tools())
        
        # Process management tools
        from supyagent.core.process_tools import get_process_management_tools
        tools.extend(get_process_management_tools())
        
        return tools
    
    def _dispatch_tool_call(self, tool_call) -> dict:
        """
        Route a tool call to the correct handler.
        
        Handles: delegation tools, process tools, supypowers tools.
        """
        name = tool_call.function.name
        
        # 1. Delegation tools
        if self.delegation_mgr and self.delegation_mgr.is_delegation_tool(name):
            return self.delegation_mgr.execute_delegation(tool_call)
        
        # 2. Process management tools
        from supyagent.core.process_tools import is_process_tool, execute_process_tool
        if is_process_tool(name):
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                return {"ok": False, "error": "Invalid JSON arguments"}
            return execute_process_tool(name, args)
        
        # 3. Supypowers tools (always through supervisor)
        return self._execute_supypowers_tool(tool_call)
    
    def _execute_supypowers_tool(self, tool_call) -> dict:
        """Execute a supypowers tool through the supervisor."""
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return {"ok": False, "error": f"Invalid JSON: {tool_call.function.arguments}"}
        
        if "__" not in name:
            return {"ok": False, "error": f"Invalid tool name format: {name}"}
        
        script, func = name.split("__", 1)
        secrets = self._get_secrets()
        
        # Supervisor handles all pattern matching internally
        return execute_tool(
            script, func, args, secrets=secrets,
            timeout=self.config.supervisor.default_timeout,
            background=False,
            use_supervisor=True,
        )
    
    @abstractmethod
    def _get_secrets(self) -> dict[str, str]:
        """Get secrets for tool execution."""
        ...
    
    def _run_loop(
        self,
        messages: list[dict],
        max_iterations: int,
        stream: bool = False,
        on_message=None,      # Called with (content, tool_calls_list)
        on_tool_result=None,  # Called with (tool_call_id, name, result)
    ) -> str:
        """
        Core LLM-tool loop. Returns final text content.
        
        Both Agent and ExecutionRunner call this with their own
        hooks for persistence and progress reporting.
        """
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            if stream:
                content, tool_calls = self._stream_response(messages)
            else:
                content, tool_calls = self._non_stream_response(messages)
            
            # Notify subclass
            if on_message:
                on_message(content, tool_calls)
            
            # No tool calls → done
            if not tool_calls:
                return content
            
            # Execute each tool call
            for tc in tool_calls:
                tc_obj = ToolCallObj(tc["id"], tc["function"]["name"], tc["function"]["arguments"])
                result = self._dispatch_tool_call(tc_obj)
                
                if on_tool_result:
                    on_tool_result(tc["id"], tc["function"]["name"], result)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result),
                })
        
        return content or ""
    
    def _stream_response(self, messages):
        """Stream LLM response, collect content and tool calls."""
        # ... (shared streaming logic, extracted from agent.py lines 374-430)
        pass
    
    def _non_stream_response(self, messages):
        """Non-streaming LLM call, returns (content, tool_calls_list)."""
        response = self.llm.chat(messages=messages, tools=self.tools or None)
        msg = response.choices[0].message
        
        content = msg.content or ""
        tool_calls = []
        
        if msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        
        # Add to messages
        msg_dict = {"role": "assistant", "content": content}
        if tool_calls:
            msg_dict["tool_calls"] = tool_calls
        messages.append(msg_dict)
        
        return content, tool_calls
```

**What to do:**

- [ ] Create `supyagent/core/engine.py` with `BaseAgentEngine`
- [ ] Refactor `Agent` to extend `BaseAgentEngine`, keeping only:
  - Session management (`SessionManager`, `ContextManager`)
  - Credential prompting (`_handle_credential_request`)
  - Message reconstruction from session
  - `/commands` support (stays in CLI)
- [ ] Refactor `ExecutionRunner` to extend `BaseAgentEngine`, keeping only:
  - Stateless operation (no session)
  - `_format_output()` (JSON/markdown parsing)
  - `_format_structured_input()`
  - Progress callbacks
- [ ] Update all imports across the codebase

**Impact:** Cuts ~400 lines of duplication. Future features (parallel tools, cost tracking) are implemented once.

---

### 9.3 Remove Duplicated Pattern Matching

Both `agent.py` (lines 581-607) and `executor.py` (lines 408-433) contain identical `fnmatch` blocks:

```python
# This block is duplicated in both files:
import fnmatch
force_background = any(
    fnmatch.fnmatch(name, pattern) 
    for pattern in sup_settings.force_background_patterns
)
force_sync = any(
    fnmatch.fnmatch(name, pattern) 
    for pattern in sup_settings.force_sync_patterns
)
# Per-tool settings override
for pattern, tool_settings in sup_settings.tool_settings.items():
    if fnmatch.fnmatch(name, pattern):
        ...
```

The supervisor already has `_should_force_background()` and `_should_force_sync()` methods.

**What to do:**

- [ ] Move all pattern matching into the supervisor's `execute()` method (it already does this when `tool_name` is passed)
- [ ] Remove the fnmatch blocks from both `agent.py` and `executor.py` (or from the new `BaseAgentEngine._execute_supypowers_tool()`)
- [ ] Have `_execute_supypowers_tool()` simply pass `tool_name=name` to `execute_tool()`, letting the supervisor decide
- [ ] Ensure per-tool `ToolExecutionSettings` from `agent_config.py` are passed through to the supervisor (currently they're only checked in the duplicated agent/executor code)

**Impact:** One place for all execution-mode decisions.

---

### 9.4 Cross-Process Depth Enforcement

Currently, delegation depth is only tracked within a single process via `AgentRegistry.MAX_DEPTH`. Each `supyagent exec` subprocess starts fresh.

**What to do:**

#### Step 1: Add `--depth` flag to `supyagent exec`

```python
# supyagent/cli/main.py — exec_agent command
@cli.command("exec")
@click.argument("agent_name")
@click.option("--task", "-t", required=True)
@click.option("--context", "-c", default="{}")
@click.option("--depth", "-d", type=int, default=0, help="Current delegation depth")
@click.option("--output", "-o", "output_fmt", type=click.Choice(["json", "text"]), default="json")
@click.option("--timeout", type=float, default=300)
def exec_agent(agent_name, task, context, depth, output_fmt, timeout):
    # Reject if too deep
    if depth >= AgentRegistry.MAX_DEPTH:
        result = {
            "ok": False,
            "error": f"Maximum delegation depth ({AgentRegistry.MAX_DEPTH}) reached across processes."
        }
        if output_fmt == "json":
            click.echo(json.dumps(result))
        sys.exit(1)
    
    # Pass depth to child agent so it can pass depth+1 to its children
    os.environ["SUPYAGENT_DELEGATION_DEPTH"] = str(depth)
    # ... rest of exec_agent
```

- [ ] Add `--depth` option to `exec_agent` command
- [ ] Reject execution if depth >= MAX_DEPTH at the CLI level

#### Step 2: Pass depth through delegation

```python
# supyagent/core/delegation.py — _delegate_subprocess
def _delegate_subprocess(self, agent_name, full_task, context, background, timeout):
    # Calculate child depth
    current_depth = int(os.environ.get("SUPYAGENT_DELEGATION_DEPTH", "0"))
    child_depth = current_depth + self.registry.get_depth(self.parent_id) + 1
    
    cmd = [
        "supyagent", "exec", agent_name,
        "--task", full_task,
        "--context", json.dumps({...}),
        "--depth", str(child_depth),  # NEW
        "--output", "json",
    ]
    # ...
```

- [ ] Update `_delegate_subprocess()` to compute and pass `--depth`
- [ ] Store depth in `DelegationContext` so it's part of serialized context

#### Step 3: Add depth to DelegationContext

```python
# supyagent/core/context.py
@dataclass
class DelegationContext:
    parent_agent: str = ""
    parent_task: str = ""
    conversation_summary: str | None = None
    relevant_facts: list[str] = field(default_factory=list)
    depth: int = 0  # NEW: cross-process delegation depth
```

- [ ] Add `depth` field to `DelegationContext`
- [ ] Serialize/deserialize depth in context JSON

---

### 9.5 CLI Version Fix (Quick Win)

The CLI version is hardcoded as `"0.2.6"` in `@click.version_option()` but `pyproject.toml` says `"0.2.7"`.

**What to do:**

```python
# supyagent/cli/main.py
from importlib.metadata import version as pkg_version

@click.group()
@click.version_option(version=pkg_version("supyagent"), prog_name="supyagent")
def cli():
    ...
```

- [ ] Replace hardcoded version with `importlib.metadata.version("supyagent")`
- [ ] Alternatively, read from `supyagent.__init__.__version__` if one exists

---

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `supyagent/core/models.py` | **Create** | Shared `ToolCallObj` class |
| `supyagent/core/engine.py` | **Create** | `BaseAgentEngine` with shared loop |
| `supyagent/core/agent.py` | **Major refactor** | Extend `BaseAgentEngine`, remove ~300 lines |
| `supyagent/core/executor.py` | **Major refactor** | Extend `BaseAgentEngine`, remove ~250 lines |
| `supyagent/core/delegation.py` | **Modify** | Pass `--depth` in subprocess delegation |
| `supyagent/core/context.py` | **Modify** | Add `depth` field to `DelegationContext` |
| `supyagent/cli/main.py` | **Modify** | Add `--depth` to `exec`, fix version string |
| `supyagent/core/tools.py` | **Modify** | Remove old `subprocess.run()` path, simplify |
| `tests/test_engine.py` | **Create** | Tests for `BaseAgentEngine` |
| `tests/test_agent.py` | **Update** | Adjust for new class hierarchy |
| `tests/test_executor.py` | **Update** | Adjust for new class hierarchy |
| `tests/test_delegation.py` | **Update** | Test depth propagation |
| `tests/test_delegation_nested.py` | **Update** | Test cross-process depth enforcement |

---

## Acceptance Criteria

1. **All existing tests pass** — No behavioral changes for users
2. **ToolCallObj** is defined in exactly one place
3. **Agent** and **ExecutionRunner** both extend `BaseAgentEngine`
4. **Zero fnmatch blocks** in `agent.py` or `executor.py` — all pattern matching is in the supervisor
5. **Cross-process depth** — Delegation chain deeper than `MAX_DEPTH` is rejected even across subprocesses
6. **CLI version** — `supyagent --version` reports the correct version from `pyproject.toml`

---

## Test Scenarios

### Test 1: Verify unified loop

```python
def test_agent_and_executor_share_engine():
    """Both Agent and ExecutionRunner use BaseAgentEngine._run_loop."""
    assert issubclass(Agent, BaseAgentEngine)
    assert issubclass(ExecutionRunner, BaseAgentEngine)
    assert Agent._run_loop is BaseAgentEngine._run_loop
```

### Test 2: Cross-process depth enforcement

```bash
# Create agents that delegate to each other: A→B→A→B→...
# Agent A delegates to B, agent B delegates to A
supyagent exec a_agent --task "Do something" --depth 0
# After MAX_DEPTH hops, the chain should be rejected with an error
```

```python
def test_cross_process_depth_rejected():
    """Delegation at MAX_DEPTH is rejected even in subprocess."""
    result = subprocess.run(
        ["supyagent", "exec", "test_agent", "--task", "test", "--depth", "5"],
        capture_output=True, text=True
    )
    output = json.loads(result.stdout)
    assert output["ok"] is False
    assert "Maximum delegation depth" in output["error"]
```

### Test 3: Pattern matching only in supervisor

```python
def test_no_fnmatch_in_agent():
    """agent.py should not import or use fnmatch."""
    import inspect
    source = inspect.getsource(Agent)
    assert "fnmatch" not in source

def test_no_fnmatch_in_executor():
    """executor.py should not import or use fnmatch."""
    import inspect
    source = inspect.getsource(ExecutionRunner)
    assert "fnmatch" not in source
```

### Test 4: CLI version matches pyproject

```bash
supyagent --version
# Should output: supyagent, version 0.2.7 (not 0.2.6)
```

---

## Migration Guide

Since `BaseAgentEngine` is an internal refactor, **no user-facing changes** are expected. However:

- **Custom subclasses** of `Agent` or `ExecutionRunner` (if any exist in user code) will need to update their inheritance chain.
- **Patching in tests** — Tests that mock `Agent._execute_tool_call` should now mock `BaseAgentEngine._dispatch_tool_call` or use the tool execution path.

---

## Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| Streaming behavior diverges subtly between Agent and ExecutionRunner | Extract streaming into `_stream_response()` on the base class; both modes call the same code |
| Breaking credential prompting in interactive mode | `Agent` overrides `_dispatch_tool_call` to check for credential requests before calling `super()` |
| Session persistence breaks during refactor | Keep `_on_assistant_message` and `_on_tool_result` hooks in `Agent`, tested independently |
| Cross-process depth env var leaks | Use unique env var name (`SUPYAGENT_DELEGATION_DEPTH`), clean up in tests |

---

## Notes

- The old `subprocess.run()` path in `tools.py` can be kept as a fallback for environments without asyncio, but should be clearly marked as deprecated.
- The `execute_tool()` function signature remains unchanged for backward compatibility.
- Consider adding a `--max-depth` override to `supyagent exec` for advanced users who need deeper chains (e.g., `--max-depth 10`).
