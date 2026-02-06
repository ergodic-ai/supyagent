# Sprint 10: Robustness & Reliability

**Goal**: Harden error handling across the tool/LLM pipeline, fix token counting gaps that can silently overflow the context window, and improve session management efficiency.

**Duration**: ~4-5 days

**Depends on**: Sprint 9 (Architecture Refactoring)

---

## Problem Statement

The core agent loop works, but several reliability gaps exist that cause silent failures, wasted tokens, or degraded performance in production use:

1. **Error handling is inconsistent** — Tool results are raw dicts (`{"ok": False, "error": str(e)}`), losing error type, duration, and process context. LLM transient failures (rate limits, network errors) have no retry logic. If a tool fails repeatedly, the agent keeps calling it.

2. **Token counting has blind spots** — Tool definitions are not counted toward the context budget (a large toolset can silently exceed the window). The hardcoded `CONTEXT_LIMITS` map in `tokens.py` is already stale. The summarization prompt itself consumes tokens that aren't accounted for.

3. **Session saves are expensive** — `_save_session()` rewrites the entire JSONL file on every title update. For long sessions, this becomes a performance bottleneck.

---

## Deliverables

### 10.1 Structured `ToolResult` Model

Replace raw `{"ok": ..., "error": ...}` dicts with a typed Pydantic model throughout the pipeline.

**What to do:**

```python
# supyagent/models/tool_result.py (new file)

from pydantic import BaseModel, Field
from typing import Any, Optional
from datetime import datetime

class ToolResult(BaseModel):
    """Structured result from any tool execution."""
    
    ok: bool = Field(..., description="Whether the tool execution succeeded")
    data: Any = Field(None, description="Result data (if ok=True)")
    error: Optional[str] = Field(None, description="Error message (if ok=False)")
    error_type: Optional[str] = Field(None, description="Error classification")
    
    # Execution metadata
    tool_name: Optional[str] = Field(None, description="Full tool name (script__func)")
    duration_ms: Optional[int] = Field(None, description="Execution time in milliseconds")
    process_id: Optional[str] = Field(None, description="Supervisor process ID (if async)")
    
    # For LLM consumption (serialized to JSON string)
    def to_llm_content(self) -> str:
        """Serialize for passing back to LLM as tool result content."""
        # Keep it simple for the LLM — just data or error
        if self.ok:
            if isinstance(self.data, str):
                return self.data
            return json.dumps(self.data) if self.data is not None else ""
        else:
            return json.dumps({"error": self.error, "error_type": self.error_type})
    
    # For internal logging/debugging
    def to_dict(self) -> dict[str, Any]:
        """Full dict including metadata."""
        return self.model_dump(exclude_none=True)
```

**Error type classification:**

| `error_type` | Meaning | Agent action |
|---|---|---|
| `"invalid_args"` | Bad arguments from LLM | Retry with corrected args |
| `"tool_not_found"` | Tool/script doesn't exist | Don't retry, inform user |
| `"execution_error"` | Tool crashed | Maybe retry once |
| `"timeout"` | Exceeded time limit | Check process status |
| `"permission_denied"` | OS-level denial | Don't retry |
| `"network_error"` | HTTP/connection failure | Retry after delay |

**Migration path:**

- [ ] Create `ToolResult` model
- [ ] Update `execute_tool()` in `tools.py` to return `ToolResult`
- [ ] Update `execute_process_tool()` in `process_tools.py` to return `ToolResult`
- [ ] Update `BaseAgentEngine._dispatch_tool_call()` to use `ToolResult.to_llm_content()` for LLM messages
- [ ] Keep backward-compatible `.model_dump()` for existing code that expects dicts
- [ ] Add duration tracking (start/end time around tool execution)

---

### 10.2 LLM Retry Logic

`LLMClient.chat()` currently makes a single call with no error handling. LiteLLM supports retries, but we haven't configured them.

**What to do:**

```python
# supyagent/core/llm.py

import litellm
from litellm import completion
from litellm.exceptions import (
    RateLimitError,
    ServiceUnavailableError,
    APIConnectionError,
    Timeout as LiteLLMTimeout,
)

class LLMClient:
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_retries: int = 3,           # NEW
        retry_delay: float = 1.0,        # NEW: seconds between retries
        retry_backoff: float = 2.0,      # NEW: exponential backoff multiplier
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff
    
    def chat(self, messages, tools=None, stream=False) -> ModelResponse:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        last_error = None
        delay = self.retry_delay
        
        for attempt in range(self.max_retries + 1):
            try:
                return completion(**kwargs)
            except (RateLimitError, ServiceUnavailableError, APIConnectionError, LiteLLMTimeout) as e:
                last_error = e
                if attempt < self.max_retries:
                    import time
                    logger.warning(
                        f"LLM call failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= self.retry_backoff
                else:
                    raise
        
        raise last_error  # Should not reach here
```

- [ ] Add retry parameters to `LLMClient.__init__`
- [ ] Implement retry loop with exponential backoff
- [ ] Only retry on transient errors (rate limit, service unavailable, connection, timeout)
- [ ] Log warnings on retry attempts
- [ ] Make retry params configurable via `AgentConfig.model` (add `max_retries`, `retry_delay` fields to `ModelConfig`)

---

### 10.3 Tool Failure Circuit Breaker

If a tool fails 3+ times consecutively in the same turn, the agent should be warned instead of looping indefinitely.

**What to do:**

```python
# In BaseAgentEngine (from Sprint 9)

class BaseAgentEngine:
    def __init__(self, config):
        ...
        self._tool_failure_counts: dict[str, int] = {}  # tool_name → consecutive failures
        self._tool_failure_threshold: int = 3
    
    def _dispatch_tool_call(self, tool_call) -> dict:
        name = tool_call.function.name
        
        # Check if tool has failed too many times
        if self._tool_failure_counts.get(name, 0) >= self._tool_failure_threshold:
            return {
                "ok": False,
                "error": (
                    f"Tool '{name}' has failed {self._tool_failure_threshold} times consecutively. "
                    "Please try a different approach or tool."
                ),
                "error_type": "circuit_breaker",
            }
        
        result = self._execute_tool_dispatch(tool_call)
        
        # Track failures
        if not result.get("ok", False):
            self._tool_failure_counts[name] = self._tool_failure_counts.get(name, 0) + 1
        else:
            self._tool_failure_counts[name] = 0  # Reset on success
        
        return result
```

- [ ] Add failure tracking to `BaseAgentEngine`
- [ ] Return a clear message to the LLM when circuit breaker trips
- [ ] Reset counter on success
- [ ] Make threshold configurable via `AgentConfig.limits`

---

### 10.4 Graceful `KeyboardInterrupt` Handling

`ExecutionRunner.run()` does not handle `KeyboardInterrupt`. If the user presses Ctrl+C during tool execution, the process may leave orphan subprocesses.

**What to do:**

```python
# In BaseAgentEngine._run_loop()

def _run_loop(self, messages, max_iterations, ...):
    try:
        # ... existing loop
        pass
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        # Kill any running background processes
        from supyagent.core.supervisor import get_supervisor
        supervisor = get_supervisor()
        running = supervisor.list_processes(include_completed=False)
        for proc in running:
            try:
                asyncio.run(supervisor.kill(proc["process_id"]))
            except Exception:
                pass
        raise  # Re-raise so caller can handle
```

- [ ] Wrap the main loop in a try/except for `KeyboardInterrupt`
- [ ] Kill all running managed processes on interrupt
- [ ] Re-raise so the caller (CLI) can show a clean "Cancelled" message
- [ ] Also handle in `Agent.send_message_stream()` — currently the generator doesn't clean up

---

### 10.5 Token Counting: Count Tool Definitions

Currently, `build_messages_for_llm()` accounts for system prompt and conversation messages, but the tool definitions array is not counted. A large toolset (30+ tools with detailed schemas) can consume 5,000-10,000 tokens that silently reduce the available context.

**What to do:**

```python
# supyagent/core/tokens.py

def count_tools_tokens(tools: list[dict], model: str = "default") -> int:
    """
    Count tokens consumed by tool definitions.
    
    Tool definitions are sent as a separate parameter but consume context tokens.
    We serialize them to JSON and count — this is an approximation,
    as the actual token count depends on the provider's serialization.
    """
    if not tools:
        return 0
    
    # Serialize tools the way they'd be sent to the API
    tools_text = json.dumps(tools)
    return count_tokens(tools_text, model)
```

```python
# supyagent/core/context_manager.py — build_messages_for_llm()

def build_messages_for_llm(
    self,
    system_prompt: str,
    all_messages: list[dict],
    tools: list[dict] | None = None,    # NEW parameter
) -> list[dict]:
    """Build messages, accounting for tool definition token cost."""
    available_tokens = self.context_limit - self.response_reserve
    
    # Subtract tool definition tokens
    if tools:
        from supyagent.core.tokens import count_tools_tokens
        tools_tokens = count_tools_tokens(tools, self.model)
        available_tokens -= tools_tokens
    
    # ... rest of existing logic with reduced budget
```

- [ ] Add `count_tools_tokens()` to `tokens.py`
- [ ] Update `build_messages_for_llm()` to accept and account for tools
- [ ] Update callers (`Agent._get_messages_for_llm()`, `BaseAgentEngine._run_loop()`) to pass tools
- [ ] Add tool token count to `/context` command output

---

### 10.6 Token Counting: Dynamic Context Limits

The hardcoded `CONTEXT_LIMITS` dict in `tokens.py` is incomplete and will always be stale:

```python
# Current (stale):
CONTEXT_LIMITS = {
    "gpt-4o": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16384,
    "claude-3-5-sonnet": 200000,
    "claude-3-opus": 200000,
    "claude-3-haiku": 200000,
    "kimi": 128000,
    "deepseek": 64000,
    "default": 128000,
}
```

Missing models: Gemini (1M+ context), Llama, Mistral, Qwen, etc.

**What to do:**

```python
# supyagent/core/tokens.py

def get_context_limit(model: str) -> int:
    """
    Get the context window limit for a model.
    
    Strategy:
    1. Try LiteLLM's model cost data (most up-to-date)
    2. Fall back to our hardcoded map
    3. Fall back to conservative default
    """
    # Strategy 1: LiteLLM's model info
    try:
        import litellm
        model_info = litellm.get_model_info(model)
        if model_info and "max_input_tokens" in model_info:
            return model_info["max_input_tokens"]
    except Exception:
        pass
    
    # Strategy 2: Our hardcoded map (prefix matching)
    model_lower = model.lower()
    for prefix, limit in CONTEXT_LIMITS.items():
        if prefix in model_lower:
            return limit
    
    # Strategy 3: Conservative default
    return CONTEXT_LIMITS["default"]
```

- [ ] Try `litellm.get_model_info()` first for dynamic context limits
- [ ] Keep hardcoded map as fallback
- [ ] Add Gemini, Llama, Mistral to the fallback map
- [ ] Log a warning when falling back to default (user might need to configure)

---

### 10.7 Token Counting: Account for Summarization Prompt

`generate_summary()` in `context_manager.py` uses an LLM call with a prompt that itself consumes tokens, but this isn't accounted for in the budget calculation.

**What to do:**

```python
# supyagent/core/context_manager.py — generate_summary()

def generate_summary(self, all_messages, up_to_idx=None):
    ...
    conversation_text = self._format_messages_for_summary(messages_to_summarize)
    
    summary_prompt = f"""Summarize this conversation concisely..."""
    
    # Check if summarization prompt fits in context
    prompt_tokens = count_tokens(summary_prompt, self.model)
    if prompt_tokens > self.context_limit * 0.8:
        # Conversation too long even for summarization — truncate input
        max_chars = int(self.context_limit * 2)  # rough chars-to-tokens ratio
        conversation_text = conversation_text[:max_chars] + "\n... [truncated for summarization]"
        summary_prompt = f"""Summarize this conversation concisely...\n\n{conversation_text}\n\nSummary:"""
    
    response = self.llm.chat([{"role": "user", "content": summary_prompt}])
    ...
```

- [ ] Count summarization prompt tokens before sending
- [ ] If prompt exceeds 80% of context, truncate the conversation text
- [ ] Log a warning when truncation occurs

---

### 10.8 Panic Mode: Context Overflow Recovery

If messages still exceed the context window after summarization (e.g., a single tool result is 50K tokens), the current code silently sends everything and lets the API reject it.

**What to do:**

```python
# supyagent/core/context_manager.py

def build_messages_for_llm(self, system_prompt, all_messages, tools=None):
    ...
    messages.extend(recent_messages)
    
    # PANIC MODE: Final safety check
    total_tokens = count_messages_tokens(messages, self.model)
    if tools:
        total_tokens += count_tools_tokens(tools, self.model)
    
    if total_tokens > self.context_limit - self.response_reserve:
        logger.warning(
            f"Context overflow detected ({total_tokens} tokens, limit {self.context_limit}). "
            "Truncating oldest non-system messages."
        )
        messages = self._emergency_truncate(
            messages, 
            target_tokens=self.context_limit - self.response_reserve - (count_tools_tokens(tools, self.model) if tools else 0)
        )
    
    return messages

def _emergency_truncate(self, messages, target_tokens):
    """
    Last-resort truncation when context is still too large.
    
    Strategy:
    1. Keep system prompt (index 0)
    2. Keep summary message (index 1, if exists)
    3. Keep the last min_recent_messages
    4. Truncate large tool results in the middle
    """
    # Always keep first 2 messages (system + optional summary) and last N
    protected_start = 2 if len(messages) > 2 else 1
    protected_end = self.min_recent_messages
    
    # Try truncating large content in middle messages
    for i in range(protected_start, len(messages) - protected_end):
        msg = messages[i]
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > 2000:
            messages[i] = {**msg, "content": content[:1000] + "\n...[truncated]...\n" + content[-500:]}
    
    # Check if that's enough
    current = count_messages_tokens(messages, self.model)
    if current <= target_tokens:
        return messages
    
    # Still too big — drop middle messages one by one
    while current > target_tokens and len(messages) > protected_start + protected_end:
        # Remove the oldest non-protected message
        messages.pop(protected_start)
        current = count_messages_tokens(messages, self.model)
    
    return messages
```

- [ ] Add `_emergency_truncate()` to `ContextManager`
- [ ] Call it as a final safety check in `build_messages_for_llm()`
- [ ] Log a warning when panic mode activates
- [ ] Truncate large tool results first, then drop oldest messages
- [ ] Never drop the system prompt or the last `min_recent_messages`

---

### 10.9 Session Management: Append-Only Writes

`_update_title()` calls `_save_session()`, which rewrites the entire JSONL file. For a session with 500 messages, this means rewriting 500 lines just to update the title in the first line.

**What to do:**

```python
# supyagent/core/session_manager.py

def _update_title(self, session: Session, first_message: str) -> None:
    """Generate and save a title from the first user message."""
    title = first_message[:50]
    if len(first_message) > 50:
        title += "..."
    session.meta.title = title
    
    # Only rewrite the meta line, not the entire file
    self._update_meta(session)

def _update_meta(self, session: Session) -> None:
    """Update only the metadata (first line) of the session file."""
    path = self._session_path(session.meta.agent, session.meta.session_id)
    if not path.exists():
        self._save_session(session)
        return
    
    # Read all lines, replace first line (meta), write back
    with open(path) as f:
        lines = f.readlines()
    
    if not lines:
        self._save_session(session)
        return
    
    # Build new meta line
    meta_dict = session.meta.model_dump()
    meta_dict["type"] = "meta"
    meta_dict["created_at"] = session.meta.created_at.isoformat()
    meta_dict["updated_at"] = session.meta.updated_at.isoformat()
    
    lines[0] = json.dumps(meta_dict) + "\n"
    
    with open(path, "w") as f:
        f.writelines(lines)
```

This is still O(n) for the file read, but avoids re-serializing all Message objects.

**Better long-term approach:**

```python
def _update_meta_inplace(self, session: Session) -> None:
    """
    Update meta by reading only the first line, then rewriting it.
    
    For truly large sessions, consider a separate meta file:
      <session_id>.meta.json  — just metadata
      <session_id>.jsonl      — append-only messages
    """
    pass  # Implement if profiling shows the simpler approach is too slow
```

- [ ] Add `_update_meta()` method that only updates the first line
- [ ] Change `_update_title()` to call `_update_meta()` instead of `_save_session()`
- [ ] Consider splitting into `<id>.meta.json` + `<id>.jsonl` for very long sessions (can be done later)

---

### 10.10 Session Search and Cleanup

**What to do:**

```python
# supyagent/core/session_manager.py

def search_sessions(
    self, 
    agent: str, 
    query: str | None = None,
    since: datetime | None = None,
    before: datetime | None = None,
) -> list[SessionMeta]:
    """
    Search sessions by keyword in title, or filter by date range.
    """
    sessions = self.list_sessions(agent)
    results = []
    
    for s in sessions:
        if query and s.title and query.lower() not in s.title.lower():
            continue
        if since and s.updated_at < since:
            continue
        if before and s.updated_at > before:
            continue
        results.append(s)
    
    return results

def delete_all_sessions(self, agent: str) -> int:
    """Delete all sessions for an agent. Returns count deleted."""
    sessions = self.list_sessions(agent)
    count = 0
    for s in sessions:
        if self.delete_session(agent, s.session_id):
            count += 1
    return count
```

- [ ] Add `search_sessions()` method
- [ ] Add `delete_all_sessions()` method
- [ ] Add `supyagent sessions --search <query>` CLI option
- [ ] Add `supyagent sessions --delete-all <agent>` CLI command

---

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `supyagent/models/tool_result.py` | **Create** | Structured `ToolResult` model |
| `supyagent/core/llm.py` | **Modify** | Add retry logic with exponential backoff |
| `supyagent/core/engine.py` | **Modify** | Add circuit breaker, KeyboardInterrupt handling |
| `supyagent/core/tokens.py` | **Modify** | Add `count_tools_tokens()`, dynamic context limits |
| `supyagent/core/context_manager.py` | **Modify** | Account for tools tokens, panic mode truncation, summarization prompt budget |
| `supyagent/core/tools.py` | **Modify** | Return `ToolResult` objects |
| `supyagent/core/process_tools.py` | **Modify** | Return `ToolResult` objects |
| `supyagent/core/session_manager.py` | **Modify** | Append-only writes, search, delete-all |
| `supyagent/models/agent_config.py` | **Modify** | Add `max_retries`, `retry_delay` to `ModelConfig`; `circuit_breaker_threshold` to limits |
| `supyagent/cli/main.py` | **Modify** | Session search/delete commands |
| `tests/test_tool_result.py` | **Create** | Tests for `ToolResult` model |
| `tests/test_llm_retry.py` | **Create** | Tests for retry logic |
| `tests/test_context_manager.py` | **Create/Update** | Tests for tools token counting, panic mode |
| `tests/test_session_manager.py` | **Create/Update** | Tests for append-only writes, search |

---

## Acceptance Criteria

1. **ToolResult** — All tool execution returns a `ToolResult` instance (backward-compatible via `.model_dump()`)
2. **LLM retries** — Rate limit errors retry 3 times with exponential backoff before raising
3. **Circuit breaker** — After 3 consecutive failures of the same tool in one turn, the agent gets a "try something else" message
4. **KeyboardInterrupt** — Ctrl+C during tool execution kills managed processes and shows "Cancelled"
5. **Tool tokens counted** — `/context` shows tool definition token usage
6. **Dynamic limits** — `get_context_limit("openrouter/google/gemini-3-flash-preview")` returns the correct value (not the hardcoded default)
7. **Panic mode** — Messages exceeding context limit are truncated with a warning, not sent as-is
8. **Session efficiency** — Title update does NOT rewrite all messages
9. **Session search** — `supyagent sessions tester --search "delegation"` filters sessions
10. **Session cleanup** — `supyagent sessions tester --delete-all` removes all sessions

---

## Test Scenarios

### Test 1: LLM retry on rate limit

```python
@patch("supyagent.core.llm.completion")
def test_retry_on_rate_limit(mock_completion):
    """LLM client retries on RateLimitError."""
    from litellm.exceptions import RateLimitError
    
    # Fail twice, succeed third time
    mock_completion.side_effect = [
        RateLimitError("Rate limited", "model", "provider"),
        RateLimitError("Rate limited", "model", "provider"),
        mock_response,  # Success
    ]
    
    client = LLMClient("test-model", max_retries=3, retry_delay=0.01)
    result = client.chat([{"role": "user", "content": "test"}])
    assert result == mock_response
    assert mock_completion.call_count == 3
```

### Test 2: Circuit breaker

```python
def test_circuit_breaker_trips():
    """Tool is blocked after 3 consecutive failures."""
    engine = BaseAgentEngine(config)
    
    # Simulate 3 failures
    for _ in range(3):
        engine._tool_failure_counts["shell__run_command"] = _ + 1
    
    # 4th call should be blocked
    result = engine._dispatch_tool_call(mock_tool_call("shell__run_command"))
    assert result["ok"] is False
    assert "circuit_breaker" in result.get("error_type", "")
```

### Test 3: Context overflow recovery

```python
def test_panic_mode_truncates():
    """Messages exceeding context are truncated, not rejected."""
    cm = ContextManager(model="gpt-4", llm=mock_llm)
    
    # Create messages that exceed context (8192 tokens for gpt-4)
    huge_tool_result = {"role": "tool", "content": "x" * 50000, "tool_call_id": "1"}
    messages = [huge_tool_result] * 10
    
    result = cm.build_messages_for_llm("You are a test agent.", messages)
    
    # Should not exceed context
    total = count_messages_tokens(result, "gpt-4")
    assert total < 8192
    # Should still have system prompt and recent messages
    assert result[0]["role"] == "system"
```

### Test 4: Tool token counting

```python
def test_tools_reduce_available_context():
    """Tool definitions consume tokens from available budget."""
    tools = [generate_large_tool_schema() for _ in range(30)]
    tools_tokens = count_tools_tokens(tools, "gpt-4")
    
    assert tools_tokens > 0
    
    # With tools, fewer messages should fit
    cm_no_tools = ContextManager(model="gpt-4", llm=mock_llm)
    cm_with_tools = ContextManager(model="gpt-4", llm=mock_llm)
    
    messages = generate_messages(50)
    
    result_no_tools = cm_no_tools.build_messages_for_llm("system", messages)
    result_with_tools = cm_with_tools.build_messages_for_llm("system", messages, tools=tools)
    
    assert len(result_with_tools) <= len(result_no_tools)
```

### Test 5: Session append-only

```python
def test_title_update_does_not_rewrite_all_messages(tmp_path):
    """Title update only modifies meta line, not all messages."""
    mgr = SessionManager(base_dir=tmp_path)
    session = mgr.create_session("test", "test-model")
    
    # Add many messages
    for i in range(100):
        mgr.append_message(session, Message(type="user", content=f"msg {i}"))
    
    # Record file content before title update
    path = tmp_path / "test" / f"{session.meta.session_id}.jsonl"
    before_lines = path.read_text().splitlines()
    
    # Update title
    mgr._update_title(session, "New title for testing")
    
    after_lines = path.read_text().splitlines()
    
    # Only first line should have changed
    assert before_lines[0] != after_lines[0]  # Meta changed
    assert before_lines[1:] == after_lines[1:]  # Messages unchanged
```

---

## Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| `ToolResult` model breaks existing dict-based code | `ToolResult.model_dump()` returns a dict identical to current format; migrate callers incrementally |
| Retry logic masks real errors (e.g., invalid API key) | Only retry on transient errors; permanent errors (auth, invalid request) raise immediately |
| Circuit breaker is too aggressive | Set threshold at 3 (not 1); reset on success; make configurable |
| `litellm.get_model_info()` returns stale/wrong data | Fall back to hardcoded map, then conservative default; log which strategy was used |
| Panic mode drops important context | Always keep system prompt + summary + last N messages; truncate tool results before dropping messages |
| Session meta rewrite is still O(n) for reading | Acceptable for now; the expensive part was re-serializing all messages. Can split files later if profiling shows need |

---

## Notes

- The `ToolResult` model can be introduced incrementally — start by using it in `execute_tool()` and `execute_process_tool()`, then expand to delegation results.
- LiteLLM has built-in retry support via `litellm.num_retries`, but implementing our own gives us control over logging, backoff strategy, and which errors to retry.
- The circuit breaker state is per-turn (resets between user messages). A persistent circuit breaker across turns would need different logic and is not recommended for v1.
- Session search is basic keyword matching for now. Vector search (embedding-based) over session content is a Priority 4 feature.
