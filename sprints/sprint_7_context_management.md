# Sprint 7: Context Window Management

**Goal**: Implement intelligent context management to handle long conversations without losing history.

**Duration**: ~3-4 days

**Depends on**: Sprint 6

---

## Problem Statement

When conversations become long, the total tokens can exceed the model's context window limit. Currently:
- Messages accumulate indefinitely
- Eventually the LLM call fails or truncates unpredictably
- No mechanism to manage this gracefully

**Requirements:**
1. Never lose conversation history (full persistence)
2. Manage what gets sent to the LLM intelligently
3. Preserve important context even when truncating
4. Make this transparent to the user
5. Trigger summarization by **either N messages OR K tokens** (whichever comes first)
6. Each agent has their own configurable (N, K) thresholds
7. Provide sensible defaults: N=30 messages, K=128,000 tokens

---

## Deliverables

### 7.1 Token Counting

Add token counting utilities:

```python
# supyagent/core/tokens.py
import tiktoken
from typing import Any

# Model to encoding mapping (approximate for non-OpenAI models)
MODEL_ENCODINGS = {
    "gpt-4": "cl100k_base",
    "gpt-3.5": "cl100k_base", 
    "claude": "cl100k_base",  # Approximation
    "default": "cl100k_base",
}


def get_encoding(model: str) -> tiktoken.Encoding:
    """Get the tiktoken encoding for a model."""
    for prefix, encoding in MODEL_ENCODINGS.items():
        if prefix in model.lower():
            return tiktoken.get_encoding(encoding)
    return tiktoken.get_encoding(MODEL_ENCODINGS["default"])


def count_tokens(text: str, model: str = "default") -> int:
    """Count tokens in a text string."""
    encoding = get_encoding(model)
    return len(encoding.encode(text))


def count_message_tokens(message: dict[str, Any], model: str = "default") -> int:
    """
    Count tokens in a message dict.
    
    Accounts for message overhead (role, formatting).
    """
    encoding = get_encoding(model)
    tokens = 4  # Base overhead per message
    
    for key, value in message.items():
        if isinstance(value, str):
            tokens += len(encoding.encode(value))
        elif isinstance(value, list):  # tool_calls
            tokens += len(encoding.encode(str(value)))
    
    return tokens


def count_messages_tokens(messages: list[dict[str, Any]], model: str = "default") -> int:
    """Count total tokens across all messages."""
    total = 3  # Conversation overhead
    for msg in messages:
        total += count_message_tokens(msg, model)
    return total


# Context window limits (conservative estimates)
CONTEXT_LIMITS = {
    "gpt-4o": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16384,
    "claude-3-5-sonnet": 200000,
    "claude-3-opus": 200000,
    "claude-3-haiku": 200000,
    "default": 8192,  # Conservative default
}


def get_context_limit(model: str) -> int:
    """Get the context window limit for a model."""
    for prefix, limit in CONTEXT_LIMITS.items():
        if prefix in model.lower():
            return limit
    return CONTEXT_LIMITS["default"]
```

### 7.2 Context Summary Model

Store and manage context summaries:

```python
# supyagent/models/context.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import json


@dataclass
class ContextSummary:
    """
    A compressed summary of older conversation history.
    """
    content: str  # The summary text
    messages_summarized: int  # Number of messages this summarizes
    first_message_idx: int  # Index of first summarized message
    last_message_idx: int  # Index of last summarized message
    created_at: datetime = field(default_factory=datetime.utcnow)
    token_count: int = 0  # Tokens in the summary
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "messages_summarized": self.messages_summarized,
            "first_message_idx": self.first_message_idx,
            "last_message_idx": self.last_message_idx,
            "created_at": self.created_at.isoformat(),
            "token_count": self.token_count,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextSummary":
        return cls(
            content=data["content"],
            messages_summarized=data["messages_summarized"],
            first_message_idx=data["first_message_idx"],
            last_message_idx=data["last_message_idx"],
            created_at=datetime.fromisoformat(data["created_at"]),
            token_count=data.get("token_count", 0),
        )
    
    def to_message(self) -> dict[str, str]:
        """Convert to a system message for injection."""
        return {
            "role": "system",
            "content": f"[Context Summary - {self.messages_summarized} previous messages]\n\n{self.content}"
        }
```

### 7.3 Context Manager

Core context management logic:

```python
# supyagent/core/context_manager.py
from typing import Any
from pathlib import Path
import json

from supyagent.core.llm import LLMClient
from supyagent.core.tokens import (
    count_messages_tokens,
    get_context_limit,
    count_tokens,
)
from supyagent.models.context import ContextSummary


class ContextManager:
    """
    Manages conversation context to stay within token limits.
    
    Strategy:
    1. Keep ALL messages in session storage (full persistence)
    2. When building messages for LLM:
       - Always include system prompt
       - Include context summary (if exists)
       - Include recent N messages that fit in budget
    3. Trigger summarization when EITHER:
       - N messages since last summary (max_messages_before_summary)
       - K total tokens exceeded (max_tokens_before_summary)
    """
    
    def __init__(
        self,
        model: str,
        llm: LLMClient | None = None,
        summary_storage_path: Path | None = None,
        # Per-agent configurable thresholds
        max_messages_before_summary: int = 30,  # N
        max_tokens_before_summary: int = 128_000,  # K
        min_recent_messages: int = 6,
        response_reserve: int = 4096,
    ):
        self.model = model
        self.llm = llm
        self.context_limit = get_context_limit(model)
        self.summary_storage_path = summary_storage_path
        self._summary: ContextSummary | None = None
        
        # Per-agent thresholds
        self.max_messages_before_summary = max_messages_before_summary  # N
        self.max_tokens_before_summary = max_tokens_before_summary  # K
        self.min_recent_messages = min_recent_messages
        self.response_reserve = response_reserve
        
        # Load existing summary if available
        if summary_storage_path and summary_storage_path.exists():
            self._load_summary()
    
    def _load_summary(self):
        """Load summary from disk."""
        try:
            with open(self.summary_storage_path) as f:
                data = json.load(f)
                self._summary = ContextSummary.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            self._summary = None
    
    def _save_summary(self):
        """Save summary to disk."""
        if self._summary and self.summary_storage_path:
            self.summary_storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.summary_storage_path, "w") as f:
                json.dump(self._summary.to_dict(), f, indent=2)
    
    def build_messages_for_llm(
        self,
        system_prompt: str,
        all_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Build the message list to send to the LLM.
        
        Args:
            system_prompt: The system prompt
            all_messages: All conversation messages (excluding system)
        
        Returns:
            Messages list optimized to fit context window
        """
        available_tokens = self.context_limit - self.response_reserve
        
        # Start with system prompt
        messages = [{"role": "system", "content": system_prompt}]
        system_tokens = count_messages_tokens(messages, self.model)
        available_tokens -= system_tokens
        
        # Add summary if exists and applicable
        summary_tokens = 0
        summary_covers_idx = -1
        
        if self._summary and self._summary.last_message_idx < len(all_messages):
            summary_msg = self._summary.to_message()
            summary_tokens = count_messages_tokens([summary_msg], self.model)
            
            if summary_tokens < available_tokens * 0.3:  # Summary shouldn't exceed 30%
                messages.append(summary_msg)
                available_tokens -= summary_tokens
                summary_covers_idx = self._summary.last_message_idx
        
        # Add recent messages (from newest to oldest until budget exhausted)
        recent_messages = []
        start_idx = summary_covers_idx + 1  # Start after summarized messages
        
        for i in range(len(all_messages) - 1, start_idx - 1, -1):
            msg = all_messages[i]
            msg_tokens = count_messages_tokens([msg], self.model)
            
            if msg_tokens <= available_tokens or len(recent_messages) < self.min_recent_messages:
                recent_messages.insert(0, msg)
                available_tokens -= msg_tokens
            else:
                break
        
        messages.extend(recent_messages)
        return messages
    
    def should_summarize(self, all_messages: list[dict[str, Any]]) -> bool:
        """
        Determine if we should generate/update the summary.
        
        Triggers (whichever comes first):
        - N messages since last summary (max_messages_before_summary)
        - K total tokens exceeded (max_tokens_before_summary)
        """
        # Need minimum messages to summarize meaningfully
        min_messages_to_summarize = self.min_recent_messages + 4
        if len(all_messages) < min_messages_to_summarize:
            return False
        
        # Calculate messages since last summary
        if self._summary:
            messages_since_summary = len(all_messages) - self._summary.last_message_idx - 1
        else:
            messages_since_summary = len(all_messages)
        
        # TRIGGER 1: N messages threshold
        if messages_since_summary >= self.max_messages_before_summary:
            return True
        
        # TRIGGER 2: K tokens threshold
        total_tokens = count_messages_tokens(all_messages, self.model)
        if total_tokens >= self.max_tokens_before_summary:
            return True
        
        return False
    
    def get_trigger_status(self, all_messages: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Get current status relative to summarization triggers.
        Useful for /context command display.
        """
        if self._summary:
            messages_since_summary = len(all_messages) - self._summary.last_message_idx - 1
        else:
            messages_since_summary = len(all_messages)
        
        total_tokens = count_messages_tokens(all_messages, self.model)
        
        return {
            "messages_since_summary": messages_since_summary,
            "messages_threshold": self.max_messages_before_summary,
            "messages_percent": messages_since_summary / self.max_messages_before_summary * 100,
            "total_tokens": total_tokens,
            "tokens_threshold": self.max_tokens_before_summary,
            "tokens_percent": total_tokens / self.max_tokens_before_summary * 100,
            "will_trigger": self.should_summarize(all_messages),
        }
    
    async def generate_summary(
        self,
        all_messages: list[dict[str, Any]],
        up_to_idx: int | None = None,
    ) -> ContextSummary:
        """
        Generate a summary of messages.
        
        Args:
            all_messages: All conversation messages
            up_to_idx: Summarize up to this index (default: len - MIN_RECENT)
        
        Returns:
            ContextSummary object
        """
        if not self.llm:
            raise ValueError("LLM client required for summarization")
        
        if up_to_idx is None:
            up_to_idx = max(0, len(all_messages) - self.min_recent_messages - 1)
        
        # Get messages to summarize
        messages_to_summarize = all_messages[:up_to_idx + 1]
        
        if not messages_to_summarize:
            raise ValueError("No messages to summarize")
        
        # Build summarization prompt
        conversation_text = self._format_messages_for_summary(messages_to_summarize)
        
        summary_prompt = f"""Summarize this conversation concisely. Focus on:
1. Key topics discussed
2. Important decisions or conclusions
3. Any tasks completed or pending
4. Relevant context for continuing the conversation

Keep the summary under 500 words.

Conversation:
{conversation_text}

Summary:"""
        
        # Generate summary
        response = self.llm.chat([
            {"role": "user", "content": summary_prompt}
        ])
        
        summary_content = response.choices[0].message.content
        
        # Create summary object
        self._summary = ContextSummary(
            content=summary_content,
            messages_summarized=len(messages_to_summarize),
            first_message_idx=0,
            last_message_idx=up_to_idx,
            token_count=count_tokens(summary_content, self.model),
        )
        
        # Persist
        self._save_summary()
        
        return self._summary
    
    def _format_messages_for_summary(self, messages: list[dict[str, Any]]) -> str:
        """Format messages into readable text for summarization."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            
            if role == "TOOL":
                # Truncate tool results
                content = content[:500] + "..." if len(content) > 500 else content
                lines.append(f"[Tool Result]: {content}")
            elif role == "ASSISTANT" and msg.get("tool_calls"):
                tools = [tc.get("function", {}).get("name", "?") for tc in msg["tool_calls"]]
                lines.append(f"ASSISTANT: [Called tools: {', '.join(tools)}]")
                if content:
                    lines.append(f"ASSISTANT: {content}")
            else:
                lines.append(f"{role}: {content}")
        
        return "\n\n".join(lines)
    
    @property
    def summary(self) -> ContextSummary | None:
        return self._summary
```

### 7.4 Integrate with Agent

Update the Agent class to use ContextManager with per-agent settings:

```python
# supyagent/core/agent.py (updates)

class Agent:
    def __init__(self, config: AgentConfig, ...):
        # ... existing init ...
        
        # Initialize context manager with agent-specific settings
        summary_path = None
        if self.session:
            summary_path = Path(f".supyagent/sessions/{config.name}/{self.session.meta.id}_summary.json")
        
        # Get context settings from agent config (with defaults)
        ctx = config.context  # ContextSettings from YAML
        
        self.context_manager = ContextManager(
            model=config.model.provider,
            llm=self.llm,
            summary_storage_path=summary_path,
            # Pass per-agent thresholds
            max_messages_before_summary=ctx.max_messages_before_summary,  # N
            max_tokens_before_summary=ctx.max_tokens_before_summary,  # K
            min_recent_messages=ctx.min_recent_messages,
            response_reserve=ctx.response_reserve,
        )
    
    def _get_messages_for_llm(self) -> list[dict[str, Any]]:
        """
        Get messages to send to LLM, managed for context limits.
        """
        system_prompt = get_full_system_prompt(self.config)
        
        # Filter out system messages from history (we add it fresh)
        conversation_messages = [
            m for m in self.messages if m.get("role") != "system"
        ]
        
        return self.context_manager.build_messages_for_llm(
            system_prompt,
            conversation_messages,
        )
    
    def send_message(self, content: str) -> str:
        # ... existing code to add user message ...
        
        while iterations < max_iterations:
            iterations += 1
            
            # Use managed messages instead of self.messages directly
            messages_for_llm = self._get_messages_for_llm()
            
            response = self.llm.chat(
                messages=messages_for_llm,  # Changed from self.messages
                tools=self.tools if self.tools else None,
            )
            
            # ... rest of existing code ...
        
        # Check if we should summarize after this exchange
        if self.context_manager.should_summarize(self.messages):
            # Could do async or schedule for later
            self._trigger_summarization()
    
    def _trigger_summarization(self):
        """Trigger background summarization."""
        try:
            # For now, synchronous. Could be made async.
            conversation_messages = [
                m for m in self.messages if m.get("role") != "system"
            ]
            self.context_manager.generate_summary(conversation_messages)
        except Exception:
            pass  # Don't fail the conversation if summarization fails
```

### 7.5 Session Summary Storage

Update session management to handle summaries:

```python
# supyagent/core/session_manager.py (additions)

class SessionManager:
    def get_summary_path(self, agent_name: str, session_id: str) -> Path:
        """Get the path to a session's context summary."""
        return self.sessions_dir / agent_name / f"{session_id}_summary.json"
    
    def delete_session(self, agent_name: str, session_id: str) -> bool:
        """Delete a session and its summary."""
        # ... existing deletion code ...
        
        # Also delete summary if exists
        summary_path = self.get_summary_path(agent_name, session_id)
        if summary_path.exists():
            summary_path.unlink()
        
        return True
```

### 7.6 CLI Integration

Add context status to CLI:

```python
# supyagent/cli/main.py (additions)

# In the chat command, add /context meta-command:
elif cmd == "context":
    # Show context status with trigger thresholds
    status = agent.context_manager.get_trigger_status(agent.messages)
    
    console.print(f"\n[cyan]Context Status[/cyan]")
    
    if agent.context_manager.summary:
        summary = agent.context_manager.summary
        console.print(f"  [dim]Last summary:[/dim] {summary.messages_summarized} messages → {summary.token_count} tokens")
        console.print(f"  [dim]Created:[/dim] {summary.created_at.strftime('%Y-%m-%d %H:%M')}")
    else:
        console.print("  [dim]No summary yet[/dim]")
    
    console.print(f"\n[cyan]Summarization Triggers (N messages OR K tokens)[/cyan]")
    
    # Messages trigger (N)
    msg_bar = "█" * int(status['messages_percent'] / 5) + "░" * (20 - int(status['messages_percent'] / 5))
    console.print(f"  Messages: {status['messages_since_summary']:,} / {status['messages_threshold']:,} ({status['messages_percent']:.0f}%)")
    console.print(f"           [{msg_bar}]")
    
    # Tokens trigger (K)
    tok_bar = "█" * int(status['tokens_percent'] / 5) + "░" * (20 - int(status['tokens_percent'] / 5))
    console.print(f"  Tokens:   {status['total_tokens']:,} / {status['tokens_threshold']:,} ({status['tokens_percent']:.0f}%)")
    console.print(f"           [{tok_bar}]")
    
    if status['will_trigger']:
        console.print(f"\n  [yellow]⚡ Summarization will trigger on next message[/yellow]")
    continue

elif cmd == "summarize":
    # Force summarization
    console.print("[dim]Generating context summary...[/dim]")
    try:
        summary = agent.context_manager.generate_summary(agent.messages)
        console.print(f"[green]✓[/green] Summarized {summary.messages_summarized} messages")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
    continue
```

### 7.7 AgentConfig Extension

Add context management settings to agent config:

```python
# supyagent/models/agent_config.py (additions)

class ContextSettings(BaseModel):
    """Context window management settings."""
    
    auto_summarize: bool = Field(
        default=True,
        description="Automatically summarize when trigger thresholds are reached"
    )
    
    # Summarization triggers (whichever comes first)
    max_messages_before_summary: int = Field(
        default=30,
        description="Trigger summarization after N messages (since last summary)"
    )
    max_tokens_before_summary: int = Field(
        default=128_000,
        description="Trigger summarization when total tokens exceed K"
    )
    
    # Other settings
    min_recent_messages: int = Field(
        default=6,
        description="Minimum recent messages to always include (never summarized)"
    )
    response_reserve: int = Field(
        default=4096,
        description="Tokens to reserve for response"
    )


class AgentConfig(BaseModel):
    # ... existing fields ...
    
    context: ContextSettings = Field(default_factory=ContextSettings)
```

Example in YAML:
```yaml
name: assistant
# ... other config ...

context:
  auto_summarize: true
  max_messages_before_summary: 30   # N: summarize after 30 messages
  max_tokens_before_summary: 128000 # K: or when tokens exceed 128k
  min_recent_messages: 6
```

Different agents can have different thresholds:
```yaml
# agents/quick_assistant.yaml - shorter context, more frequent summaries
name: quick_assistant
context:
  max_messages_before_summary: 15
  max_tokens_before_summary: 32000

# agents/researcher.yaml - longer context for deep research
name: researcher  
context:
  max_messages_before_summary: 50
  max_tokens_before_summary: 180000
```

---

## Acceptance Criteria

1. **Token Counting**: Accurate token estimates for messages
2. **Summary Generation**: Can generate coherent summaries of conversations
3. **Context Building**: Messages sent to LLM stay within limits
4. **Full Persistence**: All messages still saved to session file
5. **N Messages Trigger**: Summarization triggers after N messages (configurable per agent)
6. **K Tokens Trigger**: Summarization triggers when total tokens exceed K (configurable per agent)
7. **Per-Agent Config**: Each agent can have different (N, K) thresholds via YAML
8. **Sensible Defaults**: Default N=30 messages, K=128,000 tokens work well for most cases
9. **Transparent Operation**: Works automatically without user intervention
10. **Manual Control**: Users can view status (with progress bars) and force summarization
11. **Session Continuity**: Summaries persist across session resumes

---

## Test Scenarios

### Scenario 1: N Messages Trigger
```
1. Create agent with max_messages_before_summary: 10
2. Have 10+ message exchanges
3. Verify summarization triggered by message count
4. Verify all messages still in session file
```

### Scenario 2: K Tokens Trigger
```
1. Create agent with max_tokens_before_summary: 5000
2. Send long messages that exceed 5000 tokens
3. Verify summarization triggered by token count
4. Verify messages in LLM call are reduced
```

### Scenario 3: Context Status Display
```
You> /context

Context Status
  Last summary: 25 messages → 850 tokens
  Created: 2024-01-15 10:30

Summarization Triggers (N messages OR K tokens)
  Messages: 8 / 30 (27%)
           [█████░░░░░░░░░░░░░░░]
  Tokens:   45,000 / 128,000 (35%)
           [███████░░░░░░░░░░░░░]
```

### Scenario 4: Approaching Trigger
```
You> /context
...
  Messages: 28 / 30 (93%)
           [██████████████████░░]
  Tokens:   95,000 / 128,000 (74%)
           [██████████████░░░░░░]

  ⚡ Summarization will trigger on next message
```

### Scenario 5: Per-Agent Thresholds
```yaml
# Test different agents with different thresholds
# agents/quick.yaml
context:
  max_messages_before_summary: 15
  max_tokens_before_summary: 32000

# agents/researcher.yaml  
context:
  max_messages_before_summary: 50
  max_tokens_before_summary: 180000
```

### Scenario 6: Resume with Summary
```bash
$ supyagent chat assistant --session abc123

Resuming session abc123
52 messages in history (25 summarized)
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Session Storage                          │
│  ┌─────────────────────┐  ┌─────────────────────┐              │
│  │  messages.jsonl     │  │  summary.json       │              │
│  │  (full history)     │  │  (context summary)  │              │
│  └─────────────────────┘  └─────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Context Manager                            │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Token        │  │ Summary      │  │ Message      │          │
│  │ Counter      │  │ Generator    │  │ Builder      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
│  Input: All Messages ──────────────────────────────────────────│
│  Output: Optimized Messages for LLM                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         LLM API                                 │
│  [system] + [summary] + [recent messages]                       │
│  (fits within context window)                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Dependencies

Add to `pyproject.toml`:
```toml
dependencies = [
    # ... existing ...
    "tiktoken>=0.5.0",  # For token counting
]
```

---

## Notes

- Token counting is approximate for non-OpenAI models (using cl100k_base)
- Summarization uses the same LLM as the agent (could be configurable)
- Consider adding "importance scoring" for messages in future iteration
- Could add async summarization to avoid blocking the conversation
- Summary quality depends on LLM capability

---

## Future Enhancements

1. **Hierarchical Memory**: Multiple summary levels (recent, medium, long-term)
2. **Importance Scoring**: Mark critical messages to always include
3. **Semantic Chunking**: Group related messages before summarizing
4. **Async Summarization**: Background summarization without blocking
5. **Summary Caching**: Cache summaries across similar conversations
6. **Configurable Summarizer**: Use different/cheaper model for summarization
