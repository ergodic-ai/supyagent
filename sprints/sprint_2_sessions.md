# Sprint 2: Session Management

**Goal**: Implement persistent sessions so conversations survive between runs.

**Duration**: ~2-3 days

**Depends on**: Sprint 1

---

## Deliverables

### 2.1 Session Model

Define session data structures:

```python
# supyagent/models/session.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal
import uuid

class SessionMeta(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    model: str
    title: str | None = None  # Auto-generated from first message

class Message(BaseModel):
    type: Literal["user", "assistant", "tool_result", "system"]
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None  # For tool results
    ts: datetime = Field(default_factory=datetime.utcnow)

class Session(BaseModel):
    meta: SessionMeta
    messages: list[Message] = Field(default_factory=list)
```

### 2.2 Session Storage

JSONL-based persistence:

```python
# supyagent/core/session_manager.py
from pathlib import Path
import json
from ..models.session import Session, SessionMeta, Message

class SessionManager:
    def __init__(self, base_dir: Path = Path(".supyagent/sessions")):
        self.base_dir = base_dir
    
    def _agent_dir(self, agent: str) -> Path:
        path = self.base_dir / agent
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def _session_path(self, agent: str, session_id: str) -> Path:
        return self._agent_dir(agent) / f"{session_id}.jsonl"
    
    def _current_path(self, agent: str) -> Path:
        return self._agent_dir(agent) / "current.json"
    
    def create_session(self, agent: str, model: str) -> Session:
        """Create a new session for an agent."""
        meta = SessionMeta(agent=agent, model=model)
        session = Session(meta=meta)
        self._save_session(session)
        self._set_current(agent, meta.session_id)
        return session
    
    def load_session(self, agent: str, session_id: str) -> Session | None:
        """Load a session from disk."""
        path = self._session_path(agent, session_id)
        if not path.exists():
            return None
        
        messages = []
        meta = None
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                if data.get("type") == "meta":
                    meta = SessionMeta(**data)
                else:
                    messages.append(Message(**data))
        
        return Session(meta=meta, messages=messages) if meta else None
    
    def get_current_session(self, agent: str) -> Session | None:
        """Get the current active session for an agent."""
        current_path = self._current_path(agent)
        if not current_path.exists():
            return None
        
        with open(current_path) as f:
            data = json.load(f)
        
        return self.load_session(agent, data["session_id"])
    
    def _set_current(self, agent: str, session_id: str):
        """Set the current session for an agent."""
        with open(self._current_path(agent), "w") as f:
            json.dump({"session_id": session_id}, f)
    
    def append_message(self, session: Session, message: Message):
        """Append a message to session and persist."""
        session.messages.append(message)
        session.meta.updated_at = message.ts
        
        path = self._session_path(session.meta.agent, session.meta.session_id)
        with open(path, "a") as f:
            f.write(message.model_dump_json() + "\n")
    
    def _save_session(self, session: Session):
        """Save full session (used on create)."""
        path = self._session_path(session.meta.agent, session.meta.session_id)
        with open(path, "w") as f:
            # Write meta first
            meta_dict = session.meta.model_dump()
            meta_dict["type"] = "meta"
            f.write(json.dumps(meta_dict, default=str) + "\n")
            # Write messages
            for msg in session.messages:
                f.write(msg.model_dump_json() + "\n")
    
    def list_sessions(self, agent: str) -> list[SessionMeta]:
        """List all sessions for an agent."""
        agent_dir = self._agent_dir(agent)
        sessions = []
        for path in agent_dir.glob("*.jsonl"):
            with open(path) as f:
                first_line = f.readline()
                data = json.loads(first_line)
                if data.get("type") == "meta":
                    sessions.append(SessionMeta(**data))
        return sorted(sessions, key=lambda s: s.updated_at, reverse=True)
    
    def delete_session(self, agent: str, session_id: str) -> bool:
        """Delete a session."""
        path = self._session_path(agent, session_id)
        if path.exists():
            path.unlink()
            return True
        return False
```

### 2.3 Integrate Sessions into Agent

Update Agent class to use sessions:

```python
# supyagent/core/agent.py (updated)
class Agent:
    def __init__(self, config: AgentConfig, session: Session | None = None):
        self.config = config
        self.llm = LLMClient(config.model.provider, config.model.temperature)
        self.tools = self._load_tools()
        self.session_manager = SessionManager()
        
        if session:
            self.session = session
            self.messages = self._reconstruct_messages(session)
        else:
            self.session = self.session_manager.create_session(
                config.name, config.model.provider
            )
            self.messages = [{"role": "system", "content": config.system_prompt}]
    
    def _reconstruct_messages(self, session: Session) -> list[dict]:
        """Convert session messages to LLM message format."""
        messages = [{"role": "system", "content": self.config.system_prompt}]
        for msg in session.messages:
            if msg.type == "user":
                messages.append({"role": "user", "content": msg.content})
            elif msg.type == "assistant":
                m = {"role": "assistant", "content": msg.content}
                if msg.tool_calls:
                    m["tool_calls"] = msg.tool_calls
                messages.append(m)
            elif msg.type == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content
                })
        return messages
    
    def send_message(self, content: str) -> str:
        """Send user message, persist, get response."""
        # Record user message
        user_msg = Message(type="user", content=content)
        self.session_manager.append_message(self.session, user_msg)
        self.messages.append({"role": "user", "content": content})
        
        while True:
            response = self.llm.chat(self.messages, tools=self.tools)
            assistant_msg = response.choices[0].message
            
            # Record assistant message
            asst_record = Message(
                type="assistant",
                content=assistant_msg.content,
                tool_calls=[tc.model_dump() for tc in (assistant_msg.tool_calls or [])]
            )
            self.session_manager.append_message(self.session, asst_record)
            self.messages.append(assistant_msg.model_dump())
            
            if not assistant_msg.tool_calls:
                return assistant_msg.content
            
            # Execute and record tool calls
            for tool_call in assistant_msg.tool_calls:
                result = self._execute_tool_call(tool_call)
                tool_msg = Message(
                    type="tool_result",
                    tool_call_id=tool_call.id,
                    name=tool_call.function.name,
                    content=json.dumps(result)
                )
                self.session_manager.append_message(self.session, tool_msg)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
```

### 2.4 CLI Session Commands

Update CLI with session management:

```python
# supyagent/cli/main.py (additions)

@cli.command()
@click.argument("agent_name")
@click.option("--new", is_flag=True, help="Start a new session")
@click.option("--session", "session_id", help="Resume specific session")
def chat(agent_name: str, new: bool, session_id: str | None):
    """Start interactive chat with an agent."""
    config = load_agent_config(agent_name)
    session_mgr = SessionManager()
    
    if session_id:
        session = session_mgr.load_session(agent_name, session_id)
        if not session:
            click.echo(f"Session '{session_id}' not found")
            return
    elif new:
        session = None  # Agent will create new
    else:
        session = session_mgr.get_current_session(agent_name)
    
    agent = Agent(config, session=session)
    
    if session and session.messages:
        click.echo(f"Resuming session {agent.session.meta.session_id}")
        click.echo(f"({len(session.messages)} messages in history)\n")
    else:
        click.echo(f"Starting new session {agent.session.meta.session_id}\n")
    
    # ... rest of chat loop

@cli.command()
@click.argument("agent_name")
def sessions(agent_name: str):
    """List all sessions for an agent."""
    session_mgr = SessionManager()
    sessions = session_mgr.list_sessions(agent_name)
    
    if not sessions:
        click.echo(f"No sessions found for '{agent_name}'")
        return
    
    current = session_mgr.get_current_session(agent_name)
    current_id = current.meta.session_id if current else None
    
    for s in sessions:
        marker = " (current)" if s.session_id == current_id else ""
        title = s.title or "(untitled)"
        click.echo(f"  {s.session_id}: {title}{marker}")
        click.echo(f"    Created: {s.created_at.strftime('%Y-%m-%d %H:%M')}")
        click.echo(f"    Updated: {s.updated_at.strftime('%Y-%m-%d %H:%M')}")
        click.echo()
```

### 2.5 Auto-Title Generation

Generate session titles from first user message:

```python
# supyagent/core/session_manager.py (addition)

def generate_title(self, session: Session, llm: LLMClient):
    """Generate a short title from the first user message."""
    first_user_msg = next(
        (m for m in session.messages if m.type == "user"), 
        None
    )
    if not first_user_msg or session.meta.title:
        return
    
    # Use LLM to generate title (or simple truncation)
    # For now, simple approach:
    title = first_user_msg.content[:50]
    if len(first_user_msg.content) > 50:
        title += "..."
    
    session.meta.title = title
    # Update the meta line in the file (or just keep in memory)
```

---

## Acceptance Criteria

1. **Sessions persist**: Close and reopen chat, conversation continues
2. **New session flag**: `--new` starts fresh conversation
3. **Session switching**: Can specify `--session <id>` to resume old session
4. **Session listing**: `supyagent sessions <agent>` shows all sessions
5. **JSONL format**: Session files are valid JSONL, one message per line
6. **Graceful resume**: Reconstructed context allows LLM to continue coherently

---

## Test Scenarios

### Scenario 1: Session Persistence
```bash
$ supyagent chat researcher
Starting new session abc123

You> What is the capital of France?
researcher: The capital of France is Paris.

You> /quit

$ supyagent chat researcher
Resuming session abc123
(2 messages in history)

You> What did I just ask you?
researcher: You asked me about the capital of France, which is Paris.
```

### Scenario 2: Multiple Sessions
```bash
$ supyagent chat researcher --new
Starting new session def456

You> Tell me about Python
...
You> /quit

$ supyagent sessions researcher
  def456: Tell me about Python... (current)
    Created: 2024-01-15 10:30
    Updated: 2024-01-15 10:35
    
  abc123: What is the capital of France...
    Created: 2024-01-15 10:00
    Updated: 2024-01-15 10:05

$ supyagent chat researcher --session abc123
Resuming session abc123
```

---

## Notes

- Session files in `.supyagent/` should be gitignored
- Consider adding session export to markdown (future sprint)
- Title generation can be enhanced with LLM later
- Tool results should be stored but may need size limits for large outputs
