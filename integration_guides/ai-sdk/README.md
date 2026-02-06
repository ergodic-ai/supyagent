# Supyagent + Vercel AI SDK Integration Guide

This guide describes how to build a frontend UI on top of the supyagent server using Vercel's AI SDK (`ai` package) and its `useChat` React hook.

## Architecture

```
Browser (React/Next.js)              supyagent server (FastAPI)
========================              ==========================

useChat() ----POST /api/chat-------> Agent.send_message_stream()
  |           (streaming)                    |
  |    <--- AI SDK Data Stream ---          JSONL session files
  |         Protocol lines                   (server-authoritative)
  |
  +--- fetch /api/agents/X/sessions/{id}/messages  (hydrate on load)
  +--- fetch /api/agents                            (agent picker)
  +--- fetch /api/agents/X/sessions                 (session list)
```

### Key principle: server-authoritative persistence

The server owns all session state. The client is a display cache.

- The server's `Agent` maintains the real message history, including tool calls, tool results, system prompts, and context summarization
- `useChat` manages the **current turn's streaming state** (optimistic user message + streaming assistant response)
- On session load or after each completed response, the client should **hydrate from the server** via the messages endpoint
- No localStorage persistence needed on the client side

Why: the agent's internal context includes things the client can't see or manage (multi-step tool chains, credential requests, context window summarization). The server is the only reliable source of truth.

---

## Server API Reference

Base URL: `http://localhost:8000` (default, configurable via `supyagent serve --port`)

### Chat (streaming)

```
POST /api/chat
Content-Type: application/json
```

Request body:
```json
{
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "agent": "assistant",
  "sessionId": "abc123"     // optional, omit to start new session
}
```

Response: streaming `text/plain; charset=utf-8` with header `x-vercel-ai-data-stream: v1`

The `messages` array is required by useChat but the server only extracts the **last user message**. Server-side session history is authoritative.

If `sessionId` is omitted, a new session is created. The session ID is available after the first response via the sessions API.

### Agents

```
GET /api/agents                    -> AgentInfo[]
GET /api/agents/{name}             -> AgentInfo
```

```typescript
interface AgentInfo {
  name: string;
  description: string;
  type: "interactive" | "execution";
  model: string;        // e.g. "openai", "anthropic", "openrouter/..."
  tools_count: number;
}
```

### Sessions

```
GET    /api/agents/{name}/sessions              -> SessionInfo[]
GET    /api/agents/{name}/sessions/{id}         -> SessionInfo
GET    /api/agents/{name}/sessions/{id}/messages -> MessageInfo[]
DELETE /api/agents/{name}/sessions/{id}         -> { ok: true }
```

```typescript
interface SessionInfo {
  session_id: string;
  agent: string;
  title: string | null;    // first user message, set automatically
  created_at: string;      // ISO 8601
  updated_at: string;      // ISO 8601
  message_count: number;
}

interface MessageInfo {
  type: "user" | "assistant" | "tool_result";
  content: string | null;
  tool_calls: ToolCall[] | null;   // present on assistant messages that invoke tools
  tool_call_id: string | null;     // present on tool_result messages
  name: string | null;             // tool name for tool_result messages
  ts: string;                      // ISO 8601
}

interface ToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;   // JSON string
  };
}
```

### Tools

```
GET /api/tools -> ToolInfo[]
```

```typescript
interface ToolInfo {
  name: string;
  description: string;
}
```

### Health

```
GET /health -> { status: "ok", service: "supyagent" }
```

---

## AI SDK Data Stream Protocol

The streaming response uses prefix-coded lines. Each line is `{code}:{json}\n`.

| Code | Name | Payload | When |
|------|------|---------|------|
| `f` | message start | `{"messageId": "..."}` | First line of every response |
| `0` | text delta | `"chunk of text"` | Each token as it arrives |
| `9` | tool call | `{"toolCallId": "...", "toolName": "...", "args": {...}}` | Agent invokes a tool |
| `a` | tool result | `{"toolCallId": "...", "result": ...}` | Tool returns a result |
| `e` | step finish | `{"finishReason": "stop"|"tool-calls", "usage": {...}, "isContinued": bool}` | After each LLM step |
| `d` | message finish | `{"finishReason": "stop", "usage": {...}}` | Final line |
| `3` | error | `"error message string"` | On any error |

Example stream for a simple text response:
```
f:{"messageId": "abc123"}
0:"Hello"
0:", how"
0:" are"
0:" you?"
e:{"finishReason": "stop", "usage": {"promptTokens": 0, "completionTokens": 0}, "isContinued": false}
d:{"finishReason": "stop", "usage": {"promptTokens": 0, "completionTokens": 0}}
```

Example stream with tool usage:
```
f:{"messageId": "def456"}
0:"Let me check that."
9:{"toolCallId": "call_abc", "toolName": "shell__run_command", "args": {"command": "ls"}}
a:{"toolCallId": "call_abc", "result": "file1.txt\nfile2.txt"}
e:{"finishReason": "tool-calls", "usage": {"promptTokens": 0, "completionTokens": 0}, "isContinued": true}
0:"Here are your files: file1.txt and file2.txt"
e:{"finishReason": "stop", "usage": {"promptTokens": 0, "completionTokens": 0}, "isContinued": false}
d:{"finishReason": "stop", "usage": {"promptTokens": 0, "completionTokens": 0}}
```

---

## Frontend Implementation

### Dependencies

```bash
npm install ai @ai-sdk/react
```

### Basic useChat Integration

```typescript
import { useChat } from "@ai-sdk/react";

function Chat({ agent, sessionId }: { agent: string; sessionId?: string }) {
  const { messages, input, handleInputChange, handleSubmit, isLoading, error } =
    useChat({
      api: "http://localhost:8000/api/chat",
      body: { agent, sessionId },
    });

  return (
    <div>
      {messages.map((m) => (
        <div key={m.id}>
          <strong>{m.role}:</strong> {m.content}
        </div>
      ))}
      <form onSubmit={handleSubmit}>
        <input value={input} onChange={handleInputChange} />
        <button type="submit" disabled={isLoading}>Send</button>
      </form>
    </div>
  );
}
```

### Session Hydration (server-authoritative pattern)

On mount or session switch, fetch the real history from the server and replace useChat's state:

```typescript
import { useChat } from "@ai-sdk/react";
import { useEffect } from "react";

function Chat({ agent, sessionId }: { agent: string; sessionId: string }) {
  const { messages, setMessages, input, handleInputChange, handleSubmit, isLoading } =
    useChat({
      api: "http://localhost:8000/api/chat",
      body: { agent, sessionId },
    });

  // Hydrate from server on mount / session change
  useEffect(() => {
    if (!sessionId) return;

    fetch(`http://localhost:8000/api/agents/${agent}/sessions/${sessionId}/messages`)
      .then((r) => r.json())
      .then((serverMessages: MessageInfo[]) => {
        // Convert server messages to useChat format
        const chatMessages = serverMessages
          .filter((m) => m.type === "user" || m.type === "assistant")
          .map((m, i) => ({
            id: `msg-${i}`,
            role: m.type as "user" | "assistant",
            content: m.content ?? "",
            createdAt: new Date(m.ts),
          }));
        setMessages(chatMessages);
      });
  }, [agent, sessionId]);

  return (/* ... same as above ... */);
}
```

### Rendering Tool Calls

`useChat` exposes `toolInvocations` on assistant messages when tools are used. Render them as collapsible UI elements:

```typescript
import { Message } from "ai";

function ChatMessage({ message }: { message: Message }) {
  return (
    <div>
      <p>{message.content}</p>

      {message.toolInvocations?.map((tool) => (
        <div key={tool.toolCallId} className="tool-call">
          <div className="tool-header">
            Tool: {tool.toolName}
          </div>
          {tool.state === "result" && (
            <pre className="tool-result">
              {typeof tool.result === "string"
                ? tool.result
                : JSON.stringify(tool.result, null, 2)}
            </pre>
          )}
        </div>
      ))}
    </div>
  );
}
```

### Agent Picker

```typescript
import { useEffect, useState } from "react";

interface AgentInfo {
  name: string;
  description: string;
  type: string;
  model: string;
  tools_count: number;
}

function AgentPicker({ onSelect }: { onSelect: (name: string) => void }) {
  const [agents, setAgents] = useState<AgentInfo[]>([]);

  useEffect(() => {
    fetch("http://localhost:8000/api/agents")
      .then((r) => r.json())
      .then(setAgents);
  }, []);

  return (
    <div>
      {agents.map((a) => (
        <button key={a.name} onClick={() => onSelect(a.name)}>
          <strong>{a.name}</strong>
          <span>{a.description}</span>
          <span>{a.tools_count} tools</span>
        </button>
      ))}
    </div>
  );
}
```

### Session List & Resume

```typescript
interface SessionInfo {
  session_id: string;
  agent: string;
  title: string | null;
  created_at: string;
  updated_at: string;
  message_count: number;
}

function SessionList({
  agent,
  onSelect,
}: {
  agent: string;
  onSelect: (sessionId: string) => void;
}) {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);

  useEffect(() => {
    fetch(`http://localhost:8000/api/agents/${agent}/sessions`)
      .then((r) => r.json())
      .then(setSessions);
  }, [agent]);

  return (
    <div>
      {sessions.map((s) => (
        <button key={s.session_id} onClick={() => onSelect(s.session_id)}>
          <span>{s.title ?? "Untitled"}</span>
          <span>{s.message_count} messages</span>
          <span>{new Date(s.updated_at).toLocaleDateString()}</span>
        </button>
      ))}
    </div>
  );
}
```

---

## Lifecycle Flows

### New conversation

1. User selects an agent (or uses default)
2. User sends first message
3. `useChat` POSTs to `/api/chat` with `{ messages, agent }` (no sessionId)
4. Server creates a new session, streams response
5. After stream completes, fetch session list to get the new session ID for subsequent requests

### Resume conversation

1. User picks a session from the session list
2. Client fetches `/api/agents/{agent}/sessions/{id}/messages` and hydrates `useChat` state via `setMessages`
3. User sends a message
4. `useChat` POSTs to `/api/chat` with `{ messages, agent, sessionId }`
5. Server loads the existing session, appends the new exchange

### Session management

- **Delete**: `DELETE /api/agents/{agent}/sessions/{id}` then refresh the session list
- **Rename**: Not yet supported via API (title is auto-set from first user message)
- **New chat**: Clear `useChat` state and omit `sessionId` on next POST

---

## CORS

The server includes CORS middleware. Default allows all origins (`*`).

For production, restrict origins:
```bash
supyagent serve --cors-origin http://localhost:3000 --cors-origin https://myapp.com
```

---

## Error Handling

Errors arrive as `3:` lines in the stream. Common cases:

| Error | Cause | UI action |
|-------|-------|-----------|
| `Agent 'X' not found` | Invalid agent name | Show agent picker |
| `No user message found` | Empty messages array | Client bug, should not happen |
| `Missing ... API Key` | Agent's LLM provider not configured | Show setup instructions |
| `CredentialRequiredError` | Tool needs a secret | Prompt user or show `supyagent config set` instructions |

`useChat` exposes errors via the `error` property:

```typescript
const { error } = useChat({ /* ... */ });

if (error) {
  return <div className="error">{error.message}</div>;
}
```

---

## Running the Server

```bash
# Install with server dependencies
pip install supyagent[serve]
# or during development:
uv pip install -e ".[serve]"

# Start
supyagent serve                         # localhost:8000
supyagent serve --port 3001             # custom port
supyagent serve --host 0.0.0.0         # expose to network
supyagent serve --reload               # auto-reload on code changes

# Interactive API docs
open http://localhost:8000/docs
```

---

## Tested curl Examples

These were verified against a running server:

```bash
# Health
curl http://localhost:8000/health

# List agents
curl http://localhost:8000/api/agents

# Get agent details
curl http://localhost:8000/api/agents/assistant

# List sessions
curl http://localhost:8000/api/agents/assistant/sessions

# Get session messages
curl http://localhost:8000/api/agents/assistant/sessions/2f4d4bb1/messages

# List tools
curl http://localhost:8000/api/tools

# Chat (new session, streaming)
curl -N -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"agent":"assistant"}'

# Chat (resume session)
curl -N -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What did we talk about?"}],"agent":"assistant","sessionId":"2f4d4bb1"}'

# Delete session
curl -X DELETE http://localhost:8000/api/agents/assistant/sessions/2f4d4bb1
```
