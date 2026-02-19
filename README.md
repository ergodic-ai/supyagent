# Supyagent

[![PyPI version](https://badge.fury.io/py/supyagent.svg)](https://badge.fury.io/py/supyagent)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Give your AI agents 50+ cloud tools — Gmail, Slack, GitHub, Calendar, Drive, and more — with one CLI.

## Get Started in 60 Seconds

```bash
# 1. Install
pip install supyagent

# 2. Connect your accounts (opens browser to authorize)
supyagent connect

# 3. Run a tool
supyagent service run gmail_list_messages '{"maxResults": 5}'
```

That's it. You just read your Gmail from the command line.

---

## What Is Supyagent?

Supyagent connects AI agents and developer tools to third-party services through a unified CLI. You authenticate once on the [dashboard](https://app.supyagent.com), connect your integrations (Google, Slack, GitHub, etc.), and then call any tool from the terminal or from your agent framework.

**Supported services:** Gmail, Google Calendar, Google Drive, Google Slides, Google Sheets, Google Docs, Slack, GitHub, Discord, Notion, Microsoft 365 (Outlook, Calendar, OneDrive), Twitter/X, LinkedIn, HubSpot, Telegram, WhatsApp, Calendly, Linear, Pipedrive, Resend, and more.

## CLI Reference

| Command | Description |
|---------|-------------|
| `supyagent connect` | Authenticate with the service (device auth flow) |
| `supyagent disconnect` | Remove stored credentials |
| `supyagent status` | Show connection status and available tools |
| `supyagent service tools` | List all available cloud tools |
| `supyagent service run <tool> '<json>'` | Execute a cloud tool |
| `supyagent inbox` | View and manage incoming webhook events |
| `supyagent skills generate` | Generate skill files for AI coding assistants |
| `supyagent config set/list/delete` | Manage encrypted API keys |
| `supyagent doctor` | Diagnose your setup |

---

## Using Tools

### List available tools

```bash
supyagent service tools
```

Filter by provider:

```bash
supyagent service tools --provider google
supyagent service tools --provider slack
```

### Run a tool

```bash
# Send a Slack message
supyagent service run slack_send_message '{"channel": "#general", "text": "Hello from supyagent"}'

# List calendar events
supyagent service run calendar_list_events '{"maxResults": 10}'

# Create a GitHub issue
supyagent service run github_create_issue '{"owner": "myorg", "repo": "myrepo", "title": "Bug fix", "body": "Details here"}'
```

You can also use colon syntax (`gmail:list_messages`) or read args from a file:

```bash
echo '{"q": "from:boss@company.com"}' | supyagent service run gmail_list_messages --input -
supyagent service run gmail_list_messages --input args.json
```

### Output format

Every tool returns JSON to stdout:

```json
{
  "ok": true,
  "data": {
    "messages": [
      {"id": "abc123", "from": "alice@example.com", "subject": "Hello", "snippet": "..."}
    ]
  }
}
```

On error:

```json
{
  "ok": false,
  "error": "Permission denied for gmail_send_message: Forbidden"
}
```

Status messages go to stderr, so the JSON output is always clean for piping.

---

## Generate Skill Files for AI Coding Assistants

Supyagent can generate skill files that let AI coding assistants (Claude Code, Codex CLI, OpenCode, Cursor, Copilot, Windsurf) use your connected services directly.

### Auto-detect and generate

```bash
supyagent skills generate
```

This detects AI tool folders (`.claude/`, `.codex/`, `.agents/`, `.cursor/`, `.copilot/`, `.windsurf/`) in the current directory and generates skill files in each.

### Generate for a specific tool

```bash
# Write to a specific directory
supyagent skills generate -o .claude/skills

# Write to all detected folders without prompting
supyagent skills generate --all

# Preview to stdout
supyagent skills generate --stdout
```

### What gets generated

Each connected integration gets its own skill file. For example, if you have Google and Slack connected, you get:

```
.claude/skills/
  supy-cloud-gmail/SKILL.md
  supy-cloud-calendar/SKILL.md
  supy-cloud-drive/SKILL.md
  supy-cloud-slack/SKILL.md
```

Each SKILL.md contains YAML frontmatter and documentation for every tool in that integration:

```markdown
---
name: supy-gmail
description: >-
  Use supyagent to interact with Gmail. Available actions: list emails
  from gmail inbox, get a specific email by its message id, send an
  email via gmail. Use when the user asks to interact with Gmail.
---

# Gmail

Execute tools: `supyagent service run <tool_name> '<json>'`

### gmail_list_messages

List emails from Gmail inbox.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `maxResults` | integer | no | Number of messages to return |
| `q` | string | no | Search query using Gmail syntax |

​```bash
supyagent service run gmail_list_messages '{"q": "from:boss@company.com", "maxResults": 10}'
​```
```

After generating, your AI coding assistant will automatically use these tools when you ask it to interact with connected services.

---

## Integrate with Agent Frameworks

Supyagent tools can be used from any agent framework. There are two integration paths:

### Option A: JSON tool definitions (for LangChain, CrewAI, etc.)

Export your tools as JSON:

```bash
supyagent service tools --json
```

This returns a list of tool definitions with parameter schemas:

```json
[
  {
    "name": "google:gmail_list_messages",
    "description": "List emails from Gmail inbox.",
    "provider": "google",
    "service": "gmail",
    "method": "GET",
    "parameters": {
      "type": "object",
      "properties": {
        "maxResults": {"type": "integer", "description": "Number of messages to return"},
        "q": {"type": "string", "description": "Search query using Gmail syntax"}
      }
    }
  }
]
```

Convert these into OpenAI function-calling format for your framework:

```python
import json
import subprocess

# Get tool definitions
result = subprocess.run(["supyagent", "service", "tools", "--json"], capture_output=True, text=True)
tools = json.loads(result.stdout)

# Convert to OpenAI function-calling format
openai_tools = [
    {
        "type": "function",
        "function": {
            "name": tool["name"].split(":")[-1],
            "description": tool["description"],
            "parameters": tool["parameters"],
        },
    }
    for tool in tools
]
```

### Option B: Execute tools via subprocess

When the LLM calls a tool, execute it through supyagent:

```python
import json
import subprocess

def execute_supyagent_tool(tool_name: str, arguments: dict) -> dict:
    """Execute a supyagent cloud tool and return the result."""
    result = subprocess.run(
        ["supyagent", "service", "run", tool_name, json.dumps(arguments)],
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)

# Example: LLM decides to send an email
result = execute_supyagent_tool("gmail_send_message", {
    "to": "alice@example.com",
    "subject": "Meeting notes",
    "body": "Here are the notes from today...",
})

if result["ok"]:
    print("Email sent:", result["data"])
else:
    print("Error:", result["error"])
```

### Full example: LangChain integration

```python
import json
import subprocess
from langchain_core.tools import StructuredTool

# Load supyagent tools
result = subprocess.run(["supyagent", "service", "tools", "--json"], capture_output=True, text=True)
supyagent_tools = json.loads(result.stdout)

def make_tool(tool_def):
    """Create a LangChain tool from a supyagent tool definition."""
    tool_name = tool_def["name"].split(":")[-1]

    def run_tool(**kwargs):
        result = subprocess.run(
            ["supyagent", "service", "run", tool_name, json.dumps(kwargs)],
            capture_output=True, text=True,
        )
        return result.stdout

    return StructuredTool.from_function(
        func=run_tool,
        name=tool_name,
        description=tool_def["description"],
    )

langchain_tools = [make_tool(t) for t in supyagent_tools]
```

---

## Inbox

View webhook events from connected integrations:

```bash
# List unread events
supyagent inbox

# Filter by provider
supyagent inbox -p github

# View a specific event
supyagent inbox -i EVENT_ID

# Archive an event
supyagent inbox -a EVENT_ID

# Archive all
supyagent inbox --archive-all
```

---

## Config Management

Supyagent stores API keys encrypted in `~/.supyagent/config/`:

```bash
# Set a key interactively
supyagent config set

# Set a specific key
supyagent config set OPENAI_API_KEY

# List stored keys
supyagent config list

# Import from .env
supyagent config import .env

# Export to .env
supyagent config export backup.env

# Delete a key
supyagent config delete MY_KEY
```

---

## Development

```bash
git clone https://github.com/ergodic-ai/supyagent
cd supyagent

# Install
uv pip install -e ".[dev]"

# Test
pytest

# Lint
ruff check .
```

Only 4 runtime dependencies: `click`, `rich`, `cryptography`, `httpx`.

## License

MIT
