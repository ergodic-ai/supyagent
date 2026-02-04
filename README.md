# Supyagent

[![PyPI version](https://badge.fury.io/py/supyagent.svg)](https://badge.fury.io/py/supyagent)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LLM agents powered by [supypowers](https://github.com/ergodic-ai/supypowers) tools.

## Features

- 🤖 **Interactive & Execution Agents** - Chat interactively or run automated pipelines
- 🔧 **Tool Integration** - Use any supypowers function as an agent tool
- 🌐 **Any LLM Provider** - Via [LiteLLM](https://docs.litellm.ai/docs/) (OpenAI, Anthropic, Ollama, etc.)
- 📝 **YAML Configuration** - Simple, declarative agent definitions
- 🔄 **Session Persistence** - Conversations persist across sessions
- 🔐 **Credential Management** - Secure storage for API keys and secrets
- 🎭 **Multi-Agent Orchestration** - Agents can delegate tasks to other agents
- 📦 **Batch Processing** - Process multiple inputs from JSONL/CSV files

## Installation

```bash
pip install supyagent
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv pip install supyagent
```

## Quick Start

```bash
# Initialize supyagent (sets up default tools)
supyagent init

# Set up your API key (stored securely)
supyagent config set ANTHROPIC_API_KEY

# Create your first agent
supyagent new myagent

# Start chatting
supyagent chat myagent
```

## Usage

### Interactive Mode

```bash
# Create an interactive agent
supyagent new researcher

# Chat with it
supyagent chat researcher

# Resume a previous session
supyagent chat researcher --session <session-id>

# Start fresh
supyagent chat researcher --new
```

### Execution Mode

For automation and pipelines:

```bash
# Create an execution agent
supyagent new summarizer --type execution

# Run with inline input
supyagent run summarizer "Summarize this text..."

# Run with file input
supyagent run summarizer --input document.txt

# Get JSON output
supyagent run summarizer --input doc.txt --output json

# Pipe from stdin
cat article.txt | supyagent run summarizer
```

### Batch Processing

```bash
# Process multiple inputs
supyagent batch summarizer inputs.jsonl --output results.jsonl

# From CSV
supyagent batch summarizer data.csv --format csv
```

### Multi-Agent Orchestration

```bash
# Use the planning agent to orchestrate complex tasks
supyagent plan "Build a Python library for data validation"

# The planner will delegate to specialist agents (coder, writer, researcher)
```

## Configuration

### Setting Up API Keys

Supyagent securely stores your LLM API keys so you don't need to export them every time:

```bash
# Interactive setup (recommended)
supyagent config set
# Shows a menu of common providers to choose from

# Set a specific key
supyagent config set ANTHROPIC_API_KEY
supyagent config set OPENAI_API_KEY

# Import from a .env file
supyagent config import .env

# Import only specific keys
supyagent config import .env --filter OPENAI

# List configured keys
supyagent config list

# Export keys to backup
supyagent config export backup.env

# Delete a key
supyagent config delete OPENAI_API_KEY
```

Keys are encrypted and stored in `~/.supyagent/config/`. They're automatically loaded when running any agent command.

### Supported Providers

Supyagent uses LiteLLM, supporting 100+ providers:

```yaml
model:
  # OpenAI
  provider: openai/gpt-4o
  
  # Anthropic
  provider: anthropic/claude-3-5-sonnet-20241022
  
  # Ollama (local)
  provider: ollama/llama3
  
  # Azure OpenAI
  provider: azure/gpt-4
  
  # Google
  provider: gemini/gemini-pro
```

## Agent Configuration

Agents are defined in YAML files in the `agents/` directory:

```yaml
name: researcher
description: An AI research assistant
type: interactive  # or "execution"

model:
  provider: anthropic/claude-3-5-sonnet-20241022
  temperature: 0.7
  max_tokens: 4096

system_prompt: |
  You are a helpful research assistant...

tools:
  allow:
    - "*"           # Allow all tools
    # or be specific:
    # - "web:*"     # All functions in web.py
    # - "math:calc" # Specific function
  deny:
    - "dangerous:*" # Block specific tools

# For multi-agent support
delegates:
  - coder
  - writer
```

## CLI Reference

### Core Commands

| Command | Description |
|---------|-------------|
| `supyagent init` | Initialize project with default tools |
| `supyagent new <name>` | Create a new agent |
| `supyagent list` | List all agents |
| `supyagent show <name>` | Show agent details |
| `supyagent chat <name>` | Interactive chat session |
| `supyagent run <name> <task>` | Execute agent (non-interactive) |
| `supyagent batch <name> <file>` | Process multiple inputs |
| `supyagent plan <task>` | Run task through planning agent |

### Session Commands

| Command | Description |
|---------|-------------|
| `supyagent sessions <name>` | List sessions for an agent |
| `supyagent chat <name> --session <id>` | Resume a session |
| `supyagent chat <name> --new` | Start a new session |

### Agent Management

| Command | Description |
|---------|-------------|
| `supyagent agents` | List active agent instances |
| `supyagent cleanup` | Remove completed instances |

### Config Commands

| Command | Description |
|---------|-------------|
| `supyagent config set [KEY]` | Set an API key (interactive menu if no key specified) |
| `supyagent config list` | List all configured keys |
| `supyagent config import <file>` | Import keys from .env file |
| `supyagent config export <file>` | Export keys to .env file |
| `supyagent config delete <key>` | Delete a stored key |

### In-Chat Commands

While chatting, use these commands:

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/tools` | List available tools |
| `/history` | Show conversation history |
| `/sessions` | List your sessions |
| `/switch <id>` | Switch to another session |
| `/new` | Start a new session |
| `/save <title>` | Save session with title |
| `/export <file>` | Export session to file |
| `/model <name>` | Switch LLM model |
| `/creds` | Manage stored credentials |
| `/clear` | Clear conversation history |
| `/quit` | Exit the chat |

## Bundled Tools

Running `supyagent init` installs these default tools:

### Shell (`shell.py`)
- `run_command` - Execute shell commands
- `run_script` - Run multi-line bash scripts
- `which` - Find executable paths
- `get_env` - Get environment variables

### Files (`files.py`)
- `read_file` / `write_file` - File I/O
- `list_directory` - List files with glob patterns
- `copy_file` / `move_file` / `delete_file` - File operations
- `create_directory` - Create directories
- `file_info` - Get file metadata

You can add your own tools by creating Python files in `supypowers/`.

## Project Structure

```
your-project/
├── agents/              # Agent YAML definitions
│   ├── assistant.yaml
│   ├── planner.yaml
│   └── researcher.yaml
├── supypowers/          # Tool definitions (Python)
│   ├── shell.py         # Shell commands (bundled)
│   ├── files.py         # File operations (bundled)
│   └── my_tools.py      # Your custom tools
└── .supyagent/          # Runtime data (gitignore this)
    ├── sessions/        # Conversation history
    ├── credentials/     # Encrypted secrets
    └── registry.json    # Agent instances
```

## Credential Management

Supyagent securely manages API keys and secrets:

```bash
# In chat, agents can request credentials
🔑 Credential requested: SLACK_API_TOKEN
   Purpose: To send messages to Slack
Enter value (or press Enter to skip): ****
Save for future sessions? [Y/n]: y

# View stored credentials
/creds
```

Credentials are encrypted at rest using Fernet encryption.

## Multi-Agent Architecture

Create orchestrator agents that delegate to specialists:

```yaml
# agents/planner.yaml
name: planner
type: interactive

delegates:
  - researcher  # Can delegate research tasks
  - coder       # Can delegate coding tasks
  - writer      # Can delegate writing tasks

system_prompt: |
  You are a planning agent. Break down complex tasks
  and delegate to specialist agents...
```

The planner can then:
1. Analyze tasks
2. Create execution plans
3. Delegate subtasks to appropriate agents
4. Synthesize results

## Development

```bash
# Clone the repo
git clone https://github.com/ergodic-ai/supyagent
cd supyagent

# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .
```

## License

MIT

## Related Projects

- [supypowers](https://github.com/ergodic-ai/supypowers) - Tool execution framework
- [LiteLLM](https://github.com/BerriAI/litellm) - LLM provider abstraction
