# Sprint 1: Foundation

**Goal**: Establish project structure and core agent loop with LiteLLM integration.

**Duration**: ~3-4 days

---

## Deliverables

### 1.1 Project Setup

- [ ] Initialize Python project with `pyproject.toml`
- [ ] Set up dependencies:
  ```toml
  [project]
  dependencies = [
    "litellm>=1.30.0",
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "click>=8.0",
    "rich>=13.0",
  ]
  ```
- [ ] Create directory structure:
  ```
  supyagent/
  ├── __init__.py
  ├── __main__.py          # CLI entry point
  ├── cli/
  │   ├── __init__.py
  │   └── main.py          # Click/Typer commands
  ├── core/
  │   ├── __init__.py
  │   ├── agent.py         # Agent class
  │   ├── llm.py           # LiteLLM wrapper
  │   └── tools.py         # supypowers integration
  ├── models/
  │   ├── __init__.py
  │   ├── agent_config.py  # Agent YAML schema
  │   └── messages.py      # Message types
  └── utils/
      ├── __init__.py
      └── paths.py         # Path resolution
  ```

### 1.2 Agent Configuration Schema

Create Pydantic models for agent YAML:

```python
# supyagent/models/agent_config.py
from pydantic import BaseModel, Field
from typing import Literal

class ModelConfig(BaseModel):
    provider: str = Field(..., description="LiteLLM model string")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=4096)

class ToolPermissions(BaseModel):
    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)

class AgentConfig(BaseModel):
    name: str
    description: str = ""
    version: str = "1.0"
    type: Literal["interactive", "execution"] = "interactive"
    model: ModelConfig
    system_prompt: str
    tools: ToolPermissions = Field(default_factory=ToolPermissions)
    delegates: list[str] = Field(default_factory=list)
```

### 1.3 LiteLLM Integration

Wrap LiteLLM for consistent usage:

```python
# supyagent/core/llm.py
import litellm
from litellm import completion

class LLMClient:
    def __init__(self, model: str, temperature: float = 0.7):
        self.model = model
        self.temperature = temperature
    
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = False
    ):
        """Send messages to LLM, optionally with tools."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
        
        return completion(**kwargs, stream=stream)
```

### 1.4 Supypowers Tool Discovery

Parse available tools from supypowers:

```python
# supyagent/core/tools.py
import subprocess
import json

def discover_tools() -> list[dict]:
    """Run `supypowers docs --format json` and parse output."""
    result = subprocess.run(
        ["supypowers", "docs", "--format", "json"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return []
    return json.loads(result.stdout)

def execute_tool(script: str, func: str, args: dict, secrets: dict = None) -> dict:
    """Execute a supypowers function and return result."""
    cmd = ["supypowers", "run", f"{script}:{func}", json.dumps(args)]
    if secrets:
        # Pass secrets as --secrets KEY=VALUE
        for k, v in secrets.items():
            cmd.extend(["--secrets", f"{k}={v}"])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout)

def tools_to_openai_schema(tools: list[dict], permissions: ToolPermissions) -> list[dict]:
    """Convert supypowers tools to OpenAI function calling format."""
    # Filter by permissions and convert to OpenAI schema
    pass
```

### 1.5 Basic Agent Class

Core agent loop without persistence:

```python
# supyagent/core/agent.py
from .llm import LLMClient
from .tools import discover_tools, execute_tool, tools_to_openai_schema

class Agent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = LLMClient(config.model.provider, config.model.temperature)
        self.messages = [{"role": "system", "content": config.system_prompt}]
        self.tools = self._load_tools()
    
    def _load_tools(self) -> list[dict]:
        """Discover and filter tools based on permissions."""
        all_tools = discover_tools()
        return tools_to_openai_schema(all_tools, self.config.tools)
    
    def send_message(self, content: str) -> str:
        """Send user message and get response, handling tool calls."""
        self.messages.append({"role": "user", "content": content})
        
        while True:
            response = self.llm.chat(self.messages, tools=self.tools)
            assistant_msg = response.choices[0].message
            self.messages.append(assistant_msg.model_dump())
            
            if not assistant_msg.tool_calls:
                return assistant_msg.content
            
            # Execute tool calls
            for tool_call in assistant_msg.tool_calls:
                result = self._execute_tool_call(tool_call)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
    
    def _execute_tool_call(self, tool_call) -> dict:
        """Parse and execute a tool call."""
        name = tool_call.function.name  # e.g., "web_search__search"
        script, func = name.split("__")  # Convention: double underscore
        args = json.loads(tool_call.function.arguments)
        return execute_tool(script, func, args)
```

### 1.6 Minimal CLI

Just enough to test the agent loop:

```python
# supyagent/cli/main.py
import click
from pathlib import Path
from ..core.agent import Agent
from ..models.agent_config import AgentConfig
import yaml

@click.group()
def cli():
    """Supyagent - LLM agents powered by supypowers"""
    pass

@cli.command()
@click.argument("agent_name")
def chat(agent_name: str):
    """Start interactive chat with an agent."""
    config_path = Path(f"agents/{agent_name}.yaml")
    if not config_path.exists():
        click.echo(f"Agent '{agent_name}' not found")
        return
    
    with open(config_path) as f:
        config = AgentConfig(**yaml.safe_load(f))
    
    agent = Agent(config)
    click.echo(f"Chatting with {config.name}. Type /quit to exit.\n")
    
    while True:
        user_input = click.prompt("You", prompt_suffix="> ")
        if user_input.strip() == "/quit":
            break
        
        response = agent.send_message(user_input)
        click.echo(f"\n{config.name}: {response}\n")

# Entry point
if __name__ == "__main__":
    cli()
```

---

## Acceptance Criteria

1. **Project runs**: `python -m supyagent chat test-agent` starts without errors
2. **LiteLLM works**: Can send messages to any LiteLLM-supported provider
3. **Tool discovery**: `discover_tools()` returns valid tool list from supypowers
4. **Tool execution**: Can execute a supypowers function and get result
5. **Basic loop**: User → LLM → Tool → LLM → Response works end-to-end

---

## Test Scenario

Create a test agent:

```yaml
# agents/test.yaml
name: test
description: A simple test agent
type: interactive

model:
  provider: anthropic/claude-3-5-sonnet-20241022
  temperature: 0.7
  max_tokens: 1024

system_prompt: |
  You are a helpful test assistant. You have access to tools via supypowers.
  When asked to do something, try to use your available tools.

tools:
  allow:
    - "*"  # Allow all for testing
```

Run: `python -m supyagent chat test`

Expected behavior:
1. User can type messages
2. Agent responds using LLM
3. Agent can call supypowers tools when appropriate
4. Tool results are incorporated into responses

---

## Notes

- Don't worry about persistence yet (Sprint 2)
- Don't worry about rich formatting yet (Sprint 5)
- Focus on getting the core loop working reliably
- Test with at least 2 different LLM providers (e.g., OpenAI + Anthropic)
