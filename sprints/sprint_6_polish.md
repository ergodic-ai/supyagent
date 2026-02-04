# Sprint 6: Polish & Production Readiness

**Goal**: Add streaming, rich UI, error handling, documentation, and production features.

**Duration**: ~3-4 days

**Depends on**: Sprint 5

---

## Deliverables

### 6.1 Streaming Responses

Real-time streaming of LLM responses:

```python
# supyagent/core/llm.py (updated)
from litellm import completion
from typing import Generator

class LLMClient:
    def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None
    ) -> Generator[str, None, dict]:
        """
        Stream chat responses, yielding content chunks.
        Returns the final complete message after streaming.
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True
        }
        if tools:
            kwargs["tools"] = tools
        
        response = completion(**kwargs)
        
        full_content = ""
        tool_calls = []
        
        for chunk in response:
            delta = chunk.choices[0].delta
            
            # Content streaming
            if delta.content:
                full_content += delta.content
                yield delta.content
            
            # Tool call accumulation
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    # Accumulate tool call data
                    pass
        
        # Return final message
        return {
            "role": "assistant",
            "content": full_content,
            "tool_calls": tool_calls if tool_calls else None
        }
```

### 6.2 Rich Terminal UI

Enhanced terminal output with rich:

```python
# supyagent/cli/ui.py
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

class ChatUI:
    """Rich UI for interactive chat."""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.console = Console()
    
    def print_welcome(self, session_id: str, message_count: int = 0):
        """Print welcome banner."""
        if message_count:
            self.console.print(
                Panel(
                    f"Resuming session [bold cyan]{session_id}[/bold cyan]\n"
                    f"[dim]{message_count} messages in history[/dim]",
                    title=f"💬 {self.agent_name}",
                    border_style="cyan"
                )
            )
        else:
            self.console.print(
                Panel(
                    f"New session [bold cyan]{session_id}[/bold cyan]\n"
                    f"[dim]Type /help for commands[/dim]",
                    title=f"💬 {self.agent_name}",
                    border_style="green"
                )
            )
        self.console.print()
    
    def print_streaming_response(self, stream_generator):
        """Print a streaming response with live updates."""
        self.console.print(f"[bold green]{self.agent_name}[/bold green]")
        
        with Live("", console=self.console, refresh_per_second=10) as live:
            full_text = ""
            for chunk in stream_generator:
                full_text += chunk
                # Render as markdown
                live.update(Markdown(full_text))
        
        self.console.print()
    
    def print_tool_call(self, tool_name: str, status: str = "running"):
        """Show tool execution status."""
        if status == "running":
            self.console.print(f"  [dim]⚙️  Calling {tool_name}...[/dim]")
        elif status == "success":
            self.console.print(f"  [green]✓[/green] [dim]{tool_name}[/dim]")
        elif status == "error":
            self.console.print(f"  [red]✗[/red] [dim]{tool_name}[/dim]")
    
    def print_delegation(self, agent_name: str, task_preview: str):
        """Show delegation to sub-agent."""
        self.console.print(
            Panel(
                f"[dim]{task_preview[:100]}{'...' if len(task_preview) > 100 else ''}[/dim]",
                title=f"📤 Delegating to [bold]{agent_name}[/bold]",
                border_style="yellow"
            )
        )
    
    def print_error(self, message: str):
        """Print an error message."""
        self.console.print(f"[red]Error:[/red] {message}")
    
    def print_command_output(self, output: str):
        """Print command output."""
        self.console.print(output)
    
    def prompt_credential(self, name: str, description: str) -> tuple[str, bool] | None:
        """Prompt for a credential with nice formatting."""
        self.console.print()
        self.console.print(
            Panel(
                f"[bold]{name}[/bold]\n\n{description}",
                title="🔑 Credential Required",
                border_style="yellow"
            )
        )
        
        from rich.prompt import Prompt
        import getpass
        
        value = getpass.getpass("Enter value (or press Enter to skip): ")
        if not value:
            return None
        
        save = Prompt.ask("Save for future sessions?", choices=["y", "n"], default="y")
        return value, save == "y"
    
    def show_tools_table(self, tools: list[dict]):
        """Display available tools in a table."""
        table = Table(title="Available Tools")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="dim")
        
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "No description")[:60]
            table.add_row(name, desc)
        
        self.console.print(table)
    
    def show_sessions_table(self, sessions: list, current_id: str):
        """Display sessions in a table."""
        table = Table(title="Sessions")
        table.add_column("ID", style="cyan")
        table.add_column("Title")
        table.add_column("Updated", style="dim")
        table.add_column("", style="green")
        
        for s in sessions:
            marker = "← current" if s.session_id == current_id else ""
            title = s.title or "(untitled)"
            updated = s.updated_at.strftime("%Y-%m-%d %H:%M")
            table.add_row(s.session_id, title, updated, marker)
        
        self.console.print(table)
```

### 6.3 Comprehensive Error Handling

Graceful error handling throughout:

```python
# supyagent/core/errors.py
from typing import Any

class SupyagentError(Exception):
    """Base exception for supyagent."""
    pass

class AgentNotFoundError(SupyagentError):
    """Agent configuration not found."""
    def __init__(self, name: str):
        self.name = name
        super().__init__(f"Agent '{name}' not found. Check agents/ directory.")

class SessionNotFoundError(SupyagentError):
    """Session not found."""
    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"Session '{session_id}' not found.")

class ToolExecutionError(SupyagentError):
    """Tool execution failed."""
    def __init__(self, tool_name: str, error: str):
        self.tool_name = tool_name
        self.error = error
        super().__init__(f"Tool '{tool_name}' failed: {error}")

class LLMError(SupyagentError):
    """LLM API error."""
    def __init__(self, provider: str, error: str):
        self.provider = provider
        self.error = error
        super().__init__(f"LLM error ({provider}): {error}")

class CredentialError(SupyagentError):
    """Credential-related error."""
    def __init__(self, name: str, message: str):
        self.name = name
        super().__init__(f"Credential '{name}': {message}")

class DelegationError(SupyagentError):
    """Agent delegation failed."""
    def __init__(self, target: str, error: str):
        self.target = target
        self.error = error
        super().__init__(f"Delegation to '{target}' failed: {error}")


def handle_llm_error(e: Exception) -> str:
    """Convert LLM exceptions to user-friendly messages."""
    import litellm
    
    if isinstance(e, litellm.AuthenticationError):
        return "Authentication failed. Check your API key."
    elif isinstance(e, litellm.RateLimitError):
        return "Rate limit exceeded. Please wait and try again."
    elif isinstance(e, litellm.APIConnectionError):
        return "Could not connect to the LLM provider. Check your network."
    elif isinstance(e, litellm.Timeout):
        return "Request timed out. The model may be overloaded."
    else:
        return f"LLM error: {str(e)}"
```

### 6.4 Configuration Validation

Validate agent configs on load:

```python
# supyagent/models/agent_config.py (enhanced)
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal
import yaml
from pathlib import Path

class ModelConfig(BaseModel):
    provider: str = Field(..., description="LiteLLM model identifier")
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=4096, gt=0)
    
    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        # Basic validation - could add more
        if "/" not in v and v not in ["gpt-4", "gpt-3.5-turbo"]:
            # Most LiteLLM providers use provider/model format
            pass  # Allow for now, will fail at runtime if invalid
        return v

class AgentConfig(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    description: str = ""
    version: str = "1.0"
    type: Literal["interactive", "execution"] = "interactive"
    model: ModelConfig
    system_prompt: str = Field(..., min_length=10)
    tools: ToolPermissions = Field(default_factory=ToolPermissions)
    delegates: list[str] = Field(default_factory=list)
    credentials: list[CredentialSpec] = Field(default_factory=list)
    limits: dict = Field(default_factory=dict)
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Agent name must be alphanumeric (with _ or -)")
        return v
    
    @model_validator(mode="after")
    def validate_delegates_exist(self):
        # Check that delegated agents exist
        for delegate in self.delegates:
            delegate_path = Path(f"agents/{delegate}.yaml")
            if not delegate_path.exists():
                # Warning, not error - might be created later
                pass
        return self


def load_agent_config(name: str) -> AgentConfig:
    """Load and validate an agent configuration."""
    config_path = Path(f"agents/{name}.yaml")
    
    if not config_path.exists():
        raise AgentNotFoundError(name)
    
    with open(config_path) as f:
        data = yaml.safe_load(f)
    
    try:
        return AgentConfig(**data)
    except Exception as e:
        raise SupyagentError(f"Invalid agent config '{name}': {e}")
```

### 6.5 Logging System

Structured logging for debugging:

```python
# supyagent/utils/logging.py
import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(verbose: bool = False, log_file: Path = None):
    """Configure logging for supyagent."""
    
    level = logging.DEBUG if verbose else logging.INFO
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "[%(levelname)s] %(message)s" if not verbose else
        "[%(levelname)s] %(name)s: %(message)s"
    )
    console_handler.setFormatter(console_format)
    
    # File handler (if specified)
    handlers = [console_handler]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        file_handler.setFormatter(file_format)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(level=level, handlers=handlers)
    
    # Reduce noise from libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)


# Usage in code
logger = logging.getLogger("supyagent")

# In agent.py:
# logger.debug(f"Sending message: {content[:50]}...")
# logger.info(f"Tool call: {tool_name}")
# logger.error(f"Tool execution failed: {error}")
```

### 6.6 CLI Enhancements

Polished CLI with global options:

```python
# supyagent/cli/main.py (enhanced)
import click
from pathlib import Path
from ..utils.logging import setup_logging

@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--log-file", type=click.Path(), help="Write logs to file")
@click.option("--config", type=click.Path(exists=True), help="Config file path")
@click.version_option(version="0.1.0")
@click.pass_context
def cli(ctx, verbose, log_file, config):
    """
    Supyagent - LLM agents powered by supypowers
    
    Create and interact with AI agents that can use tools, manage sessions,
    and delegate to other agents.
    
    Quick start:
    
        supyagent new myagent     # Create an agent
        supyagent chat myagent    # Start chatting
        
    For more information, visit: https://github.com/ergodic-ai/supyagent
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    
    setup_logging(
        verbose=verbose,
        log_file=Path(log_file) if log_file else None
    )


@cli.command()
@click.option("--type", "-t", "agent_type", type=click.Choice(["interactive", "execution"]), default="interactive")
@click.argument("name")
def new(name: str, agent_type: str):
    """
    Create a new agent from template.
    
    Examples:
    
        supyagent new researcher
        supyagent new summarizer --type execution
    """
    from ..templates import create_agent_from_template
    
    agents_dir = Path("agents")
    agents_dir.mkdir(exist_ok=True)
    
    agent_path = agents_dir / f"{name}.yaml"
    if agent_path.exists():
        if not click.confirm(f"Agent '{name}' already exists. Overwrite?"):
            return
    
    create_agent_from_template(name, agent_type, agent_path)
    
    click.echo(f"✓ Created agent: {agent_path}")
    click.echo(f"\nNext steps:")
    click.echo(f"  1. Edit {agent_path} to customize")
    click.echo(f"  2. Run: supyagent chat {name}")


@cli.command()
def doctor():
    """
    Check supyagent installation and configuration.
    """
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    table = Table(title="Supyagent Health Check")
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Details", style="dim")
    
    # Check supypowers
    try:
        import subprocess
        result = subprocess.run(["supypowers", "--version"], capture_output=True)
        if result.returncode == 0:
            table.add_row("supypowers", "[green]✓ OK[/green]", result.stdout.decode().strip())
        else:
            table.add_row("supypowers", "[red]✗ Error[/red]", "Command failed")
    except FileNotFoundError:
        table.add_row("supypowers", "[red]✗ Missing[/red]", "Install with: uv tool install supypowers")
    
    # Check LiteLLM
    try:
        import litellm
        table.add_row("litellm", "[green]✓ OK[/green]", f"v{litellm.__version__}")
    except ImportError:
        table.add_row("litellm", "[red]✗ Missing[/red]", "Install with: pip install litellm")
    
    # Check agents directory
    agents_dir = Path("agents")
    if agents_dir.exists():
        agents = list(agents_dir.glob("*.yaml"))
        table.add_row("agents/", "[green]✓ OK[/green]", f"{len(agents)} agent(s) found")
    else:
        table.add_row("agents/", "[yellow]! Missing[/yellow]", "Run: supyagent new <name>")
    
    # Check API keys
    import os
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
        if os.environ.get(key):
            table.add_row(key, "[green]✓ Set[/green]", "")
        else:
            table.add_row(key, "[dim]○ Not set[/dim]", "")
    
    console.print(table)
```

### 6.7 Documentation

README and usage examples:

```markdown
# README.md (to be created)

# Supyagent

LLM agents powered by [supypowers](https://github.com/ergodic-ai/supypowers) tools.

## Features

- 🤖 **Interactive & Execution Agents** - Chat or automate
- 💾 **Persistent Sessions** - Resume conversations anytime
- 🔑 **Credential Management** - Secure API key handling
- 🔗 **Multi-Agent Architecture** - Agents that delegate to other agents
- 🌐 **Any LLM Provider** - Via LiteLLM (OpenAI, Anthropic, Ollama, etc.)

## Quick Start

```bash
# Install
pip install supyagent

# Create your first agent
supyagent new myagent

# Start chatting
supyagent chat myagent
```

## Documentation

- [Getting Started](docs/getting-started.md)
- [Agent Configuration](docs/agent-config.md)
- [CLI Reference](docs/cli-reference.md)
- [Multi-Agent Patterns](docs/multi-agent.md)
```

---

## Acceptance Criteria

1. **Streaming works**: Responses stream in real-time
2. **Rich UI**: Formatted output with colors, panels, tables
3. **Error messages**: User-friendly error messages
4. **Config validation**: Invalid configs caught early
5. **Health check**: `supyagent doctor` diagnoses issues
6. **Logging**: Debug logs available with `-v`
7. **Documentation**: README and basic docs exist

---

## Test Scenarios

### Scenario 1: Streaming Response
```
You> Write a haiku about coding

assistant: (streaming word by word)
Fingers on keyboard
Logic flows like morning dew
Bug fixed, peace returns
```

### Scenario 2: Doctor Command
```bash
$ supyagent doctor

┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Check                 ┃ Status    ┃ Details                 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ supypowers            │ ✓ OK      │ v0.1.0                  │
│ litellm               │ ✓ OK      │ v1.30.0                 │
│ agents/               │ ✓ OK      │ 3 agent(s) found        │
│ OPENAI_API_KEY        │ ✓ Set     │                         │
│ ANTHROPIC_API_KEY     │ ○ Not set │                         │
└───────────────────────┴───────────┴─────────────────────────┘
```

### Scenario 3: Verbose Mode
```bash
$ supyagent chat test -v

[DEBUG] Loading agent config: agents/test.yaml
[DEBUG] Discovered 5 tools from supypowers
[DEBUG] Session loaded: abc123 (4 messages)
[INFO] Starting chat session

You> hello
[DEBUG] Sending message: hello
[DEBUG] LLM response received in 1.2s
...
```

---

## Notes

- Streaming requires careful handling of tool calls mid-stream
- Rich UI should degrade gracefully in non-TTY environments
- Consider adding `--no-color` flag for piping output
- Documentation should include examples for common use cases
- Add shell completion scripts (bash, zsh, fish)
