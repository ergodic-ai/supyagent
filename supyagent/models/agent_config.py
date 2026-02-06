"""
Agent configuration models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """LLM model configuration."""

    provider: str = Field(..., description="LiteLLM model identifier (e.g., 'anthropic/claude-3-5-sonnet-20241022')")
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=4096, gt=0)
    max_retries: int = Field(default=3, ge=0, description="Max retries on transient LLM errors")
    retry_delay: float = Field(default=1.0, gt=0, description="Initial retry delay in seconds")
    retry_backoff: float = Field(default=2.0, gt=1, description="Exponential backoff multiplier")


class ToolPermissions(BaseModel):
    """Tool permission settings."""

    allow: list[str] = Field(default_factory=list, description="Allowed tool patterns (e.g., 'web_search:*')")
    deny: list[str] = Field(default_factory=list, description="Denied tool patterns")


class CredentialSpec(BaseModel):
    """Credential specification."""

    name: str = Field(..., description="Environment variable name")
    description: str = Field(default="", description="Description of what this credential is for")
    required: bool = Field(default=False)


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


class ToolExecutionSettings(BaseModel):
    """Per-tool execution settings."""

    timeout: float = Field(default=30, description="Seconds before auto-backgrounding")
    mode: Literal["sync", "background", "auto"] = Field(
        default="auto",
        description="Execution mode: sync (wait), background (immediate return), auto (timeout-based)"
    )


class SupervisorSettings(BaseModel):
    """Process supervisor settings for tool and agent execution."""

    # Default timeout before auto-backgrounding (seconds)
    default_timeout: float = Field(
        default=30,
        description="Default seconds to wait before auto-backgrounding a tool"
    )

    # Behavior when timeout is reached
    on_timeout: Literal["background", "kill", "wait"] = Field(
        default="background",
        description="Action when timeout reached: background (keep running), kill (terminate), wait (block forever)"
    )

    # Max concurrent background processes
    max_background_processes: int = Field(
        default=10,
        description="Maximum number of concurrent background processes"
    )

    # Patterns for tools that should always run in background
    force_background_patterns: list[str] = Field(
        default_factory=lambda: ["*__start_server*", "*__run_server*", "*__serve*"],
        description="Glob patterns for tools that always run in background (e.g., servers)"
    )

    # Patterns for tools that should always run synchronously
    force_sync_patterns: list[str] = Field(
        default_factory=lambda: ["*__read_*", "*__list_*", "*__get_*"],
        description="Glob patterns for tools that always wait for completion"
    )

    # Per-tool settings (overrides defaults)
    tool_settings: dict[str, ToolExecutionSettings] = Field(
        default_factory=dict,
        description="Per-tool execution settings, keyed by tool name pattern"
    )

    # Default delegation execution mode
    default_delegation_mode: Literal["subprocess", "in_process"] = Field(
        default="subprocess",
        description="Default mode for delegating to child agents"
    )

    # Delegation timeout
    delegation_timeout: float = Field(
        default=300,
        description="Default timeout for delegated agent tasks (seconds)"
    )

    def resolve_tool_settings(self, tool_name: str) -> tuple[float, bool, bool]:
        """
        Resolve execution settings for a specific tool.

        Returns:
            Tuple of (timeout, force_background, force_sync)
        """
        import fnmatch

        timeout = self.default_timeout
        force_background = any(
            fnmatch.fnmatch(tool_name, p) for p in self.force_background_patterns
        )
        force_sync = any(
            fnmatch.fnmatch(tool_name, p) for p in self.force_sync_patterns
        )

        # Per-tool settings override
        for pattern, settings in self.tool_settings.items():
            if fnmatch.fnmatch(tool_name, pattern):
                timeout = settings.timeout
                if settings.mode == "background":
                    force_background = True
                elif settings.mode == "sync":
                    force_sync = True
                break

        return timeout, force_background, force_sync


class AgentConfig(BaseModel):
    """
    Agent configuration loaded from YAML.
    """

    name: str = Field(..., min_length=1, max_length=50)
    description: str = Field(default="")
    version: str = Field(default="1.0")
    type: Literal["interactive", "execution"] = Field(default="interactive")
    model: ModelConfig
    system_prompt: str = Field(..., min_length=1)
    tools: ToolPermissions = Field(default_factory=ToolPermissions)
    delegates: list[str] = Field(default_factory=list)
    credentials: list[CredentialSpec] = Field(default_factory=list)
    limits: dict = Field(default_factory=dict)
    will_create_tools: bool = Field(
        default=False,
        description="If true, agent will be instructed how to create new supypowers tools"
    )
    context: ContextSettings = Field(
        default_factory=ContextSettings,
        description="Context window management settings"
    )
    supervisor: SupervisorSettings = Field(
        default_factory=SupervisorSettings,
        description="Process supervisor settings for tool and agent execution"
    )


# Instructions injected when will_create_tools is True
TOOL_CREATION_INSTRUCTIONS = """

---

## Creating New Tools

**If you need a capability that doesn't exist as a tool, you should create one.**

You can create new tools using supypowers. Tools are Python functions that you can call.

### How to Create a New Tool

1. **Create a Python file** in the `supypowers/` directory (e.g., `supypowers/my_tool.py`)

2. **Use this template:**

```python
# /// script
# dependencies = ["pydantic"]  # Add any packages you need (httpx, beautifulsoup4, etc.)
# ///
from pydantic import BaseModel, Field

class MyToolInput(BaseModel):
    \"\"\"Input for my_tool function.\"\"\"
    value: str = Field(..., description="Describe what this field is for")
    optional_field: int = Field(default=10, description="Optional with default")

class MyToolOutput(BaseModel):
    \"\"\"Output for my_tool function.\"\"\"
    ok: bool
    result: str | None = None
    error: str | None = None

def my_tool(input: MyToolInput) -> MyToolOutput:
    \"\"\"
    Describe what this function does.
    
    Examples:
        >>> my_tool({"value": "test"})
    \"\"\"
    try:
        # Your implementation here
        return MyToolOutput(ok=True, result=f"Processed: {input.value}")
    except Exception as e:
        return MyToolOutput(ok=False, error=str(e))
```

### Rules (Important!)

1. **One parameter named `input`** — Function must be `def func(input: MyModel)`
2. **Pydantic BaseModel** — Both input and output must be Pydantic models
3. **No print()** — It breaks JSON output
4. **Return errors, don't raise** — Use `ok: bool` pattern in output
5. **Declare dependencies** — Add all packages to the `# /// script` block
6. **Use environment variables for secrets** — Access via `os.environ.get("API_KEY")`

### Common Patterns

**HTTP requests:**
```python
# /// script
# dependencies = ["pydantic", "httpx"]
# ///
import httpx

def fetch_url(input: FetchInput) -> FetchOutput:
    resp = httpx.get(input.url)
    return FetchOutput(ok=True, status=resp.status_code, body=resp.text)
```

**Using secrets:**
```python
import os

def call_api(input: ApiInput) -> ApiOutput:
    api_key = os.environ.get("API_KEY")
    if not api_key:
        return ApiOutput(ok=False, error="API_KEY not set")
    # Use api_key in your request
```

After creating the tool, it will be automatically available for use.
"""


def get_full_system_prompt(config: "AgentConfig") -> str:
    """
    Get the full system prompt, including tool creation instructions if enabled.
    """
    prompt = config.system_prompt
    if config.will_create_tools:
        prompt += TOOL_CREATION_INSTRUCTIONS
    return prompt


class AgentNotFoundError(Exception):
    """Raised when an agent configuration is not found."""

    def __init__(self, name: str):
        self.name = name
        super().__init__(f"Agent '{name}' not found. Check agents/ directory.")


class AgentConfigError(Exception):
    """Raised when an agent configuration is invalid, with a user-friendly message."""

    def __init__(self, name: str, issues: list[str]):
        self.name = name
        self.issues = issues
        msg = f"Invalid configuration for agent '{name}':\n" + "\n".join(
            f"  - {issue}" for issue in issues
        )
        super().__init__(msg)


def _friendly_validation_errors(name: str, exc: ValidationError) -> AgentConfigError:
    """Convert Pydantic ValidationError to a user-friendly AgentConfigError."""
    issues: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(x) for x in error["loc"])
        msg = error["msg"]
        err_type = error["type"]

        if err_type == "missing":
            hint = ""
            if loc == "model.provider":
                hint = " (e.g., 'anthropic/claude-3-5-sonnet-20241022' or 'openrouter/google/gemini-2.5-flash')"
            elif loc == "system_prompt":
                hint = " (multi-line string starting with '|' in YAML)"
            elif loc == "name":
                hint = " (1-50 characters)"
            issues.append(f"{loc} is required{hint}")
        elif "less_than_or_equal" in err_type or "greater_than" in err_type:
            issues.append(f"{loc}: {msg}")
        elif err_type == "string_too_short":
            issues.append(f"{loc} cannot be empty")
        elif err_type == "literal_error":
            allowed = error.get("ctx", {}).get("expected", "")
            issues.append(f"{loc}: must be one of {allowed}")
        else:
            issues.append(f"{loc}: {msg}")

    return AgentConfigError(name, issues)


def validate_agent_config(
    config: AgentConfig,
    agents_dir: Path | None = None,
) -> list[str]:
    """
    Validate an agent configuration beyond Pydantic type checks.

    Checks:
    - Model provider is recognized by litellm
    - Delegate names reference existing agent files
    - Tool permission patterns are non-empty strings

    Args:
        config: Validated AgentConfig to check
        agents_dir: Directory containing agent YAML files

    Returns:
        List of warning/issue strings (empty = all good)
    """
    if agents_dir is None:
        agents_dir = Path("agents")

    issues: list[str] = []

    # Check model provider
    try:
        import litellm

        litellm.get_model_info(config.model.provider)
    except Exception:
        # Not fatal — could be a custom endpoint or unknown model
        issues.append(
            f"model.provider: '{config.model.provider}' not recognized by litellm. "
            "It may still work if you have a custom endpoint configured."
        )

    # Check delegates exist
    for delegate in config.delegates:
        delegate_path = agents_dir / f"{delegate}.yaml"
        if not delegate_path.exists():
            available = [f.stem for f in agents_dir.glob("*.yaml")] if agents_dir.exists() else []
            hint = f" Available: {', '.join(available)}" if available else ""
            issues.append(f"delegates: '{delegate}' not found in {agents_dir}/.{hint}")

    # Check tool patterns are valid
    for pattern in config.tools.allow + config.tools.deny:
        if not pattern or not isinstance(pattern, str):
            issues.append(f"tools: empty or invalid pattern found in allow/deny list")

    return issues


def load_agent_config(name: str, agents_dir: Path | None = None) -> AgentConfig:
    """
    Load and validate an agent configuration from YAML.

    Args:
        name: Agent name (without .yaml extension)
        agents_dir: Directory containing agent YAML files (default: ./agents)

    Returns:
        Validated AgentConfig

    Raises:
        AgentNotFoundError: If agent YAML doesn't exist
        AgentConfigError: If YAML is invalid (with friendly messages)
    """
    if agents_dir is None:
        agents_dir = Path("agents")

    config_path = agents_dir / f"{name}.yaml"

    if not config_path.exists():
        raise AgentNotFoundError(name)

    with open(config_path) as f:
        data = yaml.safe_load(f)

    if data is None:
        raise AgentConfigError(name, ["YAML file is empty"])

    try:
        return AgentConfig(**data)
    except ValidationError as e:
        raise _friendly_validation_errors(name, e) from e
