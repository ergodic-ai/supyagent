"""
Supypowers tool integration.

Discovers and executes tools from supypowers.

Supports both synchronous (subprocess.run) and async (ProcessSupervisor) execution.
"""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING, Any

from supyagent.models.agent_config import ToolPermissions
from supyagent.models.tool_result import ToolResult, timed_execution

if TYPE_CHECKING:
    from supyagent.core.sandbox import SandboxManager, WorkspaceValidator

# Special tool that allows LLM to request credentials from user
REQUEST_CREDENTIAL_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "request_credential",
        "description": (
            "Request an API key, token, or other credential from the user. "
            "Use this when you need authentication to use a service or API. "
            "The user will be prompted to enter the credential securely."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The environment variable name (e.g., SLACK_API_TOKEN, OPENAI_API_KEY)",
                },
                "description": {
                    "type": "string",
                    "description": "Explain to the user why this credential is needed and how to get it",
                },
                "service": {
                    "type": "string",
                    "description": "The service this credential is for (e.g., 'Slack', 'GitHub', 'OpenAI')",
                },
            },
            "required": ["name", "description"],
        },
    },
}


MEMORY_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_search",
        "description": (
            "Search your long-term memory for facts, entities, and past experiences. "
            "Use this to recall information from previous conversations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 10)",
                },
            },
            "required": ["query"],
        },
    },
}

MEMORY_WRITE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_write",
        "description": (
            "Explicitly store a fact or relationship in long-term memory. "
            "Use this when the user asks you to remember something or when you learn "
            "an important fact worth preserving."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "The entity this fact is about",
                },
                "subject_type": {
                    "type": "string",
                    "description": "Entity type (Person, Project, Technology, etc.)",
                },
                "relationship": {
                    "type": "string",
                    "description": "Relationship type (prefers, works_on, etc.)",
                },
                "object": {
                    "type": "string",
                    "description": "The related entity",
                },
                "object_type": {
                    "type": "string",
                    "description": "Entity type of the object",
                },
                "fact": {
                    "type": "string",
                    "description": "Natural language description of this fact",
                },
            },
            "required": ["subject", "fact"],
        },
    },
}

MEMORY_RECALL_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_recall",
        "description": (
            "Recall everything known about a specific entity (person, project, technology, etc.)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entity_name": {
                    "type": "string",
                    "description": "Name of the entity to recall",
                },
            },
            "required": ["entity_name"],
        },
    },
}

MEMORY_TOOLS = [MEMORY_SEARCH_TOOL, MEMORY_WRITE_TOOL, MEMORY_RECALL_TOOL]


def is_memory_tool(name: str) -> bool:
    """Check if a tool name is a memory tool."""
    return name in ("memory_search", "memory_write", "memory_recall")


def is_credential_request(tool_call: Any) -> bool:
    """Check if a tool call is a credential request."""
    return tool_call.function.name == "request_credential"


def discover_tools(
    sandbox: SandboxManager | None = None,
) -> list[dict[str, Any]]:
    """
    Discover available tools from supypowers.

    Runs `supypowers docs --format json` and parses the output.
    When a sandbox is provided, discovery runs inside the container.

    Args:
        sandbox: Optional sandbox manager for container-based discovery

    Returns:
        List of tool definitions from supypowers
    """
    try:
        cmd = ["supypowers", "docs", "--format", "json"]
        if sandbox:
            sandbox.ensure_started()
            cmd = sandbox.wrap_command(cmd)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return []

        return json.loads(result.stdout)

    except FileNotFoundError:
        # supypowers not installed
        return []
    except json.JSONDecodeError:
        return []
    except subprocess.TimeoutExpired:
        return []


def execute_tool(
    script: str,
    func: str,
    args: dict[str, Any],
    secrets: dict[str, str] | None = None,
    timeout: float | None = None,
    background: bool = False,
    use_supervisor: bool = False,
    sandbox: SandboxManager | None = None,
    workspace_validator: WorkspaceValidator | None = None,
) -> dict[str, Any]:
    """
    Execute a supypowers function.

    Args:
        script: Script name (e.g., 'web_search')
        func: Function name (e.g., 'search')
        args: Function arguments as a dict
        secrets: Optional secrets to pass as environment variables
        timeout: Override default timeout (seconds)
        background: Force background execution (returns immediately)
        use_supervisor: Use ProcessSupervisor for execution (enables timeout/background)
        sandbox: Optional sandbox manager for container-based execution
        workspace_validator: Optional validator for soft workspace boundary checks

    Returns:
        Result dict with 'ok' and 'data' or 'error'
    """
    tool_name = f"{script}__{func}"

    # Soft workspace validation (when sandbox is off but workspace is set)
    if workspace_validator:
        error = workspace_validator.check_tool_args(tool_name, args)
        if error:
            return ToolResult.fail(
                error,
                error_type="workspace_violation",
                tool_name=tool_name,
            ).to_dict()

    # If supervisor features requested, use async path
    if use_supervisor or background or timeout is not None:
        with timed_execution() as timing:
            result = _execute_tool_sync_via_supervisor(
                script, func, args, secrets, timeout, background, sandbox=sandbox
            )
        result.setdefault("tool_name", tool_name)
        result.setdefault("duration_ms", timing.get("duration_ms"))
        return result

    # Original sync implementation for backwards compatibility
    cmd = ["supypowers", "run", f"{script}:{func}", json.dumps(args)]

    # Add secrets
    if secrets:
        for key, value in secrets.items():
            cmd.extend(["--secrets", f"{key}={value}"])

    # Wrap command for sandbox execution
    if sandbox:
        cmd = sandbox.wrap_command(cmd)

    with timed_execution() as timing:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout for tool execution
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                return ToolResult.fail(
                    error_msg.strip(),
                    error_type="execution_error",
                    tool_name=tool_name,
                    duration_ms=timing.get("duration_ms"),
                ).to_dict()

            raw = json.loads(result.stdout)
            raw.setdefault("tool_name", tool_name)
            raw.setdefault("duration_ms", timing.get("duration_ms"))
            return raw

        except FileNotFoundError:
            return ToolResult.fail(
                "supypowers not installed",
                error_type="tool_not_found",
                tool_name=tool_name,
            ).to_dict()
        except json.JSONDecodeError as e:
            return ToolResult.fail(
                f"Invalid JSON response: {e}",
                error_type="execution_error",
                tool_name=tool_name,
                duration_ms=timing.get("duration_ms"),
            ).to_dict()
        except subprocess.TimeoutExpired:
            return ToolResult.fail(
                "Tool execution timed out",
                error_type="timeout",
                tool_name=tool_name,
                duration_ms=timing.get("duration_ms"),
            ).to_dict()


async def execute_tool_async(
    script: str,
    func: str,
    args: dict[str, Any],
    secrets: dict[str, str] | None = None,
    timeout: float | None = None,
    background: bool = False,
    sandbox: SandboxManager | None = None,
) -> dict[str, Any]:
    """
    Execute a supypowers function asynchronously using the ProcessSupervisor.

    Features:
    - Non-blocking execution
    - Automatic timeout handling with auto-backgrounding
    - Process lifecycle management

    Args:
        script: Script name (e.g., 'web_search')
        func: Function name (e.g., 'search')
        args: Function arguments as a dict
        secrets: Optional secrets to pass as environment variables
        timeout: Override default timeout (seconds)
        background: Force background execution (returns immediately)
        sandbox: Optional sandbox manager for container-based execution

    Returns:
        Result dict with 'ok' and 'data' or 'error'
    """
    from supyagent.core.supervisor import get_supervisor

    tool_name = f"{script}__{func}"
    cmd = ["supypowers", "run", f"{script}:{func}", json.dumps(args)]

    # Add secrets to command
    if secrets:
        for key, value in secrets.items():
            cmd.extend(["--secrets", f"{key}={value}"])

    # Wrap command for sandbox execution
    if sandbox:
        cmd = sandbox.wrap_command(cmd)

    supervisor = get_supervisor()

    return await supervisor.execute(
        cmd,
        process_type="tool",
        tool_name=tool_name,
        timeout=timeout,
        force_background=background,
        metadata={"script": script, "func": func, "args": args},
    )


def _execute_tool_sync_via_supervisor(
    script: str,
    func: str,
    args: dict[str, Any],
    secrets: dict[str, str] | None = None,
    timeout: float | None = None,
    background: bool = False,
    sandbox: SandboxManager | None = None,
) -> dict[str, Any]:
    """
    Synchronous wrapper for supervisor-based tool execution.

    Uses the supervisor's persistent event loop to avoid creating/destroying
    loops and the associated 'Event loop is closed' errors.
    """
    from supyagent.core.supervisor import run_supervisor_coroutine

    try:
        return run_supervisor_coroutine(
            execute_tool_async(script, func, args, secrets, timeout, background, sandbox=sandbox)
        )
    except Exception as e:
        return {"ok": False, "error": f"Supervisor execution failed: {e}"}


def _matches_pattern(
    name: str, pattern: str, metadata: dict[str, Any] | None = None
) -> bool:
    """
    Check if a tool name matches a permission pattern.

    Patterns:
    - "*" matches everything
    - "web_search:*" matches all functions in web_search
    - "web_search:search" matches exactly web_search:search
    - "service:*" matches all service tools (tools with metadata)
    - "service:gmail:*" matches service tools where service starts with 'gmail'
    - "service:google:*" matches service tools where provider is 'google'
    """
    if pattern == "*":
        return True

    # Service tool patterns: service:*, service:gmail:*, service:google:*
    if pattern.startswith("service:"):
        if metadata is None:
            return False  # Not a service tool
        service_pattern = pattern[len("service:"):]
        if service_pattern == "*":
            return True  # Match all service tools
        provider = metadata.get("provider", "")
        service = metadata.get("service", "")
        if service_pattern.endswith(":*"):
            prefix = service_pattern[:-2]
            return provider == prefix or service.startswith(prefix)
        # Exact match on service or provider
        return service == service_pattern or provider == service_pattern

    if pattern.endswith(":*"):
        script_pattern = pattern[:-2]
        script_name = name.split("__")[0] if "__" in name else name.split(":")[0]
        return script_name == script_pattern

    # Exact match (convert : to __ for comparison)
    normalized_pattern = pattern.replace(":", "__")
    return name == normalized_pattern or name == pattern


def filter_tools(
    tools: list[dict[str, Any]],
    permissions: ToolPermissions,
) -> list[dict[str, Any]]:
    """
    Filter tools based on permissions.

    Args:
        tools: List of tool definitions
        permissions: Allow/deny patterns

    Returns:
        Filtered list of tools
    """
    if not permissions.allow and not permissions.deny:
        # No restrictions - allow all
        return tools

    filtered = []

    for tool in tools:
        name = tool.get("function", {}).get("name", "")
        metadata = tool.get("metadata")

        # Check deny list first
        denied = any(
            _matches_pattern(name, pattern, metadata) for pattern in permissions.deny
        )
        if denied:
            continue

        # Check allow list (if specified)
        if permissions.allow:
            allowed = any(
                _matches_pattern(name, pattern, metadata) for pattern in permissions.allow
            )
            if not allowed:
                continue

        filtered.append(tool)

    return filtered


def supypowers_to_openai_tools(sp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert supypowers tool definitions to OpenAI function calling format.

    Supypowers docs output format:
    [
        {
            "script": "/path/to/script.py",
            "functions": [
                {
                    "name": "function_name",
                    "description": "...",
                    "input_schema": {...}
                }
            ]
        }
    ]

    OpenAI format:
    {
        "type": "function",
        "function": {
            "name": "script__function_name",
            "description": "...",
            "parameters": {...}
        }
    }
    """
    import os

    openai_tools = []

    for script_entry in sp_tools:
        script_path = script_entry.get("script", "")
        functions = script_entry.get("functions", [])

        # Extract script name from path (e.g., "files" from "/path/to/files.py")
        script_name = os.path.splitext(os.path.basename(script_path))[0]

        # Skip __init__ files with no functions
        if script_name == "__init__" and not functions:
            continue

        for func_def in functions:
            func_name = func_def.get("name", "")
            description = func_def.get("description", "No description")
            input_schema = func_def.get("input_schema", {"type": "object", "properties": {}})

            # Use double underscore to join script:function (since : isn't allowed in function names)
            name = f"{script_name}__{func_name}"

            openai_tool = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": input_schema,
                },
            }

            openai_tools.append(openai_tool)

    return openai_tools
