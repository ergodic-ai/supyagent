"""
Supypowers tool integration.

Discovers and executes tools from supypowers.

Supports both synchronous (subprocess.run) and async (ProcessSupervisor) execution.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from typing import Any

from supyagent.models.agent_config import ToolPermissions


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


def is_credential_request(tool_call: Any) -> bool:
    """Check if a tool call is a credential request."""
    return tool_call.function.name == "request_credential"


def discover_tools() -> list[dict[str, Any]]:
    """
    Discover available tools from supypowers.

    Runs `supypowers docs --format json` and parses the output.

    Returns:
        List of tool definitions from supypowers
    """
    try:
        result = subprocess.run(
            ["supypowers", "docs", "--format", "json"],
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

    Returns:
        Result dict with 'ok' and 'data' or 'error'
    """
    # If supervisor features requested, use async path
    if use_supervisor or background or timeout is not None:
        return _execute_tool_sync_via_supervisor(
            script, func, args, secrets, timeout, background
        )

    # Original sync implementation for backwards compatibility
    cmd = ["supypowers", "run", f"{script}:{func}", json.dumps(args)]

    # Add secrets
    if secrets:
        for key, value in secrets.items():
            cmd.extend(["--secrets", f"{key}={value}"])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for tool execution
        )

        if result.returncode != 0:
            # Try to parse error from stderr or stdout
            error_msg = result.stderr or result.stdout or "Unknown error"
            return {"ok": False, "error": error_msg.strip()}

        return json.loads(result.stdout)

    except FileNotFoundError:
        return {"ok": False, "error": "supypowers not installed"}
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"Invalid JSON response: {e}"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Tool execution timed out"}


async def execute_tool_async(
    script: str,
    func: str,
    args: dict[str, Any],
    secrets: dict[str, str] | None = None,
    timeout: float | None = None,
    background: bool = False,
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
) -> dict[str, Any]:
    """
    Synchronous wrapper for supervisor-based tool execution.

    Uses the supervisor's persistent event loop to avoid creating/destroying
    loops and the associated 'Event loop is closed' errors.
    """
    from supyagent.core.supervisor import run_supervisor_coroutine

    try:
        return run_supervisor_coroutine(
            execute_tool_async(script, func, args, secrets, timeout, background)
        )
    except Exception as e:
        return {"ok": False, "error": f"Supervisor execution failed: {e}"}


def _matches_pattern(name: str, pattern: str) -> bool:
    """
    Check if a tool name matches a permission pattern.

    Patterns:
    - "web_search:*" matches all functions in web_search
    - "web_search:search" matches exactly web_search:search
    - "*" matches everything
    """
    if pattern == "*":
        return True

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

        # Check deny list first
        denied = any(_matches_pattern(name, pattern) for pattern in permissions.deny)
        if denied:
            continue

        # Check allow list (if specified)
        if permissions.allow:
            allowed = any(_matches_pattern(name, pattern) for pattern in permissions.allow)
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
