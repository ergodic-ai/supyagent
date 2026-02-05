"""
Process management tools for LLM.

These tools allow the agent to check on, interact with,
and terminate background processes.
"""

from __future__ import annotations

from typing import Any

from supyagent.core.supervisor import get_supervisor


def get_process_management_tools() -> list[dict[str, Any]]:
    """
    Get tool schemas for process management.

    Returns:
        List of OpenAI-format tool definitions
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "list_processes",
                "description": (
                    "List all running background processes (tools and agents). "
                    "Use this to see what's currently running."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "include_completed": {
                            "type": "boolean",
                            "default": False,
                            "description": "Include completed/failed processes in the list",
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_process",
                "description": (
                    "Check the status of a specific background process. "
                    "Returns status, output, and other details."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "process_id": {
                            "type": "string",
                            "description": "The process ID to check",
                        }
                    },
                    "required": ["process_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_process_output",
                "description": (
                    "Get the output (stdout/stderr) from a background process. "
                    "Useful for checking what a long-running process has produced."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "process_id": {
                            "type": "string",
                            "description": "The process ID",
                        },
                        "tail": {
                            "type": "integer",
                            "default": 100,
                            "description": "Number of lines to return from end of output",
                        },
                    },
                    "required": ["process_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "kill_process",
                "description": (
                    "Terminate a running background process. "
                    "Use when you no longer need a process or it's misbehaving."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "process_id": {
                            "type": "string",
                            "description": "The process ID to kill",
                        }
                    },
                    "required": ["process_id"],
                },
            },
        },
    ]


def is_process_tool(tool_name: str) -> bool:
    """Check if a tool name is a process management tool."""
    return tool_name in (
        "list_processes",
        "check_process",
        "get_process_output",
        "kill_process",
    )


async def execute_process_tool_async(
    tool_name: str,
    args: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute a process management tool asynchronously.

    Args:
        tool_name: Name of the tool to execute
        args: Tool arguments

    Returns:
        Result dict with ok/data or ok/error
    """
    supervisor = get_supervisor()

    if tool_name == "list_processes":
        processes = supervisor.list_processes(
            include_completed=args.get("include_completed", False)
        )
        # Simplify output for LLM
        simplified = []
        for proc in processes:
            entry = {
                "process_id": proc.get("process_id"),
                "type": proc.get("process_type"),
                "status": proc.get("status"),
                "pid": proc.get("pid"),
            }
            # Add relevant metadata
            meta = proc.get("metadata", {})
            if "agent_name" in meta:
                entry["agent"] = meta["agent_name"]
            elif "script" in meta:
                entry["tool"] = f"{meta['script']}__{meta.get('func', '?')}"
            simplified.append(entry)

        return {"ok": True, "data": simplified}

    elif tool_name == "check_process":
        process_id = args.get("process_id")
        if not process_id:
            return {"ok": False, "error": "process_id is required"}

        process = supervisor.get_process(process_id)
        if process:
            # Return relevant info for LLM
            result = {
                "process_id": process["process_id"],
                "type": process["process_type"],
                "status": process["status"],
                "pid": process.get("pid"),
                "started_at": process.get("started_at"),
                "completed_at": process.get("completed_at"),
                "exit_code": process.get("exit_code"),
            }
            # Add metadata
            meta = process.get("metadata", {})
            if meta:
                result["metadata"] = meta
            return {"ok": True, "data": result}
        else:
            return {"ok": False, "error": f"Process {process_id} not found"}

    elif tool_name == "get_process_output":
        process_id = args.get("process_id")
        if not process_id:
            return {"ok": False, "error": "process_id is required"}

        return await supervisor.get_output(
            process_id,
            tail=args.get("tail", 100),
        )

    elif tool_name == "kill_process":
        process_id = args.get("process_id")
        if not process_id:
            return {"ok": False, "error": "process_id is required"}

        return await supervisor.kill(process_id)

    return {"ok": False, "error": f"Unknown process tool: {tool_name}"}


def execute_process_tool(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """
    Execute a process management tool (sync wrapper).

    Uses the supervisor's persistent event loop.

    Args:
        tool_name: Name of the tool to execute
        args: Tool arguments

    Returns:
        Result dict with ok/data or ok/error
    """
    from supyagent.core.supervisor import run_supervisor_coroutine

    try:
        return run_supervisor_coroutine(
            execute_process_tool_async(tool_name, args)
        )
    except Exception as e:
        return {"ok": False, "error": f"Process tool failed: {e}"}
