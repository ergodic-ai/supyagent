"""Tool listing endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from supyagent.core.tools import discover_tools, supypowers_to_openai_tools
from supyagent.server.models import ToolInfo

router = APIRouter(tags=["tools"])


@router.get("/tools")
async def list_tools() -> list[ToolInfo]:
    """List all available supypowers tools."""
    sp_tools = discover_tools()
    openai_tools = supypowers_to_openai_tools(sp_tools)
    return [
        ToolInfo(
            name=t["function"]["name"],
            description=t["function"].get("description", ""),
        )
        for t in openai_tools
    ]
