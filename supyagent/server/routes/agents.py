"""Agent listing endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from supyagent.models.agent_config import AgentNotFoundError, load_agent_config
from supyagent.server.models import AgentInfo

router = APIRouter(tags=["agents"])


@router.get("/agents")
async def list_agents() -> list[AgentInfo]:
    """List all available agents."""
    agents_dir = Path("agents")
    if not agents_dir.exists():
        return []

    results = []
    for yaml_file in sorted(agents_dir.glob("*.yaml")):
        try:
            config = load_agent_config(yaml_file.stem)
            results.append(
                AgentInfo(
                    name=config.name,
                    description=config.description,
                    type=config.type,
                    model=config.model.provider,
                    tools_count=len(config.tools.allow),
                )
            )
        except Exception:
            continue
    return results


@router.get("/agents/{agent_name}")
async def get_agent(agent_name: str) -> AgentInfo:
    """Get details for a specific agent."""
    try:
        config = load_agent_config(agent_name)
    except AgentNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    return AgentInfo(
        name=config.name,
        description=config.description,
        type=config.type,
        model=config.model.provider,
        tools_count=len(config.tools.allow),
    )
