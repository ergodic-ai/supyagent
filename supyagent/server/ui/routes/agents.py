"""Agents dashboard UI routes."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

router = APIRouter()


# ── Request models ──────────────────────────────────────────────────


class CleanupBody(BaseModel):
    mode: str = "stale"  # "stale", "completed", "all"


class RemoveInstanceBody(BaseModel):
    instance_id: str


class DeleteSessionBody(BaseModel):
    session_id: str


# ── HTML page ───────────────────────────────────────────────────────


@router.get("/agents", response_class=HTMLResponse)
async def agents_page():
    """Serve the agents dashboard page."""
    from supyagent.server.ui.templates import render_template

    html = render_template("agents.html")
    return HTMLResponse(html)


# ── State endpoint (auto-refreshed) ─────────────────────────────────


@router.get("/api/agents/state")
async def agents_state():
    """Full dashboard state: overview stats, agent configs, and all instances."""
    from supyagent.core.registry import AgentRegistry
    from supyagent.core.session_manager import SessionManager
    from supyagent.models.agent_config import load_agent_config

    registry = AgentRegistry()
    session_mgr = SessionManager()
    agents_dir = Path("agents")

    # Collect agent configs
    agent_configs: list[dict[str, Any]] = []
    if agents_dir.exists():
        for f in sorted(agents_dir.glob("*.yaml")):
            name = f.stem
            try:
                config = load_agent_config(name)
                sessions = session_mgr.list_sessions(name)
                agent_configs.append({
                    "name": config.name,
                    "description": config.description or "",
                    "type": config.type,
                    "model": config.model.provider,
                    "delegates": config.delegates,
                    "workspace": config.workspace or "",
                    "service_enabled": config.service.enabled,
                    "memory_enabled": config.memory.enabled,
                    "schedule": (
                        {
                            "interval": config.schedule.interval,
                            "max_events": config.schedule.max_events_per_cycle,
                            "prompt": config.schedule.prompt or "",
                        }
                        if config.type == "daemon"
                        else None
                    ),
                    "session_count": len(sessions),
                    "credential_specs": [
                        c.name for c in (config.credentials or [])
                    ],
                })
            except Exception as e:
                agent_configs.append({
                    "name": name,
                    "error": str(e),
                })

    # Collect instances
    instances: list[dict[str, Any]] = []
    for inst in registry.list_all():
        instances.append({
            "instance_id": inst.instance_id,
            "name": inst.name,
            "status": inst.status,
            "parent_id": inst.parent_id,
            "depth": inst.depth,
            "created_at": inst.created_at.isoformat(),
        })

    # Count instances per agent
    instance_counts: dict[str, int] = {}
    active_counts: dict[str, int] = {}
    for inst in instances:
        instance_counts[inst["name"]] = instance_counts.get(inst["name"], 0) + 1
        if inst["status"] == "active":
            active_counts[inst["name"]] = active_counts.get(inst["name"], 0) + 1

    for agent in agent_configs:
        if "error" not in agent:
            agent["instance_count"] = instance_counts.get(agent["name"], 0)
            agent["active_count"] = active_counts.get(agent["name"], 0)

    # Stats
    stats = registry.stats()
    total_sessions = sum(
        a.get("session_count", 0) for a in agent_configs if "error" not in a
    )

    return {
        "stats": {
            **stats,
            "configs": len(agent_configs),
            "total_sessions": total_sessions,
        },
        "agents": agent_configs,
        "instances": instances,
    }


# ── Detail endpoint (on demand) ─────────────────────────────────────


@router.get("/api/agents/{name}/detail")
async def agent_detail(name: str):
    """Full detail for one agent: sessions, telemetry, memory, credentials."""
    from supyagent.core.credentials import CredentialManager
    from supyagent.core.session_manager import SessionManager
    from supyagent.models.agent_config import AgentNotFoundError, load_agent_config

    try:
        config = load_agent_config(name)
    except AgentNotFoundError:
        return {"ok": False, "error": f"Agent '{name}' not found"}

    session_mgr = SessionManager()
    cred_mgr = CredentialManager()

    # Sessions
    sessions = []
    for meta in session_mgr.list_sessions(name):
        loaded = session_mgr.load_session(name, meta.session_id)
        msg_count = len(loaded.messages) if loaded else 0
        sessions.append({
            "session_id": meta.session_id,
            "title": meta.title or "(untitled)",
            "model": meta.model,
            "created_at": meta.created_at.isoformat(),
            "updated_at": meta.updated_at.isoformat(),
            "message_count": msg_count,
        })

    # Telemetry
    telemetry = _aggregate_telemetry(name)

    # Memory stats
    memory = _get_memory_stats(name)

    # Credentials
    credentials = cred_mgr.list_credentials(name)

    return {
        "ok": True,
        "name": name,
        "description": config.description or "",
        "type": config.type,
        "model": config.model.provider,
        "sessions": sessions,
        "telemetry": telemetry,
        "memory": memory,
        "credentials": credentials,
    }


# ── Session messages endpoint ────────────────────────────────────────


@router.get("/api/agents/{name}/sessions/{session_id}")
async def session_messages(name: str, session_id: str):
    """Last 20 messages of a session."""
    from supyagent.core.session_manager import SessionManager
    from supyagent.utils.media import content_to_text

    session_mgr = SessionManager()
    session = session_mgr.load_session(name, session_id)

    if not session:
        return {"ok": False, "error": "Session not found"}

    messages = []
    for msg in session.messages[-20:]:
        content = ""
        if isinstance(msg.content, str):
            content = msg.content[:500]
        elif isinstance(msg.content, list):
            content = content_to_text(msg.content)[:500]
        elif msg.content is not None:
            content = str(msg.content)[:500]

        tool_names = []
        if msg.tool_calls:
            tool_names = [
                tc.get("function", {}).get("name", "?")
                for tc in msg.tool_calls
            ]

        messages.append({
            "type": msg.type,
            "content": content,
            "tool_names": tool_names,
            "tool_call_id": msg.tool_call_id,
            "name": msg.name,
            "ts": msg.ts.isoformat(),
        })

    return {
        "ok": True,
        "session_id": session_id,
        "title": session.meta.title or "(untitled)",
        "messages": messages,
    }


# ── Action endpoints ─────────────────────────────────────────────────


@router.post("/api/agents/cleanup")
async def agents_cleanup(body: CleanupBody):
    """Cleanup registry entries."""
    from supyagent.core.registry import AgentRegistry

    registry = AgentRegistry()
    removed = 0

    if body.mode == "stale":
        removed = registry.prune_stale()
    elif body.mode == "completed":
        removed = registry.cleanup_completed()
    elif body.mode == "all":
        all_inst = registry.list_all()
        for inst in all_inst:
            registry.cleanup(inst.instance_id)
        removed = len(all_inst)

    return {"ok": True, "removed": removed}


@router.post("/api/agents/remove-instance")
async def remove_instance(body: RemoveInstanceBody):
    """Remove a specific registry instance."""
    from supyagent.core.registry import AgentRegistry

    registry = AgentRegistry()
    inst = registry.get_instance(body.instance_id)
    if not inst:
        return {"ok": False, "error": "Instance not found"}

    registry.cleanup(body.instance_id)
    return {"ok": True, "removed": body.instance_id}


@router.post("/api/agents/{name}/delete-session")
async def delete_session(name: str, body: DeleteSessionBody):
    """Delete a session."""
    from supyagent.core.session_manager import SessionManager

    session_mgr = SessionManager()
    deleted = session_mgr.delete_session(name, body.session_id)

    if deleted:
        return {"ok": True, "deleted": body.session_id}
    return {"ok": False, "error": "Session not found"}


# ── Helpers ──────────────────────────────────────────────────────────


def _aggregate_telemetry(agent_name: str) -> dict[str, Any]:
    """Aggregate telemetry across all sessions for an agent."""
    telemetry_dir = Path(".supyagent/telemetry") / agent_name

    if not telemetry_dir.exists():
        return {"available": False}

    totals: dict[str, Any] = {
        "available": True,
        "sessions": 0,
        "turns": 0,
        "tool_calls": 0,
        "llm_calls": 0,
        "errors": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_duration_ms": 0,
    }

    for f in telemetry_dir.glob("*.jsonl"):
        totals["sessions"] += 1
        try:
            for line in f.read_text().strip().splitlines():
                event = json.loads(line)
                etype = event.get("type", "")
                if etype == "turn":
                    totals["turns"] += 1
                    totals["total_duration_ms"] += event.get("duration_ms", 0)
                elif etype == "tool_call":
                    totals["tool_calls"] += 1
                elif etype == "llm_call":
                    totals["llm_calls"] += 1
                    totals["input_tokens"] += event.get("input_tokens", 0)
                    totals["output_tokens"] += event.get("output_tokens", 0)
                elif etype == "error":
                    totals["errors"] += 1
        except (json.JSONDecodeError, OSError):
            continue

    return totals


def _get_memory_stats(agent_name: str) -> dict[str, Any]:
    """Read memory stats directly from SQLite (no LLM client needed)."""
    db_path = Path.home() / ".supyagent" / "memory" / agent_name / "memory.db"

    if not db_path.exists():
        return {"available": False}

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        entities = 0
        edges = 0
        episodes = 0

        try:
            cursor.execute("SELECT COUNT(*) FROM entity_nodes")
            entities = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("SELECT COUNT(*) FROM entity_edges")
            edges = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("SELECT COUNT(*) FROM episodes")
            episodes = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            pass

        conn.close()

        return {
            "available": True,
            "entities": entities,
            "edges": edges,
            "episodes": episodes,
        }
    except Exception:
        return {"available": False}
