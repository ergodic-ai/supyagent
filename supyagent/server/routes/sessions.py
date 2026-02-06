"""Session CRUD endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from supyagent.server.dependencies import get_agent_pool
from supyagent.server.models import MessageInfo, SessionInfo

router = APIRouter(tags=["sessions"])


@router.get("/agents/{agent_name}/sessions")
async def list_sessions(agent_name: str) -> list[SessionInfo]:
    """List all sessions for an agent."""
    pool = get_agent_pool()
    sm = pool.session_manager
    sessions = sm.list_sessions(agent_name)

    results = []
    for meta in sessions:
        loaded = sm.load_session(agent_name, meta.session_id)
        msg_count = len(loaded.messages) if loaded else 0
        results.append(
            SessionInfo(
                session_id=meta.session_id,
                agent=meta.agent,
                title=meta.title,
                created_at=meta.created_at.isoformat(),
                updated_at=meta.updated_at.isoformat(),
                message_count=msg_count,
            )
        )
    return results


@router.get("/agents/{agent_name}/sessions/{session_id}")
async def get_session(agent_name: str, session_id: str) -> SessionInfo:
    """Get a specific session."""
    pool = get_agent_pool()
    sm = pool.session_manager
    session = sm.load_session(agent_name, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionInfo(
        session_id=session.meta.session_id,
        agent=session.meta.agent,
        title=session.meta.title,
        created_at=session.meta.created_at.isoformat(),
        updated_at=session.meta.updated_at.isoformat(),
        message_count=len(session.messages),
    )


@router.get("/agents/{agent_name}/sessions/{session_id}/messages")
async def get_session_messages(agent_name: str, session_id: str) -> list[MessageInfo]:
    """Get all messages in a session."""
    pool = get_agent_pool()
    sm = pool.session_manager
    session = sm.load_session(agent_name, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return [
        MessageInfo(
            type=msg.type,
            content=msg.content,
            tool_calls=msg.tool_calls,
            tool_call_id=msg.tool_call_id,
            name=msg.name,
            ts=msg.ts.isoformat(),
        )
        for msg in session.messages
    ]


@router.delete("/agents/{agent_name}/sessions/{session_id}")
async def delete_session(agent_name: str, session_id: str) -> dict:
    """Delete a session."""
    pool = get_agent_pool()
    sm = pool.session_manager
    if sm.delete_session(agent_name, session_id):
        return {"ok": True}
    raise HTTPException(status_code=404, detail="Session not found")
