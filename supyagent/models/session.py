"""
Session models for persistent conversation history.
"""

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Get current UTC time."""
    return datetime.now(UTC)


class SessionMeta(BaseModel):
    """Metadata for a session."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent: str
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    model: str
    title: str | None = None  # Auto-generated from first message


class Message(BaseModel):
    """A single message in the conversation."""

    type: Literal["user", "assistant", "tool_result", "system"]
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None  # For tool results
    ts: datetime = Field(default_factory=_utcnow)


class Session(BaseModel):
    """A conversation session with history."""

    meta: SessionMeta
    messages: list[Message] = Field(default_factory=list)
