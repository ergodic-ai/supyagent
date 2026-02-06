"""Pydantic models for the API layer."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A message in the useChat format."""

    role: Literal["user", "assistant", "system", "tool"]
    content: str | list[dict[str, Any]]
    id: str | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatRequest(BaseModel):
    """POST /api/chat request body from useChat."""

    messages: list[ChatMessage]
    agent: str = Field(default="assistant", description="Agent name")
    session_id: str | None = Field(default=None, alias="sessionId", description="Session ID to resume")

    model_config = {"populate_by_name": True}


class AgentInfo(BaseModel):
    """Agent summary for list responses."""

    name: str
    description: str
    type: str
    model: str
    tools_count: int


class SessionInfo(BaseModel):
    """Session summary for list responses."""

    session_id: str
    agent: str
    title: str | None
    created_at: str
    updated_at: str
    message_count: int


class MessageInfo(BaseModel):
    """Message in a session."""

    type: str
    content: str | list[dict[str, Any]] | None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    ts: str


class ToolInfo(BaseModel):
    """Tool summary for list responses."""

    name: str
    description: str
