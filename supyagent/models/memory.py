"""
Pydantic models for the entity-graph memory system.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class EntityNode(BaseModel):
    """A node in the entity graph (person, project, technology, etc.)."""

    id: str
    name: str
    entity_type: str  # Person, Project, Technology, etc.
    summary: str = ""
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    source_session: str = ""


class EntityEdge(BaseModel):
    """A relationship edge between two entities."""

    id: str
    source_id: str
    target_id: str
    relationship: str  # prefers, works_on, etc.
    fact: str  # Natural language: "User prefers Python"
    confidence: float = 1.0
    valid_from: datetime | None = None
    valid_until: datetime | None = None  # None = still valid
    created_at: datetime
    source_session: str = ""


class Episode(BaseModel):
    """A summary of an interaction episode (conversation segment)."""

    id: str
    session_id: str
    agent_id: str
    summary: str
    observations: list[str] = Field(default_factory=list)
    outcome: str | None = None
    entity_refs: list[str] = Field(default_factory=list)  # Entity IDs mentioned
    timestamp: datetime


class OntologyType(BaseModel):
    """A type in the dynamic ontology (entity type or edge type)."""

    name: str
    description: str = ""
    base_type: str | None = None  # Parent in hierarchy
    source: str = "seed"  # seed | llm_proposed | user_defined
    usage_count: int = 0
    first_seen: datetime | None = None


class MemoryExtractionResult(BaseModel):
    """Result of extracting memories from conversation messages."""

    entities: list[EntityNode] = Field(default_factory=list)
    edges: list[EntityEdge] = Field(default_factory=list)
    episode: Episode | None = None
