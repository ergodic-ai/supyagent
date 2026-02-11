"""
Entity-graph memory system with SQLite + FTS5 storage.

Provides long-term memory across sessions via entity extraction,
relationship tracking, and hybrid keyword search.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from supyagent.models.memory import (
    EntityEdge,
    EntityNode,
    Episode,
    MemoryExtractionResult,
)

if TYPE_CHECKING:
    from supyagent.core.llm import LLMClient

logger = logging.getLogger(__name__)

# Seed ontology types
_SEED_ENTITY_TYPES = [
    ("Person", "A human individual"),
    ("Organization", "A company, team, or group"),
    ("Project", "A software project or initiative"),
    ("Technology", "A programming language, framework, library, or tool"),
    ("Concept", "An abstract idea, pattern, or methodology"),
    ("Location", "A physical or virtual location"),
    ("Artifact", "A file, document, URL, or other concrete artifact"),
    ("Event", "A meeting, deadline, incident, or other time-bound occurrence"),
]

_SEED_EDGE_TYPES = [
    ("prefers", "Subject prefers or favors the object"),
    ("works_on", "Subject is working on the object"),
    ("depends_on", "Subject depends on or requires the object"),
    ("created", "Subject created the object"),
    ("relates_to", "General relationship between subject and object"),
    ("occurred_at", "Event occurred at a location or time"),
    ("caused_by", "Subject was caused by the object"),
    ("resolved_by", "Subject was resolved by the object"),
    ("knows", "Subject knows or is familiar with the object"),
    ("part_of", "Subject is part of the object"),
]

# Signal detection keywords (case-insensitive)
_SIGNAL_KEYWORDS = [
    r"\bi prefer\b",
    r"\bi use\b",
    r"\bi like\b",
    r"\bi want\b",
    r"\bremember\b",
    r"\bwe decided\b",
    r"\bthe plan is\b",
    r"\blet's go with\b",
    r"\bmy name is\b",
    r"\bi work\b",
    r"\bimportant\b",
    r"\balways\b",
    r"\bnever\b",
]

_SIGNAL_PATTERN = re.compile("|".join(_SIGNAL_KEYWORDS), re.IGNORECASE)


def _short_id() -> str:
    return uuid4().hex[:8]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryManager:
    """
    Manages entity-graph memory with SQLite + FTS5.

    Storage lives at ~/.supyagent/memory/{agent_name}/memory.db.
    """

    def __init__(
        self,
        agent_name: str,
        llm: LLMClient,
        enabled: bool = True,
        extraction_threshold: int = 5,
        retrieval_limit: int = 10,
    ):
        self.agent_name = agent_name
        self.llm = llm
        self.enabled = enabled
        self.retrieval_limit = retrieval_limit
        self.db_path = Path.home() / ".supyagent" / "memory" / agent_name / "memory.db"
        self._pending_messages: list[dict] = []
        self._extraction_threshold = extraction_threshold
        self._init_db()

    # ── Database setup ──────────────────────────────────────────────

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    summary TEXT DEFAULT '',
                    properties TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    source_session TEXT DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS edges (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relationship TEXT NOT NULL,
                    fact TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    valid_from TEXT,
                    valid_until TEXT,
                    created_at TEXT NOT NULL,
                    source_session TEXT DEFAULT '',
                    FOREIGN KEY (source_id) REFERENCES entities(id),
                    FOREIGN KEY (target_id) REFERENCES entities(id)
                );

                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    observations TEXT DEFAULT '[]',
                    outcome TEXT,
                    entity_refs TEXT DEFAULT '[]',
                    timestamp TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS entity_types (
                    name TEXT PRIMARY KEY,
                    description TEXT DEFAULT '',
                    base_type TEXT,
                    source TEXT DEFAULT 'seed',
                    usage_count INTEGER DEFAULT 0,
                    first_seen TEXT
                );

                CREATE TABLE IF NOT EXISTS edge_types (
                    name TEXT PRIMARY KEY,
                    description TEXT DEFAULT '',
                    base_type TEXT,
                    source TEXT DEFAULT 'seed',
                    usage_count INTEGER DEFAULT 0,
                    first_seen TEXT
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
                    name, summary, content=entities, content_rowid=rowid
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS edges_fts USING fts5(
                    fact, content=edges, content_rowid=rowid
                );
            """)

            # Seed ontology types
            now_str = _now().isoformat()
            for name, desc in _SEED_ENTITY_TYPES:
                conn.execute(
                    "INSERT OR IGNORE INTO entity_types (name, description, source, first_seen) VALUES (?, ?, 'seed', ?)",
                    (name, desc, now_str),
                )
            for name, desc in _SEED_EDGE_TYPES:
                conn.execute(
                    "INSERT OR IGNORE INTO edge_types (name, description, source, first_seen) VALUES (?, ?, 'seed', ?)",
                    (name, desc, now_str),
                )

            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ── Storage methods ─────────────────────────────────────────────

    def add_entity(self, entity: EntityNode) -> str:
        conn = self._connect()
        try:
            existing = conn.execute("SELECT id FROM entities WHERE id = ?", (entity.id,)).fetchone()
            if existing:
                conn.execute(
                    """UPDATE entities SET name=?, entity_type=?, summary=?, properties=?,
                       updated_at=?, source_session=? WHERE id=?""",
                    (
                        entity.name,
                        entity.entity_type,
                        entity.summary,
                        json.dumps(entity.properties),
                        entity.updated_at.isoformat(),
                        entity.source_session,
                        entity.id,
                    ),
                )
                # Update FTS
                rowid = conn.execute("SELECT rowid FROM entities WHERE id = ?", (entity.id,)).fetchone()
                if rowid:
                    conn.execute(
                        "INSERT OR REPLACE INTO entities_fts(rowid, name, summary) VALUES (?, ?, ?)",
                        (rowid[0], entity.name, entity.summary),
                    )
            else:
                conn.execute(
                    """INSERT INTO entities (id, name, entity_type, summary, properties,
                       created_at, updated_at, source_session)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        entity.id,
                        entity.name,
                        entity.entity_type,
                        entity.summary,
                        json.dumps(entity.properties),
                        entity.created_at.isoformat(),
                        entity.updated_at.isoformat(),
                        entity.source_session,
                    ),
                )
                rowid = conn.execute("SELECT rowid FROM entities WHERE id = ?", (entity.id,)).fetchone()
                if rowid:
                    conn.execute(
                        "INSERT INTO entities_fts(rowid, name, summary) VALUES (?, ?, ?)",
                        (rowid[0], entity.name, entity.summary),
                    )

            self._ensure_ontology_type(entity.entity_type, "entity", conn)
            conn.commit()
        finally:
            conn.close()
        return entity.id

    def add_edge(self, edge: EntityEdge) -> str:
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO edges (id, source_id, target_id, relationship, fact,
                   confidence, valid_from, valid_until, created_at, source_session)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    edge.id,
                    edge.source_id,
                    edge.target_id,
                    edge.relationship,
                    edge.fact,
                    edge.confidence,
                    edge.valid_from.isoformat() if edge.valid_from else None,
                    edge.valid_until.isoformat() if edge.valid_until else None,
                    edge.created_at.isoformat(),
                    edge.source_session,
                ),
            )
            rowid = conn.execute("SELECT rowid FROM edges WHERE id = ?", (edge.id,)).fetchone()
            if rowid:
                conn.execute(
                    "INSERT INTO edges_fts(rowid, fact) VALUES (?, ?)",
                    (rowid[0], edge.fact),
                )
            self._ensure_ontology_type(edge.relationship, "edge", conn)
            conn.commit()
        finally:
            conn.close()
        return edge.id

    def add_episode(self, episode: Episode) -> str:
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO episodes (id, session_id, agent_id, summary, observations,
                   outcome, entity_refs, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    episode.id,
                    episode.session_id,
                    episode.agent_id,
                    episode.summary,
                    json.dumps(episode.observations),
                    episode.outcome,
                    json.dumps(episode.entity_refs),
                    episode.timestamp.isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return episode.id

    def invalidate_edge(self, edge_id: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE edges SET valid_until = ? WHERE id = ?",
                (_now().isoformat(), edge_id),
            )
            conn.commit()
        finally:
            conn.close()

    def get_entity(self, entity_id: str) -> EntityNode | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
            if not row:
                return None
            return self._row_to_entity(row)
        finally:
            conn.close()

    def get_entity_by_name(self, name: str) -> EntityNode | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM entities WHERE LOWER(name) = LOWER(?)", (name,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_entity(row)
        finally:
            conn.close()

    def get_edges_for_entity(self, entity_id: str) -> list[EntityEdge]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT * FROM edges
                   WHERE (source_id = ? OR target_id = ?) AND valid_until IS NULL""",
                (entity_id, entity_id),
            ).fetchall()
            return [self._row_to_edge(r) for r in rows]
        finally:
            conn.close()

    def _ensure_ontology_type(
        self, type_name: str, category: str, conn: sqlite3.Connection
    ) -> None:
        table = "entity_types" if category == "entity" else "edge_types"
        existing = conn.execute(f"SELECT name FROM {table} WHERE name = ?", (type_name,)).fetchone()
        if existing:
            conn.execute(f"UPDATE {table} SET usage_count = usage_count + 1 WHERE name = ?", (type_name,))
        else:
            conn.execute(
                f"INSERT INTO {table} (name, source, usage_count, first_seen) VALUES (?, 'llm_proposed', 1, ?)",
                (type_name, _now().isoformat()),
            )

    # ── Retrieval (FTS5 hybrid search) ──────────────────────────────

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Hybrid search: FTS5 keyword search over entities and edges."""
        conn = self._connect()
        results: list[dict] = []
        try:
            # Escape FTS5 special characters
            fts_query = self._fts_escape(query)
            if not fts_query.strip():
                return results

            # Search entities
            entity_rows = conn.execute(
                """SELECT e.*, rank FROM entities_fts
                   JOIN entities e ON entities_fts.rowid = e.rowid
                   WHERE entities_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, limit),
            ).fetchall()

            for row in entity_rows:
                entity = self._row_to_entity(row)
                edges = self.get_edges_for_entity(entity.id)
                results.append({
                    "type": "entity",
                    "entity": entity,
                    "edges": edges,
                    "score": abs(row["rank"]),
                })

            # Search edges
            edge_rows = conn.execute(
                """SELECT ed.*, rank FROM edges_fts
                   JOIN edges ed ON edges_fts.rowid = ed.rowid
                   WHERE edges_fts MATCH ? AND ed.valid_until IS NULL
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, limit),
            ).fetchall()

            for row in edge_rows:
                edge = self._row_to_edge(row)
                # Only add if not already covered by an entity result
                already_covered = any(
                    r["type"] == "entity"
                    and (edge.source_id == r["entity"].id or edge.target_id == r["entity"].id)
                    for r in results
                )
                if not already_covered:
                    results.append({
                        "type": "edge",
                        "edge": edge,
                        "score": abs(row["rank"]),
                    })

            # Sort by score (higher is better, FTS5 rank is negative)
            results.sort(key=lambda r: r["score"], reverse=True)
            return results[:limit]
        finally:
            conn.close()

    @staticmethod
    def _fts_escape(query: str) -> str:
        """Escape a query for FTS5: extract alphanumeric tokens only."""
        # Strip everything except letters, numbers, spaces
        cleaned = re.sub(r"[^\w\s]", " ", query)
        tokens = cleaned.split()
        if not tokens:
            return ""
        # Quote each token to treat as literal, connect with OR
        # Drop single-character tokens (noise)
        escaped = " OR ".join(f'"{t}"' for t in tokens if len(t) > 1)
        return escaped or ""

    # ── Entity Resolution ───────────────────────────────────────────

    def _resolve_entity(self, name: str, entity_type: str) -> EntityNode | None:
        """Find existing entity by name match (case-insensitive)."""
        # Exact match first
        entity = self.get_entity_by_name(name)
        if entity:
            return entity

        # FTS search for fuzzy match
        conn = self._connect()
        try:
            fts_query = self._fts_escape(name)
            if not fts_query.strip():
                return None
            rows = conn.execute(
                """SELECT e.* FROM entities_fts
                   JOIN entities e ON entities_fts.rowid = e.rowid
                   WHERE entities_fts MATCH ?
                   LIMIT 5""",
                (fts_query,),
            ).fetchall()

            for row in rows:
                # Check if it's a close enough match (same type, similar name)
                if row["entity_type"] == entity_type and row["name"].lower() == name.lower():
                    return self._row_to_entity(row)
            return None
        finally:
            conn.close()

    # ── Extraction Pipeline ─────────────────────────────────────────

    def extract_from_messages(
        self, messages: list[dict], session_id: str
    ) -> MemoryExtractionResult:
        """Run LLM extraction on messages, deduplicate, and store."""
        # Format messages as conversation text
        conversation = self._format_conversation(messages)

        # Get existing ontology for context
        ontology_context = self._get_ontology_context()

        extraction_prompt = f"""Extract structured knowledge from this conversation.

Known entity types: {ontology_context['entity_types']}
Known relationship types: {ontology_context['edge_types']}

You may propose new types if the existing ones don't fit.

Return ONLY valid JSON (no markdown, no explanation) with this structure:
{{
  "entities": [
    {{"name": "...", "type": "...", "summary": "one-line description", "properties": {{}}}}
  ],
  "relationships": [
    {{"source": "entity name", "target": "entity name", "relationship": "type", "fact": "natural language fact", "confidence": 0.0-1.0}}
  ],
  "episode": {{
    "summary": "what happened in this conversation segment",
    "observations": ["key observation 1", "key observation 2"],
    "outcome": "result or null"
  }}
}}

Rules:
- Only extract facts that are explicitly stated or strongly implied
- Use existing entity/relationship types when possible
- Set confidence < 1.0 for inferred facts
- Keep summaries concise (one sentence)
- If nothing noteworthy, return empty lists

Conversation:
{conversation}"""

        try:
            response = self.llm.chat([{"role": "user", "content": extraction_prompt}])
            raw_content = response.choices[0].message.content or ""

            # Parse JSON from response (handle markdown code blocks)
            extracted = self._parse_json_response(raw_content)
        except Exception as e:
            logger.warning("Memory extraction failed: %s", e)
            return MemoryExtractionResult()

        if not extracted:
            return MemoryExtractionResult()

        now = _now()
        result_entities: list[EntityNode] = []
        result_edges: list[EntityEdge] = []

        # Process entities
        for ent_data in extracted.get("entities", []):
            name = ent_data.get("name", "").strip()
            entity_type = ent_data.get("type", "Concept").strip()
            if not name:
                continue

            # Resolve against existing entities
            existing = self._resolve_entity(name, entity_type)
            if existing:
                # Update existing entity
                existing.summary = ent_data.get("summary", existing.summary)
                existing.updated_at = now
                existing.properties.update(ent_data.get("properties", {}))
                self.add_entity(existing)
                result_entities.append(existing)
            else:
                # Create new entity
                entity = EntityNode(
                    id=_short_id(),
                    name=name,
                    entity_type=entity_type,
                    summary=ent_data.get("summary", ""),
                    properties=ent_data.get("properties", {}),
                    created_at=now,
                    updated_at=now,
                    source_session=session_id,
                )
                self.add_entity(entity)
                result_entities.append(entity)

        # Process relationships
        for rel_data in extracted.get("relationships", []):
            source_name = rel_data.get("source", "").strip()
            target_name = rel_data.get("target", "").strip()
            if not source_name or not target_name:
                continue

            # Resolve source and target entities
            source = self.get_entity_by_name(source_name)
            target = self.get_entity_by_name(target_name)
            if not source or not target:
                continue

            edge = EntityEdge(
                id=_short_id(),
                source_id=source.id,
                target_id=target.id,
                relationship=rel_data.get("relationship", "relates_to"),
                fact=rel_data.get("fact", ""),
                confidence=rel_data.get("confidence", 1.0),
                valid_from=now,
                created_at=now,
                source_session=session_id,
            )
            self.add_edge(edge)
            result_edges.append(edge)

        # Process episode
        episode_data = extracted.get("episode")
        result_episode = None
        if episode_data and episode_data.get("summary"):
            episode = Episode(
                id=_short_id(),
                session_id=session_id,
                agent_id=self.agent_name,
                summary=episode_data["summary"],
                observations=episode_data.get("observations", []),
                outcome=episode_data.get("outcome"),
                entity_refs=[e.id for e in result_entities],
                timestamp=now,
            )
            self.add_episode(episode)
            result_episode = episode

        return MemoryExtractionResult(
            entities=result_entities,
            edges=result_edges,
            episode=result_episode,
        )

    # ── Signal Detection ────────────────────────────────────────────

    def has_memory_signal(self, messages: list[dict]) -> bool:
        """Quick heuristic check — should we bother extracting from these messages?"""
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                # Multimodal: extract text parts
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
            if not isinstance(content, str):
                continue

            # Check for signal keywords
            if _SIGNAL_PATTERN.search(content):
                return True

            # Long messages often contain extractable info
            if len(content) > 500:
                return True

            # Tool calls suggest actionable context
            if msg.get("tool_calls"):
                return True

        return False

    # ── Pending Message Tracking ────────────────────────────────────

    def mark_pending(self, messages: list[dict]) -> None:
        """Add messages to pending extraction queue."""
        self._pending_messages.extend(messages)

    def flush_pending(
        self, session_id: str, force: bool = False
    ) -> MemoryExtractionResult | None:
        """Extract from pending messages if threshold met or force=True."""
        if not self._pending_messages:
            return None

        if not force and len(self._pending_messages) < self._extraction_threshold:
            return None

        messages = list(self._pending_messages)
        self._pending_messages.clear()

        try:
            return self.extract_from_messages(messages, session_id)
        except Exception as e:
            logger.warning("Memory extraction from pending messages failed: %s", e)
            return None

    # ── Context Injection ───────────────────────────────────────────

    def get_memory_context(self, query: str, limit: int | None = None) -> str:
        """Search memories relevant to query, format as context string for system prompt."""
        if limit is None:
            limit = self.retrieval_limit
        results = self.search(query, limit=limit)
        if not results:
            return ""

        lines: list[str] = ["[Agent Memory]"]

        # Collect entities and facts
        entities_seen: list[str] = []
        facts: list[str] = []

        for r in results:
            if r["type"] == "entity":
                entity: EntityNode = r["entity"]
                label = f"{entity.name} ({entity.entity_type})"
                entities_seen.append(label)

                for edge in r.get("edges", []):
                    fact_line = f"- {edge.fact}"
                    if edge.confidence < 1.0:
                        fact_line += f" (confidence: {edge.confidence:.2f})"
                    facts.append(fact_line)
            elif r["type"] == "edge":
                edge = r["edge"]
                fact_line = f"- {edge.fact}"
                if edge.confidence < 1.0:
                    fact_line += f" (confidence: {edge.confidence:.2f})"
                facts.append(fact_line)

        if entities_seen:
            lines.append(f"Known entities: {', '.join(entities_seen)}")
        if facts:
            lines.append("Key facts:")
            lines.extend(facts)

        # Recent episodes
        episodes = self._get_recent_episodes(limit=3)
        if episodes:
            lines.append("Recent episodes:")
            for ep in episodes:
                lines.append(f"- {ep.summary}")

        return "\n".join(lines)

    # ── Ontology Management ─────────────────────────────────────────

    def get_ontology_stats(self) -> dict:
        """Return current entity/edge type counts and usage stats."""
        conn = self._connect()
        try:
            entity_types = conn.execute(
                "SELECT name, usage_count, source FROM entity_types ORDER BY usage_count DESC"
            ).fetchall()
            edge_types = conn.execute(
                "SELECT name, usage_count, source FROM edge_types ORDER BY usage_count DESC"
            ).fetchall()

            entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            edge_count = conn.execute(
                "SELECT COUNT(*) FROM edges WHERE valid_until IS NULL"
            ).fetchone()[0]
            episode_count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

            return {
                "entity_count": entity_count,
                "edge_count": edge_count,
                "episode_count": episode_count,
                "entity_types": [
                    {"name": r["name"], "usage_count": r["usage_count"], "source": r["source"]}
                    for r in entity_types
                ],
                "edge_types": [
                    {"name": r["name"], "usage_count": r["usage_count"], "source": r["source"]}
                    for r in edge_types
                ],
            }
        finally:
            conn.close()

    # ── Internal helpers ────────────────────────────────────────────

    def _row_to_entity(self, row: sqlite3.Row) -> EntityNode:
        props = row["properties"]
        if isinstance(props, str):
            props = json.loads(props)
        return EntityNode(
            id=row["id"],
            name=row["name"],
            entity_type=row["entity_type"],
            summary=row["summary"] or "",
            properties=props,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            source_session=row["source_session"] or "",
        )

    @staticmethod
    def _row_to_edge(row: sqlite3.Row) -> EntityEdge:
        return EntityEdge(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            relationship=row["relationship"],
            fact=row["fact"],
            confidence=row["confidence"],
            valid_from=datetime.fromisoformat(row["valid_from"]) if row["valid_from"] else None,
            valid_until=datetime.fromisoformat(row["valid_until"]) if row["valid_until"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            source_session=row["source_session"] or "",
        )

    def _get_recent_episodes(self, limit: int = 3) -> list[Episode]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            return [
                Episode(
                    id=r["id"],
                    session_id=r["session_id"],
                    agent_id=r["agent_id"],
                    summary=r["summary"],
                    observations=json.loads(r["observations"]) if r["observations"] else [],
                    outcome=r["outcome"],
                    entity_refs=json.loads(r["entity_refs"]) if r["entity_refs"] else [],
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                )
                for r in rows
            ]
        finally:
            conn.close()

    def _get_ontology_context(self) -> dict[str, str]:
        conn = self._connect()
        try:
            entity_types = conn.execute(
                "SELECT name FROM entity_types ORDER BY usage_count DESC LIMIT 20"
            ).fetchall()
            edge_types = conn.execute(
                "SELECT name FROM edge_types ORDER BY usage_count DESC LIMIT 20"
            ).fetchall()
            return {
                "entity_types": ", ".join(r["name"] for r in entity_types),
                "edge_types": ", ".join(r["name"] for r in edge_types),
            }
        finally:
            conn.close()

    @staticmethod
    def _format_conversation(messages: list[dict]) -> str:
        lines: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
            if not isinstance(content, str):
                content = str(content)

            if role == "TOOL":
                # Truncate long tool results
                if len(content) > 500:
                    content = content[:500] + "..."
                lines.append(f"[Tool Result]: {content}")
            elif role == "ASSISTANT" and msg.get("tool_calls"):
                names = [
                    tc.get("function", {}).get("name", "?") for tc in msg.get("tool_calls", [])
                ]
                lines.append(f"ASSISTANT: [Called: {', '.join(names)}]")
                if content:
                    lines.append(f"ASSISTANT: {content}")
            elif role != "SYSTEM":
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _parse_json_response(raw: str) -> dict | None:
        """Parse JSON from LLM response, handling markdown code blocks."""
        raw = raw.strip()

        # Try direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None
