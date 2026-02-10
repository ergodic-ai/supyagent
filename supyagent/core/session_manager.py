"""
Session manager for persistent conversation storage.

Sessions are stored as JSONL files with the following format:
- First line: Session metadata with type="meta"
- Subsequent lines: Messages in chronological order
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from supyagent.models.session import Message, Session, SessionMeta
from supyagent.utils.media import content_to_text


class SessionManager:
    """
    Manages session persistence using JSONL files.

    Directory structure:
        .supyagent/sessions/<agent>/<session_id>.jsonl
        .supyagent/sessions/<agent>/current.json
    """

    def __init__(self, base_dir: Path | None = None):
        """
        Initialize the session manager.

        Args:
            base_dir: Base directory for session storage (default: .supyagent/sessions)
        """
        if base_dir is None:
            base_dir = Path(".supyagent/sessions")
        self.base_dir = base_dir

    def _agent_dir(self, agent: str) -> Path:
        """Get the directory for an agent's sessions, creating if needed."""
        path = self.base_dir / agent
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _session_path(self, agent: str, session_id: str) -> Path:
        """Get the path to a session file."""
        return self._agent_dir(agent) / f"{session_id}.jsonl"

    def _summary_path(self, agent: str, session_id: str) -> Path:
        """Get the path to a session's context summary."""
        return self._agent_dir(agent) / f"{session_id}_summary.json"

    def _current_path(self, agent: str) -> Path:
        """Get the path to the current session pointer file."""
        return self._agent_dir(agent) / "current.json"

    def _aliases_path(self, agent: str) -> Path:
        """Get the path to session aliases file."""
        return self._agent_dir(agent) / "aliases.json"

    def _load_aliases(self, agent: str) -> dict[str, str]:
        """Load session aliases (name → session_id)."""
        path = self._aliases_path(agent)
        if not path.exists():
            return {}
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_aliases(self, agent: str, aliases: dict[str, str]) -> None:
        """Save session aliases."""
        with open(self._aliases_path(agent), "w") as f:
            json.dump(aliases, f, indent=2)

    def set_alias(self, agent: str, alias: str, session_id: str) -> None:
        """Set a named alias for a session."""
        aliases = self._load_aliases(agent)
        aliases[alias] = session_id
        self._save_aliases(agent, aliases)

    def remove_alias(self, agent: str, alias: str) -> bool:
        """Remove a session alias. Returns True if it existed."""
        aliases = self._load_aliases(agent)
        if alias in aliases:
            del aliases[alias]
            self._save_aliases(agent, aliases)
            return True
        return False

    def resolve_alias(self, agent: str, name_or_id: str) -> str | None:
        """
        Resolve a name or ID to a session ID.

        Checks aliases first, then tries as a direct session ID / prefix.
        """
        aliases = self._load_aliases(agent)
        if name_or_id in aliases:
            return aliases[name_or_id]
        return None

    def get_alias(self, agent: str, session_id: str) -> str | None:
        """Get the alias for a session ID, if any."""
        aliases = self._load_aliases(agent)
        for alias, sid in aliases.items():
            if sid == session_id:
                return alias
        return None

    def create_session(
        self, agent: str, model: str, session_id: str | None = None
    ) -> Session:
        """
        Create a new session for an agent.

        Args:
            agent: Agent name
            model: Model being used
            session_id: Optional session ID to use (instead of auto-generating)

        Returns:
            The newly created session
        """
        kwargs: dict[str, Any] = {"agent": agent, "model": model}
        if session_id:
            kwargs["session_id"] = session_id
        meta = SessionMeta(**kwargs)
        session = Session(meta=meta)
        self._save_session(session)
        self._set_current(agent, meta.session_id)
        return session

    def load_session(self, agent: str, session_id: str) -> Session | None:
        """
        Load a session from disk.

        Args:
            agent: Agent name
            session_id: Session ID to load

        Returns:
            The loaded session, or None if not found
        """
        path = self._session_path(agent, session_id)
        if not path.exists():
            return None

        messages: list[Message] = []
        meta: SessionMeta | None = None

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)

                if data.get("type") == "meta":
                    # Remove the type field before parsing as SessionMeta
                    data.pop("type")
                    meta = SessionMeta(**data)
                else:
                    messages.append(Message(**data))

        if meta is None:
            return None

        return Session(meta=meta, messages=messages)

    def get_current_session(self, agent: str) -> Session | None:
        """
        Get the current active session for an agent.

        Args:
            agent: Agent name

        Returns:
            The current session, or None if no current session
        """
        current_path = self._current_path(agent)
        if not current_path.exists():
            return None

        with open(current_path) as f:
            data = json.load(f)

        return self.load_session(agent, data["session_id"])

    def _set_current(self, agent: str, session_id: str) -> None:
        """Set the current session for an agent."""
        with open(self._current_path(agent), "w") as f:
            json.dump({"session_id": session_id}, f)

    def append_message(self, session: Session, message: Message) -> None:
        """
        Append a message to a session and persist it.

        Args:
            session: The session to append to
            message: The message to append
        """
        session.messages.append(message)
        session.meta.updated_at = message.ts

        # Append to file
        path = self._session_path(session.meta.agent, session.meta.session_id)
        with open(path, "a") as f:
            f.write(message.model_dump_json() + "\n")

        # Update title if this is the first user message
        if message.type == "user" and session.meta.title is None:
            self._update_title(session, message.content or "")

    def _update_title(self, session: Session, first_message: str | list) -> None:
        """Generate and save a title from the first user message."""
        # Extract text from multimodal content
        if isinstance(first_message, list):
            first_message = content_to_text(first_message)
        # Simple truncation for now
        title = first_message[:50]
        if len(first_message) > 50:
            title += "..."

        session.meta.title = title

        # Only rewrite the meta line, not the entire file
        self._update_meta(session)

    def _update_meta(self, session: Session) -> None:
        """Update only the metadata (first line) of the session file."""
        path = self._session_path(session.meta.agent, session.meta.session_id)
        if not path.exists():
            self._save_session(session)
            return

        # Read all lines, replace first line (meta), write back
        with open(path) as f:
            lines = f.readlines()

        if not lines:
            self._save_session(session)
            return

        # Build new meta line
        meta_dict: dict[str, Any] = session.meta.model_dump()
        meta_dict["type"] = "meta"
        meta_dict["created_at"] = session.meta.created_at.isoformat()
        meta_dict["updated_at"] = session.meta.updated_at.isoformat()

        lines[0] = json.dumps(meta_dict) + "\n"

        with open(path, "w") as f:
            f.writelines(lines)

    def _save_session(self, session: Session) -> None:
        """Save the full session to disk."""
        path = self._session_path(session.meta.agent, session.meta.session_id)

        with open(path, "w") as f:
            # Write meta first with type marker
            meta_dict: dict[str, Any] = session.meta.model_dump()
            meta_dict["type"] = "meta"
            # Convert datetime to string
            meta_dict["created_at"] = session.meta.created_at.isoformat()
            meta_dict["updated_at"] = session.meta.updated_at.isoformat()
            f.write(json.dumps(meta_dict) + "\n")

            # Write messages
            for msg in session.messages:
                f.write(msg.model_dump_json() + "\n")

    def list_sessions(self, agent: str) -> list[SessionMeta]:
        """
        List all sessions for an agent.

        Args:
            agent: Agent name

        Returns:
            List of session metadata, sorted by updated_at (newest first)
        """
        agent_dir = self._agent_dir(agent)
        sessions: list[SessionMeta] = []

        for path in agent_dir.glob("*.jsonl"):
            try:
                with open(path) as f:
                    first_line = f.readline().strip()
                    if not first_line:
                        continue

                    data = json.loads(first_line)
                    if data.get("type") == "meta":
                        data.pop("type")
                        sessions.append(SessionMeta(**data))
            except (json.JSONDecodeError, KeyError):
                # Skip invalid session files
                continue

        return sorted(sessions, key=lambda s: s.updated_at, reverse=True)

    def delete_session(self, agent: str, session_id: str) -> bool:
        """
        Delete a session and its associated summary.

        Args:
            agent: Agent name
            session_id: Session ID to delete

        Returns:
            True if deleted, False if not found
        """
        path = self._session_path(agent, session_id)
        if path.exists():
            path.unlink()

            # Also delete summary if exists
            summary_path = self._summary_path(agent, session_id)
            if summary_path.exists():
                summary_path.unlink()

            # Also delete media directory if exists
            media_dir = self._agent_dir(agent) / f"{session_id}_media"
            if media_dir.exists():
                shutil.rmtree(media_dir)

            # If this was the current session, clear the current pointer
            current_path = self._current_path(agent)
            if current_path.exists():
                with open(current_path) as f:
                    data = json.load(f)
                if data.get("session_id") == session_id:
                    current_path.unlink()

            return True
        return False

    def search_sessions(
        self,
        agent: str,
        query: str | None = None,
        since: datetime | None = None,
        before: datetime | None = None,
    ) -> list[SessionMeta]:
        """
        Search sessions by keyword in title, or filter by date range.

        Args:
            agent: Agent name
            query: Search keyword (matched against title, case-insensitive)
            since: Only include sessions updated after this time
            before: Only include sessions updated before this time

        Returns:
            Filtered list of session metadata
        """
        sessions = self.list_sessions(agent)
        results = []

        for s in sessions:
            if query and (not s.title or query.lower() not in s.title.lower()):
                continue
            if since and s.updated_at < since:
                continue
            if before and s.updated_at > before:
                continue
            results.append(s)

        return results

    def export_session(
        self,
        agent: str,
        session_id: str,
        fmt: str = "markdown",
    ) -> str | None:
        """
        Export a session to a string in the specified format.

        Args:
            agent: Agent name
            session_id: Session ID to export
            fmt: Export format ('markdown' or 'json')

        Returns:
            Formatted string, or None if session not found
        """
        session = self.load_session(agent, session_id)
        if not session:
            return None

        if fmt == "json":
            data = {
                "session_id": session.meta.session_id,
                "agent": session.meta.agent,
                "model": session.meta.model,
                "title": session.meta.title,
                "created_at": session.meta.created_at.isoformat(),
                "updated_at": session.meta.updated_at.isoformat(),
                "messages": [],
            }
            for msg in session.messages:
                entry: dict[str, Any] = {
                    "type": msg.type,
                    "ts": msg.ts.isoformat(),
                }
                if msg.content:
                    entry["content"] = msg.content
                if msg.name:
                    entry["tool"] = msg.name
                data["messages"].append(entry)
            return json.dumps(data, indent=2)

        # Default: markdown
        alias = self.get_alias(agent, session_id)
        title_line = session.meta.title or "Untitled conversation"
        if alias:
            title_line += f" ({alias})"

        lines = [
            f"# {title_line}",
            "",
            f"**Agent:** {session.meta.agent}  ",
            f"**Model:** {session.meta.model}  ",
            f"**Session:** {session.meta.session_id}  ",
            f"**Created:** {session.meta.created_at.strftime('%Y-%m-%d %H:%M')}  ",
            f"**Updated:** {session.meta.updated_at.strftime('%Y-%m-%d %H:%M')}",
            "",
            "---",
            "",
        ]

        prev_role = None
        for msg in session.messages:
            content_text = msg.content
            if isinstance(content_text, list):
                content_text = content_to_text(content_text)

            if msg.type == "user":
                lines.append(f"**You:** {content_text}\n")
                prev_role = "user"
            elif msg.type == "assistant":
                # Skip empty assistant messages (e.g. thinking-only or empty before tool calls)
                if not content_text or not content_text.strip():
                    continue
                # Collapse consecutive assistant messages into one block
                if prev_role == "assistant":
                    lines.append(f"{content_text}\n")
                else:
                    lines.append(f"**{agent}:** {content_text}\n")
                prev_role = "assistant"
            elif msg.type == "tool_result":
                tool_name = msg.name or "unknown"
                preview = (content_text or "")[:200]
                if len(content_text or "") > 200:
                    preview += "..."
                lines.append(f"> *Tool: {tool_name}*  \n> {preview}\n")
                prev_role = "tool_result"

        return "\n".join(lines)

    def delete_all_sessions(self, agent: str) -> int:
        """
        Delete all sessions for an agent.

        Args:
            agent: Agent name

        Returns:
            Number of sessions deleted
        """
        sessions = self.list_sessions(agent)
        count = 0
        for s in sessions:
            if self.delete_session(agent, s.session_id):
                count += 1
        return count
