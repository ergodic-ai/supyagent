"""
Session manager for persistent conversation storage.

Sessions are stored as JSONL files with the following format:
- First line: Session metadata with type="meta"
- Subsequent lines: Messages in chronological order
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from supyagent.models.session import Message, Session, SessionMeta


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

    def create_session(self, agent: str, model: str) -> Session:
        """
        Create a new session for an agent.

        Args:
            agent: Agent name
            model: Model being used

        Returns:
            The newly created session
        """
        meta = SessionMeta(agent=agent, model=model)
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

    def _update_title(self, session: Session, first_message: str) -> None:
        """Generate and save a title from the first user message."""
        # Simple truncation for now
        title = first_message[:50]
        if len(first_message) > 50:
            title += "..."

        session.meta.title = title

        # Rewrite the session file to update meta
        self._save_session(session)

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

            # If this was the current session, clear the current pointer
            current_path = self._current_path(agent)
            if current_path.exists():
                with open(current_path) as f:
                    data = json.load(f)
                if data.get("session_id") == session_id:
                    current_path.unlink()

            return True
        return False
