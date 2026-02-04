"""
Tests for session management and persistence.
"""

import json
from datetime import datetime

import pytest

from supyagent.core.session_manager import SessionManager
from supyagent.models.session import Message, Session, SessionMeta


class TestSessionMeta:
    """Tests for SessionMeta model."""

    def test_session_meta_defaults(self):
        """Test SessionMeta generates defaults."""
        meta = SessionMeta(agent="test", model="test/model")

        assert meta.session_id is not None
        assert len(meta.session_id) == 8
        assert meta.agent == "test"
        assert meta.model == "test/model"
        assert meta.title is None
        assert isinstance(meta.created_at, datetime)
        assert isinstance(meta.updated_at, datetime)

    def test_session_meta_custom_id(self):
        """Test SessionMeta with custom ID."""
        meta = SessionMeta(
            session_id="custom123",
            agent="test",
            model="test/model",
        )
        assert meta.session_id == "custom123"


class TestMessage:
    """Tests for Message model."""

    def test_user_message(self):
        """Test creating a user message."""
        msg = Message(type="user", content="Hello!")
        assert msg.type == "user"
        assert msg.content == "Hello!"
        assert msg.tool_calls is None
        assert isinstance(msg.ts, datetime)

    def test_assistant_message_with_tool_calls(self):
        """Test creating an assistant message with tool calls."""
        tool_calls = [{"id": "call_1", "name": "test"}]
        msg = Message(
            type="assistant",
            content="Using a tool...",
            tool_calls=tool_calls,
        )
        assert msg.type == "assistant"
        assert msg.tool_calls == tool_calls

    def test_tool_result_message(self):
        """Test creating a tool result message."""
        msg = Message(
            type="tool_result",
            tool_call_id="call_1",
            name="test__func",
            content='{"ok": true}',
        )
        assert msg.type == "tool_result"
        assert msg.tool_call_id == "call_1"
        assert msg.name == "test__func"


class TestSession:
    """Tests for Session model."""

    def test_session_creation(self):
        """Test creating a session."""
        meta = SessionMeta(agent="test", model="test/model")
        session = Session(meta=meta)

        assert session.meta == meta
        assert session.messages == []

    def test_session_with_messages(self):
        """Test session with messages."""
        meta = SessionMeta(agent="test", model="test/model")
        messages = [
            Message(type="user", content="Hello"),
            Message(type="assistant", content="Hi there!"),
        ]
        session = Session(meta=meta, messages=messages)

        assert len(session.messages) == 2


class TestSessionManager:
    """Tests for SessionManager."""

    def test_create_session(self, sessions_dir):
        """Test creating a new session."""
        mgr = SessionManager(base_dir=sessions_dir)
        session = mgr.create_session("test-agent", "test/model")

        assert session.meta.agent == "test-agent"
        assert session.meta.model == "test/model"
        assert session.messages == []

        # Check file was created
        session_file = sessions_dir / "test-agent" / f"{session.meta.session_id}.jsonl"
        assert session_file.exists()

        # Check current.json was created
        current_file = sessions_dir / "test-agent" / "current.json"
        assert current_file.exists()

    def test_load_session(self, sessions_dir):
        """Test loading an existing session."""
        mgr = SessionManager(base_dir=sessions_dir)

        # Create a session
        created = mgr.create_session("test-agent", "test/model")
        session_id = created.meta.session_id

        # Load it back
        loaded = mgr.load_session("test-agent", session_id)

        assert loaded is not None
        assert loaded.meta.session_id == session_id
        assert loaded.meta.agent == "test-agent"

    def test_load_nonexistent_session(self, sessions_dir):
        """Test loading a non-existent session returns None."""
        mgr = SessionManager(base_dir=sessions_dir)
        result = mgr.load_session("test-agent", "nonexistent")
        assert result is None

    def test_get_current_session(self, sessions_dir):
        """Test getting the current session."""
        mgr = SessionManager(base_dir=sessions_dir)

        # No current session initially
        assert mgr.get_current_session("test-agent") is None

        # Create a session (becomes current)
        created = mgr.create_session("test-agent", "test/model")

        # Get current
        current = mgr.get_current_session("test-agent")
        assert current is not None
        assert current.meta.session_id == created.meta.session_id

    def test_append_message(self, sessions_dir):
        """Test appending messages to a session."""
        mgr = SessionManager(base_dir=sessions_dir)
        session = mgr.create_session("test-agent", "test/model")

        # Append a message
        msg = Message(type="user", content="Hello!")
        mgr.append_message(session, msg)

        assert len(session.messages) == 1
        assert session.messages[0].content == "Hello!"

        # Reload and verify persistence
        loaded = mgr.load_session("test-agent", session.meta.session_id)
        assert len(loaded.messages) == 1
        assert loaded.messages[0].content == "Hello!"

    def test_append_multiple_messages(self, sessions_dir):
        """Test appending multiple messages."""
        mgr = SessionManager(base_dir=sessions_dir)
        session = mgr.create_session("test-agent", "test/model")

        # Append messages
        mgr.append_message(session, Message(type="user", content="Hello"))
        mgr.append_message(session, Message(type="assistant", content="Hi!"))
        mgr.append_message(session, Message(type="user", content="How are you?"))

        assert len(session.messages) == 3

        # Reload and verify order
        loaded = mgr.load_session("test-agent", session.meta.session_id)
        assert loaded.messages[0].content == "Hello"
        assert loaded.messages[1].content == "Hi!"
        assert loaded.messages[2].content == "How are you?"

    def test_auto_title_generation(self, sessions_dir):
        """Test that title is auto-generated from first user message."""
        mgr = SessionManager(base_dir=sessions_dir)
        session = mgr.create_session("test-agent", "test/model")

        # No title initially
        assert session.meta.title is None

        # Add first user message
        mgr.append_message(session, Message(type="user", content="What is Python?"))

        # Title should be set
        assert session.meta.title == "What is Python?"

        # Reload and verify
        loaded = mgr.load_session("test-agent", session.meta.session_id)
        assert loaded.meta.title == "What is Python?"

    def test_title_truncation(self, sessions_dir):
        """Test that long titles are truncated."""
        mgr = SessionManager(base_dir=sessions_dir)
        session = mgr.create_session("test-agent", "test/model")

        long_message = "This is a very long message that should be truncated " * 5
        mgr.append_message(session, Message(type="user", content=long_message))

        assert len(session.meta.title) <= 53  # 50 + "..."
        assert session.meta.title.endswith("...")

    def test_list_sessions(self, sessions_dir):
        """Test listing all sessions for an agent."""
        mgr = SessionManager(base_dir=sessions_dir)

        # Create multiple sessions
        s1 = mgr.create_session("test-agent", "model1")
        s2 = mgr.create_session("test-agent", "model2")
        s3 = mgr.create_session("test-agent", "model3")

        # List sessions
        sessions = mgr.list_sessions("test-agent")

        assert len(sessions) == 3
        session_ids = {s.session_id for s in sessions}
        assert s1.meta.session_id in session_ids
        assert s2.meta.session_id in session_ids
        assert s3.meta.session_id in session_ids

    def test_list_sessions_sorted_by_updated(self, sessions_dir):
        """Test that sessions are sorted by updated_at (newest first)."""
        mgr = SessionManager(base_dir=sessions_dir)

        # Create sessions and add messages to make different update times
        s1 = mgr.create_session("test-agent", "model")
        mgr.append_message(s1, Message(type="user", content="First"))

        s2 = mgr.create_session("test-agent", "model")
        mgr.append_message(s2, Message(type="user", content="Second"))

        s3 = mgr.create_session("test-agent", "model")
        mgr.append_message(s3, Message(type="user", content="Third"))

        # List sessions
        sessions = mgr.list_sessions("test-agent")

        # Most recent should be first
        assert sessions[0].session_id == s3.meta.session_id

    def test_delete_session(self, sessions_dir):
        """Test deleting a session."""
        mgr = SessionManager(base_dir=sessions_dir)
        session = mgr.create_session("test-agent", "test/model")
        session_id = session.meta.session_id

        # Verify it exists
        assert mgr.load_session("test-agent", session_id) is not None

        # Delete it
        result = mgr.delete_session("test-agent", session_id)
        assert result is True

        # Verify it's gone
        assert mgr.load_session("test-agent", session_id) is None

    def test_delete_nonexistent_session(self, sessions_dir):
        """Test deleting a non-existent session returns False."""
        mgr = SessionManager(base_dir=sessions_dir)
        result = mgr.delete_session("test-agent", "nonexistent")
        assert result is False

    def test_delete_current_session_clears_pointer(self, sessions_dir):
        """Test that deleting current session clears the current pointer."""
        mgr = SessionManager(base_dir=sessions_dir)
        session = mgr.create_session("test-agent", "test/model")

        # Verify it's current
        assert mgr.get_current_session("test-agent") is not None

        # Delete it
        mgr.delete_session("test-agent", session.meta.session_id)

        # Current should be None now
        assert mgr.get_current_session("test-agent") is None

    def test_jsonl_format(self, sessions_dir):
        """Test that session files are valid JSONL."""
        mgr = SessionManager(base_dir=sessions_dir)
        session = mgr.create_session("test-agent", "test/model")

        mgr.append_message(session, Message(type="user", content="Hello"))
        mgr.append_message(session, Message(type="assistant", content="Hi!"))

        # Read the file directly
        session_file = sessions_dir / "test-agent" / f"{session.meta.session_id}.jsonl"
        lines = session_file.read_text().strip().split("\n")

        # Should be 3 lines: meta + 2 messages
        assert len(lines) == 3

        # Each line should be valid JSON
        for line in lines:
            data = json.loads(line)
            assert isinstance(data, dict)

        # First line should be meta
        meta = json.loads(lines[0])
        assert meta["type"] == "meta"
        assert meta["agent"] == "test-agent"

    def test_multiple_agents_isolated(self, sessions_dir):
        """Test that sessions from different agents are isolated."""
        mgr = SessionManager(base_dir=sessions_dir)

        s1 = mgr.create_session("agent-a", "model")
        s2 = mgr.create_session("agent-b", "model")

        # List sessions for each agent
        a_sessions = mgr.list_sessions("agent-a")
        b_sessions = mgr.list_sessions("agent-b")

        assert len(a_sessions) == 1
        assert len(b_sessions) == 1
        assert a_sessions[0].session_id == s1.meta.session_id
        assert b_sessions[0].session_id == s2.meta.session_id
