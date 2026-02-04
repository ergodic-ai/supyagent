"""
Tests for the Agent class.
"""

from unittest.mock import MagicMock, patch

import pytest

from supyagent.core.agent import Agent
from supyagent.core.session_manager import SessionManager
from supyagent.models.session import Message, Session, SessionMeta


class TestAgentInitialization:
    """Tests for Agent initialization."""

    @patch("supyagent.core.agent.discover_tools")
    def test_agent_creates_new_session(self, mock_discover, sample_agent_config, sessions_dir):
        """Test that Agent creates a new session if none provided."""
        mock_discover.return_value = []
        session_mgr = SessionManager(base_dir=sessions_dir)

        agent = Agent(sample_agent_config, session_manager=session_mgr)

        assert agent.session is not None
        assert agent.session.meta.agent == "test-agent"
        assert len(agent.messages) == 1  # System prompt only
        assert agent.messages[0]["role"] == "system"

    @patch("supyagent.core.agent.discover_tools")
    def test_agent_uses_provided_session(self, mock_discover, sample_agent_config, sessions_dir):
        """Test that Agent uses an existing session if provided."""
        mock_discover.return_value = []
        session_mgr = SessionManager(base_dir=sessions_dir)

        # Create a session with history
        meta = SessionMeta(agent="test-agent", model="test/model")
        existing_session = Session(meta=meta, messages=[
            Message(type="user", content="Previous message"),
            Message(type="assistant", content="Previous response"),
        ])

        agent = Agent(sample_agent_config, session=existing_session, session_manager=session_mgr)

        assert agent.session == existing_session
        # Should have system prompt + reconstructed messages
        assert len(agent.messages) == 3  # system + user + assistant

    @patch("supyagent.core.agent.discover_tools")
    def test_agent_reconstructs_messages(self, mock_discover, sample_agent_config, sessions_dir):
        """Test that Agent correctly reconstructs LLM messages from session."""
        mock_discover.return_value = []
        session_mgr = SessionManager(base_dir=sessions_dir)

        meta = SessionMeta(agent="test-agent", model="test/model")
        existing_session = Session(meta=meta, messages=[
            Message(type="user", content="Hello"),
            Message(type="assistant", content="Hi there!"),
            Message(type="user", content="How are you?"),
        ])

        agent = Agent(sample_agent_config, session=existing_session, session_manager=session_mgr)

        # Check reconstructed messages
        assert agent.messages[0]["role"] == "system"
        assert agent.messages[1]["role"] == "user"
        assert agent.messages[1]["content"] == "Hello"
        assert agent.messages[2]["role"] == "assistant"
        assert agent.messages[2]["content"] == "Hi there!"
        assert agent.messages[3]["role"] == "user"
        assert agent.messages[3]["content"] == "How are you?"


class TestAgentSendMessage:
    """Tests for Agent.send_message()."""

    @patch("supyagent.core.agent.discover_tools")
    def test_send_message_simple_response(
        self, mock_discover, sample_agent_config, sessions_dir, mock_llm_response
    ):
        """Test sending a message and getting a simple response."""
        mock_discover.return_value = []
        session_mgr = SessionManager(base_dir=sessions_dir)

        agent = Agent(sample_agent_config, session_manager=session_mgr)

        with patch.object(agent.llm, "chat", return_value=mock_llm_response):
            response = agent.send_message("Hello!")

        assert response == "Hello! I'm a test response."
        # Session should have 2 messages (user + assistant)
        assert len(agent.session.messages) == 2
        assert agent.session.messages[0].type == "user"
        assert agent.session.messages[1].type == "assistant"

    @patch("supyagent.core.agent.discover_tools")
    @patch("supyagent.core.agent.execute_tool")
    def test_send_message_with_tool_call(
        self,
        mock_execute_tool,
        mock_discover,
        sample_agent_config,
        sessions_dir,
        mock_llm_response_with_tool_call,
        mock_llm_response,
    ):
        """Test sending a message that triggers a tool call."""
        mock_discover.return_value = []
        mock_execute_tool.return_value = {"ok": True, "data": "Tool result"}
        session_mgr = SessionManager(base_dir=sessions_dir)

        agent = Agent(sample_agent_config, session_manager=session_mgr)

        # First call returns tool call, second returns final response
        with patch.object(
            agent.llm, "chat", side_effect=[mock_llm_response_with_tool_call, mock_llm_response]
        ):
            response = agent.send_message("Use a tool please")

        # Tool should have been executed (with empty secrets since none stored)
        mock_execute_tool.assert_called_once_with("test", "hello", {"name": "World"}, secrets={})

        # Session should have: user, assistant (with tool call), tool_result, assistant (final)
        assert len(agent.session.messages) == 4
        assert agent.session.messages[0].type == "user"
        assert agent.session.messages[1].type == "assistant"
        assert agent.session.messages[2].type == "tool_result"
        assert agent.session.messages[3].type == "assistant"

    @patch("supyagent.core.agent.discover_tools")
    def test_messages_persisted_to_session(
        self, mock_discover, sample_agent_config, sessions_dir, mock_llm_response
    ):
        """Test that messages are persisted and can be reloaded."""
        mock_discover.return_value = []
        session_mgr = SessionManager(base_dir=sessions_dir)

        agent = Agent(sample_agent_config, session_manager=session_mgr)
        session_id = agent.session.meta.session_id

        with patch.object(agent.llm, "chat", return_value=mock_llm_response):
            agent.send_message("Hello!")

        # Reload session
        loaded = session_mgr.load_session("test-agent", session_id)

        assert len(loaded.messages) == 2
        assert loaded.messages[0].content == "Hello!"


class TestAgentToolExecution:
    """Tests for tool execution."""

    @patch("supyagent.core.agent.discover_tools")
    @patch("supyagent.core.agent.execute_tool")
    def test_execute_tool_call_success(
        self, mock_execute_tool, mock_discover, sample_agent_config, sessions_dir
    ):
        """Test successful tool execution."""
        mock_discover.return_value = []
        mock_execute_tool.return_value = {"ok": True, "data": "result"}
        session_mgr = SessionManager(base_dir=sessions_dir)

        agent = Agent(sample_agent_config, session_manager=session_mgr)

        # Create a mock tool call
        tool_call = MagicMock()
        tool_call.function.name = "script__function"
        tool_call.function.arguments = '{"arg": "value"}'

        result = agent._execute_tool_call(tool_call)

        mock_execute_tool.assert_called_once_with("script", "function", {"arg": "value"}, secrets={})
        assert result == {"ok": True, "data": "result"}

    @patch("supyagent.core.agent.discover_tools")
    def test_execute_tool_call_invalid_name(
        self, mock_discover, sample_agent_config, sessions_dir
    ):
        """Test tool execution with invalid name format."""
        mock_discover.return_value = []
        session_mgr = SessionManager(base_dir=sessions_dir)

        agent = Agent(sample_agent_config, session_manager=session_mgr)

        # Create a mock tool call with invalid name (no __)
        tool_call = MagicMock()
        tool_call.function.name = "invalidname"
        tool_call.function.arguments = '{}'

        result = agent._execute_tool_call(tool_call)

        assert result["ok"] is False
        assert "Invalid tool name" in result["error"]

    @patch("supyagent.core.agent.discover_tools")
    def test_execute_tool_call_invalid_json(
        self, mock_discover, sample_agent_config, sessions_dir
    ):
        """Test tool execution with invalid JSON arguments."""
        mock_discover.return_value = []
        session_mgr = SessionManager(base_dir=sessions_dir)

        agent = Agent(sample_agent_config, session_manager=session_mgr)

        # Create a mock tool call with invalid JSON
        tool_call = MagicMock()
        tool_call.function.name = "script__func"
        tool_call.function.arguments = 'not valid json'

        result = agent._execute_tool_call(tool_call)

        assert result["ok"] is False
        assert "Invalid JSON" in result["error"]


class TestAgentClearHistory:
    """Tests for clearing history."""

    @patch("supyagent.core.agent.discover_tools")
    def test_clear_history_creates_new_session(
        self, mock_discover, sample_agent_config, sessions_dir, mock_llm_response
    ):
        """Test that clear_history creates a new session."""
        mock_discover.return_value = []
        session_mgr = SessionManager(base_dir=sessions_dir)

        agent = Agent(sample_agent_config, session_manager=session_mgr)
        old_session_id = agent.session.meta.session_id

        # Add some messages
        with patch.object(agent.llm, "chat", return_value=mock_llm_response):
            agent.send_message("Hello!")

        # Clear history
        agent.clear_history()

        # Should have new session
        assert agent.session.meta.session_id != old_session_id
        assert len(agent.session.messages) == 0
        assert len(agent.messages) == 1  # Only system prompt


class TestAgentAvailableTools:
    """Tests for get_available_tools()."""

    @patch("supyagent.core.agent.discover_tools")
    @patch("supyagent.core.agent.supypowers_to_openai_tools")
    @patch("supyagent.core.agent.filter_tools")
    def test_get_available_tools(
        self,
        mock_filter,
        mock_convert,
        mock_discover,
        sample_agent_config,
        sessions_dir,
    ):
        """Test getting list of available tools."""
        mock_discover.return_value = [{"script": "test", "function": "func"}]
        mock_convert.return_value = [{"type": "function", "function": {"name": "test__func"}}]
        mock_filter.return_value = mock_convert.return_value

        session_mgr = SessionManager(base_dir=sessions_dir)
        agent = Agent(sample_agent_config, session_manager=session_mgr)

        tools = agent.get_available_tools()

        assert "test__func" in tools
