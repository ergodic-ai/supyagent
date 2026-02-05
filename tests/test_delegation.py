"""
Tests for DelegationManager and DelegationContext.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from supyagent.core.context import DelegationContext, summarize_conversation
from supyagent.core.delegation import DelegationManager
from supyagent.core.registry import AgentRegistry
from supyagent.models.agent_config import AgentConfig, ModelConfig, ToolPermissions


@pytest.fixture
def parent_config():
    """Create a parent agent config with delegates."""
    return AgentConfig(
        name="planner",
        description="A planning agent",
        version="1.0",
        type="interactive",
        model=ModelConfig(
            provider="test/model",
            temperature=0.7,
        ),
        system_prompt="You are a planning agent.",
        tools=ToolPermissions(allow=["*"]),
        delegates=["coder", "writer"],
    )


@pytest.fixture
def delegate_config():
    """Create a delegate agent config."""
    return AgentConfig(
        name="coder",
        description="A coding agent",
        version="1.0",
        type="execution",
        model=ModelConfig(
            provider="test/model",
            temperature=0.3,
        ),
        system_prompt="You are a coding agent.",
        tools=ToolPermissions(allow=["*"]),
        delegates=[],
    )


class TestDelegationContext:
    """Tests for DelegationContext."""

    def test_basic_context(self):
        """Test creating a basic context."""
        context = DelegationContext(
            parent_agent="planner",
            parent_task="Build a web app",
        )

        prompt = context.to_prompt()
        assert "planner" in prompt
        assert "Build a web app" in prompt

    def test_context_with_summary(self):
        """Test context with conversation summary."""
        context = DelegationContext(
            parent_agent="planner",
            parent_task="Build a web app",
            conversation_summary="User wants a React app with TypeScript",
        )

        prompt = context.to_prompt()
        assert "React app" in prompt
        assert "TypeScript" in prompt

    def test_context_with_facts(self):
        """Test context with relevant facts."""
        context = DelegationContext(
            parent_agent="planner",
            parent_task="Build a web app",
            relevant_facts=[
                "Using React 18",
                "Must support dark mode",
            ],
        )

        prompt = context.to_prompt()
        assert "React 18" in prompt
        assert "dark mode" in prompt

    def test_context_with_shared_data(self):
        """Test context with shared data."""
        context = DelegationContext(
            parent_agent="planner",
            parent_task="Build a web app",
            shared_data={"framework": "React", "version": "18"},
        )

        prompt = context.to_prompt()
        assert "framework" in prompt
        assert "React" in prompt


class TestSummarizeConversation:
    """Tests for conversation summarization."""

    def test_summarize_empty_messages(self):
        """Test summarization with no messages."""
        mock_llm = MagicMock()
        result = summarize_conversation([], mock_llm)
        assert result is None

    def test_summarize_with_messages(self):
        """Test summarization with messages."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "User wants to build a web app."
        mock_llm.chat.return_value = mock_response

        messages = [
            {"role": "user", "content": "Build me a web app"},
            {"role": "assistant", "content": "Sure, I can help with that."},
        ]

        result = summarize_conversation(messages, mock_llm)
        assert result == "User wants to build a web app."
        mock_llm.chat.assert_called_once()

    def test_summarize_filters_tool_messages(self):
        """Test that tool messages are filtered out."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Summary"
        mock_llm.chat.return_value = mock_response

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "tool", "content": "tool result"},  # Should be filtered
            {"role": "assistant", "content": "Hi there"},
        ]

        summarize_conversation(messages, mock_llm)

        # Check that the prompt doesn't include tool content
        call_args = mock_llm.chat.call_args
        prompt = call_args[0][0][0]["content"]
        assert "tool result" not in prompt


class TestDelegationManager:
    """Tests for DelegationManager."""

    def test_get_delegation_tools(self, parent_config, temp_dir):
        """Test generating delegation tools."""
        registry = AgentRegistry(base_dir=temp_dir)

        mock_agent = MagicMock()
        mock_agent.config = parent_config
        mock_agent.messages = []
        mock_agent.llm = MagicMock()

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            # Mock the delegate configs
            mock_load.return_value = AgentConfig(
                name="coder",
                description="A coding agent",
                version="1.0",
                type="execution",
                model=ModelConfig(provider="test/model"),
                system_prompt="Code.",
                tools=ToolPermissions(allow=[]),
                delegates=[],
            )

            manager = DelegationManager(registry, mock_agent)
            tools = manager.get_delegation_tools()

        # Should have delegate_to_X tools + spawn_agent
        assert len(tools) >= 1
        tool_names = [t["function"]["name"] for t in tools]
        assert "spawn_agent" in tool_names

    def test_is_delegation_tool(self, parent_config, temp_dir):
        """Test checking if a tool is a delegation tool."""
        registry = AgentRegistry(base_dir=temp_dir)

        mock_agent = MagicMock()
        mock_agent.config = parent_config
        mock_agent.messages = []
        mock_agent.llm = MagicMock()

        with patch("supyagent.core.delegation.load_agent_config"):
            manager = DelegationManager(registry, mock_agent)

        assert manager.is_delegation_tool("delegate_to_coder") is True
        assert manager.is_delegation_tool("spawn_agent") is True
        assert manager.is_delegation_tool("some_other_tool") is False

    def test_execute_delegation_unknown_tool(self, parent_config, temp_dir):
        """Test executing an unknown delegation tool."""
        registry = AgentRegistry(base_dir=temp_dir)

        mock_agent = MagicMock()
        mock_agent.config = parent_config
        mock_agent.messages = []
        mock_agent.llm = MagicMock()

        with patch("supyagent.core.delegation.load_agent_config"):
            manager = DelegationManager(registry, mock_agent)

        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "unknown_tool"
        mock_tool_call.function.arguments = "{}"

        result = manager.execute_delegation(mock_tool_call)
        assert result["ok"] is False
        assert "Unknown delegation tool" in result["error"]

    def test_execute_delegation_invalid_json(self, parent_config, temp_dir):
        """Test executing with invalid JSON arguments."""
        registry = AgentRegistry(base_dir=temp_dir)

        mock_agent = MagicMock()
        mock_agent.config = parent_config
        mock_agent.messages = []
        mock_agent.llm = MagicMock()

        with patch("supyagent.core.delegation.load_agent_config"):
            manager = DelegationManager(registry, mock_agent)

        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "delegate_to_coder"
        mock_tool_call.function.arguments = "invalid json"

        result = manager.execute_delegation(mock_tool_call)
        assert result["ok"] is False
        assert "Invalid JSON" in result["error"]

    def test_spawn_agent_not_in_delegates(self, parent_config, temp_dir):
        """Test spawning an agent not in delegates list."""
        registry = AgentRegistry(base_dir=temp_dir)

        mock_agent = MagicMock()
        mock_agent.config = parent_config
        mock_agent.messages = []
        mock_agent.llm = MagicMock()

        with patch("supyagent.core.delegation.load_agent_config"):
            manager = DelegationManager(registry, mock_agent)

        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "spawn_agent"
        mock_tool_call.function.arguments = json.dumps({
            "agent_type": "unknown_agent",
            "task": "Do something",
        })

        result = manager.execute_delegation(mock_tool_call)
        assert result["ok"] is False
        assert "not in the delegates list" in result["error"]

    def test_delegate_to_execution_agent(self, parent_config, delegate_config, temp_dir):
        """Test delegating to an execution agent in-process mode."""
        registry = AgentRegistry(base_dir=temp_dir)

        mock_agent = MagicMock()
        mock_agent.config = parent_config
        mock_agent.messages = []
        mock_agent.llm = MagicMock()

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            mock_load.return_value = delegate_config

            manager = DelegationManager(registry, mock_agent)

            # ExecutionRunner is now imported inside the function, so patch the module
            with patch("supyagent.core.executor.ExecutionRunner") as mock_runner_class:
                mock_runner = MagicMock()
                mock_runner.run.return_value = {"ok": True, "data": "Code written"}
                mock_runner_class.return_value = mock_runner

                mock_tool_call = MagicMock()
                mock_tool_call.function.name = "delegate_to_coder"
                mock_tool_call.function.arguments = json.dumps({
                    "task": "Write a Python function",
                    "mode": "in_process",  # Use in_process mode to test the original code path
                })

                result = manager.execute_delegation(mock_tool_call)

        assert result["ok"] is True
        assert result["data"] == "Code written"

    def test_delegate_max_depth_reached(self, parent_config, temp_dir):
        """Test that delegation fails when max depth is reached."""
        registry = AgentRegistry(base_dir=temp_dir)

        # Create a chain to max depth
        # After MAX_DEPTH iterations, current_id will be at depth MAX_DEPTH - 1
        # Then the DelegationManager will register at depth MAX_DEPTH
        current_id = None
        for i in range(AgentRegistry.MAX_DEPTH):
            agent = MagicMock()
            agent.config.name = f"agent-{i}"
            current_id = registry.register(agent, parent_id=current_id)

        # current_id is now at depth MAX_DEPTH - 1 (e.g., depth 4 if MAX_DEPTH=5)
        mock_agent = MagicMock()
        mock_agent.config = parent_config
        mock_agent.messages = []
        mock_agent.llm = MagicMock()

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            mock_load.return_value = AgentConfig(
                name="coder",
                description="Coder",
                version="1.0",
                type="interactive",  # Interactive so it tries to create sub-agent
                model=ModelConfig(provider="test/model"),
                system_prompt="Code.",
                tools=ToolPermissions(allow=[]),
                delegates=[],
            )

            # This registers the manager at depth MAX_DEPTH (e.g., depth 5 if MAX_DEPTH=5)
            manager = DelegationManager(
                registry, mock_agent, grandparent_instance_id=current_id
            )

            # At this point, the manager's parent is at MAX_DEPTH
            # Any delegation would exceed MAX_DEPTH since sub-agent would be at MAX_DEPTH+1
            assert registry.get_depth(manager.parent_id) == AgentRegistry.MAX_DEPTH

            mock_tool_call = MagicMock()
            mock_tool_call.function.name = "delegate_to_coder"
            mock_tool_call.function.arguments = json.dumps({"task": "Write code"})

            result = manager.execute_delegation(mock_tool_call)

        assert result["ok"] is False
        assert "Maximum delegation depth" in result["error"]
