"""
Tests for CLI commands.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from supyagent.cli.main import cli


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def isolated_runner():
    """Create an isolated CLI test runner with temp directory."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        yield runner


class TestCliHelp:
    """Tests for CLI help commands."""

    def test_main_help(self, runner):
        """Test main --help."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Supyagent" in result.output
        assert "chat" in result.output
        assert "new" in result.output

    def test_version(self, runner):
        """Test --version."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "supyagent, version" in result.output


class TestNewCommand:
    """Tests for 'supyagent new' command."""

    def test_new_interactive_agent(self, isolated_runner):
        """Test creating a new interactive agent."""
        result = isolated_runner.invoke(cli, ["new", "myagent"])

        assert result.exit_code == 0
        assert "Created agent" in result.output

        # Check file was created
        agent_file = Path("agents/myagent.yaml")
        assert agent_file.exists()

        content = agent_file.read_text()
        assert "name: myagent" in content
        assert "type: interactive" in content

    def test_new_execution_agent(self, isolated_runner):
        """Test creating a new execution agent."""
        result = isolated_runner.invoke(cli, ["new", "myagent", "--type", "execution"])

        assert result.exit_code == 0

        agent_file = Path("agents/myagent.yaml")
        content = agent_file.read_text()
        assert "type: execution" in content
        assert "temperature: 0.3" in content  # Lower for execution agents

    def test_new_agent_overwrite_prompt(self, isolated_runner):
        """Test that overwriting existing agent prompts."""
        # Create first
        isolated_runner.invoke(cli, ["new", "myagent"])

        # Try to create again without confirmation
        result = isolated_runner.invoke(cli, ["new", "myagent"], input="n\n")
        assert result.exit_code == 0

        # With confirmation
        result = isolated_runner.invoke(cli, ["new", "myagent"], input="y\n")
        assert result.exit_code == 0
        assert "Created agent" in result.output


class TestListCommand:
    """Tests for 'supyagent list' command."""

    def test_list_no_agents_dir(self, isolated_runner):
        """Test list when no agents directory exists."""
        result = isolated_runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "No agents directory" in result.output

    def test_list_empty_agents_dir(self, isolated_runner):
        """Test list when agents directory is empty."""
        Path("agents").mkdir()
        result = isolated_runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "No agents found" in result.output

    def test_list_agents(self, isolated_runner):
        """Test listing agents."""
        # Create some agents
        isolated_runner.invoke(cli, ["new", "agent1"])
        isolated_runner.invoke(cli, ["new", "agent2", "--type", "execution"])

        result = isolated_runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "agent1" in result.output
        assert "agent2" in result.output
        assert "interactive" in result.output
        assert "execution" in result.output


class TestShowCommand:
    """Tests for 'supyagent show' command."""

    def test_show_agent(self, isolated_runner):
        """Test showing agent details."""
        isolated_runner.invoke(cli, ["new", "myagent"])

        result = isolated_runner.invoke(cli, ["show", "myagent"])
        assert result.exit_code == 0
        assert "myagent" in result.output
        assert "Type:" in result.output
        assert "Model:" in result.output
        assert "System Prompt:" in result.output

    def test_show_nonexistent_agent(self, isolated_runner):
        """Test showing non-existent agent."""
        Path("agents").mkdir()
        result = isolated_runner.invoke(cli, ["show", "nonexistent"])
        assert result.exit_code == 1
        assert "not found" in result.output


class TestSessionsCommand:
    """Tests for 'supyagent sessions' command."""

    def test_sessions_no_sessions(self, isolated_runner):
        """Test listing sessions when none exist."""
        isolated_runner.invoke(cli, ["new", "myagent"])

        result = isolated_runner.invoke(cli, ["sessions", "myagent"])
        assert result.exit_code == 0
        assert "No sessions found" in result.output


class TestChatCommand:
    """Tests for 'supyagent chat' command."""

    def test_chat_help(self, runner):
        """Test chat --help."""
        result = runner.invoke(cli, ["chat", "--help"])
        assert result.exit_code == 0
        assert "--new" in result.output
        assert "--session" in result.output

    def test_chat_nonexistent_agent(self, isolated_runner):
        """Test chatting with non-existent agent."""
        Path("agents").mkdir()
        result = isolated_runner.invoke(cli, ["chat", "nonexistent"])
        assert result.exit_code == 1
        assert "not found" in result.output

    @patch("supyagent.core.agent.discover_tools", return_value=[])
    @patch("supyagent.cli.main.Agent")
    def test_chat_quit_command(self, mock_agent_class, mock_discover, isolated_runner):
        """Test /quit command exits chat."""
        # Create an agent
        isolated_runner.invoke(cli, ["new", "myagent"])

        # Mock the Agent class
        mock_agent = mock_agent_class.return_value
        mock_agent.session.meta.session_id = "test123"
        mock_agent.session.messages = []
        mock_agent.tools = []
        mock_agent.config.name = "myagent"
        mock_agent.config.description = "Test"
        mock_agent.config.model.provider = "test/model"

        result = isolated_runner.invoke(cli, ["chat", "myagent"], input="/quit\n")
        assert "Goodbye" in result.output

    @patch("supyagent.core.agent.discover_tools", return_value=[])
    @patch("supyagent.cli.main.Agent")
    def test_chat_help_command(self, mock_agent_class, mock_discover, isolated_runner):
        """Test /help command shows help."""
        isolated_runner.invoke(cli, ["new", "myagent"])

        mock_agent = mock_agent_class.return_value
        mock_agent.session.meta.session_id = "test123"
        mock_agent.session.messages = []
        mock_agent.tools = []
        mock_agent.config.name = "myagent"
        mock_agent.config.description = "Test"
        mock_agent.config.model.provider = "test/model"

        result = isolated_runner.invoke(
            cli, ["chat", "myagent"], input="/help\n/quit\n"
        )
        assert "Available commands" in result.output
        assert "/tools" in result.output
        assert "/sessions" in result.output


class TestChatWithFlags:
    """Tests for chat command with session flags."""

    @patch("supyagent.cli.main.Agent")
    def test_chat_new_flag(self, mock_agent_class, isolated_runner):
        """Test --new flag starts fresh session."""
        isolated_runner.invoke(cli, ["new", "myagent"])

        mock_agent = mock_agent_class.return_value
        mock_agent.session.meta.session_id = "new123"
        mock_agent.session.messages = []
        mock_agent.tools = []
        mock_agent.config.name = "myagent"
        mock_agent.config.description = "Test"
        mock_agent.config.model.provider = "test/model"

        result = isolated_runner.invoke(
            cli, ["chat", "myagent", "--new"], input="/quit\n"
        )

        assert result.exit_code == 0
        # Agent should be called with session=None when --new is used
        call_kwargs = mock_agent_class.call_args[1]
        assert call_kwargs.get("session") is None

    def test_chat_session_flag_invalid(self, isolated_runner):
        """Test --session with invalid ID."""
        isolated_runner.invoke(cli, ["new", "myagent"])

        result = isolated_runner.invoke(
            cli, ["chat", "myagent", "--session", "invalid123"]
        )

        assert result.exit_code == 1
        assert "not found" in result.output
