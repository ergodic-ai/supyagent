"""
Tests for supypowers tool integration.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from supyagent.core.tools import (
    _matches_pattern,
    discover_tools,
    execute_tool,
    filter_tools,
    supypowers_to_openai_tools,
)
from supyagent.models.agent_config import ToolPermissions


class TestMatchesPattern:
    """Tests for pattern matching."""

    def test_wildcard_matches_all(self):
        """Test that * matches everything."""
        assert _matches_pattern("anything__function", "*")
        assert _matches_pattern("test__hello", "*")

    def test_script_wildcard(self):
        """Test script:* pattern matches all functions in script."""
        assert _matches_pattern("web_search__search", "web_search:*")
        assert _matches_pattern("web_search__advanced", "web_search:*")
        assert not _matches_pattern("other__search", "web_search:*")

    def test_exact_match_colon(self):
        """Test exact match with colon separator."""
        assert _matches_pattern("test__hello", "test:hello")
        assert not _matches_pattern("test__other", "test:hello")

    def test_exact_match_double_underscore(self):
        """Test exact match with double underscore."""
        assert _matches_pattern("test__hello", "test__hello")
        assert not _matches_pattern("test__other", "test__hello")


class TestFilterTools:
    """Tests for filtering tools by permissions."""

    def test_no_permissions_allows_all(self):
        """Test that no permissions allows all tools."""
        tools = [
            {"function": {"name": "tool1"}},
            {"function": {"name": "tool2"}},
        ]
        permissions = ToolPermissions()

        filtered = filter_tools(tools, permissions)
        assert len(filtered) == 2

    def test_allow_list_filters(self):
        """Test that allow list filters tools."""
        tools = [
            {"function": {"name": "web__search"}},
            {"function": {"name": "web__fetch"}},
            {"function": {"name": "db__query"}},
        ]
        permissions = ToolPermissions(allow=["web:*"])

        filtered = filter_tools(tools, permissions)
        assert len(filtered) == 2
        names = [t["function"]["name"] for t in filtered]
        assert "web__search" in names
        assert "web__fetch" in names
        assert "db__query" not in names

    def test_deny_list_blocks(self):
        """Test that deny list blocks tools."""
        tools = [
            {"function": {"name": "safe__func"}},
            {"function": {"name": "dangerous__delete"}},
        ]
        permissions = ToolPermissions(deny=["dangerous:*"])

        filtered = filter_tools(tools, permissions)
        assert len(filtered) == 1
        assert filtered[0]["function"]["name"] == "safe__func"

    def test_deny_overrides_allow(self):
        """Test that deny takes precedence over allow."""
        tools = [
            {"function": {"name": "web__search"}},
            {"function": {"name": "web__delete"}},
        ]
        permissions = ToolPermissions(
            allow=["web:*"],
            deny=["web:delete"],
        )

        filtered = filter_tools(tools, permissions)
        assert len(filtered) == 1
        assert filtered[0]["function"]["name"] == "web__search"


class TestSupypowersToOpenAITools:
    """Tests for converting supypowers format to OpenAI format."""

    def test_basic_conversion(self):
        """Test basic format conversion."""
        sp_tools = [
            {
                "script": "/path/to/hello.py",
                "functions": [
                    {
                        "name": "greet",
                        "description": "Greets someone",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"}
                            },
                            "required": ["name"],
                        },
                    }
                ]
            }
        ]

        openai_tools = supypowers_to_openai_tools(sp_tools)

        assert len(openai_tools) == 1
        tool = openai_tools[0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "hello__greet"
        assert tool["function"]["description"] == "Greets someone"
        assert "name" in tool["function"]["parameters"]["properties"]

    def test_multiple_tools(self):
        """Test converting multiple tools."""
        sp_tools = [
            {
                "script": "/path/to/a.py",
                "functions": [
                    {"name": "func1", "description": "", "input_schema": {}},
                    {"name": "func2", "description": "", "input_schema": {}},
                ]
            },
            {
                "script": "/path/to/b.py",
                "functions": [
                    {"name": "func1", "description": "", "input_schema": {}},
                ]
            },
        ]

        openai_tools = supypowers_to_openai_tools(sp_tools)

        assert len(openai_tools) == 3
        names = {t["function"]["name"] for t in openai_tools}
        assert names == {"a__func1", "a__func2", "b__func1"}

    def test_default_description(self):
        """Test that missing description gets default."""
        sp_tools = [
            {
                "script": "/path/to/test.py",
                "functions": [
                    {"name": "func", "input_schema": {}},
                ]
            },
        ]

        openai_tools = supypowers_to_openai_tools(sp_tools)

        assert openai_tools[0]["function"]["description"] == "No description"


class TestDiscoverTools:
    """Tests for tool discovery."""

    @patch("supyagent.core.tools.subprocess.run")
    def test_discover_tools_success(self, mock_run):
        """Test successful tool discovery."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps([
                {"script": "test", "function": "hello", "description": "Test", "input_schema": {}}
            ]),
        )

        tools = discover_tools()

        assert len(tools) == 1
        assert tools[0]["script"] == "test"
        mock_run.assert_called_once()

    @patch("supyagent.core.tools.subprocess.run")
    def test_discover_tools_supypowers_not_found(self, mock_run):
        """Test when supypowers is not installed."""
        mock_run.side_effect = FileNotFoundError()

        tools = discover_tools()

        assert tools == []

    @patch("supyagent.core.tools.subprocess.run")
    def test_discover_tools_invalid_json(self, mock_run):
        """Test when supypowers returns invalid JSON."""
        mock_run.return_value = MagicMock(returncode=0, stdout="not json")

        tools = discover_tools()

        assert tools == []

    @patch("supyagent.core.tools.subprocess.run")
    def test_discover_tools_error_exit(self, mock_run):
        """Test when supypowers exits with error."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Error")

        tools = discover_tools()

        assert tools == []


class TestExecuteTool:
    """Tests for tool execution."""

    @patch("supyagent.core.tools.subprocess.run")
    def test_execute_tool_success(self, mock_run):
        """Test successful tool execution."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"ok": true, "data": "Hello, World!"}',
        )

        result = execute_tool("hello", "greet", {"name": "World"})

        assert result["ok"] is True
        assert result["data"] == "Hello, World!"

        # Check command was correct
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd[0] == "supypowers"
        assert cmd[1] == "run"
        assert cmd[2] == "hello:greet"
        assert json.loads(cmd[3]) == {"name": "World"}

    @patch("supyagent.core.tools.subprocess.run")
    def test_execute_tool_with_secrets(self, mock_run):
        """Test tool execution with secrets."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"ok": true}',
        )

        execute_tool("api", "call", {}, secrets={"API_KEY": "secret123"})

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "--secrets" in cmd
        assert "API_KEY=secret123" in cmd

    @patch("supyagent.core.tools.subprocess.run")
    def test_execute_tool_supypowers_not_found(self, mock_run):
        """Test when supypowers is not installed."""
        mock_run.side_effect = FileNotFoundError()

        result = execute_tool("test", "func", {})

        assert result["ok"] is False
        assert "not installed" in result["error"]

    @patch("supyagent.core.tools.subprocess.run")
    def test_execute_tool_invalid_json_response(self, mock_run):
        """Test when tool returns invalid JSON."""
        mock_run.return_value = MagicMock(returncode=0, stdout="not json")

        result = execute_tool("test", "func", {})

        assert result["ok"] is False
        assert "Invalid JSON" in result["error"]

    @patch("supyagent.core.tools.subprocess.run")
    def test_execute_tool_timeout(self, mock_run):
        """Test when tool execution times out."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=300)

        result = execute_tool("test", "func", {})

        assert result["ok"] is False
        assert "timed out" in result["error"]

    @patch("supyagent.core.tools.subprocess.run")
    def test_execute_tool_error_exit(self, mock_run):
        """Test when tool exits with error."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Tool error message",
        )

        result = execute_tool("test", "func", {})

        assert result["ok"] is False
        assert "Tool error message" in result["error"]
