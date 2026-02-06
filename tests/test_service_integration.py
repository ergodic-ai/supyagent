"""
Tests for Phase 4: Tool Unification — service tools merged into agent discovery and dispatch.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from supyagent.core.tools import _matches_pattern, filter_tools
from supyagent.models.agent_config import AgentConfig, ModelConfig, ServiceConfig, ToolPermissions


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_service_tool(name, provider, service, permission, method="GET", path="/api/v1/test"):
    """Helper to create a service tool in OpenAI format with metadata."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Test tool: {name}",
            "parameters": {"type": "object", "properties": {}},
        },
        "metadata": {
            "provider": provider,
            "service": service,
            "permission": permission,
            "method": method,
            "path": path,
        },
    }


def _make_supypowers_tool(name, description="A supypowers tool"):
    """Helper to create a supypowers tool in OpenAI format (no metadata)."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


SAMPLE_SERVICE_TOOLS = [
    _make_service_tool(
        "gmail_list_messages", "google", "gmail", "gmail.read",
        "GET", "/api/v1/gmail/messages"
    ),
    _make_service_tool(
        "gmail_get_message", "google", "gmail", "gmail.read",
        "GET", "/api/v1/gmail/messages/{id}"
    ),
    _make_service_tool(
        "gmail_send_message", "google", "gmail", "gmail.send",
        "POST", "/api/v1/gmail/messages/send"
    ),
    _make_service_tool(
        "slack_list_channels", "slack", "slack.channels", "slack.channels.read",
        "GET", "/api/v1/slack/channels"
    ),
    _make_service_tool(
        "slack_send_message", "slack", "slack.messages", "slack.messages.write",
        "POST", "/api/v1/slack/messages"
    ),
    _make_service_tool(
        "github_list_repos", "github", "github.repos", "github.repos.read",
        "GET", "/api/v1/github/repos"
    ),
]


# ---------------------------------------------------------------------------
# _matches_pattern with service: prefix
# ---------------------------------------------------------------------------


class TestMatchesPatternService:
    def test_service_star_matches_service_tool(self):
        meta = {"provider": "google", "service": "gmail"}
        assert _matches_pattern("gmail_list_messages", "service:*", meta) is True

    def test_service_star_does_not_match_supypowers_tool(self):
        assert _matches_pattern("shell__run_command", "service:*", None) is False

    def test_service_provider_wildcard(self):
        meta = {"provider": "google", "service": "gmail"}
        assert _matches_pattern("gmail_list_messages", "service:google:*", meta) is True

    def test_service_provider_wildcard_no_match(self):
        meta = {"provider": "slack", "service": "slack.channels"}
        assert _matches_pattern("slack_list_channels", "service:google:*", meta) is False

    def test_service_name_wildcard(self):
        meta = {"provider": "google", "service": "gmail"}
        assert _matches_pattern("gmail_list_messages", "service:gmail:*", meta) is True

    def test_service_name_wildcard_prefix_match(self):
        meta = {"provider": "slack", "service": "slack.channels"}
        assert _matches_pattern("slack_list_channels", "service:slack:*", meta) is True

    def test_service_exact_match(self):
        meta = {"provider": "google", "service": "gmail"}
        assert _matches_pattern("gmail_list_messages", "service:gmail", meta) is True

    def test_service_exact_match_no_match(self):
        meta = {"provider": "google", "service": "gmail"}
        assert _matches_pattern("gmail_list_messages", "service:calendar", meta) is False

    def test_service_pattern_without_metadata(self):
        assert _matches_pattern("shell__run_command", "service:google:*", None) is False

    def test_star_matches_both_types(self):
        meta = {"provider": "google", "service": "gmail"}
        assert _matches_pattern("gmail_list_messages", "*", meta) is True
        assert _matches_pattern("shell__run_command", "*", None) is True

    def test_existing_supypowers_patterns_still_work(self):
        assert _matches_pattern("shell__run_command", "shell:*") is True
        assert _matches_pattern("shell__run_command", "shell:run_command") is True
        assert _matches_pattern("web_search__search", "web_search:*") is True


# ---------------------------------------------------------------------------
# filter_tools with mixed tool types
# ---------------------------------------------------------------------------


class TestFilterToolsMixed:
    def test_allow_all_includes_both_types(self):
        tools = [
            _make_supypowers_tool("shell__run_command"),
            _make_service_tool("gmail_list_messages", "google", "gmail", "gmail.read"),
        ]
        perms = ToolPermissions(allow=["*"])
        result = filter_tools(tools, perms)
        assert len(result) == 2

    def test_deny_service_keeps_supypowers(self):
        tools = [
            _make_supypowers_tool("shell__run_command"),
            _make_service_tool("gmail_list_messages", "google", "gmail", "gmail.read"),
        ]
        perms = ToolPermissions(allow=["*"], deny=["service:*"])
        result = filter_tools(tools, perms)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "shell__run_command"

    def test_allow_only_service(self):
        tools = [
            _make_supypowers_tool("shell__run_command"),
            _make_service_tool("gmail_list_messages", "google", "gmail", "gmail.read"),
        ]
        perms = ToolPermissions(allow=["service:*"])
        result = filter_tools(tools, perms)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "gmail_list_messages"

    def test_allow_specific_service_provider(self):
        tools = SAMPLE_SERVICE_TOOLS.copy()
        perms = ToolPermissions(allow=["service:google:*"])
        result = filter_tools(tools, perms)
        names = [t["function"]["name"] for t in result]
        assert "gmail_list_messages" in names
        assert "gmail_send_message" in names
        assert "slack_list_channels" not in names
        assert "github_list_repos" not in names

    def test_deny_specific_service(self):
        tools = SAMPLE_SERVICE_TOOLS.copy()
        perms = ToolPermissions(allow=["service:*"], deny=["service:gmail:*"])
        result = filter_tools(tools, perms)
        names = [t["function"]["name"] for t in result]
        assert "gmail_list_messages" not in names
        assert "slack_list_channels" in names
        assert "github_list_repos" in names

    def test_no_permissions_allows_all(self):
        tools = [
            _make_supypowers_tool("shell__run_command"),
            _make_service_tool("gmail_list_messages", "google", "gmail", "gmail.read"),
        ]
        perms = ToolPermissions()
        result = filter_tools(tools, perms)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Engine: tool discovery merging
# ---------------------------------------------------------------------------


class TestEngineToolDiscovery:
    def _make_engine(self, service_tools=None, service_enabled=True):
        """Create a BaseAgentEngine subclass for testing."""
        from supyagent.core.engine import BaseAgentEngine

        config = AgentConfig(
            name="test",
            model=ModelConfig(provider="anthropic/claude-3-5-sonnet-20241022"),
            system_prompt="Test",
            tools=ToolPermissions(allow=["*"]),
            service=ServiceConfig(enabled=service_enabled),
        )

        mock_service_client = None
        if service_tools is not None and service_enabled:
            mock_service_client = MagicMock()
            mock_service_client.discover_tools.return_value = service_tools

        # Create a concrete subclass
        class TestEngine(BaseAgentEngine):
            def _get_secrets(self):
                return {}

        with patch("supyagent.core.engine.discover_tools", return_value=[]), \
             patch("supyagent.core.service.get_service_client", return_value=mock_service_client):
            engine = TestEngine(config)

        return engine

    def test_no_service_client(self):
        engine = self._make_engine(service_enabled=False)
        tools = engine._load_base_tools()
        # Only process management tools (no supypowers in test env)
        service_names = [n for n in engine.get_available_tools() if n in engine._service_tool_metadata]
        assert len(service_names) == 0

    def test_service_tools_merged(self):
        service_tools = [
            _make_service_tool("gmail_list_messages", "google", "gmail", "gmail.read"),
            _make_service_tool("slack_send_message", "slack", "slack.messages", "slack.messages.write"),
        ]
        engine = self._make_engine(service_tools=service_tools)
        tools = engine._load_base_tools()

        tool_names = [t["function"]["name"] for t in tools]
        assert "gmail_list_messages" in tool_names
        assert "slack_send_message" in tool_names

    def test_metadata_stored_for_dispatch(self):
        service_tools = [
            _make_service_tool(
                "gmail_list_messages", "google", "gmail", "gmail.read",
                "GET", "/api/v1/gmail/messages"
            ),
        ]
        engine = self._make_engine(service_tools=service_tools)
        engine._load_base_tools()

        assert "gmail_list_messages" in engine._service_tool_metadata
        meta = engine._service_tool_metadata["gmail_list_messages"]
        assert meta["method"] == "GET"
        assert meta["path"] == "/api/v1/gmail/messages"
        assert meta["provider"] == "google"

    def test_metadata_stripped_from_tool_definitions(self):
        service_tools = [
            _make_service_tool("gmail_list_messages", "google", "gmail", "gmail.read"),
        ]
        engine = self._make_engine(service_tools=service_tools)
        tools = engine._load_base_tools()

        gmail_tools = [t for t in tools if t["function"]["name"] == "gmail_list_messages"]
        assert len(gmail_tools) == 1
        assert "metadata" not in gmail_tools[0]

    def test_service_tools_respect_deny_filter(self):
        service_tools = [
            _make_service_tool("gmail_list_messages", "google", "gmail", "gmail.read"),
            _make_service_tool("slack_send_message", "slack", "slack.messages", "slack.messages.write"),
        ]

        from supyagent.core.engine import BaseAgentEngine

        config = AgentConfig(
            name="test",
            model=ModelConfig(provider="anthropic/claude-3-5-sonnet-20241022"),
            system_prompt="Test",
            tools=ToolPermissions(allow=["*"], deny=["service:slack:*"]),
            service=ServiceConfig(enabled=True),
        )

        mock_service_client = MagicMock()
        mock_service_client.discover_tools.return_value = service_tools

        class TestEngine(BaseAgentEngine):
            def _get_secrets(self):
                return {}

        with patch("supyagent.core.engine.discover_tools", return_value=[]), \
             patch("supyagent.core.service.get_service_client", return_value=mock_service_client):
            engine = TestEngine(config)

        tools = engine._load_base_tools()
        tool_names = [t["function"]["name"] for t in tools]
        assert "gmail_list_messages" in tool_names
        assert "slack_send_message" not in tool_names

    def test_empty_service_tools(self):
        engine = self._make_engine(service_tools=[])
        tools = engine._load_base_tools()
        service_names = [n for n in engine.get_available_tools() if n in engine._service_tool_metadata]
        assert len(service_names) == 0


# ---------------------------------------------------------------------------
# Engine: tool dispatch routing
# ---------------------------------------------------------------------------


class TestEngineToolDispatch:
    def _make_engine_with_service(self, execute_result=None):
        """Create a test engine with a mock service client."""
        from supyagent.core.engine import BaseAgentEngine

        config = AgentConfig(
            name="test",
            model=ModelConfig(provider="anthropic/claude-3-5-sonnet-20241022"),
            system_prompt="Test",
            tools=ToolPermissions(allow=["*"]),
            service=ServiceConfig(enabled=True),
        )

        mock_client = MagicMock()
        mock_client.discover_tools.return_value = [
            _make_service_tool(
                "gmail_list_messages", "google", "gmail", "gmail.read",
                "GET", "/api/v1/gmail/messages"
            ),
        ]
        if execute_result:
            mock_client.execute_tool.return_value = execute_result
        else:
            mock_client.execute_tool.return_value = {
                "ok": True,
                "data": {"messages": [{"id": "1", "subject": "Test"}]},
            }

        class TestEngine(BaseAgentEngine):
            def _get_secrets(self):
                return {}

        with patch("supyagent.core.engine.discover_tools", return_value=[]), \
             patch("supyagent.core.service.get_service_client", return_value=mock_client):
            engine = TestEngine(config)

        # Load tools to populate _service_tool_metadata
        engine.tools = engine._load_base_tools()
        return engine, mock_client

    def test_service_tool_routed_to_http(self):
        engine, mock_client = self._make_engine_with_service()

        tool_call = MagicMock()
        tool_call.function.name = "gmail_list_messages"
        tool_call.function.arguments = '{"max_results": 10}'

        result = engine._dispatch_tool_call(tool_call)

        assert result["ok"] is True
        mock_client.execute_tool.assert_called_once_with(
            "gmail_list_messages",
            {"max_results": 10},
            engine._service_tool_metadata["gmail_list_messages"],
        )

    def test_supypowers_tool_not_routed_to_service(self):
        engine, mock_client = self._make_engine_with_service()

        tool_call = MagicMock()
        tool_call.function.name = "shell__run_command"
        tool_call.function.arguments = '{"command": "echo hello"}'

        with patch.object(engine, "_execute_supypowers_tool", return_value={"ok": True}) as mock_sp:
            result = engine._dispatch_tool_call(tool_call)

        mock_sp.assert_called_once()
        mock_client.execute_tool.assert_not_called()

    def test_service_tool_invalid_json_args(self):
        engine, _ = self._make_engine_with_service()

        tool_call = MagicMock()
        tool_call.function.name = "gmail_list_messages"
        tool_call.function.arguments = "not valid json"

        result = engine._dispatch_tool_call(tool_call)
        assert result["ok"] is False
        assert "Invalid JSON" in result["error"]

    def test_service_tool_error_tracked_by_circuit_breaker(self):
        error_result = {"ok": False, "error": "Rate limit exceeded"}
        engine, _ = self._make_engine_with_service(execute_result=error_result)

        tool_call = MagicMock()
        tool_call.function.name = "gmail_list_messages"
        tool_call.function.arguments = '{"max_results": 10}'

        # Call multiple times to trigger circuit breaker
        for _ in range(3):
            result = engine._dispatch_tool_call(tool_call)

        # Fourth call should be blocked by circuit breaker
        result = engine._dispatch_tool_call(tool_call)
        assert result["ok"] is False
        assert "circuit_breaker" in result.get("error_type", "")

    def test_service_tool_success_resets_circuit_breaker(self):
        engine, mock_client = self._make_engine_with_service()

        tool_call = MagicMock()
        tool_call.function.name = "gmail_list_messages"
        tool_call.function.arguments = '{"max_results": 10}'

        # Fail twice
        mock_client.execute_tool.return_value = {"ok": False, "error": "error"}
        engine._dispatch_tool_call(tool_call)
        engine._dispatch_tool_call(tool_call)
        assert engine._tool_failure_counts["gmail_list_messages"] == 2

        # Succeed once — should reset
        mock_client.execute_tool.return_value = {"ok": True, "data": {}}
        engine._dispatch_tool_call(tool_call)
        assert engine._tool_failure_counts["gmail_list_messages"] == 0

    def test_service_client_none_returns_error(self):
        """If service client disappears after init, return helpful error."""
        engine, _ = self._make_engine_with_service()
        engine._service_client = None

        tool_call = MagicMock()
        tool_call.function.name = "gmail_list_messages"
        tool_call.function.arguments = '{}'

        result = engine._execute_service_tool(tool_call)
        assert result["ok"] is False
        assert "connect" in result["error"].lower()


# ---------------------------------------------------------------------------
# Engine: service disabled
# ---------------------------------------------------------------------------


class TestEngineServiceDisabled:
    def test_no_service_client_when_disabled(self):
        from supyagent.core.engine import BaseAgentEngine

        config = AgentConfig(
            name="test",
            model=ModelConfig(provider="anthropic/claude-3-5-sonnet-20241022"),
            system_prompt="Test",
            service=ServiceConfig(enabled=False),
        )

        class TestEngine(BaseAgentEngine):
            def _get_secrets(self):
                return {}

        with patch("supyagent.core.engine.discover_tools", return_value=[]):
            engine = TestEngine(config)

        assert engine._service_client is None
        assert engine._service_tool_metadata == {}

    def test_service_yaml_opt_out(self):
        """Agent YAML with service: { enabled: false } should not load service tools."""
        from supyagent.core.engine import BaseAgentEngine

        config = AgentConfig(
            name="test",
            model=ModelConfig(provider="anthropic/claude-3-5-sonnet-20241022"),
            system_prompt="Test",
            service=ServiceConfig(enabled=False),
        )

        class TestEngine(BaseAgentEngine):
            def _get_secrets(self):
                return {}

        # Even if get_service_client returns a client, it should not be used
        with patch("supyagent.core.engine.discover_tools", return_value=[]):
            engine = TestEngine(config)

        assert engine._service_client is None
