"""
Phase 5: End-to-end integration tests and edge cases for service integration.

Tests the full workflow: connect → discover → execute → verify,
plus edge cases for robustness.
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from supyagent.core.service import (
    DEFAULT_SERVICE_URL,
    SERVICE_API_KEY,
    SERVICE_URL,
    ServiceClient,
    clear_service_credentials,
    store_service_credentials,
)
from supyagent.core.tools import _matches_pattern, filter_tools
from supyagent.models.agent_config import (
    AgentConfig,
    ModelConfig,
    ServiceConfig,
    ToolPermissions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service_tool(name, provider, service, permission, method="GET", path="/api/v1/test"):
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


def _make_supypowers_tool(script_name, func_name, description="A supypowers tool"):
    """Create a raw supypowers tool entry (as returned by `supypowers docs --format json`).

    This will be converted by supypowers_to_openai_tools into OpenAI format with
    name = "{script_name}__{func_name}".
    """
    return {
        "script": f"/path/to/{script_name}.py",
        "functions": [
            {
                "name": func_name,
                "description": description,
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
    }


def _make_test_engine(service_tools=None, supypowers_tools=None, service_enabled=True,
                       tool_permissions=None, execute_result=None, load_tools=False):
    """Create a BaseAgentEngine subclass with mocked dependencies.

    supypowers_tools should be in raw supypowers format (use _make_supypowers_tool).
    Set load_tools=True to also call _load_base_tools() within the patch context.
    """
    from supyagent.core.engine import BaseAgentEngine

    config = AgentConfig(
        name="test",
        model=ModelConfig(provider="anthropic/claude-3-5-sonnet-20241022"),
        system_prompt="Test agent",
        tools=tool_permissions or ToolPermissions(allow=["*"]),
        service=ServiceConfig(enabled=service_enabled),
    )

    mock_service_client = None
    if service_tools is not None and service_enabled:
        mock_service_client = MagicMock()
        mock_service_client.discover_tools.return_value = service_tools
        if execute_result is not None:
            mock_service_client.execute_tool.return_value = execute_result
        else:
            mock_service_client.execute_tool.return_value = {"ok": True, "data": {}}

    sp_tools = supypowers_tools or []

    class TestEngine(BaseAgentEngine):
        def _get_secrets(self):
            return {}

    with patch("supyagent.core.engine.discover_tools", return_value=sp_tools), \
         patch("supyagent.core.service.get_service_client", return_value=mock_service_client):
        engine = TestEngine(config)
        if load_tools:
            engine.tools = engine._load_base_tools()

    return engine, mock_service_client


# ---------------------------------------------------------------------------
# E2E: Full workflow (connect → discover → dispatch → verify)
# ---------------------------------------------------------------------------


class TestEndToEndWorkflow:
    """Full workflow: credentials stored → engine discovers tools → dispatches → result."""

    def test_full_connect_discover_execute(self):
        """Simulate: connect stores creds → engine init → discover tools → execute tool."""
        # 1. Simulate `supyagent connect` storing credentials
        mock_config = MagicMock()
        mock_config.get.side_effect = lambda key: {
            SERVICE_API_KEY: "sk_live_test123",
            SERVICE_URL: "https://test.supyagent.com",
        }.get(key)

        with patch("supyagent.core.service.get_config_manager", return_value=mock_config):
            store_service_credentials("sk_live_test123", "https://test.supyagent.com")
            mock_config.set.assert_any_call(SERVICE_API_KEY, "sk_live_test123")
            mock_config.set.assert_any_call(SERVICE_URL, "https://test.supyagent.com")

        # 2. Engine discovers service tools
        service_tools = [
            _make_service_tool(
                "gmail_list_messages", "google", "gmail", "gmail.read",
                "GET", "/api/v1/gmail/messages"
            ),
            _make_service_tool(
                "slack_send_message", "slack", "slack.messages", "slack.messages.write",
                "POST", "/api/v1/slack/messages"
            ),
        ]
        gmail_result = {
            "ok": True,
            "data": {
                "messages": [
                    {"id": "msg_1", "subject": "Hello", "from": "alice@example.com"},
                    {"id": "msg_2", "subject": "Meeting", "from": "bob@example.com"},
                ],
            },
        }
        engine, mock_client = _make_test_engine(
            service_tools=service_tools, execute_result=gmail_result
        )
        engine.tools = engine._load_base_tools()

        # 3. Verify tools were discovered
        tool_names = [t["function"]["name"] for t in engine.tools]
        assert "gmail_list_messages" in tool_names
        assert "slack_send_message" in tool_names

        # 4. Verify metadata stored for routing
        assert "gmail_list_messages" in engine._service_tool_metadata
        assert engine._service_tool_metadata["gmail_list_messages"]["method"] == "GET"

        # 5. Execute a service tool
        tool_call = MagicMock()
        tool_call.function.name = "gmail_list_messages"
        tool_call.function.arguments = json.dumps({"max_results": 5})

        result = engine._dispatch_tool_call(tool_call)

        # 6. Verify result
        assert result["ok"] is True
        assert len(result["data"]["messages"]) == 2
        assert result["data"]["messages"][0]["subject"] == "Hello"

        # 7. Verify HTTP call was made correctly
        mock_client.execute_tool.assert_called_once_with(
            "gmail_list_messages",
            {"max_results": 5},
            engine._service_tool_metadata["gmail_list_messages"],
        )

    def test_full_workflow_with_mixed_tools(self):
        """Service and supypowers tools coexist, each dispatched to correct handler."""
        service_tools = [
            _make_service_tool(
                "gmail_list_messages", "google", "gmail", "gmail.read",
                "GET", "/api/v1/gmail/messages"
            ),
        ]
        supypowers_tools = [
            _make_supypowers_tool("shell", "run_command", "Execute a shell command"),
        ]

        engine, mock_client = _make_test_engine(
            service_tools=service_tools,
            supypowers_tools=supypowers_tools,
            execute_result={"ok": True, "data": {"messages": []}},
            load_tools=True,
        )

        tool_names = [t["function"]["name"] for t in engine.tools]
        assert "gmail_list_messages" in tool_names
        assert "shell__run_command" in tool_names

        # Service tool → HTTP
        service_call = MagicMock()
        service_call.function.name = "gmail_list_messages"
        service_call.function.arguments = "{}"
        result = engine._dispatch_tool_call(service_call)
        assert result["ok"] is True
        mock_client.execute_tool.assert_called_once()

        # Supypowers tool → NOT service
        sp_call = MagicMock()
        sp_call.function.name = "shell__run_command"
        sp_call.function.arguments = '{"command": "echo hello"}'

        with patch.object(engine, "_execute_supypowers_tool", return_value={"ok": True}) as mock_sp:
            result = engine._dispatch_tool_call(sp_call)

        mock_sp.assert_called_once()
        # Service client should not have been called again
        assert mock_client.execute_tool.call_count == 1

    def test_workflow_service_not_connected(self):
        """Engine works normally without service tools when not connected."""
        supypowers_tools = [
            _make_supypowers_tool("shell", "run_command"),
        ]

        engine, _ = _make_test_engine(
            service_enabled=True,  # Enabled but no credentials (get_service_client returns None)
            supypowers_tools=supypowers_tools,
            load_tools=True,
        )

        tool_names = [t["function"]["name"] for t in engine.tools]
        assert "shell__run_command" in tool_names
        # No service tools
        assert all(n not in engine._service_tool_metadata for n in tool_names)

    def test_workflow_disconnect_removes_service_tools(self):
        """After disconnect, engine should not have service tools."""
        mock_config = MagicMock()
        mock_config.delete.return_value = True

        with patch("supyagent.core.service.get_config_manager", return_value=mock_config):
            removed = clear_service_credentials()

        assert removed is True
        mock_config.delete.assert_any_call(SERVICE_API_KEY)
        mock_config.delete.assert_any_call(SERVICE_URL)


# ---------------------------------------------------------------------------
# Edge cases: malformed / missing metadata
# ---------------------------------------------------------------------------


class TestMalformedMetadata:
    def test_missing_method_defaults_to_get(self):
        """If metadata has no method, execute_tool defaults to GET."""
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"result": "ok"})
        )
        client = ServiceClient.__new__(ServiceClient)
        client.api_key = "sk_test"
        client.base_url = "https://test.example.com"
        client._client = httpx.Client(
            base_url=client.base_url,
            transport=transport,
            headers={"Authorization": "Bearer sk_test"},
        )

        result = client.execute_tool(
            "test_tool",
            {"q": "hello"},
            {"path": "/api/v1/test"},  # No method field
        )
        assert result["ok"] is True

    def test_missing_path_returns_error(self):
        """If metadata has no path, should return error."""
        client = ServiceClient.__new__(ServiceClient)
        client.api_key = "sk_test"
        client.base_url = "https://test.example.com"
        client._client = MagicMock()

        result = client.execute_tool(
            "test_tool",
            {"q": "hello"},
            {"method": "GET"},  # No path field
        )
        assert result["ok"] is False
        assert "No API path" in result["error"]

    def test_empty_metadata(self):
        """Empty metadata should return error (no path)."""
        client = ServiceClient.__new__(ServiceClient)
        client.api_key = "sk_test"
        client.base_url = "https://test.example.com"
        client._client = MagicMock()

        result = client.execute_tool("test_tool", {}, {})
        assert result["ok"] is False
        assert "No API path" in result["error"]

    def test_tool_with_no_name_skipped_in_metadata_store(self):
        """Service tool with empty name should not pollute _service_tool_metadata."""
        service_tools = [
            {
                "type": "function",
                "function": {"name": "", "description": "Empty", "parameters": {}},
                "metadata": {"provider": "test"},
            },
            _make_service_tool("real_tool", "google", "gmail", "gmail.read"),
        ]
        engine, _ = _make_test_engine(service_tools=service_tools)
        engine.tools = engine._load_base_tools()

        assert "" not in engine._service_tool_metadata
        assert "real_tool" in engine._service_tool_metadata


# ---------------------------------------------------------------------------
# Edge cases: tool name collisions
# ---------------------------------------------------------------------------


class TestToolNameCollisions:
    def test_service_and_supypowers_same_name(self):
        """If a service tool has the same name as a supypowers tool, both are included."""
        # Service tool named "search__web" (could happen if service uses same naming)
        service_tools = [
            _make_service_tool(
                "search__web", "google", "google.search", "search.read",
                "GET", "/api/v1/google/search"
            ),
        ]
        # Supypowers tool: script "search" with function "web" → name "search__web"
        supypowers_tools = [
            _make_supypowers_tool("search", "web", "Local web search via supypowers"),
        ]

        engine, _ = _make_test_engine(
            service_tools=service_tools,
            supypowers_tools=supypowers_tools,
            load_tools=True,
        )

        # Both tools get loaded (LLM might see duplicates — but engine handles dispatch
        # based on _service_tool_metadata, which only has the service one)
        matching = [t for t in engine.tools if t["function"]["name"] == "search__web"]
        assert len(matching) == 2

        # The service version gets routed to HTTP
        assert "search__web" in engine._service_tool_metadata

    def test_duplicate_service_tool_names(self):
        """If service returns duplicate names, both appear but only last metadata stored."""
        service_tools = [
            _make_service_tool("gmail_list_messages", "google", "gmail", "gmail.read",
                               "GET", "/api/v1/gmail/messages"),
            _make_service_tool("gmail_list_messages", "google", "gmail", "gmail.read",
                               "GET", "/api/v1/gmail/messages/v2"),
        ]

        engine, _ = _make_test_engine(service_tools=service_tools)
        engine.tools = engine._load_base_tools()

        # Last one wins in metadata
        meta = engine._service_tool_metadata["gmail_list_messages"]
        assert meta["path"] == "/api/v1/gmail/messages/v2"


# ---------------------------------------------------------------------------
# Edge cases: HTTP error responses
# ---------------------------------------------------------------------------


class TestServiceToolHTTPErrors:
    def test_400_bad_request(self):
        """400 Bad Request returns a service error."""
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                400, json={"error": "Invalid parameters: 'max_results' must be positive"}
            )
        )
        client = ServiceClient.__new__(ServiceClient)
        client.api_key = "sk_test"
        client.base_url = "https://test.example.com"
        client._client = httpx.Client(
            base_url=client.base_url,
            transport=transport,
            headers={"Authorization": "Bearer sk_test"},
        )

        result = client.execute_tool(
            "gmail_list_messages",
            {"max_results": -1},
            {"method": "GET", "path": "/api/v1/gmail/messages"},
        )
        assert result["ok"] is False
        assert "400" in result["error"]
        assert "Invalid parameters" in result["error"]

    def test_500_error_with_non_json_body(self):
        """500 error with plain text body should still produce a useful error."""
        transport = httpx.MockTransport(
            lambda request: httpx.Response(500, text="Internal Server Error")
        )
        client = ServiceClient.__new__(ServiceClient)
        client.api_key = "sk_test"
        client.base_url = "https://test.example.com"
        client._client = httpx.Client(
            base_url=client.base_url,
            transport=transport,
            headers={"Authorization": "Bearer sk_test"},
        )

        result = client.execute_tool(
            "test_tool",
            {},
            {"method": "GET", "path": "/api/v1/test"},
        )
        assert result["ok"] is False
        assert "500" in result["error"]

    def test_response_with_invalid_json(self):
        """200 response with invalid JSON body should raise."""
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, text="not valid json {{}")
        )
        client = ServiceClient.__new__(ServiceClient)
        client.api_key = "sk_test"
        client.base_url = "https://test.example.com"
        client._client = httpx.Client(
            base_url=client.base_url,
            transport=transport,
            headers={"Authorization": "Bearer sk_test"},
        )

        with pytest.raises(Exception):
            client.execute_tool(
                "test_tool",
                {},
                {"method": "GET", "path": "/api/v1/test"},
            )


# ---------------------------------------------------------------------------
# Edge cases: network errors during discovery
# ---------------------------------------------------------------------------


class TestDiscoveryEdgeCases:
    def test_timeout_during_discovery(self):
        """Timeout during tool discovery should return empty list, not crash."""
        transport = httpx.MockTransport(
            lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("read timed out"))
        )
        client = ServiceClient.__new__(ServiceClient)
        client.api_key = "sk_test"
        client.base_url = "https://test.example.com"
        client._client = httpx.Client(
            base_url=client.base_url,
            transport=transport,
            headers={"Authorization": "Bearer sk_test"},
        )

        tools = client.discover_tools()
        assert tools == []

    def test_discovery_returns_non_json(self):
        """Discovery endpoint returning non-JSON should return empty list."""
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, text="<html>not json</html>")
        )
        client = ServiceClient.__new__(ServiceClient)
        client.api_key = "sk_test"
        client.base_url = "https://test.example.com"
        client._client = httpx.Client(
            base_url=client.base_url,
            transport=transport,
            headers={"Authorization": "Bearer sk_test"},
        )

        # response.json() will raise, but discover_tools catches it
        # Actually, the current code calls response.raise_for_status() then .json()
        # which may raise json.JSONDecodeError — check if it's caught
        tools = client.discover_tools()
        # Should not crash; returns [] or whatever data.get("tools", []) yields
        assert isinstance(tools, list)

    def test_discovery_returns_tools_with_missing_fields(self):
        """Discovery returns malformed tools (missing function.name)."""
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={
                "tools": [
                    {"type": "function", "function": {"description": "no name"}},
                    _make_service_tool("valid_tool", "google", "gmail", "gmail.read"),
                ],
            })
        )
        client = ServiceClient.__new__(ServiceClient)
        client.api_key = "sk_test"
        client.base_url = "https://test.example.com"
        client._client = httpx.Client(
            base_url=client.base_url,
            transport=transport,
            headers={"Authorization": "Bearer sk_test"},
        )

        tools = client.discover_tools()
        # Both tools returned (filtering is engine's job)
        assert len(tools) == 2

    def test_engine_handles_discovery_failure_gracefully(self):
        """If discovery fails, engine should still work with supypowers tools only."""
        from supyagent.core.engine import BaseAgentEngine

        config = AgentConfig(
            name="test",
            model=ModelConfig(provider="anthropic/claude-3-5-sonnet-20241022"),
            system_prompt="Test",
            service=ServiceConfig(enabled=True),
        )

        # Service client that fails on discover
        mock_client = MagicMock()
        mock_client.discover_tools.return_value = []  # Discovery failed, returns empty

        sp_tools = [_make_supypowers_tool("shell", "run_command")]

        class TestEngine(BaseAgentEngine):
            def _get_secrets(self):
                return {}

        with patch("supyagent.core.engine.discover_tools", return_value=sp_tools), \
             patch("supyagent.core.service.get_service_client", return_value=mock_client):
            engine = TestEngine(config)
            tools = engine._load_base_tools()

        tool_names = [t["function"]["name"] for t in tools]
        assert "shell__run_command" in tool_names
        assert len(engine._service_tool_metadata) == 0


# ---------------------------------------------------------------------------
# Edge cases: circuit breaker across tool types
# ---------------------------------------------------------------------------


class TestCircuitBreakerCrossType:
    def test_circuit_breaker_only_affects_specific_tool(self):
        """Circuit breaker for one service tool shouldn't affect others."""
        service_tools = [
            _make_service_tool("gmail_list", "google", "gmail", "gmail.read",
                               "GET", "/api/v1/gmail/messages"),
            _make_service_tool("slack_list", "slack", "slack.channels", "slack.channels.read",
                               "GET", "/api/v1/slack/channels"),
        ]

        engine, mock_client = _make_test_engine(
            service_tools=service_tools,
            execute_result={"ok": False, "error": "error"},
        )
        engine.tools = engine._load_base_tools()

        # Fail gmail 3 times
        gmail_call = MagicMock()
        gmail_call.function.name = "gmail_list"
        gmail_call.function.arguments = "{}"
        for _ in range(3):
            engine._dispatch_tool_call(gmail_call)

        # gmail should be circuit-broken
        result = engine._dispatch_tool_call(gmail_call)
        assert result["ok"] is False
        assert "circuit_breaker" in result.get("error_type", "")

        # slack should still work (even though it also fails)
        mock_client.execute_tool.return_value = {"ok": True, "data": {"channels": []}}
        slack_call = MagicMock()
        slack_call.function.name = "slack_list"
        slack_call.function.arguments = "{}"
        result = engine._dispatch_tool_call(slack_call)
        assert result["ok"] is True


# ---------------------------------------------------------------------------
# Edge cases: permission filtering with service tools
# ---------------------------------------------------------------------------


class TestPermissionEdgeCases:
    def test_deny_all_service_allow_specific_provider(self):
        """Deny all service, then explicitly allow a provider — deny wins."""
        tools = [
            _make_service_tool("gmail_list", "google", "gmail", "gmail.read"),
            _make_service_tool("slack_list", "slack", "slack.channels", "slack.read"),
        ]
        # Per the filter logic: deny checked first, then allow
        perms = ToolPermissions(allow=["service:google:*"], deny=["service:*"])
        result = filter_tools(tools, perms)
        assert len(result) == 0  # deny takes precedence

    def test_allow_specific_deny_specific(self):
        """Allow all service, deny only gmail — keep others."""
        tools = [
            _make_service_tool("gmail_list", "google", "gmail", "gmail.read"),
            _make_service_tool("slack_list", "slack", "slack.channels", "slack.read"),
            _make_service_tool("github_list", "github", "github.repos", "repos.read"),
        ]
        perms = ToolPermissions(allow=["service:*"], deny=["service:gmail"])
        result = filter_tools(tools, perms)
        names = [t["function"]["name"] for t in result]
        assert "gmail_list" not in names
        assert "slack_list" in names
        assert "github_list" in names

    def test_supypowers_pattern_uses_double_underscore(self):
        """Supypowers patterns like 'shell:*' match on script name (before __)."""
        # Supypowers tool: shell__run_command matches "shell:*"
        assert _matches_pattern("shell__run_command", "shell:*") is True
        # Service tool with same provider name matches via service: prefix
        meta = {"provider": "shell", "service": "shell"}
        assert _matches_pattern("shell_list", "service:shell:*", meta) is True
        # A tool without __ separator and without "service:" prefix
        # uses split(":")[0] for matching, so "shell_list" split by ":" is "shell_list" != "shell"
        assert _matches_pattern("shell_list", "shell:*", None) is False

    def test_empty_allow_and_deny_allows_everything(self):
        """No permissions specified means all tools pass through."""
        tools = [
            {  # OpenAI format supypowers tool (no metadata)
                "type": "function",
                "function": {
                    "name": "shell__run",
                    "description": "Run a command",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            _make_service_tool("gmail_list", "google", "gmail", "gmail.read"),
        ]
        perms = ToolPermissions()  # No allow, no deny
        result = filter_tools(tools, perms)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Edge cases: ServiceConfig in agent YAML
# ---------------------------------------------------------------------------


class TestServiceConfigEdgeCases:
    def test_default_service_config(self):
        """Default agent config has service enabled."""
        config = AgentConfig(
            name="test",
            model=ModelConfig(provider="test/model"),
            system_prompt="Test",
        )
        assert config.service.enabled is True
        assert config.service.url == DEFAULT_SERVICE_URL

    def test_service_disabled_skips_client_creation(self):
        """When service disabled, no service client is created even if credentials exist."""
        from supyagent.core.engine import BaseAgentEngine

        config = AgentConfig(
            name="test",
            model=ModelConfig(provider="test/model"),
            system_prompt="Test",
            service=ServiceConfig(enabled=False),
        )

        class TestEngine(BaseAgentEngine):
            def _get_secrets(self):
                return {}

        with patch("supyagent.core.engine.discover_tools", return_value=[]):
            engine = TestEngine(config)

        assert engine._service_client is None

    def test_service_custom_url_passed_through(self):
        """Custom service URL from config is used."""
        config = AgentConfig(
            name="test",
            model=ModelConfig(provider="test/model"),
            system_prompt="Test",
            service=ServiceConfig(enabled=True, url="https://custom.supyagent.com"),
        )
        assert config.service.url == "https://custom.supyagent.com"
