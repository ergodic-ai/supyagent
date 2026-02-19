"""
Tests for supyagent.core.service — ServiceClient, device auth, credential helpers.
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
    get_service_client,
    poll_for_token,
    request_device_code,
    store_service_credentials,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config_manager(tmp_path):
    """Provide a ConfigManager backed by a temp directory."""
    from supyagent.core.config import ConfigManager

    mgr = ConfigManager(base_dir=tmp_path / "config")
    with patch("supyagent.core.service.get_config_manager", return_value=mgr):
        yield mgr


@pytest.fixture
def mock_transport():
    """Create a mock httpx transport for intercepting requests."""
    return httpx.MockTransport(lambda req: httpx.Response(200, json={}))


# ---------------------------------------------------------------------------
# ServiceClient.__init__
# ---------------------------------------------------------------------------


class TestServiceClientInit:
    def test_uses_explicit_params(self, mock_config_manager):
        client = ServiceClient(api_key="sk_test_123", base_url="https://custom.example.com")
        assert client.api_key == "sk_test_123"
        assert client.base_url == "https://custom.example.com"
        client.close()

    def test_reads_from_config_manager(self, mock_config_manager):
        mock_config_manager.set(SERVICE_API_KEY, "sk_from_config")
        mock_config_manager.set(SERVICE_URL, "https://stored.example.com")

        client = ServiceClient()
        assert client.api_key == "sk_from_config"
        assert client.base_url == "https://stored.example.com"
        client.close()

    def test_defaults_to_production_url(self, mock_config_manager):
        client = ServiceClient(api_key="sk_test")
        assert client.base_url == DEFAULT_SERVICE_URL
        client.close()

    def test_strips_trailing_slash(self, mock_config_manager):
        client = ServiceClient(api_key="sk_test", base_url="https://example.com/")
        assert client.base_url == "https://example.com"
        client.close()

    def test_is_connected_true_with_key(self, mock_config_manager):
        client = ServiceClient(api_key="sk_test")
        assert client.is_connected is True
        client.close()

    def test_is_connected_false_without_key(self, mock_config_manager):
        client = ServiceClient()
        assert client.is_connected is False
        client.close()


# ---------------------------------------------------------------------------
# ServiceClient.discover_tools
# ---------------------------------------------------------------------------


class TestDiscoverTools:
    def test_returns_tools_on_success(self, mock_config_manager):
        tools_response = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "gmail_list_messages",
                        "description": "List Gmail messages",
                        "parameters": {"type": "object", "properties": {}},
                    },
                    "metadata": {
                        "provider": "google",
                        "service": "gmail",
                        "permission": "gmail.read",
                        "method": "GET",
                        "path": "/api/v1/gmail/messages",
                    },
                }
            ],
            "base_url": "https://app.supyagent.com",
            "total": 1,
        }

        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, json=tools_response)
        )
        client = ServiceClient(api_key="sk_test", base_url="https://test.local")
        client._client = httpx.Client(
            base_url="https://test.local",
            transport=transport,
            headers=client._headers(),
        )

        tools = client.discover_tools()
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "gmail_list_messages"
        client.close()

    def test_returns_empty_without_api_key(self, mock_config_manager):
        client = ServiceClient()
        assert client.discover_tools() == []
        client.close()

    def test_returns_empty_on_401(self, mock_config_manager):
        transport = httpx.MockTransport(
            lambda req: httpx.Response(401, json={"error": "Invalid API key"})
        )
        client = ServiceClient(api_key="sk_bad", base_url="https://test.local")
        client._client = httpx.Client(
            base_url="https://test.local",
            transport=transport,
            headers=client._headers(),
        )

        tools = client.discover_tools()
        assert tools == []
        client.close()

    def test_returns_empty_on_network_error(self, mock_config_manager):
        def raise_error(req):
            raise httpx.ConnectError("Connection refused")

        transport = httpx.MockTransport(raise_error)
        client = ServiceClient(api_key="sk_test", base_url="https://test.local")
        client._client = httpx.Client(
            base_url="https://test.local",
            transport=transport,
            headers=client._headers(),
        )

        tools = client.discover_tools()
        assert tools == []
        client.close()


# ---------------------------------------------------------------------------
# ServiceClient.execute_tool
# ---------------------------------------------------------------------------


class TestExecuteTool:
    def _make_client(self, mock_config_manager, handler):
        transport = httpx.MockTransport(handler)
        client = ServiceClient(api_key="sk_test", base_url="https://test.local")
        client._client = httpx.Client(
            base_url="https://test.local",
            transport=transport,
            headers=client._headers(),
        )
        return client

    def test_get_request(self, mock_config_manager):
        def handler(req):
            assert req.method == "GET"
            assert "/api/v1/gmail/messages" in str(req.url)
            return httpx.Response(200, json={"messages": [{"id": "1"}]})

        client = self._make_client(mock_config_manager, handler)
        result = client.execute_tool(
            "gmail_list_messages",
            {"max_results": 10},
            {"method": "GET", "path": "/api/v1/gmail/messages"},
        )
        assert result["ok"] is True
        assert "messages" in result["data"]
        client.close()

    def test_post_request(self, mock_config_manager):
        def handler(req):
            assert req.method == "POST"
            body = json.loads(req.content)
            assert body["to"] == "user@example.com"
            return httpx.Response(200, json={"id": "msg_123"})

        client = self._make_client(mock_config_manager, handler)
        result = client.execute_tool(
            "gmail_send_message",
            {"to": "user@example.com", "subject": "Hi", "body": "Hello"},
            {"method": "POST", "path": "/api/v1/gmail/messages/send"},
        )
        assert result["ok"] is True
        client.close()

    def test_patch_request(self, mock_config_manager):
        def handler(req):
            assert req.method == "PATCH"
            return httpx.Response(200, json={"updated": True})

        client = self._make_client(mock_config_manager, handler)
        result = client.execute_tool(
            "notion_update_page",
            {"page_id": "abc", "title": "New Title"},
            {"method": "PATCH", "path": "/api/v1/notion/pages"},
        )
        assert result["ok"] is True
        client.close()

    def test_delete_request(self, mock_config_manager):
        def handler(req):
            assert req.method == "DELETE"
            return httpx.Response(200, json={"deleted": True})

        client = self._make_client(mock_config_manager, handler)
        result = client.execute_tool(
            "github_delete_repo",
            {"repo": "test"},
            {"method": "DELETE", "path": "/api/v1/github/repos"},
        )
        assert result["ok"] is True
        client.close()

    def test_no_api_key(self, mock_config_manager):
        client = ServiceClient()
        result = client.execute_tool("test", {}, {"method": "GET", "path": "/test"})
        assert result["ok"] is False
        assert "connect" in result["error"].lower()
        client.close()

    def test_no_path(self, mock_config_manager):
        client = ServiceClient(api_key="sk_test")
        result = client.execute_tool("test", {}, {"method": "GET", "path": ""})
        assert result["ok"] is False
        assert "path" in result["error"].lower()
        client.close()

    def test_unsupported_method(self, mock_config_manager):
        client = ServiceClient(api_key="sk_test")
        result = client.execute_tool("test", {}, {"method": "OPTIONS", "path": "/test"})
        assert result["ok"] is False
        assert "Unsupported" in result["error"]
        client.close()

    def test_401_error(self, mock_config_manager):
        client = self._make_client(
            mock_config_manager,
            lambda req: httpx.Response(401, json={"error": "Unauthorized"}),
        )
        result = client.execute_tool(
            "test", {}, {"method": "GET", "path": "/test"}
        )
        assert result["ok"] is False
        assert "expired" in result["error"].lower()
        client.close()

    def test_403_error(self, mock_config_manager):
        client = self._make_client(
            mock_config_manager,
            lambda req: httpx.Response(403, json={"error": "Forbidden"}),
        )
        result = client.execute_tool(
            "test", {}, {"method": "GET", "path": "/test"}
        )
        assert result["ok"] is False
        assert "Permission denied" in result["error"]
        client.close()

    def test_429_error(self, mock_config_manager):
        client = self._make_client(
            mock_config_manager,
            lambda req: httpx.Response(
                429, json={"error": "Rate limit exceeded"}
            ),
        )
        result = client.execute_tool(
            "test", {}, {"method": "GET", "path": "/test"}
        )
        assert result["ok"] is False
        assert "Rate limit" in result["error"]
        client.close()

    def test_500_error(self, mock_config_manager):
        client = self._make_client(
            mock_config_manager,
            lambda req: httpx.Response(500, json={"error": "Internal error"}),
        )
        result = client.execute_tool(
            "test", {}, {"method": "GET", "path": "/test"}
        )
        assert result["ok"] is False
        assert "500" in result["error"]
        client.close()

    def test_network_error(self, mock_config_manager):
        def raise_error(req):
            raise httpx.ConnectError("Connection refused")

        client = self._make_client(mock_config_manager, raise_error)
        result = client.execute_tool(
            "test", {}, {"method": "GET", "path": "/test"}
        )
        assert result["ok"] is False
        assert "reach service" in result["error"].lower()
        client.close()

    def test_post_merges_body_defaults(self, mock_config_manager):
        """bodyDefaults from metadata are merged into the POST body."""
        captured_body = {}

        def handler(req):
            captured_body.update(json.loads(req.content))
            return httpx.Response(200, json={"ok": True})

        client = self._make_client(mock_config_manager, handler)
        result = client.execute_tool(
            "slides_replace_all_text",
            {"presentationId": "pres-1", "replaceText": "Acme"},
            {
                "method": "POST",
                "path": "/api/v1/slides/presentations/{presentationId}",
                "bodyDefaults": {"action": "replace_all_text"},
            },
        )
        assert result["ok"] is True
        assert captured_body["action"] == "replace_all_text"
        assert captured_body["replaceText"] == "Acme"
        # presentationId consumed by path substitution, should not be in body
        assert "presentationId" not in captured_body
        client.close()

    def test_post_explicit_args_override_body_defaults(self, mock_config_manager):
        """Explicit arguments override bodyDefaults."""
        captured_body = {}

        def handler(req):
            captured_body.update(json.loads(req.content))
            return httpx.Response(200, json={"ok": True})

        client = self._make_client(mock_config_manager, handler)
        result = client.execute_tool(
            "test_tool",
            {"action": "custom_action", "data": "value"},
            {
                "method": "POST",
                "path": "/test",
                "bodyDefaults": {"action": "default_action"},
            },
        )
        assert result["ok"] is True
        assert captured_body["action"] == "custom_action"
        client.close()

    def test_binary_response_saves_file(self, mock_config_manager, tmp_path):
        """Non-JSON responses are saved to disk and return filePath."""
        png_bytes = b"\x89PNG\r\n\x1a\nfake-png-content"

        def handler(req):
            return httpx.Response(
                200,
                content=png_bytes,
                headers={"content-type": "image/png"},
            )

        client = self._make_client(mock_config_manager, handler)

        with patch("supyagent.core.service.Path.home", return_value=tmp_path):
            result = client.execute_tool(
                "slides_get_thumbnail",
                {},
                {"method": "GET", "path": "/test/thumbnail"},
            )

        assert result["ok"] is True
        assert result["data"]["contentType"] == "image/png"
        assert result["data"]["size"] == len(png_bytes)

        from pathlib import Path
        saved = Path(result["data"]["filePath"])
        assert saved.exists()
        assert saved.read_bytes() == png_bytes
        assert saved.suffix == ".png"
        client.close()


# ---------------------------------------------------------------------------
# ServiceClient.health_check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    def test_healthy(self, mock_config_manager):
        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, json={"status": "ok"})
        )
        client = ServiceClient(api_key="sk_test", base_url="https://test.local")
        client._client = httpx.Client(
            base_url="https://test.local",
            transport=transport,
            headers=client._headers(),
        )
        assert client.health_check() is True
        client.close()

    def test_unhealthy(self, mock_config_manager):
        transport = httpx.MockTransport(
            lambda req: httpx.Response(503, text="Service Unavailable")
        )
        client = ServiceClient(api_key="sk_test", base_url="https://test.local")
        client._client = httpx.Client(
            base_url="https://test.local",
            transport=transport,
            headers=client._headers(),
        )
        assert client.health_check() is False
        client.close()

    def test_unreachable(self, mock_config_manager):
        def raise_error(req):
            raise httpx.ConnectError("Connection refused")

        transport = httpx.MockTransport(raise_error)
        client = ServiceClient(api_key="sk_test", base_url="https://test.local")
        client._client = httpx.Client(
            base_url="https://test.local",
            transport=transport,
            headers=client._headers(),
        )
        assert client.health_check() is False
        client.close()


# ---------------------------------------------------------------------------
# Device auth: request_device_code
# ---------------------------------------------------------------------------


class TestRequestDeviceCode:
    def test_success(self):
        response_data = {
            "device_code": "dev_abc123",
            "user_code": "ABCD-1234",
            "verification_uri": "https://app.supyagent.com/device",
            "expires_in": 900,
            "interval": 5,
        }

        with patch("supyagent.core.service.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_cls.return_value.__exit__ = MagicMock(return_value=False)

            mock_response = MagicMock()
            mock_response.json.return_value = response_data
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response

            result = request_device_code("https://test.local")

            assert result["device_code"] == "dev_abc123"
            assert result["user_code"] == "ABCD-1234"
            mock_client.post.assert_called_once_with(
                "https://test.local/api/v1/auth/device/code"
            )

    def test_uses_default_url(self):
        with patch("supyagent.core.service.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_cls.return_value.__exit__ = MagicMock(return_value=False)

            mock_response = MagicMock()
            mock_response.json.return_value = {"device_code": "x", "user_code": "Y"}
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response

            request_device_code()

            mock_client.post.assert_called_once_with(
                f"{DEFAULT_SERVICE_URL}/api/v1/auth/device/code"
            )


# ---------------------------------------------------------------------------
# Device auth: poll_for_token
# ---------------------------------------------------------------------------


class TestPollForToken:
    def test_approved_on_first_poll(self):
        with patch("supyagent.core.service.httpx.Client") as mock_cls, \
             patch("supyagent.core.service.time.sleep"):
            mock_client = MagicMock()
            mock_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_cls.return_value.__exit__ = MagicMock(return_value=False)

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"api_key": "sk_live_new_key"}
            mock_client.post.return_value = mock_response

            key = poll_for_token(
                base_url="https://test.local",
                device_code="dev_123",
                interval=1,
                expires_in=60,
            )
            assert key == "sk_live_new_key"

    def test_pending_then_approved(self):
        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            if call_count < 3:
                resp.status_code = 428
                resp.json.return_value = {"error": "authorization_pending"}
            else:
                resp.status_code = 200
                resp.json.return_value = {"api_key": "sk_live_delayed"}
            return resp

        with patch("supyagent.core.service.httpx.Client") as mock_cls, \
             patch("supyagent.core.service.time.sleep"):
            mock_client = MagicMock()
            mock_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = mock_post

            key = poll_for_token(
                base_url="https://test.local",
                device_code="dev_123",
                interval=1,
                expires_in=60,
            )
            assert key == "sk_live_delayed"
            assert call_count == 3

    def test_denied(self):
        with patch("supyagent.core.service.httpx.Client") as mock_cls, \
             patch("supyagent.core.service.time.sleep"):
            mock_client = MagicMock()
            mock_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_cls.return_value.__exit__ = MagicMock(return_value=False)

            mock_response = MagicMock()
            mock_response.status_code = 403
            mock_client.post.return_value = mock_response

            with pytest.raises(PermissionError, match="denied"):
                poll_for_token(
                    base_url="https://test.local",
                    device_code="dev_123",
                    interval=1,
                    expires_in=60,
                )

    def test_expired(self):
        with patch("supyagent.core.service.httpx.Client") as mock_cls, \
             patch("supyagent.core.service.time.sleep"):
            mock_client = MagicMock()
            mock_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_cls.return_value.__exit__ = MagicMock(return_value=False)

            mock_response = MagicMock()
            mock_response.status_code = 410
            mock_client.post.return_value = mock_response

            with pytest.raises(TimeoutError, match="expired"):
                poll_for_token(
                    base_url="https://test.local",
                    device_code="dev_123",
                    interval=1,
                    expires_in=60,
                )

    def test_unexpected_error(self):
        with patch("supyagent.core.service.httpx.Client") as mock_cls, \
             patch("supyagent.core.service.time.sleep"):
            mock_client = MagicMock()
            mock_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_cls.return_value.__exit__ = MagicMock(return_value=False)

            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.json.return_value = {"error": "Server error"}
            mock_client.post.return_value = mock_response

            with pytest.raises(RuntimeError, match="Unexpected"):
                poll_for_token(
                    base_url="https://test.local",
                    device_code="dev_123",
                    interval=1,
                    expires_in=60,
                )

    def test_timeout_when_always_pending(self):
        """If the deadline passes while still pending, raises TimeoutError."""
        with patch("supyagent.core.service.httpx.Client") as mock_cls, \
             patch("supyagent.core.service.time.sleep"), \
             patch("supyagent.core.service.time.time") as mock_time:
            mock_client = MagicMock()
            mock_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_cls.return_value.__exit__ = MagicMock(return_value=False)

            mock_response = MagicMock()
            mock_response.status_code = 428
            mock_client.post.return_value = mock_response

            # First call to time.time() sets the deadline, subsequent calls exceed it
            mock_time.side_effect = [100, 200]  # start=100, deadline=100+5=105, check=200

            with pytest.raises(TimeoutError, match="expired"):
                poll_for_token(
                    base_url="https://test.local",
                    device_code="dev_123",
                    interval=1,
                    expires_in=5,
                )


# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------


class TestCredentialHelpers:
    def test_store_service_credentials(self, mock_config_manager):
        store_service_credentials("sk_live_test123", "https://custom.example.com")

        assert mock_config_manager.get(SERVICE_API_KEY) == "sk_live_test123"
        assert mock_config_manager.get(SERVICE_URL) == "https://custom.example.com"

    def test_store_does_not_save_default_url(self, mock_config_manager):
        store_service_credentials("sk_live_test123")

        assert mock_config_manager.get(SERVICE_API_KEY) == "sk_live_test123"
        assert mock_config_manager.get(SERVICE_URL) is None

    def test_clear_service_credentials(self, mock_config_manager):
        mock_config_manager.set(SERVICE_API_KEY, "sk_live_test123")
        mock_config_manager.set(SERVICE_URL, "https://custom.example.com")

        removed = clear_service_credentials()
        assert removed is True
        assert mock_config_manager.get(SERVICE_API_KEY) is None
        assert mock_config_manager.get(SERVICE_URL) is None

    def test_clear_when_not_connected(self, mock_config_manager):
        removed = clear_service_credentials()
        assert removed is False

    def test_get_service_client_when_connected(self, mock_config_manager):
        mock_config_manager.set(SERVICE_API_KEY, "sk_live_test123")

        client = get_service_client()
        assert client is not None
        assert client.api_key == "sk_live_test123"
        client.close()

    def test_get_service_client_when_not_connected(self, mock_config_manager):
        client = get_service_client()
        assert client is None
