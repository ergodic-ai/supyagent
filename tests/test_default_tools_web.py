"""
Comprehensive tests for default_tools/web.py.

Covers: fetch_url, http_request
Uses httpx mocking to avoid real network calls.
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from supyagent.default_tools.web import (
    FetchUrlInput,
    HttpRequestInput,
    fetch_url,
    http_request,
    _html_to_markdown,
)


# =========================================================================
# _html_to_markdown helper
# =========================================================================


class TestHtmlToMarkdown:
    def test_basic_html(self):
        html = "<h1>Title</h1><p>Hello world</p>"
        result = _html_to_markdown(html)
        assert "Title" in result
        assert "Hello world" in result

    def test_strips_scripts(self):
        html = '<p>Text</p><script>alert("x")</script>'
        result = _html_to_markdown(html)
        assert "alert" not in result
        assert "Text" in result

    def test_strips_styles(self):
        html = "<style>body{color:red}</style><p>Visible</p>"
        result = _html_to_markdown(html)
        assert "color:red" not in result
        assert "Visible" in result

    def test_strips_nav_footer(self):
        html = "<nav>Menu</nav><main><p>Content</p></main><footer>Copyright</footer>"
        result = _html_to_markdown(html)
        assert "Menu" not in result
        assert "Copyright" not in result
        assert "Content" in result

    def test_preserves_links(self):
        html = '<p>Visit <a href="https://example.com">Example</a></p>'
        result = _html_to_markdown(html, include_links=True)
        assert "Example" in result
        assert "example.com" in result

    def test_preserves_code_blocks(self):
        html = "<pre><code>def main():\n    pass</code></pre>"
        result = _html_to_markdown(html)
        assert "def main" in result

    def test_collapses_blank_lines(self):
        html = "<p>A</p><br><br><br><br><p>B</p>"
        result = _html_to_markdown(html)
        # Should not have more than one consecutive blank line
        assert "\n\n\n" not in result

    def test_empty_html(self):
        result = _html_to_markdown("")
        assert result == ""

    def test_handles_tables(self):
        html = "<table><tr><th>Name</th><th>Age</th></tr><tr><td>Alice</td><td>30</td></tr></table>"
        result = _html_to_markdown(html)
        assert "Alice" in result
        assert "30" in result


# =========================================================================
# fetch_url
# =========================================================================


def _mock_response(
    status_code=200,
    text="<html><head><title>Test</title></head><body><p>Hello</p></body></html>",
    content_type="text/html; charset=utf-8",
    url="https://example.com",
    content=None,
):
    """Create a mock httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.content = content or text.encode()
    resp.headers = {"content-type": content_type}
    resp.url = url

    def json_method():
        return json.loads(text)

    resp.json = json_method
    return resp


class TestFetchUrl:
    @patch("supyagent.default_tools.web.httpx.Client")
    def test_fetch_html_page(self, mock_client_cls):
        client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response()

        result = fetch_url(FetchUrlInput(url="https://example.com"))
        assert result.ok is True
        assert "Hello" in result.content
        assert result.title == "Test"
        assert result.status_code == 200

    @patch("supyagent.default_tools.web.httpx.Client")
    def test_fetch_json_response(self, mock_client_cls):
        client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(
            text='{"key": "value"}',
            content_type="application/json",
        )

        result = fetch_url(FetchUrlInput(url="https://api.example.com/data"))
        assert result.ok is True
        assert '"key"' in result.content
        assert '"value"' in result.content

    @patch("supyagent.default_tools.web.httpx.Client")
    def test_fetch_plain_text(self, mock_client_cls):
        client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(
            text="Just plain text content",
            content_type="text/plain",
        )

        result = fetch_url(FetchUrlInput(url="https://example.com/robots.txt"))
        assert result.ok is True
        assert result.content == "Just plain text content"

    @patch("supyagent.default_tools.web.httpx.Client")
    def test_fetch_truncation(self, mock_client_cls):
        client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        long_html = "<html><body><p>" + "word " * 20000 + "</p></body></html>"
        client.get.return_value = _mock_response(text=long_html)

        result = fetch_url(FetchUrlInput(url="https://example.com", max_length=100))
        assert result.ok is True
        assert result.truncated is True
        assert "[truncated]" in result.content

    @patch("supyagent.default_tools.web.httpx.Client")
    def test_fetch_follows_redirects(self, mock_client_cls):
        client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(url="https://example.com/final")

        result = fetch_url(FetchUrlInput(url="https://example.com/redirect"))
        assert result.ok is True
        assert result.url == "https://example.com/final"

    @patch("supyagent.default_tools.web.httpx.Client")
    def test_fetch_timeout(self, mock_client_cls):
        import httpx

        client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.side_effect = httpx.TimeoutException("timeout")

        result = fetch_url(FetchUrlInput(url="https://slow.example.com", timeout=5))
        assert result.ok is False
        assert "timed out" in result.error.lower()

    @patch("supyagent.default_tools.web.httpx.Client")
    def test_fetch_connection_error(self, mock_client_cls):
        import httpx

        client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.side_effect = httpx.ConnectError("refused")

        result = fetch_url(FetchUrlInput(url="https://nonexistent.example.com"))
        assert result.ok is False
        assert "connection failed" in result.error.lower()

    @patch("supyagent.default_tools.web.httpx.Client")
    def test_fetch_binary_content(self, mock_client_cls):
        client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(
            text="",
            content_type="application/pdf",
            content=b"\x00\x01\x02",
        )

        result = fetch_url(FetchUrlInput(url="https://example.com/doc.pdf"))
        assert result.ok is True
        assert "Binary content" in result.content

    @patch("supyagent.default_tools.web.httpx.Client")
    def test_fetch_custom_headers(self, mock_client_cls):
        client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response()

        result = fetch_url(
            FetchUrlInput(
                url="https://example.com",
                headers={"Authorization": "Bearer token123"},
            )
        )
        assert result.ok is True

    @patch("supyagent.default_tools.web.httpx.Client")
    def test_fetch_returns_status_code(self, mock_client_cls):
        client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(status_code=404)

        result = fetch_url(FetchUrlInput(url="https://example.com/missing"))
        assert result.ok is True  # HTTP 404 is still a valid response
        assert result.status_code == 404


# =========================================================================
# http_request
# =========================================================================


class TestHttpRequest:
    @patch("supyagent.default_tools.web.httpx.Client")
    def test_simple_get(self, mock_client_cls):
        client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        resp = MagicMock()
        resp.status_code = 200
        resp.text = '{"result": "ok"}'
        resp.headers = {"content-type": "application/json"}
        resp.url = "https://api.example.com/data"
        client.request.return_value = resp

        result = http_request(HttpRequestInput(url="https://api.example.com/data"))
        assert result.ok is True
        assert result.status_code == 200
        assert "result" in result.body

    @patch("supyagent.default_tools.web.httpx.Client")
    def test_post_with_json(self, mock_client_cls):
        client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        resp = MagicMock()
        resp.status_code = 201
        resp.text = '{"id": 1}'
        resp.headers = {"content-type": "application/json"}
        resp.url = "https://api.example.com/items"
        client.request.return_value = resp

        result = http_request(
            HttpRequestInput(
                method="POST",
                url="https://api.example.com/items",
                json_body={"name": "test"},
            )
        )
        assert result.ok is True
        assert result.status_code == 201
        client.request.assert_called_once()
        call_kwargs = client.request.call_args
        assert call_kwargs[1]["json"] == {"name": "test"}

    @patch("supyagent.default_tools.web.httpx.Client")
    def test_post_with_body_string(self, mock_client_cls):
        client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        resp = MagicMock()
        resp.status_code = 200
        resp.text = "ok"
        resp.headers = {}
        resp.url = "https://example.com"
        client.request.return_value = resp

        result = http_request(
            HttpRequestInput(
                method="POST",
                url="https://example.com",
                body="raw body content",
            )
        )
        assert result.ok is True
        call_kwargs = client.request.call_args
        assert call_kwargs[1]["content"] == "raw body content"

    @patch("supyagent.default_tools.web.httpx.Client")
    def test_delete_method(self, mock_client_cls):
        client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        resp = MagicMock()
        resp.status_code = 204
        resp.text = ""
        resp.headers = {}
        resp.url = "https://api.example.com/items/1"
        client.request.return_value = resp

        result = http_request(
            HttpRequestInput(method="DELETE", url="https://api.example.com/items/1")
        )
        assert result.ok is True
        assert result.status_code == 204

    @patch("supyagent.default_tools.web.httpx.Client")
    def test_put_method(self, mock_client_cls):
        client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        resp = MagicMock()
        resp.status_code = 200
        resp.text = "{}"
        resp.headers = {}
        resp.url = "https://api.example.com/items/1"
        client.request.return_value = resp

        result = http_request(
            HttpRequestInput(
                method="PUT",
                url="https://api.example.com/items/1",
                json_body={"name": "updated"},
            )
        )
        assert result.ok is True
        client.request.assert_called_with("PUT", "https://api.example.com/items/1", headers={}, json={"name": "updated"})

    def test_invalid_method(self):
        result = http_request(
            HttpRequestInput(method="INVALID", url="https://example.com")
        )
        assert result.ok is False
        assert "Invalid HTTP method" in result.error

    @patch("supyagent.default_tools.web.httpx.Client")
    def test_custom_headers(self, mock_client_cls):
        client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        resp = MagicMock()
        resp.status_code = 200
        resp.text = "{}"
        resp.headers = {}
        resp.url = "https://api.example.com"
        client.request.return_value = resp

        result = http_request(
            HttpRequestInput(
                url="https://api.example.com",
                headers={"Authorization": "Bearer abc", "X-Custom": "value"},
            )
        )
        assert result.ok is True
        call_kwargs = client.request.call_args
        assert call_kwargs[1]["headers"]["Authorization"] == "Bearer abc"

    @patch("supyagent.default_tools.web.httpx.Client")
    def test_timeout_error(self, mock_client_cls):
        import httpx

        client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.request.side_effect = httpx.TimeoutException("timeout")

        result = http_request(
            HttpRequestInput(url="https://slow.example.com", timeout=1)
        )
        assert result.ok is False
        assert "timed out" in result.error.lower()

    @patch("supyagent.default_tools.web.httpx.Client")
    def test_connection_error(self, mock_client_cls):
        import httpx

        client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.request.side_effect = httpx.ConnectError("refused")

        result = http_request(HttpRequestInput(url="https://down.example.com"))
        assert result.ok is False
        assert "connection failed" in result.error.lower()

    @patch("supyagent.default_tools.web.httpx.Client")
    def test_response_headers_returned(self, mock_client_cls):
        client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        resp = MagicMock()
        resp.status_code = 200
        resp.text = "ok"
        resp.headers = {"x-request-id": "abc123", "content-type": "text/plain"}
        resp.url = "https://example.com"
        client.request.return_value = resp

        result = http_request(HttpRequestInput(url="https://example.com"))
        assert result.ok is True
        assert result.headers["x-request-id"] == "abc123"

    @patch("supyagent.default_tools.web.httpx.Client")
    def test_body_truncation(self, mock_client_cls):
        client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        resp = MagicMock()
        resp.status_code = 200
        resp.text = "x" * 200000
        resp.headers = {}
        resp.url = "https://example.com"
        client.request.return_value = resp

        result = http_request(HttpRequestInput(url="https://example.com"))
        assert result.ok is True
        assert len(result.body) < 200000
        assert "truncated" in result.body.lower()

    @patch("supyagent.default_tools.web.httpx.Client")
    def test_head_method(self, mock_client_cls):
        client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        resp = MagicMock()
        resp.status_code = 200
        resp.text = ""
        resp.headers = {"content-length": "1234"}
        resp.url = "https://example.com"
        client.request.return_value = resp

        result = http_request(
            HttpRequestInput(method="HEAD", url="https://example.com")
        )
        assert result.ok is True
        assert result.status_code == 200
