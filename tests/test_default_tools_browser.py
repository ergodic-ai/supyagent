"""
Comprehensive tests for browser daemon command dispatch.

Covers: browse, screenshot, click, type_text, get_page_state, close_browser
Uses mocked Playwright to avoid requiring actual browser installation.

The browser daemon (supyagent.core.browser_daemon) is the single source of
truth for all browser automation in supyagent.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from supyagent.core.browser_daemon import (
    BrowseInput,
    ClickInput,
    GetPageStateInput,
    ScreenshotInput,
    TypeTextInput,
    _dispatch_command,
    _html_to_markdown,
)


def _make_mock_page(
    url="https://example.com",
    title="Test Page",
    content="<html><body><p>Hello World</p></body></html>",
):
    """Create a mock Playwright page."""
    page = MagicMock()
    page.url = url
    page.title.return_value = title
    page.content.return_value = content
    page.is_closed.return_value = False
    page.viewport_size = {"width": 1280, "height": 720}
    page.evaluate.return_value = []
    return page


# =========================================================================
# _html_to_markdown
# =========================================================================


class TestHtmlToMarkdown:
    def test_basic_conversion(self):
        html = "<h1>Title</h1><p>Content here</p>"
        result = _html_to_markdown(html)
        assert "Title" in result
        assert "Content" in result

    def test_strips_scripts(self):
        html = '<p>Text</p><script>alert("x")</script>'
        result = _html_to_markdown(html)
        assert "alert" not in result

    def test_strips_nav_footer(self):
        html = "<nav>Nav</nav><p>Content</p><footer>Foot</footer>"
        result = _html_to_markdown(html)
        assert "Nav" not in result
        assert "Foot" not in result
        assert "Content" in result


# =========================================================================
# browse command
# =========================================================================


class TestBrowse:
    @patch("supyagent.core.browser_daemon._get_page")
    def test_browse_basic(self, mock_get_page):
        page = _make_mock_page()
        mock_get_page.return_value = page

        result = _dispatch_command("browse", {"url": "https://example.com"})
        assert result["ok"] is True
        assert result["data"]["title"] == "Test Page"
        assert result["data"]["url"] == "https://example.com"
        assert "Hello World" in result["data"]["content"]
        page.goto.assert_called_once()

    @patch("supyagent.core.browser_daemon._get_page")
    def test_browse_with_networkidle(self, mock_get_page):
        page = _make_mock_page()
        mock_get_page.return_value = page

        result = _dispatch_command(
            "browse", {"url": "https://example.com", "wait_for": "networkidle"}
        )
        assert result["ok"] is True
        page.goto.assert_called_with(
            "https://example.com", wait_until="networkidle", timeout=30000
        )

    @patch("supyagent.core.browser_daemon._get_page")
    def test_browse_with_css_selector(self, mock_get_page):
        page = _make_mock_page()
        mock_get_page.return_value = page

        result = _dispatch_command(
            "browse", {"url": "https://example.com", "wait_for": "#main-content"}
        )
        assert result["ok"] is True
        page.goto.assert_called_with(
            "https://example.com", wait_until="domcontentloaded", timeout=30000
        )
        page.wait_for_selector.assert_called_with("#main-content", timeout=30000)

    @patch("supyagent.core.browser_daemon._get_page")
    def test_browse_with_domcontentloaded(self, mock_get_page):
        page = _make_mock_page()
        mock_get_page.return_value = page

        result = _dispatch_command(
            "browse", {"url": "https://example.com", "wait_for": "domcontentloaded"}
        )
        assert result["ok"] is True

    @patch("supyagent.core.browser_daemon._get_page")
    def test_browse_truncation(self, mock_get_page):
        long_content = "<html><body><p>" + "word " * 20000 + "</p></body></html>"
        page = _make_mock_page(content=long_content)
        mock_get_page.return_value = page

        result = _dispatch_command(
            "browse", {"url": "https://example.com", "max_length": 100}
        )
        assert result["ok"] is True
        assert result["data"]["truncated"] is True
        assert "[truncated]" in result["data"]["content"]

    @patch("supyagent.core.browser_daemon._get_page")
    def test_browse_error_handling(self, mock_get_page):
        page = _make_mock_page()
        mock_get_page.return_value = page
        page.goto.side_effect = Exception("Navigation failed")

        result = _dispatch_command("browse", {"url": "https://example.com"})
        assert result["ok"] is False
        assert "Navigation failed" in result["error"]

    @patch("supyagent.core.browser_daemon._get_page")
    def test_browse_playwright_not_installed(self, mock_get_page):
        page = _make_mock_page()
        mock_get_page.return_value = page
        page.goto.side_effect = Exception("Executable doesn't exist at /path/chromium")

        result = _dispatch_command("browse", {"url": "https://example.com"})
        assert result["ok"] is False
        assert "playwright install" in result["error"].lower()

    @patch("supyagent.core.browser_daemon._get_page")
    def test_browse_updates_url_after_redirect(self, mock_get_page):
        page = _make_mock_page(url="https://example.com/final")
        mock_get_page.return_value = page

        result = _dispatch_command("browse", {"url": "https://example.com/redirect"})
        assert result["ok"] is True
        assert result["data"]["url"] == "https://example.com/final"


# =========================================================================
# screenshot command
# =========================================================================


class TestScreenshot:
    @patch("supyagent.core.browser_daemon._get_page")
    def test_screenshot_with_url(self, mock_get_page, tmp_path):
        page = _make_mock_page()
        mock_get_page.return_value = page

        output_path = str(tmp_path / "test.png")
        result = _dispatch_command(
            "screenshot", {"url": "https://example.com", "output_path": output_path}
        )
        assert result["ok"] is True
        assert result["data"]["width"] == 1280
        assert result["data"]["height"] == 720
        assert result["data"]["url"] == "https://example.com"
        page.goto.assert_called_once()
        page.screenshot.assert_called_once()

    @patch("supyagent.core.browser_daemon._get_page")
    def test_screenshot_current_page(self, mock_get_page, tmp_path):
        page = _make_mock_page()
        mock_get_page.return_value = page

        output_path = str(tmp_path / "current.png")
        result = _dispatch_command("screenshot", {"output_path": output_path})
        assert result["ok"] is True
        page.goto.assert_not_called()
        page.screenshot.assert_called_once()

    @patch("supyagent.core.browser_daemon._get_page")
    def test_screenshot_full_page(self, mock_get_page, tmp_path):
        page = _make_mock_page()
        mock_get_page.return_value = page

        output_path = str(tmp_path / "full.png")
        result = _dispatch_command(
            "screenshot",
            {"url": "https://example.com", "output_path": output_path, "full_page": True},
        )
        assert result["ok"] is True
        page.screenshot.assert_called_with(path=output_path, full_page=True)

    @patch("supyagent.core.browser_daemon._get_page")
    def test_screenshot_creates_parent_dirs(self, mock_get_page, tmp_path):
        page = _make_mock_page()
        mock_get_page.return_value = page

        output_path = str(tmp_path / "sub" / "dir" / "test.png")
        result = _dispatch_command(
            "screenshot", {"url": "https://example.com", "output_path": output_path}
        )
        assert result["ok"] is True
        assert Path(output_path).parent.exists()

    @patch("supyagent.core.browser_daemon._get_page")
    def test_screenshot_error(self, mock_get_page, tmp_path):
        page = _make_mock_page()
        mock_get_page.return_value = page
        page.screenshot.side_effect = Exception("Screenshot failed")

        result = _dispatch_command(
            "screenshot",
            {"url": "https://example.com", "output_path": str(tmp_path / "fail.png")},
        )
        assert result["ok"] is False
        assert "Screenshot failed" in result["error"]

    @patch("supyagent.core.browser_daemon._get_page")
    def test_screenshot_with_css_wait(self, mock_get_page, tmp_path):
        page = _make_mock_page()
        mock_get_page.return_value = page

        result = _dispatch_command(
            "screenshot",
            {
                "url": "https://example.com",
                "output_path": str(tmp_path / "test.png"),
                "wait_for": "#loaded",
            },
        )
        assert result["ok"] is True
        page.wait_for_selector.assert_called_with("#loaded", timeout=30000)


# =========================================================================
# click command
# =========================================================================


class TestClick:
    @patch("supyagent.core.browser_daemon._get_page")
    def test_click_by_selector(self, mock_get_page):
        page = _make_mock_page()
        mock_get_page.return_value = page

        result = _dispatch_command("click", {"selector": "#submit-btn"})
        assert result["ok"] is True
        assert result["data"]["url"] == "https://example.com"
        assert result["data"]["title"] == "Test Page"
        page.click.assert_called_with("#submit-btn", timeout=10000)

    @patch("supyagent.core.browser_daemon._get_page")
    def test_click_by_text(self, mock_get_page):
        page = _make_mock_page()
        mock_get_page.return_value = page

        result = _dispatch_command("click", {"selector": "text=Login"})
        assert result["ok"] is True
        page.click.assert_called_with("text=Login", timeout=10000)

    @patch("supyagent.core.browser_daemon._get_page")
    def test_click_waits_after(self, mock_get_page):
        page = _make_mock_page()
        mock_get_page.return_value = page

        result = _dispatch_command("click", {"selector": "#btn", "wait_after": 2000})
        assert result["ok"] is True
        page.wait_for_timeout.assert_called_with(2000)

    @patch("supyagent.core.browser_daemon._get_page")
    def test_click_element_not_found(self, mock_get_page):
        page = _make_mock_page()
        mock_get_page.return_value = page
        page.click.side_effect = Exception("Element not found: #nonexistent")

        result = _dispatch_command("click", {"selector": "#nonexistent"})
        assert result["ok"] is False
        assert "not found" in result["error"].lower()

    @patch("supyagent.core.browser_daemon._get_page")
    def test_click_with_custom_timeout(self, mock_get_page):
        page = _make_mock_page()
        mock_get_page.return_value = page

        result = _dispatch_command("click", {"selector": "#btn", "timeout": 5000})
        assert result["ok"] is True
        page.click.assert_called_with("#btn", timeout=5000)

    @patch("supyagent.core.browser_daemon._get_page")
    def test_click_url_changes(self, mock_get_page):
        page = _make_mock_page(url="https://example.com/new-page")
        mock_get_page.return_value = page

        result = _dispatch_command("click", {"selector": "a[href='/new-page']"})
        assert result["ok"] is True
        assert result["data"]["url"] == "https://example.com/new-page"


# =========================================================================
# type_text command
# =========================================================================


class TestTypeText:
    @patch("supyagent.core.browser_daemon._get_page")
    def test_type_with_clear(self, mock_get_page):
        page = _make_mock_page()
        mock_get_page.return_value = page

        result = _dispatch_command(
            "type_text", {"selector": "#search", "text": "hello world", "clear_first": True}
        )
        assert result["ok"] is True
        page.fill.assert_called_with("#search", "hello world", timeout=10000)

    @patch("supyagent.core.browser_daemon._get_page")
    def test_type_without_clear(self, mock_get_page):
        page = _make_mock_page()
        mock_get_page.return_value = page

        result = _dispatch_command(
            "type_text", {"selector": "#search", "text": "append", "clear_first": False}
        )
        assert result["ok"] is True
        page.type.assert_called_with("#search", "append", timeout=10000)
        page.fill.assert_not_called()

    @patch("supyagent.core.browser_daemon._get_page")
    def test_type_with_enter(self, mock_get_page):
        page = _make_mock_page()
        mock_get_page.return_value = page

        result = _dispatch_command(
            "type_text", {"selector": "#search", "text": "query", "press_enter": True}
        )
        assert result["ok"] is True
        page.press.assert_called_with("#search", "Enter")

    @patch("supyagent.core.browser_daemon._get_page")
    def test_type_without_enter(self, mock_get_page):
        page = _make_mock_page()
        mock_get_page.return_value = page

        result = _dispatch_command(
            "type_text", {"selector": "#search", "text": "query", "press_enter": False}
        )
        assert result["ok"] is True
        page.press.assert_not_called()

    @patch("supyagent.core.browser_daemon._get_page")
    def test_type_element_not_found(self, mock_get_page):
        page = _make_mock_page()
        mock_get_page.return_value = page
        page.fill.side_effect = Exception("Element not found")

        result = _dispatch_command("type_text", {"selector": "#nope", "text": "test"})
        assert result["ok"] is False

    @patch("supyagent.core.browser_daemon._get_page")
    def test_type_into_named_input(self, mock_get_page):
        page = _make_mock_page()
        mock_get_page.return_value = page

        result = _dispatch_command(
            "type_text", {"selector": "input[name=email]", "text": "user@example.com"}
        )
        assert result["ok"] is True
        page.fill.assert_called_with("input[name=email]", "user@example.com", timeout=10000)


# =========================================================================
# get_page_state command
# =========================================================================


class TestGetPageState:
    @patch("supyagent.core.browser_daemon._get_page")
    def test_basic_state(self, mock_get_page):
        page = _make_mock_page()
        mock_get_page.return_value = page

        result = _dispatch_command("get_page_state", {})
        assert result["ok"] is True
        assert result["data"]["url"] == "https://example.com"
        assert result["data"]["title"] == "Test Page"
        assert "Hello World" in result["data"]["text"]

    @patch("supyagent.core.browser_daemon._get_page")
    def test_state_without_text(self, mock_get_page):
        page = _make_mock_page()
        mock_get_page.return_value = page

        result = _dispatch_command("get_page_state", {"include_text": False})
        assert result["ok"] is True
        assert "text" not in result["data"]

    @patch("supyagent.core.browser_daemon._get_page")
    def test_state_with_links(self, mock_get_page):
        page = _make_mock_page()
        mock_get_page.return_value = page
        page.evaluate.return_value = [
            {"text": "Example Link", "href": "https://example.com/page"},
            {"text": "Another", "href": "https://example.com/other"},
        ]

        result = _dispatch_command(
            "get_page_state",
            {"include_links": True, "include_inputs": False, "include_text": False},
        )
        assert result["ok"] is True
        assert len(result["data"]["links"]) == 2
        assert result["data"]["links"][0]["text"] == "Example Link"

    @patch("supyagent.core.browser_daemon._get_page")
    def test_state_with_inputs(self, mock_get_page):
        page = _make_mock_page()
        mock_get_page.return_value = page
        page.evaluate.side_effect = [
            [],  # links
            [
                {
                    "type": "text",
                    "name": "username",
                    "placeholder": "Enter username",
                    "value": "",
                    "selector": 'input[name="username"]',
                },
                {
                    "type": "button",
                    "name": "submit",
                    "placeholder": None,
                    "value": "Login",
                    "selector": "#submit",
                },
            ],
        ]

        result = _dispatch_command(
            "get_page_state",
            {"include_links": True, "include_inputs": True, "include_text": False},
        )
        assert result["ok"] is True
        assert len(result["data"]["inputs"]) == 2
        assert result["data"]["inputs"][0]["type"] == "text"
        assert result["data"]["inputs"][0]["name"] == "username"
        assert result["data"]["inputs"][1]["type"] == "button"

    @patch("supyagent.core.browser_daemon._get_page")
    def test_state_without_links(self, mock_get_page):
        page = _make_mock_page()
        mock_get_page.return_value = page

        result = _dispatch_command(
            "get_page_state",
            {"include_links": False, "include_inputs": False, "include_text": False},
        )
        assert result["ok"] is True
        assert "links" not in result["data"]
        assert "inputs" not in result["data"]

    @patch("supyagent.core.browser_daemon._get_page")
    def test_state_text_truncation(self, mock_get_page):
        long_content = "<html><body><p>" + "word " * 10000 + "</p></body></html>"
        page = _make_mock_page(content=long_content)
        mock_get_page.return_value = page

        result = _dispatch_command("get_page_state", {"max_length": 100})
        assert result["ok"] is True
        assert "[truncated]" in result["data"]["text"]

    @patch("supyagent.core.browser_daemon._get_page")
    def test_state_error(self, mock_get_page):
        page = _make_mock_page()
        mock_get_page.return_value = page
        page.content.side_effect = Exception("Page crashed")

        result = _dispatch_command("get_page_state", {})
        assert result["ok"] is False
        assert "Page crashed" in result["error"]


# =========================================================================
# close_browser command
# =========================================================================


class TestCloseBrowser:
    @patch("supyagent.core.browser_daemon._close_browser")
    def test_close_browser(self, mock_close):
        result = _dispatch_command("close_browser", {})
        assert result["ok"] is True
        mock_close.assert_called_once()

    @patch("supyagent.core.browser_daemon._close_browser")
    def test_close_browser_twice(self, mock_close):
        result1 = _dispatch_command("close_browser", {})
        assert result1["ok"] is True

        result2 = _dispatch_command("close_browser", {})
        assert result2["ok"] is True


# =========================================================================
# Integration flows (mocked via daemon dispatch)
# =========================================================================


class TestBrowserFlows:
    @patch("supyagent.core.browser_daemon._get_page")
    def test_browse_then_click(self, mock_get_page):
        """Navigate to a page, then click a button."""
        page = _make_mock_page()
        mock_get_page.return_value = page

        result1 = _dispatch_command("browse", {"url": "https://example.com"})
        assert result1["ok"] is True

        # After click, URL changes
        page.url = "https://example.com/next"
        page.title.return_value = "Next Page"

        result2 = _dispatch_command("click", {"selector": "#next-btn"})
        assert result2["ok"] is True
        assert result2["data"]["url"] == "https://example.com/next"

    @patch("supyagent.core.browser_daemon._get_page")
    def test_browse_type_and_submit(self, mock_get_page):
        """Navigate, fill form, submit."""
        page = _make_mock_page()
        mock_get_page.return_value = page

        _dispatch_command("browse", {"url": "https://example.com/login"})
        _dispatch_command("type_text", {"selector": "#email", "text": "user@example.com"})
        _dispatch_command("type_text", {"selector": "#password", "text": "secret123"})

        page.url = "https://example.com/dashboard"
        result = _dispatch_command("click", {"selector": "#login-btn"})
        assert result["ok"] is True
        assert result["data"]["url"] == "https://example.com/dashboard"

    @patch("supyagent.core.browser_daemon._get_page")
    def test_browse_then_screenshot(self, mock_get_page, tmp_path):
        """Navigate then take screenshot."""
        page = _make_mock_page()
        mock_get_page.return_value = page

        _dispatch_command("browse", {"url": "https://example.com"})

        result = _dispatch_command(
            "screenshot", {"output_path": str(tmp_path / "shot.png")}
        )
        assert result["ok"] is True
        assert result["data"]["url"] == "https://example.com"

    @patch("supyagent.core.browser_daemon._get_page")
    def test_get_state_then_interact(self, mock_get_page):
        """Check page state, then interact based on findings."""
        page = _make_mock_page()
        mock_get_page.return_value = page
        page.evaluate.side_effect = [
            [{"text": "Products", "href": "https://example.com/products"}],
            [
                {
                    "type": "text",
                    "name": "search",
                    "placeholder": "Search...",
                    "value": "",
                    "selector": "#search",
                }
            ],
        ]

        state = _dispatch_command(
            "get_page_state", {"include_text": False}
        )
        assert state["ok"] is True
        assert len(state["data"]["links"]) == 1
        assert len(state["data"]["inputs"]) == 1

        # Type into the search box found in state
        result = _dispatch_command(
            "type_text",
            {
                "selector": state["data"]["inputs"][0]["selector"],
                "text": "laptop",
                "press_enter": True,
            },
        )
        assert result["ok"] is True


# =========================================================================
# Input model validation
# =========================================================================


class TestInputModels:
    def test_browse_input_defaults(self):
        inp = BrowseInput(url="https://example.com")
        assert inp.wait_for == "networkidle"
        assert inp.timeout == 30000
        assert inp.max_length == 50000

    def test_screenshot_input_defaults(self):
        inp = ScreenshotInput()
        assert inp.url is None
        assert inp.output_path == "screenshot.png"
        assert inp.full_page is False

    def test_click_input_defaults(self):
        inp = ClickInput(selector="#btn")
        assert inp.wait_after == 1000
        assert inp.timeout == 10000

    def test_type_text_input_defaults(self):
        inp = TypeTextInput(selector="#input", text="hello")
        assert inp.clear_first is True
        assert inp.press_enter is False
        assert inp.timeout == 10000

    def test_get_page_state_input_defaults(self):
        inp = GetPageStateInput()
        assert inp.include_links is True
        assert inp.include_inputs is True
        assert inp.include_text is True
        assert inp.max_length == 20000
