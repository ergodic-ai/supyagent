"""
Comprehensive tests for default_tools/browser.py.

Covers: browse, screenshot, click, type_text, get_page_state, close_browser
Uses mocked Playwright to avoid requiring actual browser installation.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from supyagent.default_tools.browser import (
    BrowseInput,
    ClickInput,
    CloseBrowserInput,
    GetPageStateInput,
    ScreenshotInput,
    TypeTextInput,
    _html_to_markdown,
    browse,
    click,
    close_browser,
    get_page_state,
    screenshot,
    type_text,
)


@pytest.fixture(autouse=True)
def reset_browser_state():
    """Reset browser module state before and after each test."""
    import supyagent.default_tools.browser as browser_mod
    browser_mod._browser = None
    browser_mod._context = None
    browser_mod._page = None
    yield
    browser_mod._browser = None
    browser_mod._context = None
    browser_mod._page = None


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


def _inject_mock_page(page_mock):
    """Inject a mock page into the browser module."""
    import supyagent.default_tools.browser as browser_mod
    browser_mod._page = page_mock
    browser_mod._browser = MagicMock()
    browser_mod._context = MagicMock()


# =========================================================================
# _html_to_markdown (browser's own copy)
# =========================================================================


class TestBrowserHtmlToMarkdown:
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
# browse
# =========================================================================


class TestBrowse:
    def test_browse_basic(self):
        page = _make_mock_page()
        _inject_mock_page(page)

        result = browse(BrowseInput(url="https://example.com"))
        assert result.ok is True
        assert result.title == "Test Page"
        assert result.url == "https://example.com"
        assert "Hello World" in result.content
        page.goto.assert_called_once()

    def test_browse_with_networkidle(self):
        page = _make_mock_page()
        _inject_mock_page(page)

        result = browse(BrowseInput(url="https://example.com", wait_for="networkidle"))
        assert result.ok is True
        page.goto.assert_called_with(
            "https://example.com", wait_until="networkidle", timeout=30000
        )

    def test_browse_with_css_selector(self):
        page = _make_mock_page()
        _inject_mock_page(page)

        result = browse(BrowseInput(url="https://example.com", wait_for="#main-content"))
        assert result.ok is True
        page.goto.assert_called_with(
            "https://example.com", wait_until="domcontentloaded", timeout=30000
        )
        page.wait_for_selector.assert_called_with("#main-content", timeout=30000)

    def test_browse_with_domcontentloaded(self):
        page = _make_mock_page()
        _inject_mock_page(page)

        result = browse(BrowseInput(url="https://example.com", wait_for="domcontentloaded"))
        assert result.ok is True

    def test_browse_truncation(self):
        long_content = "<html><body><p>" + "word " * 20000 + "</p></body></html>"
        page = _make_mock_page(content=long_content)
        _inject_mock_page(page)

        result = browse(BrowseInput(url="https://example.com", max_length=100))
        assert result.ok is True
        assert result.truncated is True
        assert "[truncated]" in result.content

    def test_browse_error_handling(self):
        page = _make_mock_page()
        _inject_mock_page(page)
        page.goto.side_effect = Exception("Navigation failed")

        result = browse(BrowseInput(url="https://example.com"))
        assert result.ok is False
        assert "Navigation failed" in result.error

    def test_browse_playwright_not_installed(self):
        page = _make_mock_page()
        _inject_mock_page(page)
        page.goto.side_effect = Exception("Executable doesn't exist at /path/chromium")

        result = browse(BrowseInput(url="https://example.com"))
        assert result.ok is False
        assert "playwright install" in result.error.lower()

    def test_browse_updates_url_after_redirect(self):
        page = _make_mock_page(url="https://example.com/final")
        _inject_mock_page(page)

        result = browse(BrowseInput(url="https://example.com/redirect"))
        assert result.ok is True
        assert result.url == "https://example.com/final"


# =========================================================================
# screenshot
# =========================================================================


class TestScreenshot:
    def test_screenshot_with_url(self, tmp_path):
        page = _make_mock_page()
        _inject_mock_page(page)

        output_path = str(tmp_path / "test.png")
        result = screenshot(
            ScreenshotInput(url="https://example.com", output_path=output_path)
        )
        assert result.ok is True
        assert result.width == 1280
        assert result.height == 720
        assert result.url == "https://example.com"
        page.goto.assert_called_once()
        page.screenshot.assert_called_once()

    def test_screenshot_current_page(self, tmp_path):
        page = _make_mock_page()
        _inject_mock_page(page)

        output_path = str(tmp_path / "current.png")
        result = screenshot(ScreenshotInput(output_path=output_path))
        assert result.ok is True
        page.goto.assert_not_called()  # No URL = screenshot current page
        page.screenshot.assert_called_once()

    def test_screenshot_full_page(self, tmp_path):
        page = _make_mock_page()
        _inject_mock_page(page)

        output_path = str(tmp_path / "full.png")
        result = screenshot(
            ScreenshotInput(
                url="https://example.com",
                output_path=output_path,
                full_page=True,
            )
        )
        assert result.ok is True
        page.screenshot.assert_called_with(path=output_path, full_page=True)

    def test_screenshot_creates_parent_dirs(self, tmp_path):
        page = _make_mock_page()
        _inject_mock_page(page)

        output_path = str(tmp_path / "sub" / "dir" / "test.png")
        result = screenshot(
            ScreenshotInput(url="https://example.com", output_path=output_path)
        )
        assert result.ok is True
        assert Path(output_path).parent.exists()

    def test_screenshot_error(self, tmp_path):
        page = _make_mock_page()
        _inject_mock_page(page)
        page.screenshot.side_effect = Exception("Screenshot failed")

        result = screenshot(
            ScreenshotInput(
                url="https://example.com",
                output_path=str(tmp_path / "fail.png"),
            )
        )
        assert result.ok is False
        assert "Screenshot failed" in result.error

    def test_screenshot_with_css_wait(self, tmp_path):
        page = _make_mock_page()
        _inject_mock_page(page)

        result = screenshot(
            ScreenshotInput(
                url="https://example.com",
                output_path=str(tmp_path / "test.png"),
                wait_for="#loaded",
            )
        )
        assert result.ok is True
        page.wait_for_selector.assert_called_with("#loaded", timeout=30000)


# =========================================================================
# click
# =========================================================================


class TestClick:
    def test_click_by_selector(self):
        page = _make_mock_page()
        _inject_mock_page(page)

        result = click(ClickInput(selector="#submit-btn"))
        assert result.ok is True
        assert result.url == "https://example.com"
        assert result.title == "Test Page"
        page.click.assert_called_with("#submit-btn", timeout=10000)

    def test_click_by_text(self):
        page = _make_mock_page()
        _inject_mock_page(page)

        result = click(ClickInput(selector="text=Login"))
        assert result.ok is True
        page.click.assert_called_with("text=Login", timeout=10000)

    def test_click_waits_after(self):
        page = _make_mock_page()
        _inject_mock_page(page)

        result = click(ClickInput(selector="#btn", wait_after=2000))
        assert result.ok is True
        page.wait_for_timeout.assert_called_with(2000)

    def test_click_element_not_found(self):
        page = _make_mock_page()
        _inject_mock_page(page)
        page.click.side_effect = Exception("Element not found: #nonexistent")

        result = click(ClickInput(selector="#nonexistent"))
        assert result.ok is False
        assert "not found" in result.error.lower()

    def test_click_with_custom_timeout(self):
        page = _make_mock_page()
        _inject_mock_page(page)

        result = click(ClickInput(selector="#btn", timeout=5000))
        assert result.ok is True
        page.click.assert_called_with("#btn", timeout=5000)

    def test_click_url_changes(self):
        page = _make_mock_page(url="https://example.com/new-page")
        _inject_mock_page(page)

        result = click(ClickInput(selector="a[href='/new-page']"))
        assert result.ok is True
        assert result.url == "https://example.com/new-page"


# =========================================================================
# type_text
# =========================================================================


class TestTypeText:
    def test_type_with_clear(self):
        page = _make_mock_page()
        _inject_mock_page(page)

        result = type_text(
            TypeTextInput(selector="#search", text="hello world", clear_first=True)
        )
        assert result.ok is True
        page.fill.assert_called_with("#search", "hello world", timeout=10000)

    def test_type_without_clear(self):
        page = _make_mock_page()
        _inject_mock_page(page)

        result = type_text(
            TypeTextInput(selector="#search", text="append", clear_first=False)
        )
        assert result.ok is True
        page.type.assert_called_with("#search", "append", timeout=10000)
        page.fill.assert_not_called()

    def test_type_with_enter(self):
        page = _make_mock_page()
        _inject_mock_page(page)

        result = type_text(
            TypeTextInput(selector="#search", text="query", press_enter=True)
        )
        assert result.ok is True
        page.press.assert_called_with("#search", "Enter")

    def test_type_without_enter(self):
        page = _make_mock_page()
        _inject_mock_page(page)

        result = type_text(
            TypeTextInput(selector="#search", text="query", press_enter=False)
        )
        assert result.ok is True
        page.press.assert_not_called()

    def test_type_element_not_found(self):
        page = _make_mock_page()
        _inject_mock_page(page)
        page.fill.side_effect = Exception("Element not found")

        result = type_text(TypeTextInput(selector="#nope", text="test"))
        assert result.ok is False

    def test_type_into_named_input(self):
        page = _make_mock_page()
        _inject_mock_page(page)

        result = type_text(
            TypeTextInput(selector="input[name=email]", text="user@example.com")
        )
        assert result.ok is True
        page.fill.assert_called_with("input[name=email]", "user@example.com", timeout=10000)


# =========================================================================
# get_page_state
# =========================================================================


class TestGetPageState:
    def test_basic_state(self):
        page = _make_mock_page()
        _inject_mock_page(page)

        result = get_page_state(GetPageStateInput())
        assert result.ok is True
        assert result.url == "https://example.com"
        assert result.title == "Test Page"
        assert result.text is not None
        assert "Hello World" in result.text

    def test_state_without_text(self):
        page = _make_mock_page()
        _inject_mock_page(page)

        result = get_page_state(GetPageStateInput(include_text=False))
        assert result.ok is True
        assert result.text is None

    def test_state_with_links(self):
        page = _make_mock_page()
        _inject_mock_page(page)
        page.evaluate.return_value = [
            {"text": "Example Link", "href": "https://example.com/page"},
            {"text": "Another", "href": "https://example.com/other"},
        ]

        result = get_page_state(
            GetPageStateInput(include_links=True, include_inputs=False, include_text=False)
        )
        assert result.ok is True
        assert result.links is not None
        assert len(result.links) == 2
        assert result.links[0].text == "Example Link"

    def test_state_with_inputs(self):
        page = _make_mock_page()
        _inject_mock_page(page)
        # First evaluate call returns links (empty), second returns inputs
        page.evaluate.side_effect = [
            [],  # links
            [
                {
                    "type": "text",
                    "name": "username",
                    "placeholder": "Enter username",
                    "value": "",
                    "selector": "input[name=\"username\"]",
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

        result = get_page_state(
            GetPageStateInput(include_links=True, include_inputs=True, include_text=False)
        )
        assert result.ok is True
        assert result.inputs is not None
        assert len(result.inputs) == 2
        assert result.inputs[0].type == "text"
        assert result.inputs[0].name == "username"
        assert result.inputs[1].type == "button"

    def test_state_without_links(self):
        page = _make_mock_page()
        _inject_mock_page(page)

        result = get_page_state(
            GetPageStateInput(include_links=False, include_inputs=False, include_text=False)
        )
        assert result.ok is True
        assert result.links is None
        assert result.inputs is None

    def test_state_text_truncation(self):
        long_content = "<html><body><p>" + "word " * 10000 + "</p></body></html>"
        page = _make_mock_page(content=long_content)
        _inject_mock_page(page)

        result = get_page_state(GetPageStateInput(max_length=100))
        assert result.ok is True
        assert "[truncated]" in result.text

    def test_state_error(self):
        page = _make_mock_page()
        _inject_mock_page(page)
        page.content.side_effect = Exception("Page crashed")

        result = get_page_state(GetPageStateInput())
        assert result.ok is False
        assert "Page crashed" in result.error


# =========================================================================
# close_browser
# =========================================================================


class TestCloseBrowser:
    def test_close_browser(self):
        import supyagent.default_tools.browser as browser_mod

        page = _make_mock_page()
        _inject_mock_page(page)

        result = close_browser(CloseBrowserInput())
        assert result.ok is True
        assert browser_mod._page is None
        assert browser_mod._browser is None
        assert browser_mod._context is None

    def test_close_browser_when_not_open(self):
        result = close_browser(CloseBrowserInput())
        assert result.ok is True  # Should not error

    def test_close_browser_twice(self):
        page = _make_mock_page()
        _inject_mock_page(page)

        result1 = close_browser(CloseBrowserInput())
        assert result1.ok is True

        result2 = close_browser(CloseBrowserInput())
        assert result2.ok is True


# =========================================================================
# Integration flows (mocked)
# =========================================================================


class TestBrowserFlows:
    def test_browse_then_click(self):
        """Navigate to a page, then click a button."""
        page = _make_mock_page()
        _inject_mock_page(page)

        result1 = browse(BrowseInput(url="https://example.com"))
        assert result1.ok is True

        # After click, URL changes
        page.url = "https://example.com/next"
        page.title.return_value = "Next Page"

        result2 = click(ClickInput(selector="#next-btn"))
        assert result2.ok is True
        assert result2.url == "https://example.com/next"

    def test_browse_type_and_submit(self):
        """Navigate, fill form, submit."""
        page = _make_mock_page()
        _inject_mock_page(page)

        browse(BrowseInput(url="https://example.com/login"))

        type_text(TypeTextInput(selector="#email", text="user@example.com"))
        type_text(TypeTextInput(selector="#password", text="secret123"))

        page.url = "https://example.com/dashboard"
        result = click(ClickInput(selector="#login-btn"))
        assert result.ok is True
        assert result.url == "https://example.com/dashboard"

    def test_browse_then_screenshot(self, tmp_path):
        """Navigate then take screenshot."""
        page = _make_mock_page()
        _inject_mock_page(page)

        browse(BrowseInput(url="https://example.com"))

        result = screenshot(ScreenshotInput(output_path=str(tmp_path / "shot.png")))
        assert result.ok is True
        assert result.url == "https://example.com"

    def test_get_state_then_interact(self):
        """Check page state, then interact based on findings."""
        page = _make_mock_page()
        _inject_mock_page(page)
        page.evaluate.side_effect = [
            [{"text": "Products", "href": "https://example.com/products"}],  # links
            [{"type": "text", "name": "search", "placeholder": "Search...", "value": "", "selector": "#search"}],  # inputs
        ]

        state = get_page_state(GetPageStateInput(include_text=False))
        assert state.ok is True
        assert len(state.links) == 1
        assert len(state.inputs) == 1

        # Now type into the search box found in state
        result = type_text(
            TypeTextInput(selector=state.inputs[0].selector, text="laptop", press_enter=True)
        )
        assert result.ok is True
