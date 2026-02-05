# /// script
# dependencies = ["pydantic", "playwright", "markdownify", "beautifulsoup4"]
# ///
"""
Browser automation tools powered by Playwright.

Provides headless Chromium browsing for JavaScript-rendered pages,
screenshots, and interactive web page manipulation. For simple static
pages, use the web.fetch_url tool instead — it's faster and lighter.

IMPORTANT: Requires Playwright browsers to be installed:
    playwright install chromium
"""

import json
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from bs4 import BeautifulSoup
from markdownify import markdownify as md
from pydantic import BaseModel, Field


# Tags to remove from HTML before converting to markdown
_REMOVE_TAGS = [
    "script",
    "style",
    "nav",
    "footer",
    "header",
    "aside",
    "noscript",
    "iframe",
    "svg",
]


def _html_to_markdown(html: str) -> str:
    """Convert HTML to clean markdown text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in _REMOVE_TAGS:
        for el in soup.find_all(tag):
            el.decompose()

    text = md(str(soup))

    # Clean up excessive whitespace
    lines = text.splitlines()
    cleaned = []
    prev_blank = False
    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            if not prev_blank:
                cleaned.append("")
                prev_blank = True
        else:
            cleaned.append(stripped)
            prev_blank = False

    return "\n".join(cleaned).strip()


# =============================================================================
# Browser session management
# =============================================================================

# Module-level browser session — reused across tool calls within the same process
_browser = None
_context = None
_page = None


def _get_page():
    """Get or create a shared browser page."""
    global _browser, _context, _page

    if _page is not None and not _page.is_closed():
        return _page

    from playwright.sync_api import sync_playwright

    pw = sync_playwright().start()
    try:
        _browser = pw.chromium.launch(headless=True)
    except Exception as e:
        if "executable doesn't exist" in str(e).lower():
            # Auto-install Chromium on first use
            import subprocess
            subprocess.run(
                ["playwright", "install", "chromium"],
                check=True,
                capture_output=True,
            )
            _browser = pw.chromium.launch(headless=True)
        else:
            raise
    _context = _browser.new_context(
        viewport={"width": 1280, "height": 720},
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    )
    _page = _context.new_page()
    return _page


def _close_browser():
    """Close the browser session."""
    global _browser, _context, _page
    try:
        if _page and not _page.is_closed():
            _page.close()
        if _context:
            _context.close()
        if _browser:
            _browser.close()
    except Exception:
        pass
    _browser = None
    _context = None
    _page = None


# =============================================================================
# Browse (navigate + get rendered content)
# =============================================================================


class BrowseInput(BaseModel):
    """Input for browse function."""

    url: str = Field(description="URL to navigate to")
    wait_for: str = Field(
        default="networkidle",
        description="Wait condition: 'load', 'domcontentloaded', 'networkidle', or a CSS selector to wait for",
    )
    timeout: int = Field(
        default=30000,
        description="Navigation timeout in milliseconds",
    )
    max_length: int = Field(
        default=50000,
        description="Maximum characters to return",
    )


class BrowseOutput(BaseModel):
    """Output for browse function."""

    ok: bool
    content: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    truncated: bool = False
    error: Optional[str] = None


def browse(input: BrowseInput) -> BrowseOutput:
    """
    Navigate to a URL using a headless browser and return the rendered page as markdown.

    Unlike fetch_url, this executes JavaScript and waits for dynamic content to load.
    Use this for Single Page Applications (SPAs), JavaScript-rendered pages, and
    sites that require a real browser.

    The browser session persists across calls, so you can navigate through multi-page flows.

    Examples:
        >>> browse({"url": "https://react.dev/learn"})
        >>> browse({"url": "https://example.com", "wait_for": "domcontentloaded"})
        >>> browse({"url": "https://app.example.com", "wait_for": "#main-content"})
    """
    try:
        page = _get_page()

        # Determine wait strategy
        wait_until = None
        wait_selector = None

        if input.wait_for in ("load", "domcontentloaded", "networkidle"):
            wait_until = input.wait_for
        else:
            # Treat as CSS selector
            wait_until = "domcontentloaded"
            wait_selector = input.wait_for

        page.goto(input.url, wait_until=wait_until, timeout=input.timeout)

        if wait_selector:
            page.wait_for_selector(wait_selector, timeout=input.timeout)

        # Get rendered HTML
        html = page.content()
        title = page.title()
        final_url = page.url

        content = _html_to_markdown(html)

        truncated = False
        if len(content) > input.max_length:
            content = content[: input.max_length] + "\n\n... [truncated]"
            truncated = True

        return BrowseOutput(
            ok=True,
            content=content,
            title=title,
            url=final_url,
            truncated=truncated,
        )

    except Exception as e:
        error_msg = str(e)
        if "playwright install" in error_msg.lower() or "executable doesn't exist" in error_msg.lower():
            error_msg = (
                f"Playwright browsers not installed. Run: playwright install chromium\n"
                f"Original error: {error_msg}"
            )
        return BrowseOutput(ok=False, error=error_msg)


# =============================================================================
# Screenshot
# =============================================================================


class ScreenshotInput(BaseModel):
    """Input for screenshot function."""

    url: Optional[str] = Field(
        default=None,
        description="URL to navigate to before taking screenshot. If None, screenshots the current page.",
    )
    output_path: str = Field(
        default="screenshot.png",
        description="File path to save the screenshot",
    )
    full_page: bool = Field(
        default=False,
        description="Capture the entire scrollable page (not just viewport)",
    )
    wait_for: str = Field(
        default="networkidle",
        description="Wait condition before taking screenshot",
    )
    timeout: int = Field(
        default=30000, description="Navigation timeout in milliseconds"
    )


class ScreenshotOutput(BaseModel):
    """Output for screenshot function."""

    ok: bool
    path: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    url: Optional[str] = None
    error: Optional[str] = None


def screenshot(input: ScreenshotInput) -> ScreenshotOutput:
    """
    Take a screenshot of a web page and save it to a file.

    If a URL is provided, navigates there first. Otherwise screenshots the current page
    (useful after a browse() call). Returns the file path and dimensions.

    Examples:
        >>> screenshot({"url": "https://example.com", "output_path": "example.png"})
        >>> screenshot({"output_path": "current_page.png"})  # screenshot current page
        >>> screenshot({"url": "https://example.com", "full_page": True, "output_path": "full.png"})
    """
    try:
        page = _get_page()

        if input.url:
            wait_until = input.wait_for if input.wait_for in ("load", "domcontentloaded", "networkidle") else "domcontentloaded"
            page.goto(input.url, wait_until=wait_until, timeout=input.timeout)

            if input.wait_for not in ("load", "domcontentloaded", "networkidle"):
                page.wait_for_selector(input.wait_for, timeout=input.timeout)

        # Ensure output directory exists
        output_path = Path(os.path.expanduser(input.output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        page.screenshot(path=str(output_path), full_page=input.full_page)

        # Get viewport size
        viewport = page.viewport_size
        width = viewport["width"] if viewport else None
        height = viewport["height"] if viewport else None

        return ScreenshotOutput(
            ok=True,
            path=str(output_path.absolute()),
            width=width,
            height=height,
            url=page.url,
        )

    except Exception as e:
        error_msg = str(e)
        if "playwright install" in error_msg.lower() or "executable doesn't exist" in error_msg.lower():
            error_msg = f"Playwright browsers not installed. Run: playwright install chromium\nOriginal error: {error_msg}"
        return ScreenshotOutput(ok=False, error=error_msg)


# =============================================================================
# Click element
# =============================================================================


class ClickInput(BaseModel):
    """Input for click function."""

    selector: str = Field(
        description="CSS selector or text to click (e.g., '#submit-btn', 'text=Login', 'button:has-text(\"Submit\")')"
    )
    wait_after: int = Field(
        default=1000,
        description="Milliseconds to wait after clicking for page to update",
    )
    timeout: int = Field(
        default=10000, description="Timeout to find the element in milliseconds"
    )


class ClickOutput(BaseModel):
    """Output for click function."""

    ok: bool
    url: Optional[str] = None
    title: Optional[str] = None
    error: Optional[str] = None


def click(input: ClickInput) -> ClickOutput:
    """
    Click an element on the current page.

    Supports CSS selectors and Playwright text selectors. Use after browse()
    to interact with page elements.

    Examples:
        >>> click({"selector": "#login-button"})
        >>> click({"selector": "text=Sign In"})
        >>> click({"selector": "button:has-text('Submit')"})
        >>> click({"selector": "a[href='/about']"})
    """
    try:
        page = _get_page()
        page.click(input.selector, timeout=input.timeout)
        page.wait_for_timeout(input.wait_after)

        return ClickOutput(
            ok=True,
            url=page.url,
            title=page.title(),
        )

    except Exception as e:
        return ClickOutput(ok=False, error=str(e))


# =============================================================================
# Type text into an input
# =============================================================================


class TypeTextInput(BaseModel):
    """Input for type_text function."""

    selector: str = Field(
        description="CSS selector for the input element (e.g., '#search', 'input[name=email]')"
    )
    text: str = Field(description="Text to type")
    clear_first: bool = Field(
        default=True, description="Clear the input before typing"
    )
    press_enter: bool = Field(
        default=False, description="Press Enter after typing"
    )
    timeout: int = Field(
        default=10000, description="Timeout to find the element in milliseconds"
    )


class TypeTextOutput(BaseModel):
    """Output for type_text function."""

    ok: bool
    error: Optional[str] = None


def type_text(input: TypeTextInput) -> TypeTextOutput:
    """
    Type text into an input field on the current page.

    Use after browse() to fill in forms. Can optionally clear the field first
    and press Enter after typing.

    Examples:
        >>> type_text({"selector": "#search", "text": "python docs", "press_enter": True})
        >>> type_text({"selector": "input[name=email]", "text": "user@example.com"})
    """
    try:
        page = _get_page()

        if input.clear_first:
            page.fill(input.selector, input.text, timeout=input.timeout)
        else:
            page.type(input.selector, input.text, timeout=input.timeout)

        if input.press_enter:
            page.press(input.selector, "Enter")

        return TypeTextOutput(ok=True)

    except Exception as e:
        return TypeTextOutput(ok=False, error=str(e))


# =============================================================================
# Get page state (current URL, title, visible elements)
# =============================================================================


class GetPageStateInput(BaseModel):
    """Input for get_page_state function."""

    include_links: bool = Field(
        default=True, description="Include a list of links on the page"
    )
    include_inputs: bool = Field(
        default=True, description="Include form inputs and buttons"
    )
    include_text: bool = Field(
        default=True, description="Include the page text content as markdown"
    )
    max_length: int = Field(
        default=20000, description="Max characters for text content"
    )


class PageLink(BaseModel):
    text: str
    href: str


class PageInput(BaseModel):
    type: str  # "text", "password", "email", "submit", "button", etc.
    name: Optional[str] = None
    placeholder: Optional[str] = None
    value: Optional[str] = None
    selector: str  # CSS selector to target this element


class PageStateOutput(BaseModel):
    """Output for get_page_state function."""

    ok: bool
    url: Optional[str] = None
    title: Optional[str] = None
    text: Optional[str] = None
    links: Optional[List[PageLink]] = None
    inputs: Optional[List[PageInput]] = None
    error: Optional[str] = None


def get_page_state(input: GetPageStateInput) -> PageStateOutput:
    """
    Get the current state of the browser page.

    Returns the URL, title, page text, links, and form elements. Use this
    to understand what's on the page before deciding what to click or type.

    Examples:
        >>> get_page_state({})
        >>> get_page_state({"include_text": False, "include_inputs": True})
    """
    try:
        page = _get_page()

        url = page.url
        title = page.title()

        text = None
        if input.include_text:
            html = page.content()
            text = _html_to_markdown(html)
            if len(text) > input.max_length:
                text = text[: input.max_length] + "\n\n... [truncated]"

        links = None
        if input.include_links:
            link_data = page.evaluate("""
                () => Array.from(document.querySelectorAll('a[href]')).slice(0, 100).map(a => ({
                    text: a.innerText.trim().substring(0, 100),
                    href: a.href
                })).filter(l => l.text && l.href)
            """)
            links = [PageLink(**l) for l in link_data]

        inputs = None
        if input.include_inputs:
            input_data = page.evaluate("""
                () => {
                    const elements = [
                        ...document.querySelectorAll('input, textarea, select, button, [role="button"]')
                    ].slice(0, 50);

                    return elements.map((el, i) => {
                        const type = el.tagName.toLowerCase() === 'button' ? 'button'
                            : el.tagName.toLowerCase() === 'textarea' ? 'textarea'
                            : el.tagName.toLowerCase() === 'select' ? 'select'
                            : el.getAttribute('type') || 'text';

                        const name = el.getAttribute('name') || el.getAttribute('id') || null;
                        const placeholder = el.getAttribute('placeholder') || null;
                        const value = el.value || el.innerText?.trim()?.substring(0, 50) || null;

                        // Build a unique selector
                        let selector = '';
                        if (el.id) selector = '#' + el.id;
                        else if (el.name) selector = `${el.tagName.toLowerCase()}[name="${el.name}"]`;
                        else selector = `${el.tagName.toLowerCase()}:nth-of-type(${i + 1})`;

                        return { type, name, placeholder, value, selector };
                    });
                }
            """)
            inputs = [PageInput(**d) for d in input_data]

        return PageStateOutput(
            ok=True,
            url=url,
            title=title,
            text=text,
            links=links,
            inputs=inputs,
        )

    except Exception as e:
        return PageStateOutput(ok=False, error=str(e))


# =============================================================================
# Close browser session
# =============================================================================


class CloseBrowserInput(BaseModel):
    """Input for close_browser function."""
    pass


class CloseBrowserOutput(BaseModel):
    """Output for close_browser function."""

    ok: bool
    error: Optional[str] = None


def close_browser(input: CloseBrowserInput) -> CloseBrowserOutput:
    """
    Close the browser session and free resources.

    Call this when you're done with browser interactions. A new session will
    be created automatically on the next browse() call.

    Examples:
        >>> close_browser({})
    """
    try:
        _close_browser()
        return CloseBrowserOutput(ok=True)
    except Exception as e:
        return CloseBrowserOutput(ok=False, error=str(e))
