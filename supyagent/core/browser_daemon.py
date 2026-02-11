"""
Persistent browser daemon for stateful Playwright sessions.

Launches the user's real Chrome/Brave/Edge browser with CDP enabled, then
connects via Playwright's connectOverCDP. This avoids headless bot detection
because the browser is a genuine user-installed binary with a persistent profile
(cookies, sessions, history all preserved across runs).

Runs as a standalone HTTP server that keeps the browser alive across tool calls.
The agent engine routes browser__* tools here instead of spawning a new
supypowers subprocess each time.

This module is the single source of truth for all browser automation in supyagent.

Usage:
    python -m supyagent.core.browser_daemon --port 0 --port-file .supyagent/tmp/browser.port
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTML → Markdown conversion
# ---------------------------------------------------------------------------

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
    from bs4 import BeautifulSoup
    from markdownify import markdownify as md

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


# ---------------------------------------------------------------------------
# Pydantic input models (used for validation and schema generation)
# ---------------------------------------------------------------------------


class BrowseInput(BaseModel):
    """Input for browse function."""

    url: str = Field(description="URL to navigate to")
    wait_for: str = Field(
        default="networkidle",
        description=(
            "Wait condition: 'load', 'domcontentloaded', 'networkidle', "
            "or a CSS selector to wait for"
        ),
    )
    timeout: int = Field(default=30000, description="Navigation timeout in milliseconds")
    max_length: int = Field(default=50000, description="Maximum characters to return")


class ScreenshotInput(BaseModel):
    """Input for screenshot function."""

    url: Optional[str] = Field(
        default=None,
        description="URL to navigate to before screenshot. If None, screenshots the current page.",
    )
    output_path: str = Field(default="screenshot.png", description="File path to save the screenshot")
    full_page: bool = Field(default=False, description="Capture the entire scrollable page")
    wait_for: str = Field(default="networkidle", description="Wait condition before screenshot")
    timeout: int = Field(default=30000, description="Navigation timeout in milliseconds")


class ClickInput(BaseModel):
    """Input for click function."""

    selector: str = Field(
        description=(
            "CSS selector or text to click "
            "(e.g., '#submit-btn', 'text=Login', 'button:has-text(\"Submit\")')"
        )
    )
    wait_after: int = Field(
        default=1000, description="Milliseconds to wait after clicking for page to update"
    )
    timeout: int = Field(default=10000, description="Timeout to find the element in milliseconds")


class TypeTextInput(BaseModel):
    """Input for type_text function."""

    selector: str = Field(
        description="CSS selector for the input element (e.g., '#search', 'input[name=email]')"
    )
    text: str = Field(description="Text to type")
    clear_first: bool = Field(default=True, description="Clear the input before typing")
    press_enter: bool = Field(default=False, description="Press Enter after typing")
    timeout: int = Field(default=10000, description="Timeout to find the element in milliseconds")


class GetPageStateInput(BaseModel):
    """Input for get_page_state function."""

    include_links: bool = Field(default=True, description="Include a list of links on the page")
    include_inputs: bool = Field(default=True, description="Include form inputs and buttons")
    include_text: bool = Field(
        default=True, description="Include the page text content as markdown"
    )
    max_length: int = Field(default=20000, description="Max characters for text content")


# ---------------------------------------------------------------------------
# Chrome / Brave / Edge executable detection
# ---------------------------------------------------------------------------

# Candidates ordered by preference: Chrome → Brave → Edge → Chromium
_BROWSER_CANDIDATES_MACOS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
    "/Applications/Arc.app/Contents/MacOS/Arc",
]

_BROWSER_CANDIDATES_LINUX = [
    "google-chrome",
    "google-chrome-stable",
    "brave-browser",
    "microsoft-edge",
    "chromium",
    "chromium-browser",
]

_BROWSER_CANDIDATES_WINDOWS = [
    os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
    os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
    os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
    os.path.expandvars(r"%ProgramFiles%\BraveSoftware\Brave-Browser\Application\brave.exe"),
    os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
]


def _detect_chrome_executable() -> str | None:
    """Detect the system's installed Chromium-based browser."""
    system = platform.system()

    if system == "Darwin":
        candidates = _BROWSER_CANDIDATES_MACOS
    elif system == "Linux":
        candidates = _BROWSER_CANDIDATES_LINUX
    elif system == "Windows":
        candidates = _BROWSER_CANDIDATES_WINDOWS
    else:
        return None

    for candidate in candidates:
        if system == "Linux":
            # Linux candidates are command names — check with `which`
            import shutil

            if shutil.which(candidate):
                return candidate
        else:
            if os.path.isfile(candidate):
                return candidate

    return None


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_cdp(port: int, timeout: float = 15.0) -> None:
    """Wait until the CDP endpoint is accepting connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.3)
    raise RuntimeError(f"Chrome CDP port {port} not ready within {timeout}s")


def _ensure_clean_exit(profile_dir: str) -> None:
    """Clean up stale lock files and mark profile as cleanly exited.

    Chrome refuses to start if another instance is using the same profile.
    Stale lock files from unclean shutdowns will block startup.
    """
    # Remove stale lock files
    for lock_file in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
        lock_path = os.path.join(profile_dir, lock_file)
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass
        except OSError:
            pass

    # Mark profile as cleanly exited to prevent "Chrome didn't shut down correctly" bar
    prefs_path = os.path.join(profile_dir, "Default", "Preferences")
    try:
        if os.path.isfile(prefs_path):
            with open(prefs_path) as f:
                import json as _json

                prefs = _json.load(f)
            prefs.setdefault("profile", {})["exit_type"] = "Normal"
            prefs.setdefault("profile", {})["exited_cleanly"] = True
            with open(prefs_path, "w") as f:
                _json.dump(prefs, f)
    except Exception:
        pass  # Best effort


# ---------------------------------------------------------------------------
# Browser state (module-level singleton, persists for the daemon's lifetime)
# ---------------------------------------------------------------------------

_browser = None
_context = None
_page = None
_chrome_process = None

# Daemon-level config (set by CLI args before serve_forever)
_headless = False
_profile_dir: str | None = None

# Default persistent profile directory (under the project's .supyagent dir)
_DEFAULT_PROFILE_DIR = os.path.join(".supyagent", "browser", "profile")


def _get_page():
    """Get or create a shared browser page.

    Launches the user's real Chrome/Brave/Edge with CDP enabled, then
    connects via Playwright's connectOverCDP. Falls back to Playwright's
    bundled Chromium if no system browser is found.
    """
    global _browser, _context, _page, _chrome_process

    if _page is not None and not _page.is_closed():
        return _page

    from playwright.sync_api import sync_playwright

    pw = sync_playwright().start()

    chrome_path = _detect_chrome_executable()

    if chrome_path:
        # --- Real browser via CDP ---
        profile_dir = os.path.abspath(_profile_dir or _DEFAULT_PROFILE_DIR)
        os.makedirs(profile_dir, exist_ok=True)
        _ensure_clean_exit(profile_dir)

        cdp_port = _find_free_port()

        chrome_args = [
            chrome_path,
            f"--remote-debugging-port={cdp_port}",
            f"--user-data-dir={profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-sync",
            "--disable-background-networking",
            "--disable-component-update",
            "--disable-features=Translate,MediaRouter",
            "--disable-session-crashed-bubble",
            "--hide-crash-restore-bubble",
            "--password-store=basic",
        ]

        if _headless:
            chrome_args.extend(["--headless=new", "--disable-gpu"])

        if platform.system() == "Linux":
            chrome_args.append("--disable-dev-shm-usage")

        chrome_args.append("about:blank")

        logger.info(f"Launching browser: {chrome_path} (CDP port {cdp_port})")
        _chrome_process = subprocess.Popen(
            chrome_args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        _wait_for_cdp(cdp_port)

        _browser = pw.chromium.connect_over_cdp(f"http://127.0.0.1:{cdp_port}")
        contexts = _browser.contexts
        if contexts:
            _context = contexts[0]
            pages = _context.pages
            _page = pages[0] if pages else _context.new_page()
        else:
            _context = _browser.new_context()
            _page = _context.new_page()

        logger.info("Connected to browser via CDP")

    else:
        # --- Fallback: Playwright's bundled Chromium ---
        logger.warning(
            "No system Chrome/Brave/Edge found, falling back to Playwright Chromium"
        )
        try:
            _browser = pw.chromium.launch(headless=_headless)
        except Exception as e:
            if "executable doesn't exist" in str(e).lower():
                subprocess.run(
                    ["playwright", "install", "chromium"],
                    check=True,
                    capture_output=True,
                )
                _browser = pw.chromium.launch(headless=_headless)
            else:
                raise
        _context = _browser.new_context(
            viewport={"width": 1280, "height": 720},
        )
        _page = _context.new_page()

        # Apply stealth only for bundled Chromium (real browsers don't need it)
        try:
            from playwright_stealth import stealth_sync

            stealth_sync(_page)
        except ImportError:
            pass

    return _page


def _close_browser():
    """Close the browser session."""
    global _browser, _context, _page, _chrome_process
    try:
        if _page and not _page.is_closed():
            _page.close()
        if _context:
            _context.close()
        if _browser:
            _browser.close()
    except Exception:
        pass
    # Terminate the Chrome process we spawned
    if _chrome_process:
        try:
            _chrome_process.terminate()
            _chrome_process.wait(timeout=5)
        except Exception:
            try:
                _chrome_process.kill()
            except Exception:
                pass
    _browser = None
    _context = None
    _page = None
    _chrome_process = None


# ---------------------------------------------------------------------------
# Command dispatch
# ---------------------------------------------------------------------------

_WAIT_STATES = ("load", "domcontentloaded", "networkidle")


def _dispatch_command(command: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute a browser command and return a result dict."""
    try:
        if command == "browse":
            page = _get_page()
            inp = BrowseInput(**args)
            wait_until = inp.wait_for if inp.wait_for in _WAIT_STATES else "domcontentloaded"
            wait_selector = None if inp.wait_for in _WAIT_STATES else inp.wait_for
            page.goto(inp.url, wait_until=wait_until, timeout=inp.timeout)
            if wait_selector:
                page.wait_for_selector(wait_selector, timeout=inp.timeout)

            html = page.content()
            content = _html_to_markdown(html)
            truncated = False
            if len(content) > inp.max_length:
                content = content[: inp.max_length] + "\n\n... [truncated]"
                truncated = True
            return {
                "ok": True,
                "data": {
                    "content": content,
                    "title": page.title(),
                    "url": page.url,
                    "truncated": truncated,
                },
            }

        elif command == "screenshot":
            page = _get_page()
            inp = ScreenshotInput(**args)
            if inp.url:
                wait_until = inp.wait_for if inp.wait_for in _WAIT_STATES else "domcontentloaded"
                page.goto(inp.url, wait_until=wait_until, timeout=inp.timeout)
                if inp.wait_for not in _WAIT_STATES:
                    page.wait_for_selector(inp.wait_for, timeout=inp.timeout)

            output_path = Path(os.path.expanduser(inp.output_path))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(output_path), full_page=inp.full_page)
            viewport = page.viewport_size
            return {
                "ok": True,
                "data": {
                    "path": str(output_path.absolute()),
                    "width": viewport["width"] if viewport else None,
                    "height": viewport["height"] if viewport else None,
                    "url": page.url,
                },
            }

        elif command == "click":
            page = _get_page()
            inp = ClickInput(**args)
            page.click(inp.selector, timeout=inp.timeout)
            page.wait_for_timeout(inp.wait_after)
            return {
                "ok": True,
                "data": {"url": page.url, "title": page.title()},
            }

        elif command == "type_text":
            page = _get_page()
            inp = TypeTextInput(**args)
            if inp.clear_first:
                page.fill(inp.selector, inp.text, timeout=inp.timeout)
            else:
                page.type(inp.selector, inp.text, timeout=inp.timeout)
            if inp.press_enter:
                page.press(inp.selector, "Enter")
            return {"ok": True, "data": {}}

        elif command == "get_page_state":
            page = _get_page()
            inp = GetPageStateInput(**args)
            data: dict[str, Any] = {"url": page.url, "title": page.title()}

            if inp.include_text:
                html = page.content()
                text = _html_to_markdown(html)
                if len(text) > inp.max_length:
                    text = text[: inp.max_length] + "\n\n... [truncated]"
                data["text"] = text

            if inp.include_links:
                data["links"] = page.evaluate(
                    """() => Array.from(document.querySelectorAll('a[href]'))
                    .slice(0, 100).map(a => ({
                        text: a.innerText.trim().substring(0, 100),
                        href: a.href
                    })).filter(l => l.text && l.href)"""
                )

            if inp.include_inputs:
                data["inputs"] = page.evaluate(
                    """() => {
                    const elements = [
                        ...document.querySelectorAll(
                            'input, textarea, select, button, [role="button"]'
                        )
                    ].slice(0, 50);
                    return elements.map((el, i) => {
                        const type = el.tagName.toLowerCase() === 'button' ? 'button'
                            : el.tagName.toLowerCase() === 'textarea' ? 'textarea'
                            : el.tagName.toLowerCase() === 'select' ? 'select'
                            : el.getAttribute('type') || 'text';
                        const name = el.getAttribute('name')
                            || el.getAttribute('id') || null;
                        const placeholder = el.getAttribute('placeholder') || null;
                        const value = el.value
                            || el.innerText?.trim()?.substring(0, 50) || null;
                        let selector = '';
                        if (el.id) selector = '#' + el.id;
                        else if (el.name)
                            selector = `${el.tagName.toLowerCase()}[name="${el.name}"]`;
                        else
                            selector = `${el.tagName.toLowerCase()}:nth-of-type(${i + 1})`;
                        return { type, name, placeholder, value, selector };
                    });
                }"""
                )

            return {"ok": True, "data": data}

        elif command == "close_browser":
            _close_browser()
            return {"ok": True, "data": {"message": "Browser closed"}}

        else:
            return {"ok": False, "error": f"Unknown command: {command}"}

    except Exception as e:
        error_msg = str(e)
        if (
            "playwright install" in error_msg.lower()
            or "executable doesn't exist" in error_msg.lower()
        ):
            error_msg = (
                "Playwright browsers not installed. Run: playwright install chromium\n"
                f"Original error: {error_msg}"
            )
        return {"ok": False, "error": error_msg}


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class BrowserDaemonHandler(BaseHTTPRequestHandler):
    """Handle POST /execute requests."""

    def do_POST(self):
        if self.path == "/execute":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                self._respond(400, {"ok": False, "error": "Invalid JSON"})
                return

            command = payload.get("command", "")
            args = payload.get("args", {})
            result = _dispatch_command(command, args)
            self._respond(200, result)

        elif self.path == "/health":
            self._respond(200, {"ok": True, "status": "running"})
        else:
            self._respond(404, {"ok": False, "error": "Not found"})

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"ok": True, "status": "running"})
        else:
            self._respond(404, {"ok": False, "error": "Not found"})

    def _respond(self, status: int, body: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def log_message(self, format, *args):
        """Suppress default stderr logging."""
        logger.debug(format, *args)


def run_daemon(
    port: int = 0,
    port_file: str | None = None,
    headless: bool = False,
    profile_dir: str | None = None,
):
    """Start the browser daemon HTTP server."""
    global _headless, _profile_dir
    _headless = headless
    _profile_dir = profile_dir

    server = HTTPServer(("127.0.0.1", port), BrowserDaemonHandler)
    actual_port = server.server_address[1]

    # Write the bound port so the session manager can discover it
    if port_file:
        with open(port_file, "w") as f:
            f.write(str(actual_port))

    # Print to stdout for supervisor to capture
    print(json.dumps({"ok": True, "data": {"port": actual_port, "status": "started"}}))
    sys.stdout.flush()

    # Clean shutdown on SIGTERM
    def _shutdown(signum, frame):
        logger.info("Browser daemon shutting down")
        _close_browser()
        server.shutdown()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    logger.info(f"Browser daemon listening on 127.0.0.1:{actual_port}")
    server.serve_forever()


# ---------------------------------------------------------------------------
# __main__ entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Browser daemon for supyagent")
    parser.add_argument("--port", type=int, default=0, help="Port to listen on (0 = auto)")
    parser.add_argument(
        "--port-file", type=str, default=None, help="File to write bound port to"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run browser in headless mode (default: headed)",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default=None,
        help="Browser profile directory (default: ~/.supyagent/browser/profile)",
    )
    args = parser.parse_args()
    run_daemon(
        port=args.port,
        port_file=args.port_file,
        headless=args.headless,
        profile_dir=args.profile_dir,
    )


if __name__ == "__main__":
    main()
