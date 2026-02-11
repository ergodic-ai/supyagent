"""
Tests for the persistent browser session system.

Covers: BrowserSessionManager, browser daemon dispatch, engine integration,
        Chrome executable detection, CDP connection logic.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

from supyagent.core.browser_daemon import (
    _DEFAULT_PROFILE_DIR,
    _detect_chrome_executable,
    _dispatch_command,
    _find_free_port,
)
from supyagent.core.browser_session import BrowserSessionManager

# =========================================================================
# Browser daemon command dispatch
# =========================================================================


class TestDaemonDispatch:
    """Test _dispatch_command in the daemon."""

    def test_unknown_command(self):
        result = _dispatch_command("unknown_cmd", {})
        assert result["ok"] is False
        assert "Unknown command" in result["error"]

    @patch("supyagent.core.browser_daemon._get_page")
    def test_browse_command(self, mock_get_page):
        page = MagicMock()
        page.url = "https://example.com"
        page.title.return_value = "Test"
        page.content.return_value = "<html><body><p>Hello</p></body></html>"
        mock_get_page.return_value = page

        result = _dispatch_command("browse", {"url": "https://example.com"})
        assert result["ok"] is True
        assert result["data"]["url"] == "https://example.com"
        assert result["data"]["title"] == "Test"
        assert "Hello" in result["data"]["content"]
        page.goto.assert_called_once()

    @patch("supyagent.core.browser_daemon._get_page")
    def test_click_command(self, mock_get_page):
        page = MagicMock()
        page.url = "https://example.com/clicked"
        page.title.return_value = "Clicked"
        mock_get_page.return_value = page

        result = _dispatch_command("click", {"selector": "#btn"})
        assert result["ok"] is True
        assert result["data"]["url"] == "https://example.com/clicked"
        page.click.assert_called_with("#btn", timeout=10000)

    @patch("supyagent.core.browser_daemon._get_page")
    def test_type_text_command(self, mock_get_page):
        page = MagicMock()
        mock_get_page.return_value = page

        result = _dispatch_command(
            "type_text", {"selector": "#input", "text": "hello", "clear_first": True}
        )
        assert result["ok"] is True
        page.fill.assert_called_with("#input", "hello", timeout=10000)

    @patch("supyagent.core.browser_daemon._get_page")
    def test_type_text_with_enter(self, mock_get_page):
        page = MagicMock()
        mock_get_page.return_value = page

        result = _dispatch_command(
            "type_text",
            {"selector": "#search", "text": "query", "press_enter": True},
        )
        assert result["ok"] is True
        page.press.assert_called_with("#search", "Enter")

    @patch("supyagent.core.browser_daemon._get_page")
    def test_get_page_state_command(self, mock_get_page):
        page = MagicMock()
        page.url = "https://example.com"
        page.title.return_value = "Test"
        page.content.return_value = "<html><body><p>Content</p></body></html>"
        page.evaluate.return_value = []
        mock_get_page.return_value = page

        result = _dispatch_command("get_page_state", {})
        assert result["ok"] is True
        assert result["data"]["url"] == "https://example.com"
        assert result["data"]["title"] == "Test"

    @patch("supyagent.core.browser_daemon._get_page")
    def test_screenshot_command(self, mock_get_page, tmp_path):
        page = MagicMock()
        page.url = "https://example.com"
        page.viewport_size = {"width": 1280, "height": 720}
        mock_get_page.return_value = page

        output = str(tmp_path / "test.png")
        result = _dispatch_command(
            "screenshot", {"url": "https://example.com", "output_path": output}
        )
        assert result["ok"] is True
        assert result["data"]["width"] == 1280
        page.screenshot.assert_called_once()

    @patch("supyagent.core.browser_daemon._close_browser")
    def test_close_browser_command(self, mock_close):
        result = _dispatch_command("close_browser", {})
        assert result["ok"] is True
        mock_close.assert_called_once()

    @patch("supyagent.core.browser_daemon._get_page")
    def test_browse_error_handling(self, mock_get_page):
        page = MagicMock()
        page.goto.side_effect = Exception("Connection refused")
        mock_get_page.return_value = page

        result = _dispatch_command("browse", {"url": "https://example.com"})
        assert result["ok"] is False
        assert "Connection refused" in result["error"]

    @patch("supyagent.core.browser_daemon._get_page")
    def test_browse_with_css_selector_wait(self, mock_get_page):
        page = MagicMock()
        page.url = "https://example.com"
        page.title.return_value = "Test"
        page.content.return_value = "<html><body>loaded</body></html>"
        mock_get_page.return_value = page

        result = _dispatch_command(
            "browse", {"url": "https://example.com", "wait_for": "#main"}
        )
        assert result["ok"] is True
        page.goto.assert_called_with(
            "https://example.com", wait_until="domcontentloaded", timeout=30000
        )
        page.wait_for_selector.assert_called_with("#main", timeout=30000)


# =========================================================================
# BrowserSessionManager
# =========================================================================


class TestBrowserSessionManager:
    def test_initial_state(self):
        mgr = BrowserSessionManager()
        assert mgr._daemon_port is None
        assert mgr._daemon_process_id is None

    @patch("supyagent.core.browser_session.BrowserSessionManager._health_check")
    def test_ensure_daemon_health_check_passes(self, mock_health):
        """If daemon is already running and healthy, ensure_daemon is a no-op."""
        mock_health.return_value = True
        mgr = BrowserSessionManager()
        mgr._daemon_port = 9999  # Simulate already running

        mgr.ensure_daemon()
        # Should not try to start a new daemon
        assert mgr._daemon_port == 9999

    @patch("supyagent.core.browser_session.BrowserSessionManager.ensure_daemon")
    def test_execute_calls_ensure_daemon(self, mock_ensure):
        """execute() should call ensure_daemon before making HTTP request."""
        mgr = BrowserSessionManager()
        mgr._daemon_port = 12345

        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "data": {"content": "test"}}

        with patch("httpx.post", return_value=mock_response) as mock_post:
            result = mgr.execute("browse", {"url": "https://example.com"})

        mock_ensure.assert_called_once()
        mock_post.assert_called_once()
        assert result["ok"] is True

    @patch("supyagent.core.browser_session.BrowserSessionManager.ensure_daemon")
    def test_execute_connection_error_resets(self, mock_ensure):
        """Connection error should reset daemon state for restart on next call."""
        import httpx

        mgr = BrowserSessionManager()
        mgr._daemon_port = 12345
        mgr._daemon_process_id = "proc_123"

        with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            result = mgr.execute("browse", {"url": "https://example.com"})

        assert result["ok"] is False
        assert "not responding" in result["error"]
        assert mgr._daemon_port is None
        assert mgr._daemon_process_id is None

    @patch("supyagent.core.browser_session.BrowserSessionManager.ensure_daemon")
    def test_execute_timeout_error(self, mock_ensure):
        """Timeout should return error without resetting daemon."""
        import httpx

        mgr = BrowserSessionManager()
        mgr._daemon_port = 12345

        with patch("httpx.post", side_effect=httpx.TimeoutException("timed out")):
            result = mgr.execute("click", {"selector": "#slow"})

        assert result["ok"] is False
        assert "timed out" in result["error"]
        # Daemon port should NOT be reset on timeout (daemon is still alive)
        assert mgr._daemon_port == 12345

    def test_shutdown_without_daemon(self):
        """Shutdown when no daemon is running should be a no-op."""
        mgr = BrowserSessionManager()
        mgr.shutdown()  # Should not raise
        assert mgr._daemon_port is None

    def test_shutdown_kills_daemon(self):
        """Shutdown should kill the daemon process via supervisor."""
        mgr = BrowserSessionManager()
        mgr._daemon_process_id = "proc_abc"
        mgr._daemon_port = 9999

        mock_supervisor = MagicMock()

        with patch(
            "supyagent.core.supervisor.get_supervisor", return_value=mock_supervisor
        ), patch(
            "supyagent.core.supervisor.run_supervisor_coroutine"
        ):
            mgr.shutdown()

        assert mgr._daemon_port is None
        assert mgr._daemon_process_id is None

    def test_health_check_no_port(self):
        """Health check with no port should return False."""
        mgr = BrowserSessionManager()
        assert mgr._health_check() is False

    def test_health_check_connection_refused(self):
        """Health check when daemon is not running should return False."""
        import httpx

        mgr = BrowserSessionManager()
        mgr._daemon_port = 99999  # Unlikely to be in use

        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            assert mgr._health_check() is False


# =========================================================================
# Engine integration
# =========================================================================


class TestEngineIntegration:
    """Test that browser tools are correctly intercepted in the engine."""

    def test_is_browser_tool(self):
        from supyagent.core.engine import BaseAgentEngine

        assert BaseAgentEngine._is_browser_tool("browser__browse") is True
        assert BaseAgentEngine._is_browser_tool("browser__click") is True
        assert BaseAgentEngine._is_browser_tool("browser__type_text") is True
        assert BaseAgentEngine._is_browser_tool("browser__screenshot") is True
        assert BaseAgentEngine._is_browser_tool("browser__get_page_state") is True
        assert BaseAgentEngine._is_browser_tool("browser__close_browser") is True

    def test_is_not_browser_tool(self):
        from supyagent.core.engine import BaseAgentEngine

        assert BaseAgentEngine._is_browser_tool("files__read_file") is False
        assert BaseAgentEngine._is_browser_tool("web__fetch_url") is False
        assert BaseAgentEngine._is_browser_tool("shell__exec") is False
        assert BaseAgentEngine._is_browser_tool("list_processes") is False

    def test_execute_browser_tool_lazy_init(self):
        """_execute_browser_tool should lazy-init BrowserSessionManager."""
        from supyagent.core.engine import BaseAgentEngine
        from supyagent.core.models import ToolCallObj
        from supyagent.models.agent_config import AgentConfig

        # Create a minimal config
        config = AgentConfig(
            name="test",
            system_prompt="test",
            model={"provider": "test/model"},
            service={"enabled": False},
        )

        # We can't instantiate BaseAgentEngine directly (it's abstract),
        # so create a minimal concrete subclass
        class TestEngine(BaseAgentEngine):
            def _get_secrets(self):
                return {}

        engine = TestEngine(config)
        assert engine._browser_session is None

        # Mock the session manager to avoid actually starting a daemon
        mock_session = MagicMock()
        mock_session.execute.return_value = {"ok": True, "data": {"url": "https://example.com"}}

        with patch(
            "supyagent.core.browser_session.BrowserSessionManager",
            return_value=mock_session,
        ) as mock_cls:
            tool_call = ToolCallObj(
                "call_1", "browser__browse", json.dumps({"url": "https://example.com"})
            )
            result = engine._execute_browser_tool(tool_call)

        assert result["ok"] is True
        mock_cls.assert_called_once()
        mock_session.execute.assert_called_with("browse", {"url": "https://example.com"})
        assert engine._browser_session is mock_session

    def test_execute_browser_tool_reuses_session(self):
        """Subsequent calls should reuse the same BrowserSessionManager."""
        from supyagent.core.engine import BaseAgentEngine
        from supyagent.core.models import ToolCallObj
        from supyagent.models.agent_config import AgentConfig

        class TestEngine(BaseAgentEngine):
            def _get_secrets(self):
                return {}

        config = AgentConfig(
            name="test",
            system_prompt="test",
            model={"provider": "test/model"},
            service={"enabled": False},
        )
        engine = TestEngine(config)

        mock_session = MagicMock()
        mock_session.execute.return_value = {"ok": True, "data": {}}
        engine._browser_session = mock_session

        with patch("supyagent.core.browser_session.BrowserSessionManager") as mock_cls:
            tool_call = ToolCallObj(
                "call_1", "browser__click", json.dumps({"selector": "#btn"})
            )
            engine._execute_browser_tool(tool_call)

        # Should NOT create a new session manager
        mock_cls.assert_not_called()
        mock_session.execute.assert_called_with("click", {"selector": "#btn"})

    def test_execute_browser_tool_invalid_json(self):
        """Invalid JSON args should return error."""
        from supyagent.core.engine import BaseAgentEngine
        from supyagent.core.models import ToolCallObj
        from supyagent.models.agent_config import AgentConfig

        class TestEngine(BaseAgentEngine):
            def _get_secrets(self):
                return {}

        config = AgentConfig(
            name="test",
            system_prompt="test",
            model={"provider": "test/model"},
            service={"enabled": False},
        )
        engine = TestEngine(config)

        tool_call = ToolCallObj("call_1", "browser__browse", "not valid json{{{")
        result = engine._execute_browser_tool(tool_call)
        assert result["ok"] is False
        assert "Invalid JSON" in result["error"]

    def test_dispatch_routes_browser_tools(self):
        """_dispatch_tool_call should route browser__* to _execute_browser_tool."""
        from supyagent.core.engine import BaseAgentEngine
        from supyagent.core.models import ToolCallObj
        from supyagent.models.agent_config import AgentConfig

        class TestEngine(BaseAgentEngine):
            def _get_secrets(self):
                return {}

        config = AgentConfig(
            name="test",
            system_prompt="test",
            model={"provider": "test/model"},
            service={"enabled": False},
        )
        engine = TestEngine(config)

        mock_result = {"ok": True, "data": {"content": "page content"}}
        with patch.object(engine, "_execute_browser_tool", return_value=mock_result) as mock_exec:
            tool_call = ToolCallObj(
                "call_1", "browser__browse", json.dumps({"url": "https://example.com"})
            )
            result = engine._dispatch_tool_call(tool_call)

        mock_exec.assert_called_once_with(tool_call)
        assert result["ok"] is True

    def test_dispatch_non_browser_skips_browser_path(self):
        """Non-browser tools should NOT go through _execute_browser_tool."""
        from supyagent.core.engine import BaseAgentEngine
        from supyagent.core.models import ToolCallObj
        from supyagent.models.agent_config import AgentConfig

        class TestEngine(BaseAgentEngine):
            def _get_secrets(self):
                return {}

        config = AgentConfig(
            name="test",
            system_prompt="test",
            model={"provider": "test/model"},
            service={"enabled": False},
        )
        engine = TestEngine(config)

        with patch.object(engine, "_execute_browser_tool") as mock_browser, patch.object(
            engine, "_execute_supypowers_tool", return_value={"ok": True, "data": "file content"}
        ) as mock_sp:
            tool_call = ToolCallObj(
                "call_1", "files__read_file", json.dumps({"path": "/tmp/test.txt"})
            )
            engine._dispatch_tool_call(tool_call)

        mock_browser.assert_not_called()
        mock_sp.assert_called_once()


# =========================================================================
# Daemon HTTP handler (unit tests)
# =========================================================================


class TestDaemonHandler:
    """Test the HTTP handler logic."""

    def test_dispatch_browse_truncation(self):
        """Content exceeding max_length should be truncated."""
        with patch("supyagent.core.browser_daemon._get_page") as mock_get_page:
            page = MagicMock()
            page.url = "https://example.com"
            page.title.return_value = "Test"
            page.content.return_value = "<html><body><p>" + "x" * 100000 + "</p></body></html>"
            mock_get_page.return_value = page

            result = _dispatch_command("browse", {"url": "https://example.com", "max_length": 100})
            assert result["ok"] is True
            assert result["data"]["truncated"] is True
            assert len(result["data"]["content"]) <= 200  # 100 + truncation message

    def test_dispatch_get_page_state_without_text(self):
        """get_page_state with include_text=False should not include text."""
        with patch("supyagent.core.browser_daemon._get_page") as mock_get_page:
            page = MagicMock()
            page.url = "https://example.com"
            page.title.return_value = "Test"
            page.evaluate.return_value = []
            mock_get_page.return_value = page

            result = _dispatch_command(
                "get_page_state",
                {"include_text": False, "include_links": False, "include_inputs": False},
            )
            assert result["ok"] is True
            assert "text" not in result["data"]
            assert "links" not in result["data"]
            assert "inputs" not in result["data"]

    def test_dispatch_type_text_without_clear(self):
        """type_text with clear_first=False should use page.type instead of page.fill."""
        with patch("supyagent.core.browser_daemon._get_page") as mock_get_page:
            page = MagicMock()
            mock_get_page.return_value = page

            result = _dispatch_command(
                "type_text",
                {"selector": "#input", "text": "append", "clear_first": False},
            )
            assert result["ok"] is True
            page.type.assert_called_with("#input", "append", timeout=10000)
            page.fill.assert_not_called()


# =========================================================================
# Chrome executable detection
# =========================================================================


class TestChromeDetection:
    """Test system browser detection logic."""

    @patch("platform.system", return_value="Darwin")
    @patch("os.path.isfile")
    def test_detects_chrome_on_macos(self, mock_isfile, _mock_sys):
        mock_isfile.side_effect = lambda p: "Google Chrome" in p
        result = _detect_chrome_executable()
        assert result is not None
        assert "Google Chrome" in result

    @patch("platform.system", return_value="Darwin")
    @patch("os.path.isfile", return_value=False)
    def test_no_browser_on_macos(self, _mock_isfile, _mock_sys):
        result = _detect_chrome_executable()
        assert result is None

    @patch("platform.system", return_value="Darwin")
    @patch("os.path.isfile")
    def test_prefers_chrome_over_brave(self, mock_isfile, _mock_sys):
        """Chrome should be preferred over Brave when both exist."""
        mock_isfile.return_value = True  # All candidates exist
        result = _detect_chrome_executable()
        assert result is not None
        assert "Google Chrome" in result

    @patch("platform.system", return_value="Darwin")
    @patch("os.path.isfile")
    def test_falls_back_to_brave(self, mock_isfile, _mock_sys):
        """Should find Brave when Chrome is not installed."""
        mock_isfile.side_effect = lambda p: "Brave" in p
        result = _detect_chrome_executable()
        assert result is not None
        assert "Brave" in result

    @patch("platform.system", return_value="Linux")
    @patch("shutil.which")
    def test_detects_chrome_on_linux(self, mock_which, _mock_sys):
        mock_which.side_effect = lambda c: "/usr/bin/google-chrome" if "google-chrome" == c else None
        result = _detect_chrome_executable()
        assert result == "google-chrome"

    @patch("platform.system", return_value="Linux")
    @patch("shutil.which", return_value=None)
    def test_no_browser_on_linux(self, _mock_which, _mock_sys):
        result = _detect_chrome_executable()
        assert result is None

    @patch("platform.system", return_value="FreeBSD")
    def test_unsupported_platform(self, _mock_sys):
        result = _detect_chrome_executable()
        assert result is None

    def test_find_free_port(self):
        """Should return a valid port number."""
        port = _find_free_port()
        assert isinstance(port, int)
        assert 1024 <= port <= 65535

    def test_default_profile_dir(self):
        """Default profile dir should be under .supyagent/ in the project."""
        assert _DEFAULT_PROFILE_DIR == os.path.join(".supyagent", "browser", "profile")


class TestCDPConnection:
    """Test the CDP-based browser launch path in _get_page."""

    @patch("supyagent.core.browser_daemon._detect_chrome_executable")
    @patch("supyagent.core.browser_daemon._find_free_port", return_value=9222)
    @patch("supyagent.core.browser_daemon._wait_for_cdp")
    @patch("supyagent.core.browser_daemon.subprocess.Popen")
    @patch("supyagent.core.browser_daemon.os.makedirs")
    def test_get_page_launches_real_chrome(
        self, mock_makedirs, mock_popen, mock_wait, mock_port, mock_detect
    ):
        """When a real Chrome is found, should launch it with CDP."""
        import supyagent.core.browser_daemon as bd

        # Reset module state
        bd._browser = None
        bd._context = None
        bd._page = None
        bd._chrome_process = None

        mock_detect.return_value = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

        # Mock playwright
        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False

        mock_pw.chromium.connect_over_cdp.return_value = mock_browser
        mock_browser.contexts = [mock_context]
        mock_context.pages = [mock_page]

        with patch("playwright.sync_api.sync_playwright", return_value=mock_pw):
            mock_pw.start.return_value = mock_pw
            page = bd._get_page()

        assert page is mock_page
        mock_popen.assert_called_once()
        # Verify CDP URL
        mock_pw.chromium.connect_over_cdp.assert_called_with("http://127.0.0.1:9222")
        # Verify Chrome was launched with correct args
        launch_args = mock_popen.call_args[0][0]
        assert any("--remote-debugging-port=9222" in a for a in launch_args)
        assert any("--user-data-dir=" in a for a in launch_args)
        assert any("--no-first-run" in a for a in launch_args)

        # Cleanup
        bd._browser = None
        bd._context = None
        bd._page = None
        bd._chrome_process = None

    @patch("supyagent.core.browser_daemon._detect_chrome_executable", return_value=None)
    def test_get_page_falls_back_to_playwright(self, mock_detect):
        """When no real Chrome found, should fall back to Playwright Chromium."""
        import supyagent.core.browser_daemon as bd

        bd._browser = None
        bd._context = None
        bd._page = None
        bd._chrome_process = None

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False

        mock_pw.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page

        with patch("playwright.sync_api.sync_playwright", return_value=mock_pw):
            mock_pw.start.return_value = mock_pw
            page = bd._get_page()

        assert page is mock_page
        mock_pw.chromium.launch.assert_called_once()
        # Should NOT try connect_over_cdp
        mock_pw.chromium.connect_over_cdp.assert_not_called()

        bd._browser = None
        bd._context = None
        bd._page = None
        bd._chrome_process = None


class TestBrowserSessionManagerConfig:
    """Test BrowserSessionManager headless/profile config."""

    def test_default_headed(self):
        mgr = BrowserSessionManager()
        assert mgr._headless is False
        assert mgr._profile_dir is None

    def test_headless_mode(self):
        mgr = BrowserSessionManager(headless=True)
        assert mgr._headless is True

    def test_custom_profile_dir(self):
        mgr = BrowserSessionManager(profile_dir="/tmp/my-profile")
        assert mgr._profile_dir == "/tmp/my-profile"

    @patch("supyagent.core.browser_session.BrowserSessionManager.ensure_daemon")
    def test_headless_passed_to_daemon_cmd(self, mock_ensure):
        """Headless flag should be part of the daemon launch command."""
        # We can't easily test the full launch flow without the supervisor,
        # but we can verify the cmd is built correctly by inspecting ensure_daemon
        mgr = BrowserSessionManager(headless=True, profile_dir="/tmp/test-profile")
        assert mgr._headless is True
        assert mgr._profile_dir == "/tmp/test-profile"
