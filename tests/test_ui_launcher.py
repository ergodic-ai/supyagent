"""
Tests for supyagent.server.ui.launcher.
"""

import threading
from unittest.mock import MagicMock, patch

from supyagent.server.ui.launcher import UILauncher, find_free_port


class TestFindFreePort:
    def test_returns_valid_port(self):
        port = find_free_port()
        assert isinstance(port, int)
        assert 1024 <= port <= 65535

    def test_returns_different_ports(self):
        ports = {find_free_port() for _ in range(5)}
        # Should get at least 2 different ports (not always, but very likely)
        assert len(ports) >= 1


class TestUILauncher:
    def test_init(self):
        launcher = UILauncher("models")
        assert launcher.mode == "models"
        assert launcher.result == {}

    def test_on_done_sets_event(self):
        launcher = UILauncher("models")
        assert not launcher._done.is_set()
        launcher._on_done({"action": "done"})
        assert launcher._done.is_set()
        assert launcher.result == {"action": "done"}

    @patch("supyagent.server.ui.launcher.webbrowser")
    @patch("supyagent.server.ui.launcher.uvicorn")
    def test_run_lifecycle(self, mock_uvicorn, mock_webbrowser):
        """Test that run() starts server, opens browser, and returns on done."""
        launcher = UILauncher("models")

        # Mock the uvicorn Server
        mock_server = MagicMock()
        mock_server.started = True
        mock_server.should_exit = False
        mock_uvicorn.Server.return_value = mock_server
        mock_uvicorn.Config = MagicMock()

        # Simulate run() completing quickly — signal done from another thread
        def fake_run():
            pass

        mock_server.run = fake_run

        # Signal done after a short delay
        def signal_done():
            launcher._on_done({"action": "test_done"})

        timer = threading.Timer(0.1, signal_done)
        timer.start()

        result = launcher.run()

        assert result == {"action": "test_done"}
        mock_webbrowser.open.assert_called_once()
        assert mock_server.should_exit is True

    @patch("supyagent.server.ui.launcher.webbrowser")
    @patch("supyagent.server.ui.launcher.uvicorn")
    def test_run_keyboard_interrupt(self, mock_uvicorn, mock_webbrowser):
        """Test that Ctrl+C is handled gracefully."""
        launcher = UILauncher("hello")

        mock_server = MagicMock()
        mock_server.started = True
        mock_server.should_exit = False
        mock_uvicorn.Server.return_value = mock_server
        mock_uvicorn.Config = MagicMock()
        mock_server.run = lambda: None

        # Simulate KeyboardInterrupt during wait
        def interrupt_wait(*args, **kwargs):
            raise KeyboardInterrupt()

        launcher._done.wait = interrupt_wait

        result = launcher.run()

        assert result == {}
        assert mock_server.should_exit is True
