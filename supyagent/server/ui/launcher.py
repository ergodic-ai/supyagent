"""
UILauncher — start an ephemeral local server, open the browser, wait for done.
"""

from __future__ import annotations

import socket
import threading
import webbrowser
from typing import Any

import uvicorn

from supyagent.server.ui import create_ui_app


def find_free_port() -> int:
    """Bind to port 0 to get an OS-assigned free port, then release it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class UILauncher:
    """
    Launches a browser UI for a given mode ("models" or "hello").

    Usage::

        launcher = UILauncher("models")
        result = launcher.run()   # blocks until user clicks Done or Ctrl+C
    """

    def __init__(self, mode: str):
        self.mode = mode
        self.result: dict[str, Any] = {}
        self._done = threading.Event()
        self._server: uvicorn.Server | None = None

    def _on_done(self, data: dict[str, Any]) -> None:
        """Callback from POST /api/done."""
        self.result = data
        self._done.set()

    def run(self) -> dict[str, Any]:
        """Start server, open browser, block until done. Returns result dict."""
        port = find_free_port()
        app = create_ui_app(mode=self.mode, done_callback=self._on_done)

        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)

        # Run uvicorn in a daemon thread
        thread = threading.Thread(target=self._server.run, daemon=True)
        thread.start()

        # Wait for server to be ready
        while not self._server.started:
            self._done.wait(timeout=0.05)
            if self._done.is_set():
                break

        url = f"http://127.0.0.1:{port}/{self.mode}"

        try:
            webbrowser.open(url)
        except Exception:
            pass

        from rich.console import Console

        console = Console(stderr=True)
        console.print(f"  [cyan]Browser UI[/cyan] opened at [link={url}]{url}[/link]")
        console.print("  [grey62]Press Ctrl+C to close.[/grey62]")

        try:
            self._done.wait()
        except KeyboardInterrupt:
            console.print("\n  [grey62]Shutting down...[/grey62]")

        self._server.should_exit = True
        thread.join(timeout=3)

        return self.result
