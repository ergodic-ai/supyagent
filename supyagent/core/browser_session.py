"""
Browser session manager for persistent browser interactions.

Bridges the agent engine's tool dispatch to a long-running browser daemon
process. The daemon is started lazily on first use and managed by the
ProcessSupervisor.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# How long to wait for the daemon to start and write its port file
_DAEMON_STARTUP_TIMEOUT = 15.0
_DAEMON_STARTUP_POLL = 0.2

# How long to wait for a single command to execute
_COMMAND_TIMEOUT = 120.0


class BrowserSessionManager:
    """
    Manages a persistent browser daemon for stateful browsing.

    The daemon is started lazily via the ProcessSupervisor on the first
    browser tool call. Subsequent calls reuse the same daemon.
    """

    def __init__(self, headless: bool = False, profile_dir: str | None = None):
        self._daemon_process_id: str | None = None
        self._daemon_port: int | None = None
        self._port_file: Path | None = None
        self._headless = headless
        self._profile_dir = profile_dir

    def ensure_daemon(self) -> None:
        """Start the browser daemon if not already running."""
        if self._daemon_port is not None:
            # Check if daemon is still responsive
            if self._health_check():
                return
            # Daemon died — reset and restart
            logger.info("Browser daemon no longer responsive, restarting")
            self._daemon_port = None
            self._daemon_process_id = None

        from supyagent.core.supervisor import get_supervisor, run_supervisor_coroutine

        supervisor = get_supervisor()

        # Create a port file under .supyagent/tmp/
        tmp_dir = Path(".supyagent", "tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        self._port_file = tmp_dir / "browser_daemon.port"
        self._port_file.write_text("")

        # Build command
        cmd = [
            sys.executable,
            "-m",
            "supyagent.core.browser_daemon",
            "--port",
            "0",
            "--port-file",
            str(self._port_file),
        ]
        if self._headless:
            cmd.append("--headless")
        if self._profile_dir:
            cmd.extend(["--profile-dir", self._profile_dir])

        # Start daemon as a backgrounded process
        result = run_supervisor_coroutine(
            supervisor.execute(
                cmd,
                process_type="daemon",
                tool_name="browser_daemon",
                force_background=True,
                metadata={"purpose": "persistent_browser"},
            )
        )

        if not result.get("ok"):
            raise RuntimeError(f"Failed to start browser daemon: {result.get('error')}")

        self._daemon_process_id = result.get("data", {}).get("process_id")

        # Wait for daemon to write its port
        deadline = time.monotonic() + _DAEMON_STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            try:
                content = self._port_file.read_text().strip()
                if content:
                    self._daemon_port = int(content)
                    logger.info(f"Browser daemon started on port {self._daemon_port}")
                    return
            except (FileNotFoundError, ValueError):
                pass
            time.sleep(_DAEMON_STARTUP_POLL)

        raise RuntimeError(
            f"Browser daemon did not start within {_DAEMON_STARTUP_TIMEOUT}s"
        )

    def execute(self, command: str, args: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a browser command on the persistent daemon.

        Args:
            command: Command name (browse, click, type_text, etc.)
            args: Command arguments

        Returns:
            Result dict with ok/data or ok/error
        """
        self.ensure_daemon()

        import httpx

        try:
            response = httpx.post(
                f"http://127.0.0.1:{self._daemon_port}/execute",
                json={"command": command, "args": args},
                timeout=_COMMAND_TIMEOUT,
            )
            return response.json()
        except httpx.ConnectError:
            # Daemon may have crashed — reset for next call
            logger.warning("Browser daemon connection failed, will restart on next call")
            self._daemon_port = None
            self._daemon_process_id = None
            return {"ok": False, "error": "Browser daemon not responding. It will restart on next call."}
        except httpx.TimeoutException:
            return {"ok": False, "error": f"Browser command '{command}' timed out after {_COMMAND_TIMEOUT}s"}
        except Exception as e:
            return {"ok": False, "error": f"Browser daemon error: {e}"}

    def shutdown(self) -> None:
        """Stop the browser daemon."""
        if self._daemon_process_id:
            try:
                from supyagent.core.supervisor import get_supervisor, run_supervisor_coroutine

                supervisor = get_supervisor()
                run_supervisor_coroutine(supervisor.kill(self._daemon_process_id))
            except Exception:
                logger.debug("Failed to kill browser daemon", exc_info=True)

        self._daemon_port = None
        self._daemon_process_id = None

        # Clean up port file
        if self._port_file:
            try:
                self._port_file.unlink(missing_ok=True)
            except Exception:
                pass
            self._port_file = None

    def _health_check(self) -> bool:
        """Check if the daemon is still responding."""
        if self._daemon_port is None:
            return False

        import httpx

        try:
            response = httpx.get(
                f"http://127.0.0.1:{self._daemon_port}/health",
                timeout=2.0,
            )
            return response.status_code == 200
        except Exception:
            return False
