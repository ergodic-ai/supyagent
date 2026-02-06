"""
Process Supervisor for managing tool and agent execution.

Provides non-blocking execution with timeout handling, auto-backgrounding,
and full lifecycle management for all external processes.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import os
import signal
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ProcessStatus(Enum):
    """Status of a managed process."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BACKGROUNDED = "backgrounded"
    KILLED = "killed"


class TimeoutAction(Enum):
    """What to do when a process times out."""

    BACKGROUND = "background"  # Keep running, return control
    KILL = "kill"  # Terminate the process
    WAIT = "wait"  # Keep waiting (use max_execution_time as hard limit)


@dataclass
class ManagedProcess:
    """Metadata for a supervised process."""

    process_id: str
    cmd: list[str]
    process: asyncio.subprocess.Process | None = None
    status: ProcessStatus = ProcessStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    log_file: Path | None = None
    process_type: str = "tool"  # "tool" or "agent"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output."""
        return {
            "process_id": self.process_id,
            "cmd": self.cmd,
            "pid": self.process.pid if self.process else None,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "exit_code": self.exit_code,
            "log_file": str(self.log_file) if self.log_file else None,
            "process_type": self.process_type,
            "metadata": self.metadata,
        }


@dataclass
class SupervisorConfig:
    """Configuration for the ProcessSupervisor."""

    # Timeout settings
    default_timeout: float = 30.0
    on_timeout: TimeoutAction = TimeoutAction.BACKGROUND
    max_execution_time: float = 300.0  # Hard limit

    # Parallelism settings
    max_parallel_tools: int = 5
    max_background_processes: int = 10

    # Pattern matching for forced modes
    force_background_patterns: list[str] = field(
        default_factory=lambda: [
            "server__*",
            "docker__run*",
            "*__serve*",
            "*__start_server*",
        ]
    )
    force_sync_patterns: list[str] = field(
        default_factory=lambda: [
            "files__read_file",
            "files__write_file",
        ]
    )

    # Logging
    log_dir: Path = field(
        default_factory=lambda: Path.home() / ".supyagent" / "process_logs"
    )
    log_retention_days: int = 7

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SupervisorConfig":
        """Create config from dictionary."""
        config = cls()

        if "default_timeout" in data:
            config.default_timeout = float(data["default_timeout"])
        if "on_timeout" in data:
            config.on_timeout = TimeoutAction(data["on_timeout"])
        if "max_execution_time" in data:
            config.max_execution_time = float(data["max_execution_time"])
        if "max_parallel_tools" in data:
            config.max_parallel_tools = int(data["max_parallel_tools"])
        if "max_background_processes" in data:
            config.max_background_processes = int(data["max_background_processes"])
        if "force_background_patterns" in data:
            config.force_background_patterns = data["force_background_patterns"]
        if "force_sync_patterns" in data:
            config.force_sync_patterns = data["force_sync_patterns"]
        if "log_dir" in data:
            config.log_dir = Path(data["log_dir"])

        return config


def _matches_pattern(name: str, pattern: str) -> bool:
    """Check if a tool name matches a glob-like pattern."""
    return fnmatch.fnmatch(name, pattern)


class ProcessSupervisor:
    """
    Unified supervisor for tool and agent execution.

    Features:
    - Non-blocking async execution
    - Automatic timeout handling with configurable action
    - Process lifecycle management (start, monitor, kill)
    - Parallel execution support
    - Background process tracking
    - Output logging and retrieval
    """

    def __init__(self, config: SupervisorConfig | None = None):
        """
        Initialize the supervisor.

        Args:
            config: Supervisor configuration (uses defaults if not provided)
        """
        self.config = config or SupervisorConfig()
        self._processes: dict[str, ManagedProcess] = {}
        self._lock = asyncio.Lock()
        self._process_counter = 0

        # Ensure log directory exists
        self.config.log_dir.mkdir(parents=True, exist_ok=True)

    def _generate_process_id(self, prefix: str = "proc") -> str:
        """Generate a unique process ID."""
        self._process_counter += 1
        timestamp = datetime.now(UTC).strftime("%H%M%S")
        return f"{prefix}_{timestamp}_{self._process_counter}"

    def _should_force_background(self, tool_name: str) -> bool:
        """Check if a tool should be forced to run in background."""
        return any(
            _matches_pattern(tool_name, pattern)
            for pattern in self.config.force_background_patterns
        )

    def _should_force_sync(self, tool_name: str) -> bool:
        """Check if a tool should be forced to run synchronously."""
        return any(
            _matches_pattern(tool_name, pattern)
            for pattern in self.config.force_sync_patterns
        )

    async def execute(
        self,
        cmd: list[str],
        *,
        process_type: str = "tool",
        tool_name: str | None = None,
        timeout: float | None = None,
        on_timeout: TimeoutAction | None = None,
        force_background: bool = False,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute a command with supervised lifecycle management.

        Args:
            cmd: Command and arguments to execute
            process_type: "tool" or "agent"
            tool_name: Name of the tool (for pattern matching)
            timeout: Seconds before timeout action (default: config.default_timeout)
            on_timeout: What to do on timeout (default: config.on_timeout)
            force_background: Force immediate background execution
            env: Additional environment variables
            cwd: Working directory
            metadata: Additional metadata to store with process

        Returns:
            Result dict with "ok", "data"/"error", and process info
        """
        # Determine execution mode
        effective_timeout = timeout if timeout is not None else self.config.default_timeout
        effective_on_timeout = on_timeout or self.config.on_timeout

        # Check pattern-based overrides
        if tool_name:
            if self._should_force_background(tool_name):
                force_background = True
            elif self._should_force_sync(tool_name):
                # Force sync: use max_execution_time as timeout, wait action
                effective_timeout = self.config.max_execution_time
                effective_on_timeout = TimeoutAction.WAIT

        # Check background process limit
        async with self._lock:
            bg_count = sum(
                1
                for p in self._processes.values()
                if p.status == ProcessStatus.BACKGROUNDED
            )
            if bg_count >= self.config.max_background_processes:
                # Clean up completed background processes
                await self._cleanup_completed()
                bg_count = sum(
                    1
                    for p in self._processes.values()
                    if p.status == ProcessStatus.BACKGROUNDED
                )
                if bg_count >= self.config.max_background_processes:
                    return {
                        "ok": False,
                        "error": f"Maximum background processes ({self.config.max_background_processes}) reached. "
                        "Use /processes to see running processes or /kill to terminate some.",
                    }

        # Create managed process entry
        process_id = self._generate_process_id(process_type)
        log_file = self.config.log_dir / f"{process_id}.log"

        managed = ManagedProcess(
            process_id=process_id,
            cmd=cmd,
            status=ProcessStatus.PENDING,
            log_file=log_file,
            process_type=process_type,
            metadata=metadata or {},
        )

        async with self._lock:
            self._processes[process_id] = managed

        try:
            return await self._run_process(
                managed,
                timeout=effective_timeout,
                on_timeout=effective_on_timeout,
                force_background=force_background,
                env=env,
                cwd=cwd,
            )
        except Exception as e:
            managed.status = ProcessStatus.FAILED
            managed.completed_at = datetime.now(UTC)
            logger.exception(f"Process {process_id} failed")
            return {"ok": False, "error": str(e)}

    async def _run_process(
        self,
        managed: ManagedProcess,
        timeout: float,
        on_timeout: TimeoutAction,
        force_background: bool,
        env: dict[str, str] | None,
        cwd: str | None,
    ) -> dict[str, Any]:
        """Internal: Run a process with timeout handling."""

        # Prepare environment
        process_env = os.environ.copy()
        if env:
            process_env.update(env)

        # Open log file
        log_handle = None
        if managed.log_file:
            managed.log_file.parent.mkdir(parents=True, exist_ok=True)
            log_handle = open(managed.log_file, "w")

        try:
            # Start the process
            managed.process = await asyncio.create_subprocess_exec(
                *managed.cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=process_env,
                cwd=cwd,
                start_new_session=True,  # Allows killing process group
            )
            managed.status = ProcessStatus.RUNNING
            managed.started_at = datetime.now(UTC)

            logger.debug(
                f"Started process {managed.process_id} (PID: {managed.process.pid})"
            )

            # If forced background, return immediately
            if force_background:
                managed.status = ProcessStatus.BACKGROUNDED

                # Start background output collection
                asyncio.create_task(
                    self._collect_background_output(managed, log_handle)
                )

                return {
                    "ok": True,
                    "data": {
                        "status": "backgrounded",
                        "process_id": managed.process_id,
                        "pid": managed.process.pid,
                        "message": "Process started in background",
                        "log_file": str(managed.log_file),
                    },
                }

            # Wait with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    managed.process.communicate(),
                    timeout=timeout,
                )

                managed.stdout = stdout.decode() if stdout else ""
                managed.stderr = stderr.decode() if stderr else ""
                managed.exit_code = managed.process.returncode
                managed.completed_at = datetime.now(UTC)

                # Write to log
                if log_handle:
                    log_handle.write(f"=== STDOUT ===\n{managed.stdout}\n")
                    log_handle.write(f"=== STDERR ===\n{managed.stderr}\n")
                    log_handle.write(f"=== EXIT CODE: {managed.exit_code} ===\n")

                if managed.exit_code == 0:
                    managed.status = ProcessStatus.COMPLETED
                    # Try to parse as JSON
                    try:
                        return json.loads(managed.stdout)
                    except json.JSONDecodeError:
                        return {"ok": True, "data": managed.stdout.strip()}
                else:
                    managed.status = ProcessStatus.FAILED
                    return {
                        "ok": False,
                        "error": managed.stderr
                        or managed.stdout
                        or f"Process exited with code {managed.exit_code}",
                    }

            except asyncio.TimeoutError:
                # Handle timeout based on action
                if on_timeout == TimeoutAction.KILL:
                    await self._kill_process(managed)
                    return {
                        "ok": False,
                        "error": f"Process killed after {timeout}s timeout",
                        "process_id": managed.process_id,
                    }

                elif on_timeout == TimeoutAction.BACKGROUND:
                    managed.status = ProcessStatus.BACKGROUNDED

                    # Start background output collection
                    asyncio.create_task(
                        self._collect_background_output(managed, log_handle)
                    )

                    return {
                        "ok": True,
                        "data": {
                            "status": "backgrounded",
                            "process_id": managed.process_id,
                            "pid": managed.process.pid,
                            "message": f"Process promoted to background after {timeout}s. "
                            f"Use /process {managed.process_id} to check status.",
                            "log_file": str(managed.log_file),
                        },
                    }

                elif on_timeout == TimeoutAction.WAIT:
                    # Continue waiting up to max_execution_time
                    remaining = self.config.max_execution_time - timeout
                    if remaining > 0:
                        try:
                            stdout, stderr = await asyncio.wait_for(
                                managed.process.communicate(),
                                timeout=remaining,
                            )
                            managed.stdout = stdout.decode() if stdout else ""
                            managed.stderr = stderr.decode() if stderr else ""
                            managed.exit_code = managed.process.returncode
                            managed.completed_at = datetime.now(UTC)
                            managed.status = (
                                ProcessStatus.COMPLETED
                                if managed.exit_code == 0
                                else ProcessStatus.FAILED
                            )

                            if log_handle:
                                log_handle.write(f"=== STDOUT ===\n{managed.stdout}\n")
                                log_handle.write(f"=== STDERR ===\n{managed.stderr}\n")
                                log_handle.write(
                                    f"=== EXIT CODE: {managed.exit_code} ===\n"
                                )

                            try:
                                return json.loads(managed.stdout)
                            except json.JSONDecodeError:
                                return {
                                    "ok": managed.exit_code == 0,
                                    "data": managed.stdout.strip(),
                                }

                        except asyncio.TimeoutError:
                            await self._kill_process(managed)
                            return {
                                "ok": False,
                                "error": f"Process killed after {self.config.max_execution_time}s hard timeout",
                            }
                    else:
                        await self._kill_process(managed)
                        return {
                            "ok": False,
                            "error": "Process killed after reaching max execution time",
                        }

                # Should not reach here
                return {"ok": False, "error": "Unexpected timeout handling state"}

        finally:
            if log_handle and managed.status not in (
                ProcessStatus.BACKGROUNDED,
                ProcessStatus.RUNNING,
            ):
                log_handle.close()

    async def _collect_background_output(
        self,
        managed: ManagedProcess,
        log_handle,
    ) -> None:
        """Collect output from a backgrounded process."""
        try:
            if managed.process is None:
                return

            stdout, stderr = await managed.process.communicate()
            managed.stdout = stdout.decode() if stdout else ""
            managed.stderr = stderr.decode() if stderr else ""
            managed.exit_code = managed.process.returncode
            managed.completed_at = datetime.now(UTC)

            # Don't overwrite KILLED status — the process may have exited
            # because we killed it, and _kill_process already set the status.
            if managed.status != ProcessStatus.KILLED:
                managed.status = (
                    ProcessStatus.COMPLETED
                    if managed.exit_code == 0
                    else ProcessStatus.FAILED
                )

            if log_handle:
                log_handle.write(f"=== STDOUT ===\n{managed.stdout}\n")
                log_handle.write(f"=== STDERR ===\n{managed.stderr}\n")
                log_handle.write(f"=== EXIT CODE: {managed.exit_code} ===\n")
                log_handle.close()

            logger.debug(
                f"Background process {managed.process_id} completed with code {managed.exit_code}"
            )

        except Exception as e:
            # Don't overwrite KILLED status on error either
            if managed.status != ProcessStatus.KILLED:
                managed.status = ProcessStatus.FAILED
            managed.stderr = str(e)
            managed.completed_at = datetime.now(UTC)
            if log_handle:
                log_handle.write(f"=== ERROR ===\n{e}\n")
                log_handle.close()
            logger.exception(f"Background process {managed.process_id} failed")

    async def _kill_process(self, managed: ManagedProcess) -> None:
        """Kill a managed process."""
        if managed.process and managed.process.returncode is None:
            try:
                # Kill the entire process group
                pgid = os.getpgid(managed.process.pid)
                os.killpg(pgid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(managed.process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                # Fallback to just the process
                try:
                    managed.process.kill()
                except ProcessLookupError:
                    pass

        managed.status = ProcessStatus.KILLED
        managed.completed_at = datetime.now(UTC)
        logger.debug(f"Killed process {managed.process_id}")

    async def kill(self, process_id: str) -> dict[str, Any]:
        """
        Kill a managed process by ID.

        Args:
            process_id: The process ID to kill

        Returns:
            Result dict
        """
        managed = self._processes.get(process_id)
        if not managed:
            return {"ok": False, "error": f"Process {process_id} not found"}

        if managed.status not in (ProcessStatus.RUNNING, ProcessStatus.BACKGROUNDED):
            return {
                "ok": False,
                "error": f"Process {process_id} is not running (status: {managed.status.value})",
            }

        await self._kill_process(managed)
        return {"ok": True, "data": f"Process {process_id} killed"}

    def list_processes(self, include_completed: bool = False) -> list[dict[str, Any]]:
        """
        List all managed processes.

        Args:
            include_completed: Include completed/failed/killed processes

        Returns:
            List of process info dicts
        """
        result = []
        for managed in self._processes.values():
            if include_completed or managed.status in (
                ProcessStatus.RUNNING,
                ProcessStatus.BACKGROUNDED,
                ProcessStatus.PENDING,
            ):
                result.append(managed.to_dict())
        return result

    def get_process(self, process_id: str) -> dict[str, Any] | None:
        """Get info about a specific process."""
        managed = self._processes.get(process_id)
        if managed:
            return managed.to_dict()
        return None

    async def get_output(self, process_id: str, tail: int = 100) -> dict[str, Any]:
        """
        Get output from a process (from log file).

        Args:
            process_id: Process ID
            tail: Number of lines to return (from end)

        Returns:
            Result dict with stdout/stderr
        """
        managed = self._processes.get(process_id)
        if not managed:
            return {"ok": False, "error": f"Process {process_id} not found"}

        if not managed.log_file or not managed.log_file.exists():
            return {
                "ok": True,
                "data": {"stdout": managed.stdout, "stderr": managed.stderr},
            }

        try:
            with open(managed.log_file) as f:
                lines = f.readlines()
                if tail:
                    lines = lines[-tail:]
                return {"ok": True, "data": {"output": "".join(lines)}}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def _cleanup_completed(self) -> int:
        """Remove completed processes from tracking."""
        to_remove = [
            pid
            for pid, p in self._processes.items()
            if p.status
            in (ProcessStatus.COMPLETED, ProcessStatus.FAILED, ProcessStatus.KILLED)
        ]
        for pid in to_remove:
            del self._processes[pid]
        return len(to_remove)

    async def cleanup_old_logs(self) -> int:
        """Remove old log files based on retention policy."""
        import time

        cutoff = time.time() - (self.config.log_retention_days * 86400)
        removed = 0

        for log_file in self.config.log_dir.glob("*.log"):
            try:
                if log_file.stat().st_mtime < cutoff:
                    log_file.unlink()
                    removed += 1
            except OSError:
                pass

        return removed


# Global supervisor instance (lazy initialized)
_supervisor: ProcessSupervisor | None = None
_supervisor_loop: asyncio.AbstractEventLoop | None = None
_supervisor_thread: "threading.Thread | None" = None


def _get_supervisor_loop() -> asyncio.AbstractEventLoop:
    """Get or create the persistent event loop for the supervisor."""
    import threading

    global _supervisor_loop, _supervisor_thread

    if _supervisor_loop is not None and not _supervisor_loop.is_closed():
        return _supervisor_loop

    _supervisor_loop = asyncio.new_event_loop()

    def _run_loop():
        asyncio.set_event_loop(_supervisor_loop)
        _supervisor_loop.run_forever()

    _supervisor_thread = threading.Thread(
        target=_run_loop, daemon=True, name="supervisor-loop"
    )
    _supervisor_thread.start()

    return _supervisor_loop


def run_supervisor_coroutine(coro) -> Any:
    """
    Run an async coroutine on the supervisor's persistent event loop.

    This avoids creating/destroying event loops and prevents
    'Event loop is closed' errors with backgrounded processes.
    """

    loop = _get_supervisor_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def get_supervisor(config: SupervisorConfig | None = None) -> ProcessSupervisor:
    """Get or create the global supervisor instance."""
    global _supervisor
    if _supervisor is None:
        # Ensure the loop is running before creating the supervisor
        _get_supervisor_loop()
        _supervisor = ProcessSupervisor(config)
    return _supervisor


def reset_supervisor() -> None:
    """Reset the global supervisor (mainly for testing)."""
    global _supervisor, _supervisor_loop, _supervisor_thread

    _supervisor = None

    if _supervisor_loop is not None and not _supervisor_loop.is_closed():
        _supervisor_loop.call_soon_threadsafe(_supervisor_loop.stop)
        if _supervisor_thread is not None:
            _supervisor_thread.join(timeout=2)
        _supervisor_loop.close()

    _supervisor_loop = None
    _supervisor_thread = None
