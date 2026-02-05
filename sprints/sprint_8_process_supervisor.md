# Sprint 8: Process Supervisor Architecture

**Goal**: Implement robust, non-blocking execution for tools and child agents with unified process management.

**Duration**: ~5-6 days

**Depends on**: Sprint 5 (Multi-Agent), Sprint 7 (Context Management)

---

## Problem Statement

Currently, both tool execution and agent delegation are **blocking and synchronous**:

1. **Tool execution** uses `subprocess.run()` — blocks until tool completes (up to 5 min timeout)
2. **Agent delegation** creates child agents **in-process** — parent blocks waiting for child
3. **No parallelism** — tools/agents run sequentially, even when independent
4. **No recovery** — if a tool hangs, the whole agent is stuck
5. **No visibility** — can't see what's running or interrupt long operations

**Solution**: A unified `ProcessSupervisor` that manages all external work (tools and agents) as subprocesses with timeout handling, auto-backgrounding, and full lifecycle control.

---

## Design Principles

### User Experience First

The complexity should be **hidden by default**. Users running `supyagent init` or `supyagent chat` should not need to understand process management.

```bash
# This should just work™
supyagent init myproject
supyagent chat assistant

# Power users can tune settings
supyagent chat assistant --supervisor.default_timeout=60
```

### Sensible Defaults

| Setting | Default | Rationale |
|---------|---------|-----------|
| `default_timeout` | 30s | Long enough for most tools, short enough to not feel stuck |
| `on_timeout` | `"background"` | Don't kill work, promote it and continue |
| `max_parallel_tools` | 5 | Balance between speed and resource usage |
| `max_background_processes` | 10 | Prevent resource exhaustion |

### Progressive Disclosure

1. **Level 0 (Beginner)**: Everything automatic, no config needed
2. **Level 1 (Intermediate)**: Tune timeouts in agent YAML
3. **Level 2 (Advanced)**: Force-background patterns, per-tool settings
4. **Level 3 (Expert)**: Custom supervisor config, async orchestration

---

## Deliverables

### 8.1 ProcessSupervisor Core

The heart of the new architecture:

```python
# supyagent/core/supervisor.py
"""
Process Supervisor for managing tool and agent execution.

Provides non-blocking execution with timeout handling, auto-backgrounding,
and full lifecycle management for all external processes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from pathlib import Path
from typing import Any, Callable

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
    KILL = "kill"              # Terminate the process
    WAIT = "wait"              # Keep waiting (use max_execution_time as hard limit)


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
    force_background_patterns: list[str] = field(default_factory=lambda: [
        "server__*",
        "docker__run*",
        "*__serve*",
        "*__start_server*",
    ])
    force_sync_patterns: list[str] = field(default_factory=lambda: [
        "files__read_file",
        "files__write_file",
    ])
    
    # Logging
    log_dir: Path = field(default_factory=lambda: Path.home() / ".supyagent" / "process_logs")
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
    import fnmatch
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
        effective_timeout = timeout or self.config.default_timeout
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
                1 for p in self._processes.values()
                if p.status == ProcessStatus.BACKGROUNDED
            )
            if bg_count >= self.config.max_background_processes:
                # Clean up completed background processes
                await self._cleanup_completed()
                bg_count = sum(
                    1 for p in self._processes.values()
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
        log_handle = open(managed.log_file, "w") if managed.log_file else None
        
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
            
            # If forced background, return immediately
            if force_background:
                managed.status = ProcessStatus.BACKGROUNDED
                return {
                    "ok": True,
                    "data": {
                        "status": "backgrounded",
                        "process_id": managed.process_id,
                        "pid": managed.process.pid,
                        "message": "Process started in background",
                        "log_file": str(managed.log_file),
                    }
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
                        return {"ok": True, "data": managed.stdout}
                else:
                    managed.status = ProcessStatus.FAILED
                    return {
                        "ok": False,
                        "error": managed.stderr or managed.stdout or f"Process exited with code {managed.exit_code}",
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
                        }
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
                            managed.status = ProcessStatus.COMPLETED if managed.exit_code == 0 else ProcessStatus.FAILED
                            
                            try:
                                return json.loads(managed.stdout)
                            except json.JSONDecodeError:
                                return {"ok": managed.exit_code == 0, "data": managed.stdout}
                                
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
                            "error": f"Process killed after reaching max execution time",
                        }
                        
        finally:
            if log_handle and managed.status not in (ProcessStatus.BACKGROUNDED, ProcessStatus.RUNNING):
                log_handle.close()
    
    async def _collect_background_output(
        self,
        managed: ManagedProcess,
        log_handle,
    ) -> None:
        """Collect output from a backgrounded process."""
        try:
            stdout, stderr = await managed.process.communicate()
            managed.stdout = stdout.decode() if stdout else ""
            managed.stderr = stderr.decode() if stderr else ""
            managed.exit_code = managed.process.returncode
            managed.completed_at = datetime.now(UTC)
            managed.status = ProcessStatus.COMPLETED if managed.exit_code == 0 else ProcessStatus.FAILED
            
            if log_handle:
                log_handle.write(f"=== STDOUT ===\n{managed.stdout}\n")
                log_handle.write(f"=== STDERR ===\n{managed.stderr}\n")
                log_handle.write(f"=== EXIT CODE: {managed.exit_code} ===\n")
                log_handle.close()
                
        except Exception as e:
            managed.status = ProcessStatus.FAILED
            managed.stderr = str(e)
            if log_handle:
                log_handle.write(f"=== ERROR ===\n{e}\n")
                log_handle.close()
    
    async def _kill_process(self, managed: ManagedProcess) -> None:
        """Kill a managed process."""
        if managed.process and managed.process.returncode is None:
            try:
                # Kill the entire process group
                os.killpg(os.getpgid(managed.process.pid), signal.SIGTERM)
                try:
                    await asyncio.wait_for(managed.process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    os.killpg(os.getpgid(managed.process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception:
                # Fallback to just the process
                managed.process.kill()
        
        managed.status = ProcessStatus.KILLED
        managed.completed_at = datetime.now(UTC)
    
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
            return {"ok": False, "error": f"Process {process_id} is not running (status: {managed.status.value})"}
        
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
            return {"ok": True, "data": {"stdout": managed.stdout, "stderr": managed.stderr}}
        
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
            pid for pid, p in self._processes.items()
            if p.status in (ProcessStatus.COMPLETED, ProcessStatus.FAILED, ProcessStatus.KILLED)
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
            if log_file.stat().st_mtime < cutoff:
                log_file.unlink()
                removed += 1
        
        return removed


# Global supervisor instance (lazy initialized)
_supervisor: ProcessSupervisor | None = None


def get_supervisor(config: SupervisorConfig | None = None) -> ProcessSupervisor:
    """Get or create the global supervisor instance."""
    global _supervisor
    if _supervisor is None:
        _supervisor = ProcessSupervisor(config)
    return _supervisor
```

**Tests for 8.1:**

```python
# tests/test_supervisor.py
import pytest
import asyncio
from pathlib import Path
from supyagent.core.supervisor import (
    ProcessSupervisor,
    SupervisorConfig,
    ProcessStatus,
    TimeoutAction,
)


@pytest.fixture
def supervisor(tmp_path):
    """Create a supervisor with temp log directory."""
    config = SupervisorConfig(
        default_timeout=2.0,
        max_execution_time=5.0,
        log_dir=tmp_path / "logs",
    )
    return ProcessSupervisor(config)


@pytest.mark.asyncio
async def test_execute_quick_command(supervisor):
    """Test executing a command that completes quickly."""
    result = await supervisor.execute(["echo", "hello"])
    assert result["ok"] is True
    assert "hello" in result.get("data", "")


@pytest.mark.asyncio
async def test_execute_with_timeout_background(supervisor):
    """Test that slow commands get backgrounded."""
    supervisor.config.default_timeout = 0.5
    supervisor.config.on_timeout = TimeoutAction.BACKGROUND
    
    result = await supervisor.execute(["sleep", "10"])
    
    assert result["ok"] is True
    assert result["data"]["status"] == "backgrounded"
    assert "process_id" in result["data"]
    
    # Kill the background process
    await supervisor.kill(result["data"]["process_id"])


@pytest.mark.asyncio
async def test_execute_with_timeout_kill(supervisor):
    """Test that slow commands get killed when configured."""
    supervisor.config.default_timeout = 0.5
    supervisor.config.on_timeout = TimeoutAction.KILL
    
    result = await supervisor.execute(["sleep", "10"])
    
    assert result["ok"] is False
    assert "killed" in result["error"].lower()


@pytest.mark.asyncio
async def test_force_background_patterns(supervisor):
    """Test pattern-based force background."""
    supervisor.config.force_background_patterns = ["*__serve*"]
    
    result = await supervisor.execute(
        ["sleep", "10"],
        tool_name="server__serve_http",
    )
    
    assert result["ok"] is True
    assert result["data"]["status"] == "backgrounded"
    
    await supervisor.kill(result["data"]["process_id"])


@pytest.mark.asyncio
async def test_list_processes(supervisor):
    """Test listing running processes."""
    # Start a background process
    result = await supervisor.execute(
        ["sleep", "10"],
        force_background=True,
    )
    process_id = result["data"]["process_id"]
    
    processes = supervisor.list_processes()
    assert len(processes) == 1
    assert processes[0]["process_id"] == process_id
    assert processes[0]["status"] == "backgrounded"
    
    await supervisor.kill(process_id)


@pytest.mark.asyncio
async def test_kill_process(supervisor):
    """Test killing a background process."""
    result = await supervisor.execute(
        ["sleep", "100"],
        force_background=True,
    )
    process_id = result["data"]["process_id"]
    
    kill_result = await supervisor.kill(process_id)
    assert kill_result["ok"] is True
    
    proc = supervisor.get_process(process_id)
    assert proc["status"] == "killed"


@pytest.mark.asyncio
async def test_max_background_limit(supervisor):
    """Test that max background process limit is enforced."""
    supervisor.config.max_background_processes = 2
    
    # Start 2 background processes
    p1 = await supervisor.execute(["sleep", "100"], force_background=True)
    p2 = await supervisor.execute(["sleep", "100"], force_background=True)
    
    # Third should fail
    p3 = await supervisor.execute(["sleep", "100"], force_background=True)
    assert p3["ok"] is False
    assert "maximum" in p3["error"].lower()
    
    # Cleanup
    await supervisor.kill(p1["data"]["process_id"])
    await supervisor.kill(p2["data"]["process_id"])


@pytest.mark.asyncio
async def test_json_output_parsing(supervisor):
    """Test that JSON output is parsed correctly."""
    result = await supervisor.execute(
        ["echo", '{"ok": true, "data": "test"}'],
    )
    
    assert result["ok"] is True
    assert result["data"] == "test"
```

---

### 8.2 Async Tool Execution

Update `tools.py` to use the supervisor:

```python
# supyagent/core/tools.py (updated execute_tool function)

import asyncio
from supyagent.core.supervisor import get_supervisor, TimeoutAction


async def execute_tool_async(
    script: str,
    func: str,
    args: dict[str, Any],
    secrets: dict[str, str] | None = None,
    timeout: float | None = None,
    background: bool = False,
) -> dict[str, Any]:
    """
    Execute a supypowers function using the process supervisor.
    
    Args:
        script: Script name (e.g., 'web_search')
        func: Function name (e.g., 'search')
        args: Function arguments as a dict
        secrets: Optional secrets to pass as environment variables
        timeout: Override default timeout
        background: Force background execution
    
    Returns:
        Result dict with 'ok' and 'data' or 'error'
    """
    tool_name = f"{script}__{func}"
    cmd = ["supypowers", "run", f"{script}:{func}", json.dumps(args)]
    
    # Add secrets to command
    if secrets:
        for key, value in secrets.items():
            cmd.extend(["--secrets", f"{key}={value}"])
    
    supervisor = get_supervisor()
    
    return await supervisor.execute(
        cmd,
        process_type="tool",
        tool_name=tool_name,
        timeout=timeout,
        force_background=background,
        metadata={"script": script, "func": func, "args": args},
    )


def execute_tool(
    script: str,
    func: str,
    args: dict[str, Any],
    secrets: dict[str, str] | None = None,
    timeout: float | None = None,
    background: bool = False,
) -> dict[str, Any]:
    """
    Synchronous wrapper for execute_tool_async.
    
    For backwards compatibility with existing code.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're already in an async context, create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    execute_tool_async(script, func, args, secrets, timeout, background)
                )
                return future.result()
        else:
            return loop.run_until_complete(
                execute_tool_async(script, func, args, secrets, timeout, background)
            )
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(
            execute_tool_async(script, func, args, secrets, timeout, background)
        )
```

**Tests for 8.2:**

```python
# tests/test_tools_supervisor.py
import pytest
from unittest.mock import patch, AsyncMock
from supyagent.core.tools import execute_tool, execute_tool_async


@pytest.mark.asyncio
async def test_execute_tool_async_basic():
    """Test basic async tool execution."""
    with patch('supyagent.core.tools.get_supervisor') as mock_get:
        mock_supervisor = AsyncMock()
        mock_supervisor.execute.return_value = {"ok": True, "data": "result"}
        mock_get.return_value = mock_supervisor
        
        result = await execute_tool_async("files", "read_file", {"path": "/tmp/test"})
        
        assert result["ok"] is True
        mock_supervisor.execute.assert_called_once()


def test_execute_tool_sync_wrapper():
    """Test synchronous wrapper works."""
    with patch('supyagent.core.tools.get_supervisor') as mock_get:
        mock_supervisor = AsyncMock()
        mock_supervisor.execute.return_value = {"ok": True, "data": "sync result"}
        mock_get.return_value = mock_supervisor
        
        result = execute_tool("files", "read_file", {"path": "/tmp/test"})
        
        assert result["ok"] is True
        assert result["data"] == "sync result"


@pytest.mark.asyncio
async def test_execute_tool_with_background():
    """Test forcing background execution."""
    with patch('supyagent.core.tools.get_supervisor') as mock_get:
        mock_supervisor = AsyncMock()
        mock_supervisor.execute.return_value = {
            "ok": True,
            "data": {"status": "backgrounded", "process_id": "test123"}
        }
        mock_get.return_value = mock_supervisor
        
        result = await execute_tool_async(
            "server", "start", {"port": 8000},
            background=True
        )
        
        assert result["data"]["status"] == "backgrounded"
        
        # Verify force_background was passed
        call_kwargs = mock_supervisor.execute.call_args.kwargs
        assert call_kwargs["force_background"] is True
```

---

### 8.3 CLI Command for Agent Subprocess Execution

Add `supyagent exec` command for running agents as subprocesses:

```python
# supyagent/cli/main.py (additions)

@cli.command()
@click.argument("agent_name")
@click.option("--task", "-t", required=True, help="Task for the agent to perform")
@click.option("--context", "-c", default="{}", help="JSON context from parent agent")
@click.option("--output", "-o", type=click.Choice(["json", "text"]), default="json")
@click.option("--timeout", type=float, default=300, help="Max execution time in seconds")
def exec(agent_name: str, task: str, context: str, output: str, timeout: float):
    """
    Execute an agent as a subprocess (used by delegation).
    
    This command is primarily used internally when a parent agent
    delegates to a child agent via the ProcessSupervisor.
    
    Example:
        supyagent exec researcher --task "Find papers on AI" --output json
    """
    import json as json_module
    import sys
    
    try:
        context_dict = json_module.loads(context)
    except json_module.JSONDecodeError:
        if output == "json":
            click.echo(json_module.dumps({"ok": False, "error": "Invalid context JSON"}))
        else:
            click.echo("Error: Invalid context JSON", err=True)
        sys.exit(1)
    
    try:
        config = load_agent_config(agent_name)
    except AgentNotFoundError:
        if output == "json":
            click.echo(json_module.dumps({"ok": False, "error": f"Agent '{agent_name}' not found"}))
        else:
            click.echo(f"Error: Agent '{agent_name}' not found", err=True)
        sys.exit(1)
    
    # Build full task with context
    if context_dict:
        full_task = _build_task_with_context(task, context_dict)
    else:
        full_task = task
    
    # Run the agent
    try:
        if config.type == "execution":
            from supyagent.core.executor import ExecutionRunner
            runner = ExecutionRunner(config)
            result = runner.run(full_task, output_format="json")
        else:
            from supyagent.core.agent import Agent
            agent = Agent(config)
            response = agent.send_message(full_task)
            result = {"ok": True, "data": response}
        
        if output == "json":
            click.echo(json_module.dumps(result))
        else:
            if result.get("ok"):
                click.echo(result.get("data", ""))
            else:
                click.echo(f"Error: {result.get('error', 'Unknown error')}", err=True)
                sys.exit(1)
                
    except Exception as e:
        if output == "json":
            click.echo(json_module.dumps({"ok": False, "error": str(e)}))
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _build_task_with_context(task: str, context: dict) -> str:
    """Build task string with context from parent agent."""
    parts = []
    
    if context.get("parent_agent"):
        parts.append(f"You are being called by the '{context['parent_agent']}' agent.")
    
    if context.get("parent_task"):
        parts.append(f"Parent's current task: {context['parent_task']}")
    
    if context.get("conversation_summary"):
        parts.append(f"\nConversation context:\n{context['conversation_summary']}")
    
    if context.get("relevant_facts"):
        parts.append("\nRelevant information:")
        for fact in context["relevant_facts"]:
            parts.append(f"- {fact}")
    
    if parts:
        parts.append(f"\n---\n\nYour task:\n{task}")
        return "\n".join(parts)
    
    return task
```

**Tests for 8.3:**

```python
# tests/test_cli_exec.py
import pytest
from click.testing import CliRunner
from supyagent.cli.main import cli
import json


@pytest.fixture
def runner():
    return CliRunner()


def test_exec_basic(runner, tmp_path, monkeypatch):
    """Test basic exec command."""
    # Create a minimal agent config
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "test_agent.yaml").write_text("""
name: test_agent
description: Test agent
type: execution
model:
  provider: openai/gpt-4o-mini
system_prompt: You are a test agent.
""")
    
    monkeypatch.chdir(tmp_path)
    
    with patch('supyagent.core.executor.ExecutionRunner') as mock_runner:
        mock_instance = mock_runner.return_value
        mock_instance.run.return_value = {"ok": True, "data": "test response"}
        
        result = runner.invoke(cli, [
            "exec", "test_agent",
            "--task", "Do something",
            "--output", "json"
        ])
        
        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output["ok"] is True


def test_exec_with_context(runner, tmp_path, monkeypatch):
    """Test exec with parent context."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "test_agent.yaml").write_text("""
name: test_agent
type: execution
model:
  provider: openai/gpt-4o-mini
system_prompt: Test
""")
    
    monkeypatch.chdir(tmp_path)
    
    context = json.dumps({
        "parent_agent": "planner",
        "parent_task": "Build a website",
        "relevant_facts": ["User wants Python", "Deadline is Friday"]
    })
    
    with patch('supyagent.core.executor.ExecutionRunner') as mock_runner:
        mock_instance = mock_runner.return_value
        mock_instance.run.return_value = {"ok": True, "data": "done"}
        
        result = runner.invoke(cli, [
            "exec", "test_agent",
            "--task", "Write the code",
            "--context", context,
            "--output", "json"
        ])
        
        assert result.exit_code == 0
        
        # Verify context was included in task
        call_args = mock_instance.run.call_args
        task_sent = call_args[0][0]
        assert "planner" in task_sent
        assert "Build a website" in task_sent


def test_exec_agent_not_found(runner):
    """Test exec with non-existent agent."""
    result = runner.invoke(cli, [
        "exec", "nonexistent",
        "--task", "Do something",
        "--output", "json"
    ])
    
    output = json.loads(result.output)
    assert output["ok"] is False
    assert "not found" in output["error"]
```

---

### 8.4 Updated Delegation Manager

Update `DelegationManager` to use the supervisor for subprocess delegation:

```python
# supyagent/core/delegation.py (updated)

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from supyagent.core.context import DelegationContext, summarize_conversation
from supyagent.core.supervisor import get_supervisor, TimeoutAction
from supyagent.core.registry import AgentRegistry
from supyagent.models.agent_config import AgentNotFoundError, load_agent_config

if TYPE_CHECKING:
    from supyagent.core.agent import Agent


class DelegationManager:
    """
    Manages agent-to-agent delegation using the ProcessSupervisor.
    
    Child agents can run either:
    - As subprocesses (default): Full isolation, managed by supervisor
    - In-process: Shared memory, faster for quick tasks
    """

    def __init__(
        self,
        registry: AgentRegistry,
        parent_agent: "Agent",
        grandparent_instance_id: str | None = None,
    ):
        self.registry = registry
        self.parent = parent_agent
        self.parent_id = registry.register(parent_agent, parent_id=grandparent_instance_id)

    def get_delegation_tools(self) -> list[dict[str, Any]]:
        """Generate tool schemas for each delegatable agent."""
        tools: list[dict[str, Any]] = []

        for delegate_name in self.parent.config.delegates:
            try:
                delegate_config = load_agent_config(delegate_name)
            except AgentNotFoundError:
                continue

            tool = {
                "type": "function",
                "function": {
                    "name": f"delegate_to_{delegate_name}",
                    "description": (
                        f"Delegate a task to the {delegate_name} agent. "
                        f"{delegate_config.description}"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "The task to delegate to this agent",
                            },
                            "context": {
                                "type": "string",
                                "description": "Optional context from the current conversation",
                            },
                            "mode": {
                                "type": "string",
                                "enum": ["subprocess", "in_process"],
                                "default": "subprocess",
                                "description": (
                                    "Execution mode. 'subprocess' for isolation (default), "
                                    "'in_process' for faster execution of quick tasks"
                                ),
                            },
                            "background": {
                                "type": "boolean",
                                "default": False,
                                "description": (
                                    "If true, returns immediately without waiting for result. "
                                    "Use for long-running tasks you don't need to wait for."
                                ),
                            },
                            "timeout": {
                                "type": "integer",
                                "default": 300,
                                "description": "Max seconds to wait before backgrounding (subprocess mode)",
                            },
                        },
                        "required": ["task"],
                    },
                },
            }
            tools.append(tool)

        # Generic spawn tool
        if self.parent.config.delegates:
            tools.append({
                "type": "function",
                "function": {
                    "name": "spawn_agent",
                    "description": (
                        "Create and run a new agent instance. "
                        f"Available agents: {', '.join(self.parent.config.delegates)}"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "agent_type": {
                                "type": "string",
                                "enum": self.parent.config.delegates,
                            },
                            "task": {"type": "string"},
                            "background": {
                                "type": "boolean",
                                "default": False,
                                "description": "Run in background without waiting",
                            },
                        },
                        "required": ["agent_type", "task"],
                    },
                },
            })

        return tools

    def is_delegation_tool(self, tool_name: str) -> bool:
        """Check if a tool name is a delegation tool."""
        return tool_name.startswith("delegate_to_") or tool_name == "spawn_agent"

    def execute_delegation(self, tool_call: Any) -> dict[str, Any]:
        """Execute a delegation tool call."""
        name = tool_call.function.name

        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return {"ok": False, "error": "Invalid JSON in tool arguments"}

        if name == "spawn_agent":
            return self._delegate_task(
                args.get("agent_type", ""),
                args.get("task", ""),
                mode="subprocess",
                background=args.get("background", False),
            )

        if name.startswith("delegate_to_"):
            agent_name = name[len("delegate_to_"):]
            return self._delegate_task(
                agent_name,
                args.get("task", ""),
                extra_context=args.get("context"),
                mode=args.get("mode", "subprocess"),
                background=args.get("background", False),
                timeout=args.get("timeout", 300),
            )

        return {"ok": False, "error": f"Unknown delegation tool: {name}"}

    def _build_context(self, task: str, extra_context: str | None = None) -> DelegationContext:
        """Build context to pass to a delegate."""
        summary = None
        if hasattr(self.parent, "messages") and self.parent.messages:
            summary = summarize_conversation(self.parent.messages, self.parent.llm)

        context = DelegationContext(
            parent_agent=self.parent.config.name,
            parent_task=task,
            conversation_summary=summary,
        )

        if extra_context:
            context.relevant_facts.append(extra_context)

        return context

    def _delegate_task(
        self,
        agent_name: str,
        task: str,
        extra_context: str | None = None,
        mode: str = "subprocess",
        background: bool = False,
        timeout: float = 300,
    ) -> dict[str, Any]:
        """
        Delegate a task to another agent.
        
        Args:
            agent_name: Name of the agent to delegate to
            task: The task to perform
            extra_context: Optional additional context
            mode: "subprocess" (isolated) or "in_process" (shared memory)
            background: Return immediately without waiting
            timeout: Seconds before auto-backgrounding (subprocess mode)
        """
        if agent_name not in self.parent.config.delegates:
            return {
                "ok": False,
                "error": f"Agent '{agent_name}' is not in the delegates list",
            }

        try:
            config = load_agent_config(agent_name)
        except AgentNotFoundError:
            return {"ok": False, "error": f"Agent '{agent_name}' not found"}

        # Check delegation depth
        parent_depth = self.registry.get_depth(self.parent_id)
        if parent_depth >= AgentRegistry.MAX_DEPTH:
            return {
                "ok": False,
                "error": f"Maximum delegation depth ({AgentRegistry.MAX_DEPTH}) reached.",
            }

        # Build context
        context = self._build_context(task, extra_context)
        full_task = f"{context.to_prompt()}\n\n---\n\nYour task:\n{task}"

        if mode == "subprocess":
            return self._delegate_subprocess(
                agent_name, full_task, context, background, timeout
            )
        else:
            return self._delegate_in_process(agent_name, config, full_task)

    def _delegate_subprocess(
        self,
        agent_name: str,
        full_task: str,
        context: DelegationContext,
        background: bool,
        timeout: float,
    ) -> dict[str, Any]:
        """Delegate via subprocess using the supervisor."""
        cmd = [
            "supyagent", "exec", agent_name,
            "--task", full_task,
            "--context", json.dumps({
                "parent_agent": context.parent_agent,
                "parent_task": context.parent_task,
                "conversation_summary": context.conversation_summary,
                "relevant_facts": context.relevant_facts,
            }),
            "--output", "json",
        ]

        supervisor = get_supervisor()
        
        # Run async in sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # Already in async context - use thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    supervisor.execute(
                        cmd,
                        process_type="agent",
                        tool_name=f"agent__{agent_name}",
                        timeout=timeout,
                        force_background=background,
                        metadata={"agent_name": agent_name, "task": full_task[:200]},
                    )
                )
                return future.result()
        else:
            return loop.run_until_complete(
                supervisor.execute(
                    cmd,
                    process_type="agent",
                    tool_name=f"agent__{agent_name}",
                    timeout=timeout,
                    force_background=background,
                    metadata={"agent_name": agent_name, "task": full_task[:200]},
                )
            )

    def _delegate_in_process(
        self,
        agent_name: str,
        config,
        full_task: str,
    ) -> dict[str, Any]:
        """Delegate in-process (original behavior)."""
        try:
            if config.type == "execution":
                from supyagent.core.executor import ExecutionRunner
                runner = ExecutionRunner(config)
                return runner.run(full_task, output_format="json")
            else:
                from supyagent.core.agent import Agent
                sub_agent = Agent(
                    config,
                    registry=self.registry,
                    parent_instance_id=self.parent_id,
                )
                response = sub_agent.send_message(full_task)
                if sub_agent.instance_id:
                    self.registry.mark_completed(sub_agent.instance_id)
                return {"ok": True, "data": response}

        except Exception as e:
            return {"ok": False, "error": f"Delegation failed: {str(e)}"}
```

---

### 8.5 Process Management Tools for LLM

Add tools that let the LLM manage background processes:

```python
# supyagent/core/process_tools.py
"""
Process management tools for LLM.

These tools allow the agent to check on, interact with,
and terminate background processes.
"""

from typing import Any
from supyagent.core.supervisor import get_supervisor


def get_process_management_tools() -> list[dict[str, Any]]:
    """Get tool schemas for process management."""
    return [
        {
            "type": "function",
            "function": {
                "name": "list_processes",
                "description": (
                    "List all running background processes (tools and agents). "
                    "Use this to see what's currently running."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "include_completed": {
                            "type": "boolean",
                            "default": False,
                            "description": "Include completed/failed processes in the list",
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_process",
                "description": (
                    "Check the status of a specific background process. "
                    "Returns status, output, and other details."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "process_id": {
                            "type": "string",
                            "description": "The process ID to check",
                        }
                    },
                    "required": ["process_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_process_output",
                "description": (
                    "Get the output (stdout/stderr) from a background process. "
                    "Useful for checking what a long-running process has produced."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "process_id": {
                            "type": "string",
                            "description": "The process ID",
                        },
                        "tail": {
                            "type": "integer",
                            "default": 100,
                            "description": "Number of lines to return from end of output",
                        }
                    },
                    "required": ["process_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "kill_process",
                "description": (
                    "Terminate a running background process. "
                    "Use when you no longer need a process or it's misbehaving."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "process_id": {
                            "type": "string",
                            "description": "The process ID to kill",
                        }
                    },
                    "required": ["process_id"],
                },
            },
        },
    ]


async def execute_process_tool(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute a process management tool."""
    supervisor = get_supervisor()
    
    if tool_name == "list_processes":
        processes = supervisor.list_processes(
            include_completed=args.get("include_completed", False)
        )
        return {"ok": True, "data": processes}
    
    elif tool_name == "check_process":
        process_id = args.get("process_id")
        if not process_id:
            return {"ok": False, "error": "process_id is required"}
        
        process = supervisor.get_process(process_id)
        if process:
            return {"ok": True, "data": process}
        else:
            return {"ok": False, "error": f"Process {process_id} not found"}
    
    elif tool_name == "get_process_output":
        process_id = args.get("process_id")
        if not process_id:
            return {"ok": False, "error": "process_id is required"}
        
        return await supervisor.get_output(
            process_id,
            tail=args.get("tail", 100)
        )
    
    elif tool_name == "kill_process":
        process_id = args.get("process_id")
        if not process_id:
            return {"ok": False, "error": "process_id is required"}
        
        return await supervisor.kill(process_id)
    
    return {"ok": False, "error": f"Unknown process tool: {tool_name}"}


def is_process_tool(tool_name: str) -> bool:
    """Check if a tool name is a process management tool."""
    return tool_name in ("list_processes", "check_process", "get_process_output", "kill_process")
```

---

### 8.6 Configuration Schema

Add supervisor configuration to agent config:

```python
# supyagent/models/agent_config.py (additions)

from pydantic import BaseModel, Field
from typing import Literal


class SupervisorSettings(BaseModel):
    """Process supervisor settings for an agent."""
    
    default_timeout: float = Field(
        default=30.0,
        description="Seconds before a tool/agent is auto-backgrounded"
    )
    
    on_timeout: Literal["background", "kill", "wait"] = Field(
        default="background",
        description="Action when timeout is reached"
    )
    
    max_execution_time: float = Field(
        default=300.0,
        description="Hard limit in seconds (kills process after this)"
    )
    
    max_parallel_tools: int = Field(
        default=5,
        description="Maximum concurrent tool executions"
    )


class ToolSettings(BaseModel):
    """Extended tool settings."""
    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)
    
    # Per-tool timeout overrides
    timeouts: dict[str, float] = Field(
        default_factory=dict,
        description="Per-tool timeout overrides, e.g. {'web_search__search': 45}"
    )
    
    # Per-tool mode overrides
    modes: dict[str, Literal["sync", "background"]] = Field(
        default_factory=dict,
        description="Per-tool execution mode, e.g. {'server__start': 'background'}"
    )


class DelegationSettings(BaseModel):
    """Delegation settings for multi-agent orchestration."""
    
    default_mode: Literal["subprocess", "in_process"] = Field(
        default="subprocess",
        description="Default execution mode for child agents"
    )
    
    timeout: float = Field(
        default=300.0,
        description="Default timeout for child agent execution"
    )
    
    max_parallel_agents: int = Field(
        default=3,
        description="Maximum concurrent child agents"
    )


class AgentConfig(BaseModel):
    """Full agent configuration."""
    name: str
    description: str = ""
    type: Literal["interactive", "execution"] = "interactive"
    version: str = "1.0"
    
    model: ModelConfig
    system_prompt: str = ""
    
    tools: ToolSettings = Field(default_factory=ToolSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    supervisor: SupervisorSettings = Field(default_factory=SupervisorSettings)
    delegation: DelegationSettings = Field(default_factory=DelegationSettings)
    
    delegates: list[str] = Field(default_factory=list)
    limits: dict[str, int] = Field(default_factory=dict)
```

Example agent YAML with new settings:

```yaml
# agents/assistant.yaml
name: assistant
description: General purpose assistant
type: interactive

model:
  provider: anthropic/claude-sonnet-4-20250514
  temperature: 0.7
  max_tokens: 4096

system_prompt: |
  You are a helpful assistant.

# Tool settings
tools:
  allow:
    - "files:*"
    - "web_search:*"
  timeouts:
    web_search__search: 45
  modes:
    server__start: background

# Supervisor settings (optional - uses sensible defaults)
supervisor:
  default_timeout: 30
  on_timeout: background

# Context management
context:
  auto_summarize: true
  max_messages_before_summary: 30

# Delegation (if this agent can delegate)
delegates:
  - researcher
  - coder
  
delegation:
  default_mode: subprocess
  timeout: 300
```

---

### 8.7 CLI Process Commands

Add CLI commands for process management:

```python
# supyagent/cli/main.py (additions)

@cli.group()
def process():
    """Manage background processes."""
    pass


@process.command("list")
@click.option("--all", "-a", "include_all", is_flag=True, help="Include completed processes")
def process_list(include_all: bool):
    """List running background processes."""
    import asyncio
    from supyagent.core.supervisor import get_supervisor
    
    supervisor = get_supervisor()
    processes = supervisor.list_processes(include_completed=include_all)
    
    if not processes:
        click.echo("No running processes")
        return
    
    for proc in processes:
        status_color = {
            "running": "green",
            "backgrounded": "yellow",
            "completed": "blue",
            "failed": "red",
            "killed": "red",
        }.get(proc["status"], "white")
        
        click.echo(f"  {proc['process_id']}: ", nl=False)
        click.secho(f"[{proc['status']}]", fg=status_color, nl=False)
        click.echo(f" {proc['process_type']} (PID: {proc.get('pid', 'N/A')})")
        
        if proc.get("metadata"):
            if "agent_name" in proc["metadata"]:
                click.echo(f"    Agent: {proc['metadata']['agent_name']}")
            if "script" in proc["metadata"]:
                click.echo(f"    Tool: {proc['metadata']['script']}__{proc['metadata']['func']}")


@process.command("show")
@click.argument("process_id")
def process_show(process_id: str):
    """Show details of a specific process."""
    from supyagent.core.supervisor import get_supervisor
    
    supervisor = get_supervisor()
    proc = supervisor.get_process(process_id)
    
    if not proc:
        click.echo(f"Process {process_id} not found", err=True)
        return
    
    click.echo(f"Process ID: {proc['process_id']}")
    click.echo(f"Status: {proc['status']}")
    click.echo(f"Type: {proc['process_type']}")
    click.echo(f"PID: {proc.get('pid', 'N/A')}")
    click.echo(f"Started: {proc.get('started_at', 'N/A')}")
    click.echo(f"Command: {' '.join(proc['cmd'][:5])}...")
    
    if proc.get("log_file"):
        click.echo(f"Log: {proc['log_file']}")


@process.command("output")
@click.argument("process_id")
@click.option("--tail", "-n", default=50, help="Number of lines to show")
def process_output(process_id: str, tail: int):
    """Show output from a background process."""
    import asyncio
    from supyagent.core.supervisor import get_supervisor
    
    supervisor = get_supervisor()
    result = asyncio.run(supervisor.get_output(process_id, tail=tail))
    
    if result["ok"]:
        click.echo(result["data"].get("output", result["data"]))
    else:
        click.echo(f"Error: {result['error']}", err=True)


@process.command("kill")
@click.argument("process_id")
@click.option("--force", "-f", is_flag=True, help="Force kill without confirmation")
def process_kill(process_id: str, force: bool):
    """Kill a running background process."""
    import asyncio
    from supyagent.core.supervisor import get_supervisor
    
    if not force:
        if not click.confirm(f"Kill process {process_id}?"):
            return
    
    supervisor = get_supervisor()
    result = asyncio.run(supervisor.kill(process_id))
    
    if result["ok"]:
        click.echo(f"Process {process_id} killed")
    else:
        click.echo(f"Error: {result['error']}", err=True)
```

---

## Acceptance Criteria

1. **Non-blocking tool execution**: Tools don't block the agent indefinitely
2. **Auto-backgrounding**: Long-running tools are automatically promoted to background
3. **Subprocess agents**: Child agents can run in isolated subprocesses
4. **Process visibility**: LLM can list and check background processes
5. **Process control**: LLM can kill background processes
6. **Configurable timeouts**: Per-agent and per-tool timeout settings
7. **Pattern matching**: Force background/sync based on tool name patterns
8. **CLI commands**: `supyagent process list/show/kill` work correctly
9. **Exec command**: `supyagent exec` runs agents as subprocesses
10. **Backwards compatible**: Existing code works without changes

---

## Test Scenarios

### Scenario 1: Auto-Background Long-Running Tool
```
You> Start a server on port 8000

Agent: I'll start a server for you.
[Calling server__start_server(port=8000)]

[Tool promoted to background after 30s. Process ID: tool_143521_1]

The server is starting in the background. You can check its status with:
- Process ID: tool_143521_1
- Use /processes to see all running processes
```

### Scenario 2: Subprocess Agent Delegation
```
You> Research quantum computing and write a summary

Planner: I'll delegate this to the researcher agent.
[Delegating to researcher via subprocess]

Researcher (subprocess): [runs isolated, returns result]

Planner: Based on the research, here's a summary of quantum computing...
```

### Scenario 3: Process Management
```
You> /processes

Running Processes:
  tool_143521_1: [backgrounded] tool (PID: 12345)
    Tool: server__start_server
  agent_143522_1: [running] agent (PID: 12346)
    Agent: researcher

You> Kill the server

Agent: [Calling kill_process(process_id="tool_143521_1")]

The server has been stopped.
```

### Scenario 4: Background Agent
```
You> Start the research in background, I'll check later

Agent: [Delegating to researcher with background=true]

Started researcher agent in background (process_id: agent_143523_1).
You can check on it anytime with /process agent_143523_1
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            Agent Process                                │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                      ProcessSupervisor                            │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │  │
│  │  │ Config      │  │ Process     │  │ Lifecycle   │               │  │
│  │  │ (timeouts,  │  │ Registry    │  │ Manager     │               │  │
│  │  │  patterns)  │  │ (tracking)  │  │ (start/stop)│               │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│         │                    │                    │                     │
│  ┌──────┴─────┐       ┌──────┴─────┐       ┌──────┴─────┐              │
│  │   Tool     │       │   Tool     │       │   Agent    │              │
│  │ Subprocess │       │ Subprocess │       │ Subprocess │              │
│  │ (30s → bg) │       │ (bg)       │       │ (isolated) │              │
│  └────────────┘       └────────────┘       └────────────┘              │
│                                                                         │
│  Process Tools for LLM:                                                 │
│  • list_processes    • check_process                                    │
│  • get_process_output • kill_process                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## User Experience Notes

### For Beginners

Everything works automatically:
```bash
supyagent init myproject    # Creates project with sensible defaults
supyagent chat assistant    # Just works, no config needed
```

If a tool takes too long, they see:
```
[Tool promoted to background. Use /processes to check status]
```

### For Power Users

Tune in agent YAML:
```yaml
supervisor:
  default_timeout: 60
  
tools:
  timeouts:
    web_search__search: 45
```

### For Advanced Users

Full control via config + CLI:
```bash
supyagent chat assistant --supervisor.default_timeout=120
supyagent process list --all
supyagent process kill proc_123
```

---

## Migration Notes

1. **No breaking changes**: Existing code continues to work
2. **Gradual adoption**: New features are opt-in via config
3. **Default behavior**: Slightly different (auto-background at 30s)
4. **Testing**: Run existing tests to ensure backwards compatibility

---

## Dependencies

No new dependencies required. Uses only:
- `asyncio` (stdlib)
- `signal` (stdlib)
- `json` (stdlib)
- `pathlib` (stdlib)

---

## Notes

- Process isolation means child agent crashes don't affect parent
- Log files help debug background processes
- Consider adding WebSocket support for real-time process output in future
- May want to add process grouping for related operations
