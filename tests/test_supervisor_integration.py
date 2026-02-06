"""
Integration tests for ProcessSupervisor – full lifecycle scenarios.

Tests cover the complete process lifecycle: create → list → inspect → output → kill → verify.
Uses real subprocesses (echo, sleep, sh) to exercise actual OS-level behavior.
"""

import asyncio
import time

import pytest

from supyagent.core.supervisor import (
    ProcessSupervisor,
    ProcessStatus,
    SupervisorConfig,
    TimeoutAction,
    get_supervisor,
    reset_supervisor,
    run_supervisor_coroutine,
)
from supyagent.core.process_tools import (
    execute_process_tool,
    execute_process_tool_async,
    is_process_tool,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def supervisor(tmp_path):
    """Fresh supervisor with a temporary log dir and short timeouts for tests."""
    reset_supervisor()
    config = SupervisorConfig(
        default_timeout=1.0,
        max_execution_time=5.0,
        on_timeout=TimeoutAction.BACKGROUND,
        log_dir=tmp_path / "logs",
        max_background_processes=5,
    )
    return ProcessSupervisor(config)


@pytest.fixture(autouse=True)
def _cleanup():
    """Ensure supervisor is reset after every test."""
    yield
    reset_supervisor()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _start_bg_process(supervisor: ProcessSupervisor, sleep_secs: int = 60):
    """Start a background process and return the process_id."""
    result = await supervisor.execute(["sleep", str(sleep_secs)], force_background=True)
    assert result["ok"] is True
    assert result["data"]["status"] == "backgrounded"
    return result["data"]["process_id"]


# ===========================================================================
# 1. Full lifecycle: create → list → check → output → kill → verify
# ===========================================================================

class TestFullLifecycle:
    """End-to-end lifecycle of a single managed process."""

    async def test_create_list_check_kill_verify(self, supervisor):
        """Walk through every lifecycle step for one process."""
        # --- Create ---
        pid = await _start_bg_process(supervisor)

        # --- List ---
        procs = supervisor.list_processes()
        assert len(procs) == 1
        assert procs[0]["process_id"] == pid
        assert procs[0]["status"] == "backgrounded"
        assert procs[0]["pid"] is not None  # real OS PID

        # --- Check ---
        info = supervisor.get_process(pid)
        assert info is not None
        assert info["status"] == "backgrounded"
        assert info["started_at"] is not None

        # --- Kill ---
        kill_result = await supervisor.kill(pid)
        assert kill_result["ok"] is True

        # --- Verify killed ---
        info = supervisor.get_process(pid)
        assert info["status"] == "killed"
        assert info["completed_at"] is not None

        # --- Should no longer appear in active list ---
        procs = supervisor.list_processes(include_completed=False)
        assert all(p["process_id"] != pid for p in procs)

        # --- But appears when include_completed=True ---
        procs = supervisor.list_processes(include_completed=True)
        matched = [p for p in procs if p["process_id"] == pid]
        assert len(matched) == 1
        assert matched[0]["status"] == "killed"

    async def test_lifecycle_with_output(self, supervisor, tmp_path):
        """Process that writes output → background → collect output after completion."""
        # Use a command that produces output and exits quickly
        result = await supervisor.execute(
            ["sh", "-c", "echo 'line1'; echo 'line2'; echo 'line3'"],
        )
        # Should complete within timeout, not backgrounded
        assert result["ok"] is True
        assert "line1" in str(result.get("data", ""))

        # Check that the log file was written
        procs = supervisor.list_processes(include_completed=True)
        assert len(procs) >= 1
        proc_id = procs[0]["process_id"]

        output = await supervisor.get_output(proc_id)
        assert output["ok"] is True

    async def test_auto_background_lifecycle(self, supervisor):
        """
        A slow process auto-backgrounds after timeout, then can be killed.
        This is the primary "safety net" scenario.
        """
        supervisor.config.default_timeout = 0.5
        supervisor.config.on_timeout = TimeoutAction.BACKGROUND

        # Starts, exceeds timeout, auto-backgrounds
        result = await supervisor.execute(["sleep", "60"])
        assert result["ok"] is True
        assert result["data"]["status"] == "backgrounded"
        pid = result["data"]["process_id"]

        # Verify it's listed
        procs = supervisor.list_processes()
        assert any(p["process_id"] == pid for p in procs)

        # Kill it
        kill = await supervisor.kill(pid)
        assert kill["ok"] is True

        # Verify killed
        info = supervisor.get_process(pid)
        assert info["status"] == "killed"


# ===========================================================================
# 2. Multiple concurrent processes
# ===========================================================================

class TestConcurrentProcesses:
    """Managing multiple processes simultaneously."""

    async def test_multiple_create_list_kill_all(self, supervisor):
        """Create several background processes, list them, kill them all."""
        pids = []
        for _ in range(3):
            pid = await _start_bg_process(supervisor)
            pids.append(pid)

        # All three should be listed
        procs = supervisor.list_processes()
        active_ids = {p["process_id"] for p in procs}
        for pid in pids:
            assert pid in active_ids

        # Kill them all
        for pid in pids:
            result = await supervisor.kill(pid)
            assert result["ok"] is True

        # None should be active
        procs = supervisor.list_processes(include_completed=False)
        assert len(procs) == 0

    async def test_selective_kill(self, supervisor):
        """Kill only some processes, verify the rest remain alive."""
        p1 = await _start_bg_process(supervisor)
        p2 = await _start_bg_process(supervisor)
        p3 = await _start_bg_process(supervisor)

        # Kill only p2
        await supervisor.kill(p2)

        procs = supervisor.list_processes()
        active_ids = {p["process_id"] for p in procs}
        assert p1 in active_ids
        assert p2 not in active_ids
        assert p3 in active_ids

        # Cleanup
        await supervisor.kill(p1)
        await supervisor.kill(p3)

    async def test_mixed_completed_and_running(self, supervisor):
        """Quick commands (completed) + long-running (backgrounded) coexist."""
        # Quick command – completes instantly
        await supervisor.execute(["echo", "done"])

        # Long-running – backgrounds
        pid = await _start_bg_process(supervisor)

        # Without include_completed, only the background process shows
        procs = supervisor.list_processes(include_completed=False)
        assert len(procs) == 1
        assert procs[0]["process_id"] == pid

        # With include_completed, both appear
        procs = supervisor.list_processes(include_completed=True)
        assert len(procs) >= 2

        await supervisor.kill(pid)


# ===========================================================================
# 3. Kill edge-cases
# ===========================================================================

class TestKillEdgeCases:
    """Robustness of the kill operation."""

    async def test_kill_nonexistent_process(self, supervisor):
        """Killing a process that was never created returns an error."""
        result = await supervisor.kill("does_not_exist")
        assert result["ok"] is False
        assert "not found" in result["error"]

    async def test_kill_already_killed_process(self, supervisor):
        """Killing a process that was already killed returns a clear error."""
        pid = await _start_bg_process(supervisor)

        first = await supervisor.kill(pid)
        assert first["ok"] is True

        second = await supervisor.kill(pid)
        assert second["ok"] is False
        assert "not running" in second["error"]

    async def test_kill_completed_process(self, supervisor):
        """Killing a process that already exited returns a clear error."""
        result = await supervisor.execute(["echo", "bye"])
        assert result["ok"] is True

        # Find the completed process
        procs = supervisor.list_processes(include_completed=True)
        assert len(procs) >= 1
        pid = procs[0]["process_id"]

        kill_result = await supervisor.kill(pid)
        assert kill_result["ok"] is False
        assert "not running" in kill_result["error"]

    async def test_kill_actually_terminates_os_process(self, supervisor):
        """Verify the underlying OS process is truly dead after kill."""
        import os
        import signal

        pid = await _start_bg_process(supervisor)
        info = supervisor.get_process(pid)
        os_pid = info["pid"]

        # The OS process should be running
        assert os_pid is not None
        try:
            os.kill(os_pid, 0)  # signal 0 = check existence
        except ProcessLookupError:
            pytest.fail("Process should be running before kill")

        await supervisor.kill(pid)

        # Give a moment for cleanup
        await asyncio.sleep(0.3)

        # The OS process should be gone
        with pytest.raises(ProcessLookupError):
            os.kill(os_pid, 0)


# ===========================================================================
# 4. Timeout behaviour
# ===========================================================================

class TestTimeoutBehavior:
    """Detailed timeout handling across all three actions."""

    async def test_timeout_background_returns_control(self, supervisor):
        """BACKGROUND action: control returns quickly, process keeps running."""
        supervisor.config.default_timeout = 0.3
        supervisor.config.on_timeout = TimeoutAction.BACKGROUND

        start = time.monotonic()
        result = await supervisor.execute(["sleep", "60"])
        elapsed = time.monotonic() - start

        assert result["ok"] is True
        assert result["data"]["status"] == "backgrounded"
        # Should return roughly around the timeout, not 60s
        assert elapsed < 3.0

        await supervisor.kill(result["data"]["process_id"])

    async def test_timeout_kill_terminates_immediately(self, supervisor):
        """KILL action: process is terminated and error is returned."""
        supervisor.config.default_timeout = 0.3
        supervisor.config.on_timeout = TimeoutAction.KILL

        result = await supervisor.execute(["sleep", "60"])
        assert result["ok"] is False
        assert "killed" in result["error"].lower() or "timeout" in result["error"].lower()

        # Should not appear in active list
        procs = supervisor.list_processes()
        assert len(procs) == 0

    async def test_timeout_wait_allows_slow_but_finite_command(self, supervisor):
        """WAIT action: process runs up to max_execution_time."""
        supervisor.config.default_timeout = 0.3
        supervisor.config.max_execution_time = 3.0
        supervisor.config.on_timeout = TimeoutAction.WAIT

        # This finishes before max_execution_time
        result = await supervisor.execute(["sleep", "0.5"])
        assert result["ok"] is True

    async def test_timeout_wait_kills_at_hard_limit(self, supervisor):
        """WAIT action: process killed if it exceeds max_execution_time."""
        supervisor.config.default_timeout = 0.3
        supervisor.config.max_execution_time = 1.0
        supervisor.config.on_timeout = TimeoutAction.WAIT

        result = await supervisor.execute(["sleep", "60"])
        assert result["ok"] is False
        assert "timeout" in result["error"].lower() or "killed" in result["error"].lower()

    async def test_per_call_timeout_overrides_default(self, supervisor):
        """A per-call timeout overrides the default_timeout from config."""
        supervisor.config.default_timeout = 10.0
        supervisor.config.on_timeout = TimeoutAction.KILL

        start = time.monotonic()
        result = await supervisor.execute(["sleep", "60"], timeout=0.3)
        elapsed = time.monotonic() - start

        assert result["ok"] is False
        assert elapsed < 5.0  # Much less than the default 10s


# ===========================================================================
# 5. Background process limits
# ===========================================================================

class TestBackgroundLimits:
    """Max background process enforcement."""

    async def test_max_background_enforced(self, supervisor):
        """Once limit is reached, new background requests are rejected."""
        supervisor.config.max_background_processes = 2

        p1 = await _start_bg_process(supervisor)
        p2 = await _start_bg_process(supervisor)

        # Third should be rejected
        result = await supervisor.execute(["sleep", "60"], force_background=True)
        assert result["ok"] is False
        assert "maximum" in result["error"].lower()

        # Cleanup
        await supervisor.kill(p1)
        await supervisor.kill(p2)

    async def test_limit_frees_after_kill(self, supervisor):
        """After killing a process, a new one can take its slot."""
        supervisor.config.max_background_processes = 1

        p1 = await _start_bg_process(supervisor)

        # At limit – reject
        result = await supervisor.execute(["sleep", "60"], force_background=True)
        assert result["ok"] is False

        # Free the slot
        await supervisor.kill(p1)

        # Clean up killed process from tracking
        await supervisor._cleanup_completed()

        # Now it should succeed
        p2 = await _start_bg_process(supervisor)
        assert p2 is not None

        await supervisor.kill(p2)


# ===========================================================================
# 6. Output collection
# ===========================================================================

class TestOutputCollection:
    """Stdout/stderr capture and retrieval."""

    async def test_stdout_captured_for_quick_command(self, supervisor):
        """Output from a synchronous completion is returned directly."""
        result = await supervisor.execute(["echo", "hello world"])
        assert result["ok"] is True
        assert "hello world" in str(result.get("data", ""))

    async def test_stderr_captured_on_failure(self, supervisor):
        """Stderr from a failing command is captured."""
        result = await supervisor.execute(
            ["sh", "-c", "echo 'boom' >&2; exit 1"]
        )
        assert result["ok"] is False
        assert "boom" in result.get("error", "")

    async def test_output_from_backgrounded_process_after_exit(self, supervisor):
        """
        A backgrounded process that later exits has its output collected
        and retrievable via get_output.
        """
        # Short command that is still backgrounded due to our short timeout
        supervisor.config.default_timeout = 0.2

        result = await supervisor.execute(
            ["sh", "-c", "sleep 0.3; echo 'background_result'"]
        )
        assert result["ok"] is True
        assert result["data"]["status"] == "backgrounded"
        pid = result["data"]["process_id"]

        # Wait for the background task to complete
        await asyncio.sleep(1.0)

        # Output should now be available
        output = await supervisor.get_output(pid)
        assert output["ok"] is True
        output_str = str(output.get("data", ""))
        assert "background_result" in output_str

    async def test_get_output_nonexistent(self, supervisor):
        """Requesting output from a nonexistent process returns an error."""
        result = await supervisor.get_output("ghost_process")
        assert result["ok"] is False
        assert "not found" in result["error"]

    async def test_log_files_created(self, supervisor, tmp_path):
        """Every execution creates a log file."""
        await supervisor.execute(["echo", "logged"])
        log_files = list((tmp_path / "logs").glob("*.log"))
        assert len(log_files) >= 1


# ===========================================================================
# 7. Pattern matching (force_background / force_sync)
# ===========================================================================

class TestPatternMatching:
    """Pattern-based execution mode overrides."""

    async def test_force_background_pattern_triggers(self, supervisor):
        """A command whose tool_name matches a background pattern is auto-backgrounded."""
        supervisor.config.force_background_patterns = ["server__*"]

        result = await supervisor.execute(
            ["sleep", "60"],
            tool_name="server__start_http",
        )
        assert result["ok"] is True
        assert result["data"]["status"] == "backgrounded"
        await supervisor.kill(result["data"]["process_id"])

    async def test_force_sync_pattern_waits_longer(self, supervisor):
        """
        A command whose tool_name matches a sync pattern uses max_execution_time
        instead of the default_timeout.
        """
        supervisor.config.default_timeout = 0.3
        supervisor.config.max_execution_time = 3.0
        supervisor.config.force_sync_patterns = ["files__*"]

        # This 0.5s sleep would normally trigger a background at 0.3s,
        # but the sync pattern overrides to wait up to 3s.
        result = await supervisor.execute(
            ["sleep", "0.5"],
            tool_name="files__read_file",
        )
        assert result["ok"] is True

    async def test_no_pattern_match_uses_default(self, supervisor):
        """When tool_name matches no pattern, default behaviour is used."""
        supervisor.config.default_timeout = 0.3
        supervisor.config.on_timeout = TimeoutAction.BACKGROUND
        supervisor.config.force_background_patterns = ["server__*"]
        supervisor.config.force_sync_patterns = ["files__*"]

        result = await supervisor.execute(
            ["sleep", "60"],
            tool_name="math__compute",
        )
        # Should be auto-backgrounded per the default
        assert result["ok"] is True
        assert result["data"]["status"] == "backgrounded"
        await supervisor.kill(result["data"]["process_id"])


# ===========================================================================
# 8. Process tools (LLM-facing wrappers)
# ===========================================================================

class TestProcessTools:
    """
    Test the LLM-accessible process management tools end-to-end,
    exercising the full create → list → check → output → kill → verify
    flow through the tool wrappers.
    """

    def test_is_process_tool_detection(self):
        """is_process_tool correctly identifies process tools."""
        assert is_process_tool("list_processes")
        assert is_process_tool("check_process")
        assert is_process_tool("get_process_output")
        assert is_process_tool("kill_process")
        assert is_process_tool("sleep")
        assert not is_process_tool("shell__run_command")
        assert not is_process_tool("files__read_file")
        assert not is_process_tool("")

    async def test_list_processes_tool_empty(self, supervisor, tmp_path):
        """list_processes returns empty when nothing is running."""
        # Initialise global supervisor so tools find it
        reset_supervisor()
        config = SupervisorConfig(
            default_timeout=1.0,
            log_dir=tmp_path / "logs",
        )
        from supyagent.core import supervisor as sup_mod
        sup_mod._supervisor = ProcessSupervisor(config)

        result = await execute_process_tool_async("list_processes", {})
        assert result["ok"] is True
        assert result["data"] == []

    async def test_tool_full_lifecycle(self, tmp_path):
        """Exercise all four process tools through a full lifecycle."""
        reset_supervisor()
        config = SupervisorConfig(
            default_timeout=1.0,
            log_dir=tmp_path / "logs",
            max_background_processes=5,
        )
        from supyagent.core import supervisor as sup_mod
        sup_mod._supervisor = ProcessSupervisor(config)
        sv = sup_mod._supervisor

        # --- Create a background process ---
        bg_result = await sv.execute(["sleep", "60"], force_background=True)
        assert bg_result["ok"] is True
        pid = bg_result["data"]["process_id"]

        # --- list_processes ---
        list_result = await execute_process_tool_async("list_processes", {})
        assert list_result["ok"] is True
        ids = [p["process_id"] for p in list_result["data"]]
        assert pid in ids

        # --- check_process ---
        check_result = await execute_process_tool_async(
            "check_process", {"process_id": pid}
        )
        assert check_result["ok"] is True
        assert check_result["data"]["status"] == "backgrounded"

        # --- get_process_output ---
        output_result = await execute_process_tool_async(
            "get_process_output", {"process_id": pid}
        )
        assert output_result["ok"] is True

        # --- kill_process ---
        kill_result = await execute_process_tool_async(
            "kill_process", {"process_id": pid}
        )
        assert kill_result["ok"] is True

        # --- Verify gone from active list ---
        list_result2 = await execute_process_tool_async("list_processes", {})
        assert list_result2["ok"] is True
        active_ids = [p["process_id"] for p in list_result2["data"]]
        assert pid not in active_ids

    async def test_check_nonexistent_via_tool(self, tmp_path):
        """check_process for a missing ID returns an error."""
        reset_supervisor()
        from supyagent.core import supervisor as sup_mod
        sup_mod._supervisor = ProcessSupervisor(
            SupervisorConfig(log_dir=tmp_path / "logs")
        )

        result = await execute_process_tool_async(
            "check_process", {"process_id": "nope"}
        )
        assert result["ok"] is False
        assert "not found" in result["error"]

    async def test_kill_nonexistent_via_tool(self, tmp_path):
        """kill_process for a missing ID returns an error."""
        reset_supervisor()
        from supyagent.core import supervisor as sup_mod
        sup_mod._supervisor = ProcessSupervisor(
            SupervisorConfig(log_dir=tmp_path / "logs")
        )

        result = await execute_process_tool_async(
            "kill_process", {"process_id": "nope"}
        )
        assert result["ok"] is False
        assert "not found" in result["error"]

    async def test_check_process_missing_arg(self, tmp_path):
        """check_process without process_id returns a useful error."""
        reset_supervisor()
        from supyagent.core import supervisor as sup_mod
        sup_mod._supervisor = ProcessSupervisor(
            SupervisorConfig(log_dir=tmp_path / "logs")
        )

        result = await execute_process_tool_async("check_process", {})
        assert result["ok"] is False
        assert "required" in result["error"].lower()


# ===========================================================================
# 8b. Sleep tool
# ===========================================================================

class TestSleepTool:
    """Tests for the built-in sleep tool."""

    async def test_sleep_short_duration(self):
        """Sleep for a very short duration and verify result."""
        result = await execute_process_tool_async("sleep", {"seconds": 0.01})
        assert result["ok"] is True
        assert result["data"]["slept_seconds"] == 0.01
        assert "Resuming now" in result["data"]["message"]

    async def test_sleep_with_reason(self):
        """Sleep with a reason and verify it's returned."""
        result = await execute_process_tool_async(
            "sleep", {"seconds": 0.01, "reason": "waiting for API"}
        )
        assert result["ok"] is True
        assert result["data"]["reason"] == "waiting for API"

    async def test_sleep_zero_seconds(self):
        """Sleep with zero seconds returns error."""
        result = await execute_process_tool_async("sleep", {"seconds": 0})
        assert result["ok"] is False
        assert "positive" in result["error"].lower()

    async def test_sleep_negative_seconds(self):
        """Sleep with negative seconds returns error."""
        result = await execute_process_tool_async("sleep", {"seconds": -10})
        assert result["ok"] is False
        assert "positive" in result["error"].lower()

    async def test_sleep_exceeds_max(self):
        """Sleep exceeding 24 hours returns error."""
        result = await execute_process_tool_async("sleep", {"seconds": 86401})
        assert result["ok"] is False
        assert "24 hours" in result["error"]

    def test_sleep_in_tool_schemas(self):
        """Sleep tool is included in the tool schemas."""
        from supyagent.core.process_tools import get_process_management_tools

        tools = get_process_management_tools()
        names = [t["function"]["name"] for t in tools]
        assert "sleep" in names


# ===========================================================================
# 9. Sync wrapper (run_supervisor_coroutine)
# ===========================================================================

class TestSyncWrapper:
    """
    The sync bridge used by agent/executor to call async supervisor
    methods from synchronous code.
    """

    def test_sync_execute_and_kill(self, tmp_path):
        """Full create-kill cycle through the sync wrapper."""
        reset_supervisor()
        config = SupervisorConfig(
            default_timeout=1.0,
            log_dir=tmp_path / "logs",
        )
        sv = get_supervisor(config)

        result = run_supervisor_coroutine(
            sv.execute(["sleep", "60"], force_background=True)
        )
        assert result["ok"] is True
        pid = result["data"]["process_id"]

        # List via sync wrapper
        procs = sv.list_processes()
        assert any(p["process_id"] == pid for p in procs)

        # Kill via sync wrapper
        kill = run_supervisor_coroutine(sv.kill(pid))
        assert kill["ok"] is True

        # Verify killed
        info = sv.get_process(pid)
        assert info["status"] == "killed"

    def test_sync_process_tools(self, tmp_path):
        """Process tools work through the synchronous execute_process_tool."""
        reset_supervisor()
        config = SupervisorConfig(
            default_timeout=1.0,
            log_dir=tmp_path / "logs",
        )
        from supyagent.core import supervisor as sup_mod
        sup_mod._supervisor = ProcessSupervisor(config)
        # Ensure the persistent loop exists
        sup_mod._get_supervisor_loop()

        sv = get_supervisor()

        # Create a background process
        result = run_supervisor_coroutine(
            sv.execute(["sleep", "60"], force_background=True)
        )
        pid = result["data"]["process_id"]

        # list_processes through sync tool
        list_result = execute_process_tool("list_processes", {})
        assert list_result["ok"] is True
        assert any(p["process_id"] == pid for p in list_result["data"])

        # kill through sync tool
        kill_result = execute_process_tool("kill_process", {"process_id": pid})
        assert kill_result["ok"] is True

        # verify gone
        list_result2 = execute_process_tool("list_processes", {})
        assert list_result2["ok"] is True
        assert all(p["process_id"] != pid for p in list_result2["data"])


# ===========================================================================
# 10. Rapid create / destroy stress test
# ===========================================================================

class TestStress:
    """Rapid-fire lifecycle operations to verify robustness."""

    async def test_rapid_create_kill_cycle(self, supervisor):
        """Create and immediately kill 10 processes in a tight loop."""
        for i in range(10):
            pid = await _start_bg_process(supervisor)
            result = await supervisor.kill(pid)
            assert result["ok"] is True

        # Nothing should be active
        procs = supervisor.list_processes()
        assert len(procs) == 0

    async def test_parallel_create_then_parallel_kill(self, supervisor):
        """Create many in parallel, then kill them all in parallel."""
        # Create in parallel
        tasks = [
            supervisor.execute(["sleep", "60"], force_background=True)
            for _ in range(5)
        ]
        results = await asyncio.gather(*tasks)
        pids = [r["data"]["process_id"] for r in results if r["ok"]]
        assert len(pids) == 5

        # Kill in parallel
        kill_tasks = [supervisor.kill(pid) for pid in pids]
        kill_results = await asyncio.gather(*kill_tasks)
        for kr in kill_results:
            assert kr["ok"] is True

        procs = supervisor.list_processes()
        assert len(procs) == 0

    async def test_interleaved_create_kill(self, supervisor):
        """
        Interleave creates and kills:
        create A, create B, kill A, create C, kill B, kill C.
        """
        a = await _start_bg_process(supervisor)
        b = await _start_bg_process(supervisor)

        await supervisor.kill(a)
        assert supervisor.get_process(a)["status"] == "killed"

        c = await _start_bg_process(supervisor)

        await supervisor.kill(b)
        assert supervisor.get_process(b)["status"] == "killed"

        # Only C should be active
        procs = supervisor.list_processes()
        assert len(procs) == 1
        assert procs[0]["process_id"] == c

        await supervisor.kill(c)


# ===========================================================================
# 11. Cleanup operations
# ===========================================================================

class TestCleanup:
    """Internal cleanup and log retention."""

    async def test_cleanup_completed_removes_finished(self, supervisor):
        """_cleanup_completed purges completed/killed entries."""
        # Create and kill a process
        pid = await _start_bg_process(supervisor)
        await supervisor.kill(pid)

        # Also run a quick command
        await supervisor.execute(["echo", "done"])

        # Before cleanup – completed processes exist
        all_procs = supervisor.list_processes(include_completed=True)
        assert len(all_procs) >= 2

        # Cleanup
        removed = await supervisor._cleanup_completed()
        assert removed >= 2

        # After cleanup – tracking is empty
        all_procs = supervisor.list_processes(include_completed=True)
        assert len(all_procs) == 0

    async def test_cleanup_does_not_touch_running(self, supervisor):
        """_cleanup_completed never removes running/backgrounded processes."""
        pid = await _start_bg_process(supervisor)
        await supervisor.execute(["echo", "done"])

        removed = await supervisor._cleanup_completed()
        # Only the completed echo should be removed, not the running one
        assert removed >= 1

        procs = supervisor.list_processes()
        assert len(procs) == 1
        assert procs[0]["process_id"] == pid

        await supervisor.kill(pid)

    async def test_cleanup_old_logs(self, supervisor, tmp_path):
        """cleanup_old_logs removes files older than retention."""
        import time

        log_dir = tmp_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create a fake old log
        old_log = log_dir / "old_process.log"
        old_log.write_text("old content")
        # Set mtime to 30 days ago
        old_time = time.time() - (30 * 86400)
        import os
        os.utime(old_log, (old_time, old_time))

        # Create a recent log
        new_log = log_dir / "new_process.log"
        new_log.write_text("new content")

        supervisor.config.log_retention_days = 7
        removed = await supervisor.cleanup_old_logs()
        assert removed == 1
        assert not old_log.exists()
        assert new_log.exists()


# ===========================================================================
# 12. Environment and working directory
# ===========================================================================

class TestEnvironmentAndCwd:
    """Processes receive the correct env vars and cwd."""

    async def test_custom_env_vars(self, supervisor):
        """Custom environment variables are passed to the subprocess."""
        result = await supervisor.execute(
            ["sh", "-c", "echo $MY_TEST_VAR"],
            env={"MY_TEST_VAR": "supersecret"},
        )
        assert result["ok"] is True
        assert "supersecret" in str(result["data"])

    async def test_custom_cwd(self, supervisor, tmp_path):
        """Working directory is respected."""
        result = await supervisor.execute(
            ["sh", "-c", "pwd"],
            cwd=str(tmp_path),
        )
        assert result["ok"] is True
        assert str(tmp_path) in str(result["data"])

    async def test_env_does_not_leak_between_calls(self, supervisor):
        """Env vars from one call don't leak into the next."""
        await supervisor.execute(
            ["echo", "setup"],
            env={"LEAK_TEST": "should_not_leak"},
        )
        result = await supervisor.execute(
            ["sh", "-c", "echo ${LEAK_TEST:-empty}"],
        )
        assert result["ok"] is True
        # Should be empty (not set) or literally "empty"
        assert "should_not_leak" not in str(result["data"]) or "empty" in str(result["data"])


# ===========================================================================
# 13. Process metadata
# ===========================================================================

class TestProcessMetadata:
    """Metadata stored with processes is accessible."""

    async def test_metadata_round_trip(self, supervisor):
        """Custom metadata is stored and retrievable."""
        meta = {"agent_name": "tester", "task": "run tests"}
        result = await supervisor.execute(
            ["sleep", "60"],
            force_background=True,
            metadata=meta,
        )
        pid = result["data"]["process_id"]

        info = supervisor.get_process(pid)
        assert info["metadata"] == meta

        await supervisor.kill(pid)

    async def test_process_type_stored(self, supervisor):
        """process_type ('tool' or 'agent') is stored correctly."""
        result = await supervisor.execute(
            ["sleep", "60"],
            force_background=True,
            process_type="agent",
        )
        pid = result["data"]["process_id"]

        info = supervisor.get_process(pid)
        assert info["process_type"] == "agent"

        await supervisor.kill(pid)
