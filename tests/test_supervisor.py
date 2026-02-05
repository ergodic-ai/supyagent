"""Tests for ProcessSupervisor."""

import asyncio
import pytest
from pathlib import Path

from supyagent.core.supervisor import (
    ProcessSupervisor,
    SupervisorConfig,
    ProcessStatus,
    TimeoutAction,
    get_supervisor,
    reset_supervisor,
)


@pytest.fixture
def supervisor(tmp_path):
    """Create a supervisor with temp log directory."""
    reset_supervisor()  # Ensure clean state
    config = SupervisorConfig(
        default_timeout=2.0,
        max_execution_time=5.0,
        log_dir=tmp_path / "logs",
        max_background_processes=5,
    )
    return ProcessSupervisor(config)


@pytest.fixture(autouse=True)
def cleanup_supervisor():
    """Ensure supervisor is reset after each test."""
    yield
    reset_supervisor()


class TestProcessSupervisorBasic:
    """Basic execution tests."""

    @pytest.mark.asyncio
    async def test_execute_quick_command(self, supervisor):
        """Test executing a command that completes quickly."""
        result = await supervisor.execute(["echo", "hello"])
        assert result["ok"] is True
        assert "hello" in str(result.get("data", ""))

    @pytest.mark.asyncio
    async def test_execute_with_json_output(self, supervisor):
        """Test that JSON output is parsed correctly."""
        result = await supervisor.execute(
            ["echo", '{"ok": true, "data": "test"}'],
        )
        assert result["ok"] is True
        assert result["data"] == "test"

    @pytest.mark.asyncio
    async def test_execute_failing_command(self, supervisor):
        """Test handling of commands that exit with error."""
        result = await supervisor.execute(["sh", "-c", "exit 1"])
        assert result["ok"] is False
        assert "exit" in result.get("error", "").lower() or result.get("error") == ""

    @pytest.mark.asyncio
    async def test_execute_with_env(self, supervisor):
        """Test passing environment variables."""
        result = await supervisor.execute(
            ["sh", "-c", "echo $TEST_VAR"],
            env={"TEST_VAR": "hello_env"},
        )
        assert result["ok"] is True
        assert "hello_env" in str(result.get("data", ""))


class TestTimeoutHandling:
    """Timeout and backgrounding tests."""

    @pytest.mark.asyncio
    async def test_timeout_with_background(self, supervisor):
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
    async def test_timeout_with_kill(self, supervisor):
        """Test that slow commands get killed when configured."""
        supervisor.config.default_timeout = 0.5
        supervisor.config.on_timeout = TimeoutAction.KILL

        result = await supervisor.execute(["sleep", "10"])

        assert result["ok"] is False
        assert "killed" in result["error"].lower() or "timeout" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_timeout_with_wait(self, supervisor):
        """Test WAIT action continues to max_execution_time."""
        supervisor.config.default_timeout = 0.3
        supervisor.config.max_execution_time = 1.0
        supervisor.config.on_timeout = TimeoutAction.WAIT

        # This should complete within max_execution_time
        result = await supervisor.execute(["sleep", "0.5"])

        assert result["ok"] is True

    @pytest.mark.asyncio
    async def test_force_background(self, supervisor):
        """Test forcing background execution."""
        result = await supervisor.execute(
            ["sleep", "10"],
            force_background=True,
        )

        assert result["ok"] is True
        assert result["data"]["status"] == "backgrounded"

        await supervisor.kill(result["data"]["process_id"])


class TestPatternMatching:
    """Pattern-based execution mode tests."""

    @pytest.mark.asyncio
    async def test_force_background_patterns(self, supervisor):
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
    async def test_force_sync_patterns(self, supervisor):
        """Test pattern-based force sync."""
        supervisor.config.force_sync_patterns = ["files__read*"]
        supervisor.config.default_timeout = 0.5

        # Even with short timeout, sync patterns should wait longer
        result = await supervisor.execute(
            ["sleep", "0.3"],
            tool_name="files__read_file",
        )

        assert result["ok"] is True


class TestProcessManagement:
    """Process listing and management tests."""

    @pytest.mark.asyncio
    async def test_list_processes(self, supervisor):
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
    async def test_list_processes_excludes_completed(self, supervisor):
        """Test that completed processes are excluded by default."""
        # Run a quick command
        await supervisor.execute(["echo", "done"])

        # Start a background process
        result = await supervisor.execute(["sleep", "10"], force_background=True)
        process_id = result["data"]["process_id"]

        # Only background process should be listed
        processes = supervisor.list_processes()
        assert len(processes) == 1
        assert processes[0]["process_id"] == process_id

        await supervisor.kill(process_id)

    @pytest.mark.asyncio
    async def test_list_processes_include_completed(self, supervisor):
        """Test including completed processes in list."""
        # Run a quick command
        await supervisor.execute(["echo", "done"])

        processes = supervisor.list_processes(include_completed=True)
        assert len(processes) >= 1

    @pytest.mark.asyncio
    async def test_get_process(self, supervisor):
        """Test getting a specific process."""
        result = await supervisor.execute(["sleep", "10"], force_background=True)
        process_id = result["data"]["process_id"]

        proc = supervisor.get_process(process_id)
        assert proc is not None
        assert proc["process_id"] == process_id
        assert proc["status"] == "backgrounded"

        await supervisor.kill(process_id)

    @pytest.mark.asyncio
    async def test_get_nonexistent_process(self, supervisor):
        """Test getting a process that doesn't exist."""
        proc = supervisor.get_process("nonexistent")
        assert proc is None

    @pytest.mark.asyncio
    async def test_kill_process(self, supervisor):
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
    async def test_kill_nonexistent_process(self, supervisor):
        """Test killing a process that doesn't exist."""
        result = await supervisor.kill("nonexistent")
        assert result["ok"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_kill_already_completed_process(self, supervisor):
        """Test killing a process that already completed."""
        # Run a quick command
        await supervisor.execute(["echo", "done"])

        # Get the process ID from the internal state
        processes = supervisor.list_processes(include_completed=True)
        if processes:
            process_id = processes[0]["process_id"]
            result = await supervisor.kill(process_id)
            assert result["ok"] is False
            assert "not running" in result["error"]


class TestOutputCollection:
    """Output and logging tests."""

    @pytest.mark.asyncio
    async def test_get_output(self, supervisor):
        """Test getting output from a completed process."""
        await supervisor.execute(["echo", "test output"])

        processes = supervisor.list_processes(include_completed=True)
        if processes:
            process_id = processes[0]["process_id"]
            result = await supervisor.get_output(process_id)
            assert result["ok"] is True

    @pytest.mark.asyncio
    async def test_get_output_nonexistent(self, supervisor):
        """Test getting output from nonexistent process."""
        result = await supervisor.get_output("nonexistent")
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_log_file_created(self, supervisor, tmp_path):
        """Test that log files are created."""
        await supervisor.execute(["echo", "logged"])

        log_files = list((tmp_path / "logs").glob("*.log"))
        assert len(log_files) >= 1


class TestLimits:
    """Resource limit tests."""

    @pytest.mark.asyncio
    async def test_max_background_limit(self, supervisor):
        """Test that max background process limit is enforced."""
        supervisor.config.max_background_processes = 2

        # Start 2 background processes
        p1 = await supervisor.execute(["sleep", "100"], force_background=True)
        p2 = await supervisor.execute(["sleep", "100"], force_background=True)

        assert p1["ok"] is True
        assert p2["ok"] is True

        # Third should fail
        p3 = await supervisor.execute(["sleep", "100"], force_background=True)
        assert p3["ok"] is False
        assert "maximum" in p3["error"].lower()

        # Cleanup
        await supervisor.kill(p1["data"]["process_id"])
        await supervisor.kill(p2["data"]["process_id"])


class TestConfig:
    """Configuration tests."""

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "default_timeout": 60,
            "on_timeout": "kill",
            "max_parallel_tools": 10,
        }
        config = SupervisorConfig.from_dict(data)

        assert config.default_timeout == 60.0
        assert config.on_timeout == TimeoutAction.KILL
        assert config.max_parallel_tools == 10

    def test_config_defaults(self):
        """Test config has sensible defaults."""
        config = SupervisorConfig()

        assert config.default_timeout == 30.0
        assert config.on_timeout == TimeoutAction.BACKGROUND
        assert config.max_background_processes == 10


class TestGlobalSupervisor:
    """Global supervisor instance tests."""

    def test_get_supervisor_creates_instance(self):
        """Test that get_supervisor creates an instance."""
        reset_supervisor()
        sup = get_supervisor()
        assert sup is not None
        assert isinstance(sup, ProcessSupervisor)

    def test_get_supervisor_returns_same_instance(self):
        """Test that get_supervisor returns the same instance."""
        reset_supervisor()
        sup1 = get_supervisor()
        sup2 = get_supervisor()
        assert sup1 is sup2

    def test_reset_supervisor(self):
        """Test that reset_supervisor clears the instance."""
        reset_supervisor()
        sup1 = get_supervisor()
        reset_supervisor()
        sup2 = get_supervisor()
        assert sup1 is not sup2
