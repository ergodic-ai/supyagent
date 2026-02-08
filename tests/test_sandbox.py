"""
Tests for workspace sandbox — container-based isolation and workspace validation.

Covers:
- SandboxManager lifecycle (runtime detection, start, wrap, stop, env inheritance)
- WorkspaceValidator (path checks, tool arg scanning)
- create_sandbox_context_prompt (system prompt generation)
- Config models (MountConfig, SandboxConfig)
- Integration with tools.py (discover_tools, execute_tool, workspace_validator)
- Integration with engine.py (shell gating, system prompt kwargs)
- Integration with agent.py (init ordering, sandbox sharing, cleanup)
- Integration with executor.py (sandbox init)
- Integration with delegation.py (in-process sharing, subprocess env vars)
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from supyagent.models.agent_config import (
    AgentConfig,
    ModelConfig,
    MountConfig,
    SandboxConfig,
    ServiceConfig,
    ToolPermissions,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_config(
    name: str = "test-agent",
    workspace: str | None = None,
    sandbox_enabled: bool = False,
    sandbox_kwargs: dict | None = None,
    **overrides,
) -> AgentConfig:
    """Build an AgentConfig with sandbox settings."""
    sandbox = SandboxConfig(enabled=sandbox_enabled, **(sandbox_kwargs or {}))
    return AgentConfig(
        name=name,
        model=ModelConfig(provider="test/model"),
        system_prompt="You are a test assistant.",
        tools=ToolPermissions(allow=["*"]),
        limits={"max_tool_calls_per_turn": 10},
        workspace=workspace,
        sandbox=sandbox,
        service=ServiceConfig(enabled=False),
        **overrides,
    )


# ── Config models ────────────────────────────────────────────────────


class TestSandboxConfig:
    """Tests for SandboxConfig and MountConfig Pydantic models."""

    def test_defaults(self):
        cfg = SandboxConfig()
        assert cfg.enabled is False
        assert cfg.image == "python:3.12-slim"
        assert cfg.runtime == "auto"
        assert cfg.network == "bridge"
        assert cfg.memory_limit == "2g"
        assert cfg.allow_shell is True
        assert cfg.extra_mounts == []
        assert cfg.env == {}
        assert cfg.setup_commands == []

    def test_explicit_values(self):
        cfg = SandboxConfig(
            enabled=True,
            image="ubuntu:22.04",
            runtime="podman",
            network="none",
            memory_limit="4g",
            allow_shell=False,
            env={"MY_VAR": "1"},
            setup_commands=["pip install pandas"],
        )
        assert cfg.enabled is True
        assert cfg.image == "ubuntu:22.04"
        assert cfg.runtime == "podman"
        assert cfg.network == "none"
        assert cfg.memory_limit == "4g"
        assert cfg.allow_shell is False
        assert cfg.env == {"MY_VAR": "1"}
        assert cfg.setup_commands == ["pip install pandas"]

    def test_mount_config_defaults(self):
        m = MountConfig(host_path="/data")
        assert m.host_path == "/data"
        assert m.container_path == ""
        assert m.readonly is True

    def test_mount_config_rw(self):
        m = MountConfig(host_path="/data", container_path="/mnt/data", readonly=False)
        assert m.container_path == "/mnt/data"
        assert m.readonly is False

    def test_agent_config_workspace_default_none(self):
        cfg = _make_config()
        assert cfg.workspace is None
        assert cfg.sandbox.enabled is False

    def test_agent_config_with_workspace_and_sandbox(self):
        cfg = _make_config(workspace="/tmp/ws", sandbox_enabled=True)
        assert cfg.workspace == "/tmp/ws"
        assert cfg.sandbox.enabled is True


# ── SandboxManager unit tests ────────────────────────────────────────


class TestSandboxManager:
    """Tests for the SandboxManager class."""

    def _make_manager(self, workspace="/tmp/ws", **kwargs):
        from supyagent.core.sandbox import SandboxManager

        config = SandboxConfig(**kwargs)
        return SandboxManager(Path(workspace), config, session_id="abcdef123456789")

    @patch("shutil.which", return_value="/usr/bin/podman")
    def test_detect_runtime_auto_podman(self, mock_which):
        mgr = self._make_manager()
        assert mgr.runtime == "podman"
        mock_which.assert_called_with("podman")

    @patch("shutil.which", side_effect=lambda cmd: "/usr/bin/docker" if cmd == "docker" else None)
    def test_detect_runtime_auto_docker_fallback(self, mock_which):
        mgr = self._make_manager()
        assert mgr.runtime == "docker"

    @patch("shutil.which", return_value=None)
    def test_detect_runtime_auto_none_raises(self, mock_which):
        from supyagent.core.sandbox import SandboxError

        mgr = self._make_manager()
        with pytest.raises(SandboxError, match="No container runtime found"):
            _ = mgr.runtime

    @patch("shutil.which", return_value="/usr/bin/podman")
    def test_detect_runtime_explicit(self, mock_which):
        mgr = self._make_manager(runtime="podman")
        assert mgr.runtime == "podman"

    @patch("shutil.which", return_value=None)
    def test_detect_runtime_explicit_missing(self, mock_which):
        from supyagent.core.sandbox import SandboxError

        mgr = self._make_manager(runtime="podman")
        with pytest.raises(SandboxError, match="not found on PATH"):
            _ = mgr.runtime

    def test_container_name_from_session_id(self):
        mgr = self._make_manager()
        assert mgr.container_name == "supyagent-abcdef123456"

    def test_container_name_stable(self):
        mgr = self._make_manager()
        name1 = mgr.container_name
        name2 = mgr.container_name
        assert name1 == name2

    @patch("supyagent.core.sandbox.subprocess.run")
    @patch("shutil.which", return_value="/usr/bin/podman")
    def test_wrap_command(self, mock_which, mock_run):
        """Test that wrap_command prepends runtime exec prefix."""
        mgr = self._make_manager()
        # Pretend already started
        mgr._started = True

        mock_run.return_value = MagicMock(returncode=0, stdout="true")  # for _is_container_running

        wrapped = mgr.wrap_command(["supypowers", "run", "test:func", "{}"])

        assert wrapped[0] == "podman"
        assert wrapped[1] == "exec"
        assert mgr.container_name in wrapped
        assert wrapped[-4:] == ["supypowers", "run", "test:func", "{}"]

    @patch("supyagent.core.sandbox.subprocess.run")
    @patch("shutil.which", return_value="/usr/bin/podman")
    def test_wrap_command_with_env(self, mock_which, mock_run):
        """Test that wrap_command passes env vars."""
        mgr = self._make_manager()
        mgr._started = True
        mock_run.return_value = MagicMock(returncode=0, stdout="true")

        wrapped = mgr.wrap_command(
            ["supypowers", "run", "test:func", "{}"],
            env={"API_KEY": "secret"},
        )

        assert "--env" in wrapped
        env_idx = wrapped.index("--env")
        assert wrapped[env_idx + 1] == "API_KEY=secret"

    @patch("supyagent.core.sandbox.subprocess.run")
    @patch("shutil.which", return_value="/usr/bin/podman")
    def test_start_container_command(self, mock_which, mock_run):
        """Test the container start command structure."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        mgr = self._make_manager(
            memory_limit="4g",
            network="none",
            image="ubuntu:22.04",
        )
        mgr._start_container()

        # First call is _remove_if_exists, second is docker/podman run
        assert mock_run.call_count >= 2
        run_call = mock_run.call_args_list[1]
        cmd = run_call[0][0]

        assert cmd[0] == "podman"
        assert "run" in cmd
        assert "--detach" in cmd
        assert "--name" in cmd
        assert mgr.container_name in cmd
        assert "--memory" in cmd
        assert "4g" in cmd
        assert "--network" in cmd
        assert "none" in cmd
        assert "ubuntu:22.04" in cmd

    @patch("supyagent.core.sandbox.subprocess.run")
    @patch("shutil.which", return_value="/usr/bin/podman")
    def test_start_container_failure_raises(self, mock_which, mock_run):
        from supyagent.core.sandbox import SandboxError

        mock_run.side_effect = [
            MagicMock(returncode=0),  # _remove_if_exists
            MagicMock(returncode=1, stderr="image not found"),  # run fails
        ]

        mgr = self._make_manager()
        with pytest.raises(SandboxError, match="Failed to start sandbox container"):
            mgr._start_container()

    @patch("supyagent.core.sandbox.subprocess.run")
    @patch("shutil.which", return_value="/usr/bin/podman")
    def test_ensure_started_lazy(self, mock_which, mock_run):
        """Test that ensure_started only starts once."""
        mock_run.return_value = MagicMock(returncode=0, stdout="true")

        mgr = self._make_manager()
        mgr._started = True

        # Container is "running"
        mgr.ensure_started()
        mgr.ensure_started()

        # _is_container_running is called, but _start_container is not
        # (no rm -f call expected since it's already running)
        for call in mock_run.call_args_list:
            cmd = call[0][0]
            assert "run" not in cmd or cmd[1] != "run"  # no `podman run`

    @patch("supyagent.core.sandbox.subprocess.run")
    @patch("shutil.which", return_value="/usr/bin/podman")
    def test_stop_container(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(returncode=0)

        mgr = self._make_manager()
        mgr._started = True
        _ = mgr.container_name  # Force name generation

        mgr.stop()

        # Should have called stop then rm
        cmds = [call[0][0] for call in mock_run.call_args_list]
        stop_called = any("stop" in cmd for cmd in cmds)
        rm_called = any("rm" in cmd for cmd in cmds)
        assert stop_called
        assert rm_called
        assert mgr._started is False

    def test_stop_noop_when_no_container(self):
        """Stop should be safe when no container was ever started."""
        mgr = self._make_manager()
        mgr.stop()  # Should not raise

    @patch.dict("os.environ", {
        "SUPYAGENT_SANDBOX_CONTAINER": "supyagent-parent123",
        "SUPYAGENT_SANDBOX_RUNTIME": "docker",
    })
    def test_inherit_container_from_env(self):
        """Test that child process inherits container from env vars."""
        mgr = self._make_manager()
        assert mgr._container_name == "supyagent-parent123"
        assert mgr._runtime == "docker"
        assert mgr._started is True

    @patch("supyagent.core.sandbox.subprocess.run")
    @patch("shutil.which", return_value="/usr/bin/podman")
    def test_extra_mounts_in_start(self, mock_which, mock_run):
        """Test that extra_mounts are included in container start."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        mgr = self._make_manager(
            extra_mounts=[
                MountConfig(host_path="/tmp/data", container_path="/mnt/data", readonly=True),
                MountConfig(host_path="/tmp/models", readonly=False),
            ]
        )
        mgr._start_container()

        run_call = mock_run.call_args_list[1]
        cmd = " ".join(run_call[0][0])
        assert "/mnt/data:ro" in cmd
        assert "models:rw" in cmd

    @patch("supyagent.core.sandbox.subprocess.run")
    @patch("shutil.which", return_value="/usr/bin/podman")
    def test_env_vars_in_start(self, mock_which, mock_run):
        """Test that env vars are passed to container."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        mgr = self._make_manager(env={"MY_KEY": "my_val"})
        mgr._start_container()

        run_call = mock_run.call_args_list[1]
        cmd = run_call[0][0]
        assert "--env" in cmd
        # Find the user-defined env var (built-in HOME/UV_CACHE_DIR/PIP_CACHE_DIR come first)
        env_pairs = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--env"]
        assert "MY_KEY=my_val" in env_pairs

    @patch("supyagent.core.sandbox.subprocess.run")
    @patch("shutil.which", return_value="/usr/bin/podman")
    def test_setup_commands_run(self, mock_which, mock_run):
        """Test that setup_commands are executed after start."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        mgr = self._make_manager(setup_commands=["pip install pandas"])
        mgr._start_container()

        # Find the setup command call (sh -c "pip install pandas")
        found_setup = False
        for call in mock_run.call_args_list:
            cmd = call[0][0]
            if "pip install pandas" in " ".join(cmd):
                found_setup = True
                break
        assert found_setup, "Setup command was not executed"


# ── WorkspaceValidator tests ─────────────────────────────────────────


class TestWorkspaceValidator:
    """Tests for soft workspace boundary validation."""

    def _make_validator(self, workspace="/tmp/workspace"):
        from supyagent.core.sandbox import WorkspaceValidator

        return WorkspaceValidator(Path(workspace))

    def test_path_inside_workspace_ok(self, tmp_path):
        from supyagent.core.sandbox import WorkspaceValidator

        ws = tmp_path / "workspace"
        ws.mkdir()
        target = ws / "data" / "file.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()

        v = WorkspaceValidator(ws)
        assert v.validate_path(str(target)) is None

    def test_path_outside_workspace_fails(self, tmp_path):
        from supyagent.core.sandbox import WorkspaceValidator

        ws = tmp_path / "workspace"
        ws.mkdir()
        outside = tmp_path / "other" / "secret.txt"
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.touch()

        v = WorkspaceValidator(ws)
        error = v.validate_path(str(outside))
        assert error is not None
        assert "outside the workspace" in error

    def test_check_tool_args_path_key(self, tmp_path):
        from supyagent.core.sandbox import WorkspaceValidator

        ws = tmp_path / "workspace"
        ws.mkdir()

        v = WorkspaceValidator(ws)

        # Absolute path outside workspace
        error = v.check_tool_args("files__read_file", {"path": "/etc/passwd"})
        assert error is not None
        assert "outside the workspace" in error

    def test_check_tool_args_file_path_key(self, tmp_path):
        from supyagent.core.sandbox import WorkspaceValidator

        ws = tmp_path / "workspace"
        ws.mkdir()

        v = WorkspaceValidator(ws)
        error = v.check_tool_args("files__write_file", {"file_path": "/etc/shadow"})
        assert error is not None

    def test_check_tool_args_relative_path_ignored(self, tmp_path):
        from supyagent.core.sandbox import WorkspaceValidator

        ws = tmp_path / "workspace"
        ws.mkdir()

        v = WorkspaceValidator(ws)
        # Relative paths are not validated (only absolute or ~)
        error = v.check_tool_args("files__read_file", {"path": "data/file.csv"})
        assert error is None

    def test_check_tool_args_non_path_key_ignored(self, tmp_path):
        from supyagent.core.sandbox import WorkspaceValidator

        ws = tmp_path / "workspace"
        ws.mkdir()

        v = WorkspaceValidator(ws)
        # "query" is not a path key, so absolute path is fine
        error = v.check_tool_args("db__query", {"query": "/etc/passwd"})
        assert error is None

    def test_check_tool_args_inside_workspace_ok(self, tmp_path):
        from supyagent.core.sandbox import WorkspaceValidator

        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "data.csv").touch()

        v = WorkspaceValidator(ws)
        error = v.check_tool_args("files__read_file", {"path": str(ws / "data.csv")})
        assert error is None

    def test_check_tool_args_home_relative(self, tmp_path):
        from supyagent.core.sandbox import WorkspaceValidator

        ws = tmp_path / "workspace"
        ws.mkdir()

        v = WorkspaceValidator(ws)
        # ~ expands to home dir which is outside workspace
        error = v.check_tool_args("files__read_file", {"path": "~/secret.txt"})
        assert error is not None


# ── create_sandbox_context_prompt tests ──────────────────────────────


class TestSandboxContextPrompt:
    """Tests for system prompt generation."""

    def test_no_workspace_empty(self):
        from supyagent.core.sandbox import create_sandbox_context_prompt

        config = _make_config()
        prompt = create_sandbox_context_prompt(config, None)
        assert prompt == ""

    def test_sandbox_active(self):
        from supyagent.core.sandbox import create_sandbox_context_prompt

        config = _make_config(workspace="/tmp/ws", sandbox_enabled=True)
        mock_sandbox = MagicMock()

        prompt = create_sandbox_context_prompt(config, mock_sandbox)
        assert "sandboxed environment" in prompt
        assert "relative paths" in prompt

    def test_sandbox_with_mounts(self):
        from supyagent.core.sandbox import create_sandbox_context_prompt

        config = _make_config(
            workspace="/tmp/ws",
            sandbox_enabled=True,
            sandbox_kwargs={
                "extra_mounts": [
                    MountConfig(host_path="/tmp/templates", container_path="/mnt/templates", readonly=True),
                ],
            },
        )
        mock_sandbox = MagicMock()

        prompt = create_sandbox_context_prompt(config, mock_sandbox)
        assert "/mnt/templates" in prompt
        assert "read-only" in prompt

    def test_sandbox_shell_disabled(self):
        from supyagent.core.sandbox import create_sandbox_context_prompt

        config = _make_config(
            workspace="/tmp/ws",
            sandbox_enabled=True,
            sandbox_kwargs={"allow_shell": False},
        )
        mock_sandbox = MagicMock()

        prompt = create_sandbox_context_prompt(config, mock_sandbox)
        assert "Shell/exec tools are disabled" in prompt

    def test_soft_mode_no_sandbox(self):
        from supyagent.core.sandbox import create_sandbox_context_prompt

        config = _make_config(workspace="/tmp/ws", sandbox_enabled=False)

        prompt = create_sandbox_context_prompt(config, None)
        assert "Workspace" in prompt
        assert "/tmp/ws" in prompt
        assert "within the workspace" in prompt


# ── Tools integration tests ──────────────────────────────────────────


class TestToolsSandboxIntegration:
    """Tests for sandbox params in tools.py functions."""

    @patch("supyagent.core.tools.subprocess.run")
    def test_discover_tools_with_sandbox(self, mock_run):
        """Test that discover_tools wraps command via sandbox."""
        from supyagent.core.tools import discover_tools

        mock_sandbox = MagicMock()
        mock_sandbox.wrap_command.return_value = [
            "podman", "exec", "supyagent-abc", "supypowers", "docs", "--format", "json"
        ]

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps([{"script": "test.py", "functions": []}]),
        )

        tools = discover_tools(sandbox=mock_sandbox)

        mock_sandbox.ensure_started.assert_called_once()
        mock_sandbox.wrap_command.assert_called_once()
        # The actual subprocess should receive the wrapped command
        call_cmd = mock_run.call_args[0][0]
        assert call_cmd[0] == "podman"
        assert len(tools) == 1

    @patch("supyagent.core.tools.subprocess.run")
    def test_discover_tools_without_sandbox(self, mock_run):
        """Test that discover_tools works normally without sandbox."""
        from supyagent.core.tools import discover_tools

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps([{"script": "test.py", "functions": []}]),
        )

        tools = discover_tools(sandbox=None)

        call_cmd = mock_run.call_args[0][0]
        assert call_cmd[0] == "supypowers"
        assert len(tools) == 1

    @patch("supyagent.core.tools.subprocess.run")
    def test_execute_tool_with_sandbox(self, mock_run):
        """Test that execute_tool wraps command via sandbox."""
        from supyagent.core.tools import execute_tool

        mock_sandbox = MagicMock()
        mock_sandbox.wrap_command.return_value = [
            "podman", "exec", "supyagent-abc",
            "supypowers", "run", "test:func", '{"x": 1}',
        ]

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"ok": true, "data": "result"}',
        )

        result = execute_tool("test", "func", {"x": 1}, sandbox=mock_sandbox)

        assert result["ok"] is True
        mock_sandbox.wrap_command.assert_called_once()
        call_cmd = mock_run.call_args[0][0]
        assert call_cmd[0] == "podman"

    @patch("supyagent.core.tools.subprocess.run")
    def test_execute_tool_workspace_validator_blocks(self, mock_run):
        """Test that workspace_validator blocks out-of-bounds paths."""
        from supyagent.core.tools import execute_tool

        mock_validator = MagicMock()
        mock_validator.check_tool_args.return_value = (
            "Path '/etc/passwd' is outside the workspace '/tmp/ws'."
        )

        result = execute_tool(
            "files", "read_file",
            {"path": "/etc/passwd"},
            workspace_validator=mock_validator,
        )

        assert result["ok"] is False
        assert "outside the workspace" in result["error"]
        mock_run.assert_not_called()  # Should NOT execute the tool

    @patch("supyagent.core.tools.subprocess.run")
    def test_execute_tool_workspace_validator_allows(self, mock_run):
        """Test that workspace_validator allows valid paths."""
        from supyagent.core.tools import execute_tool

        mock_validator = MagicMock()
        mock_validator.check_tool_args.return_value = None  # No error

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"ok": true, "data": "content"}',
        )

        result = execute_tool(
            "files", "read_file",
            {"path": "/tmp/ws/data.csv"},
            workspace_validator=mock_validator,
        )

        assert result["ok"] is True
        mock_run.assert_called_once()

    def test_execute_tool_sandbox_via_supervisor(self):
        """Test that sandbox is passed through the supervisor path."""
        from supyagent.core.tools import execute_tool

        mock_sandbox = MagicMock()

        with patch("supyagent.core.tools._execute_tool_sync_via_supervisor") as mock_sup:
            mock_sup.return_value = {"ok": True, "data": "result"}
            result = execute_tool(
                "test", "func", {},
                timeout=30,
                sandbox=mock_sandbox,
            )

        assert result["ok"] is True
        # Verify sandbox was passed to the supervisor path
        mock_sup.assert_called_once()
        call_kwargs = mock_sup.call_args
        assert call_kwargs[1]["sandbox"] is mock_sandbox


# ── Engine integration tests ─────────────────────────────────────────


class TestEngineSandboxIntegration:
    """Tests for sandbox wiring in BaseAgentEngine."""

    def test_engine_sandbox_attrs_default_none(self):
        """Test that engine has sandbox attrs initialized to None."""
        config = _make_config()

        with patch("supyagent.core.engine.discover_tools", return_value=[]):
            from supyagent.core.engine import BaseAgentEngine

            # Can't instantiate ABC directly, so check via Agent
            # Instead, verify the attributes exist on a concrete subclass
            from supyagent.core.executor import ExecutionRunner

            runner = ExecutionRunner(config)
            assert runner.sandbox_mgr is None
            assert runner.workspace_validator is None

    @patch("supyagent.core.engine.discover_tools", return_value=[])
    def test_load_base_tools_passes_sandbox(self, mock_discover):
        """Test that _load_base_tools passes sandbox to discover_tools."""
        config = _make_config()
        from supyagent.core.executor import ExecutionRunner

        runner = ExecutionRunner(config)
        runner.sandbox_mgr = MagicMock()

        runner._load_base_tools()

        mock_discover.assert_called_with(sandbox=runner.sandbox_mgr)

    @patch("supyagent.core.engine.discover_tools", return_value=[])
    def test_shell_tool_gating(self, mock_discover):
        """Test that shell tools are filtered when sandbox disallows shell."""
        from supyagent.core.engine import supypowers_to_openai_tools

        config = _make_config(
            workspace="/tmp/ws",
            sandbox_enabled=True,
            sandbox_kwargs={"allow_shell": False},
        )

        from supyagent.core.executor import ExecutionRunner

        runner = ExecutionRunner(config)

        # Simulate tools including shell tools
        mock_tools = [
            {"function": {"name": "shell__exec"}},
            {"function": {"name": "shell__run"}},
            {"function": {"name": "files__read_file"}},
        ]

        with patch("supyagent.core.engine.supypowers_to_openai_tools", return_value=mock_tools):
            with patch("supyagent.core.engine.filter_tools", return_value=mock_tools):
                tools = runner._load_base_tools()

        tool_names = [t.get("function", {}).get("name", "") for t in tools]
        assert "shell__exec" not in tool_names
        assert "shell__run" not in tool_names
        assert "files__read_file" in tool_names

    @patch("supyagent.core.engine.discover_tools", return_value=[])
    def test_shell_tools_allowed_when_flag_true(self, mock_discover):
        """Test that shell tools are kept when allow_shell is True."""
        config = _make_config(
            workspace="/tmp/ws",
            sandbox_enabled=True,
            sandbox_kwargs={"allow_shell": True},
        )

        from supyagent.core.executor import ExecutionRunner

        runner = ExecutionRunner(config)

        mock_tools = [
            {"function": {"name": "shell__exec"}},
            {"function": {"name": "files__read_file"}},
        ]

        with patch("supyagent.core.engine.supypowers_to_openai_tools", return_value=mock_tools):
            with patch("supyagent.core.engine.filter_tools", return_value=mock_tools):
                tools = runner._load_base_tools()

        tool_names = [t.get("function", {}).get("name", "") for t in tools]
        assert "shell__exec" in tool_names

    @patch("supyagent.core.engine.discover_tools", return_value=[])
    def test_system_prompt_includes_sandbox_context(self, mock_discover):
        """Test that _system_prompt_kwargs includes sandbox context."""
        config = _make_config(workspace="/tmp/ws", sandbox_enabled=True)
        from supyagent.core.executor import ExecutionRunner

        runner = ExecutionRunner(config)
        runner.sandbox_mgr = MagicMock()

        kwargs = runner._system_prompt_kwargs()
        assert "sandbox_context" in kwargs
        assert "sandboxed environment" in kwargs["sandbox_context"]

    @patch("supyagent.core.engine.discover_tools", return_value=[])
    def test_execute_supypowers_tool_passes_sandbox(self, mock_discover):
        """Test that _execute_supypowers_tool passes sandbox and validator."""
        config = _make_config()
        from supyagent.core.executor import ExecutionRunner

        runner = ExecutionRunner(config)
        runner.sandbox_mgr = MagicMock()
        runner.workspace_validator = MagicMock()
        runner.workspace_validator.check_tool_args.return_value = None

        tool_call = MagicMock()
        tool_call.function.name = "test__func"
        tool_call.function.arguments = '{"x": 1}'

        with patch("supyagent.core.engine.execute_tool") as mock_exec:
            mock_exec.return_value = {"ok": True, "data": "result"}
            runner._execute_supypowers_tool(tool_call)

        mock_exec.assert_called_once()
        call_kwargs = mock_exec.call_args[1]
        assert call_kwargs["sandbox"] is runner.sandbox_mgr
        assert call_kwargs["workspace_validator"] is runner.workspace_validator


# ── Agent integration tests ──────────────────────────────────────────


class TestAgentSandboxIntegration:
    """Tests for sandbox in Agent.__init__ and lifecycle."""

    @patch("supyagent.core.engine.discover_tools", return_value=[])
    def test_agent_no_sandbox_by_default(self, mock_discover, sessions_dir):
        from supyagent.core.agent import Agent
        from supyagent.core.session_manager import SessionManager

        config = _make_config()
        agent = Agent(config, session_manager=SessionManager(base_dir=sessions_dir))

        assert agent.sandbox_mgr is None
        assert agent.workspace_validator is None

    @patch("supyagent.core.engine.discover_tools", return_value=[])
    def test_agent_creates_sandbox_manager(self, mock_discover, sessions_dir):
        from supyagent.core.agent import Agent
        from supyagent.core.sandbox import SandboxManager
        from supyagent.core.session_manager import SessionManager

        config = _make_config(workspace="/tmp/ws", sandbox_enabled=True)

        with patch.object(SandboxManager, "_detect_runtime", return_value="podman"):
            agent = Agent(config, session_manager=SessionManager(base_dir=sessions_dir))

        assert agent.sandbox_mgr is not None
        assert isinstance(agent.sandbox_mgr, SandboxManager)

    @patch("supyagent.core.engine.discover_tools", return_value=[])
    def test_agent_creates_workspace_validator(self, mock_discover, sessions_dir):
        from supyagent.core.agent import Agent
        from supyagent.core.sandbox import WorkspaceValidator
        from supyagent.core.session_manager import SessionManager

        config = _make_config(workspace="/tmp/ws", sandbox_enabled=False)
        agent = Agent(config, session_manager=SessionManager(base_dir=sessions_dir))

        assert agent.sandbox_mgr is None
        assert agent.workspace_validator is not None
        assert isinstance(agent.workspace_validator, WorkspaceValidator)

    @patch("supyagent.core.engine.discover_tools", return_value=[])
    def test_agent_uses_shared_sandbox(self, mock_discover, sessions_dir):
        """Test that agent uses shared sandbox_mgr from parent."""
        from supyagent.core.agent import Agent
        from supyagent.core.session_manager import SessionManager

        config = _make_config(workspace="/tmp/ws", sandbox_enabled=True)
        shared_sandbox = MagicMock()

        agent = Agent(
            config,
            session_manager=SessionManager(base_dir=sessions_dir),
            sandbox_mgr=shared_sandbox,
        )

        assert agent.sandbox_mgr is shared_sandbox

    @patch("supyagent.core.engine.discover_tools", return_value=[])
    def test_agent_clear_history_stops_sandbox(self, mock_discover, sessions_dir):
        """Test that clear_history stops the sandbox container."""
        from supyagent.core.agent import Agent
        from supyagent.core.session_manager import SessionManager

        config = _make_config(workspace="/tmp/ws", sandbox_enabled=True)
        mock_sandbox = MagicMock()

        agent = Agent(
            config,
            session_manager=SessionManager(base_dir=sessions_dir),
            sandbox_mgr=mock_sandbox,
        )

        agent.clear_history()
        mock_sandbox.stop.assert_called_once()


# ── Executor integration tests ───────────────────────────────────────


class TestExecutorSandboxIntegration:
    """Tests for sandbox in ExecutionRunner."""

    @patch("supyagent.core.engine.discover_tools", return_value=[])
    def test_executor_no_sandbox_by_default(self, mock_discover):
        from supyagent.core.executor import ExecutionRunner

        config = _make_config()
        runner = ExecutionRunner(config)

        assert runner.sandbox_mgr is None
        assert runner.workspace_validator is None

    @patch("supyagent.core.engine.discover_tools", return_value=[])
    def test_executor_creates_sandbox_with_uuid(self, mock_discover):
        """Test that executor creates sandbox with uuid session_id."""
        from supyagent.core.executor import ExecutionRunner
        from supyagent.core.sandbox import SandboxManager

        config = _make_config(workspace="/tmp/ws", sandbox_enabled=True)

        with patch.object(SandboxManager, "_detect_runtime", return_value="podman"):
            runner = ExecutionRunner(config)

        assert runner.sandbox_mgr is not None
        assert runner.sandbox_mgr.container_name.startswith("supyagent-")

    @patch("supyagent.core.engine.discover_tools", return_value=[])
    def test_executor_creates_workspace_validator(self, mock_discover):
        from supyagent.core.executor import ExecutionRunner

        config = _make_config(workspace="/tmp/ws", sandbox_enabled=False)
        runner = ExecutionRunner(config)

        assert runner.workspace_validator is not None
        assert runner.sandbox_mgr is None

    @patch("supyagent.core.engine.discover_tools", return_value=[])
    def test_executor_uses_shared_sandbox(self, mock_discover):
        from supyagent.core.executor import ExecutionRunner

        config = _make_config(workspace="/tmp/ws", sandbox_enabled=True)
        shared_sandbox = MagicMock()

        runner = ExecutionRunner(config, sandbox_mgr=shared_sandbox)

        assert runner.sandbox_mgr is shared_sandbox


# ── Delegation integration tests ─────────────────────────────────────


class TestDelegationSandboxIntegration:
    """Tests for sandbox sharing in delegation."""

    def _make_parent_agent(self, sandbox_mgr=None, delegates=None):
        mock_agent = MagicMock()
        mock_agent.config = _make_config(
            workspace="/tmp/ws",
            sandbox_enabled=True,
            delegates=delegates or ["child-agent"],
        )
        mock_agent.config.delegates = delegates or ["child-agent"]
        mock_agent.sandbox_mgr = sandbox_mgr
        mock_agent.messages = []
        mock_agent.llm = MagicMock()
        return mock_agent

    @patch("supyagent.core.delegation.load_agent_config")
    def test_in_process_shares_sandbox(self, mock_load, temp_dir):
        """Test that in-process delegation passes sandbox_mgr to child."""
        from supyagent.core.delegation import DelegationManager
        from supyagent.core.registry import AgentRegistry

        child_config = _make_config(name="child-agent", type="execution")
        mock_load.return_value = child_config

        mock_sandbox = MagicMock()
        parent = self._make_parent_agent(sandbox_mgr=mock_sandbox)

        registry = AgentRegistry(base_dir=temp_dir)
        mgr = DelegationManager(registry, parent)

        with patch("supyagent.core.executor.ExecutionRunner") as MockRunner:
            mock_runner = MagicMock()
            mock_runner.run.return_value = {"ok": True, "data": "done"}
            MockRunner.return_value = mock_runner

            mgr._delegate_in_process("child-agent", child_config, "do something")

            MockRunner.assert_called_once()
            call_kwargs = MockRunner.call_args
            assert call_kwargs[1]["sandbox_mgr"] is mock_sandbox

    @patch("supyagent.core.delegation.load_agent_config")
    def test_in_process_agent_shares_sandbox(self, mock_load, temp_dir):
        """Test that in-process Agent delegation passes sandbox_mgr."""
        from supyagent.core.delegation import DelegationManager
        from supyagent.core.registry import AgentRegistry

        child_config = _make_config(name="child-agent", type="interactive")
        mock_load.return_value = child_config

        mock_sandbox = MagicMock()
        parent = self._make_parent_agent(sandbox_mgr=mock_sandbox)

        registry = AgentRegistry(base_dir=temp_dir)
        mgr = DelegationManager(registry, parent)

        with patch("supyagent.core.agent.Agent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.send_message.return_value = "done"
            mock_agent.instance_id = "test-123"
            MockAgent.return_value = mock_agent

            mgr._delegate_in_process("child-agent", child_config, "do something")

            MockAgent.assert_called_once()
            call_kwargs = MockAgent.call_args
            assert call_kwargs[1]["sandbox_mgr"] is mock_sandbox

    @patch("supyagent.core.delegation.load_agent_config")
    def test_subprocess_passes_env_vars(self, mock_load, temp_dir):
        """Test that subprocess delegation passes sandbox env vars."""
        from supyagent.core.delegation import DelegationManager
        from supyagent.core.registry import AgentRegistry

        mock_sandbox = MagicMock()
        mock_sandbox._started = True
        mock_sandbox.container_name = "supyagent-parent123"
        mock_sandbox.runtime = "podman"

        parent = self._make_parent_agent(
            sandbox_mgr=mock_sandbox,
            delegates=["child-agent"],
        )

        child_config = _make_config(name="child-agent")
        mock_load.return_value = child_config

        registry = AgentRegistry(base_dir=temp_dir)
        mgr = DelegationManager(registry, parent)

        mock_context = MagicMock()
        mock_context.parent_agent = "test-agent"
        mock_context.parent_task = "test task"
        mock_context.conversation_summary = None
        mock_context.relevant_facts = []

        with patch("supyagent.core.supervisor.run_supervisor_coroutine") as mock_run:
            with patch("supyagent.core.supervisor.get_supervisor") as mock_get_sup:
                mock_supervisor = MagicMock()
                mock_get_sup.return_value = mock_supervisor
                mock_run.return_value = {"ok": True, "data": "done"}

                mgr._delegate_subprocess(
                    "child-agent", "do something", mock_context, False, 300
                )

                # Verify env was passed with sandbox container info
                execute_call = mock_supervisor.execute
                execute_call.assert_called_once()
                call_kwargs = execute_call.call_args[1]
                assert call_kwargs["env"] == {
                    "SUPYAGENT_SANDBOX_CONTAINER": "supyagent-parent123",
                    "SUPYAGENT_SANDBOX_RUNTIME": "podman",
                }

    @patch("supyagent.core.delegation.load_agent_config")
    def test_subprocess_no_env_vars_without_sandbox(self, mock_load, temp_dir):
        """Test that subprocess delegation doesn't pass sandbox env when no sandbox."""
        from supyagent.core.delegation import DelegationManager
        from supyagent.core.registry import AgentRegistry

        parent = self._make_parent_agent(sandbox_mgr=None, delegates=["child-agent"])

        child_config = _make_config(name="child-agent")
        mock_load.return_value = child_config

        registry = AgentRegistry(base_dir=temp_dir)
        mgr = DelegationManager(registry, parent)

        mock_context = MagicMock()
        mock_context.parent_agent = "test-agent"
        mock_context.parent_task = "test task"
        mock_context.conversation_summary = None
        mock_context.relevant_facts = []

        with patch("supyagent.core.supervisor.run_supervisor_coroutine") as mock_run:
            with patch("supyagent.core.supervisor.get_supervisor") as mock_get_sup:
                mock_supervisor = MagicMock()
                mock_get_sup.return_value = mock_supervisor
                mock_run.return_value = {"ok": True, "data": "done"}

                mgr._delegate_subprocess(
                    "child-agent", "do something", mock_context, False, 300
                )

                # Verify env=None was passed (no sandbox env vars)
                execute_call = mock_supervisor.execute
                execute_call.assert_called_once()
                call_kwargs = execute_call.call_args[1]
                assert call_kwargs.get("env") is None


# ── End-to-end integration test ──────────────────────────────────────


class TestSandboxEndToEnd:
    """Higher-level integration tests combining multiple components."""

    @patch("supyagent.core.tools.subprocess.run")
    @patch("supyagent.core.engine.discover_tools", return_value=[])
    def test_tool_execution_blocked_by_validator(self, mock_discover, mock_run, sessions_dir):
        """Full path: agent with workspace, no sandbox, tool call with out-of-bounds path."""
        from supyagent.core.agent import Agent
        from supyagent.core.session_manager import SessionManager

        config = _make_config(workspace="/tmp/ws", sandbox_enabled=False)
        agent = Agent(config, session_manager=SessionManager(base_dir=sessions_dir))

        # Verify validator is set up
        assert agent.workspace_validator is not None

        # Simulate a tool call to _execute_supypowers_tool
        tool_call = MagicMock()
        tool_call.function.name = "files__read_file"
        tool_call.function.arguments = json.dumps({"path": "/etc/passwd"})

        result = agent._execute_supypowers_tool(tool_call)

        assert result["ok"] is False
        assert "outside the workspace" in result["error"]
        mock_run.assert_not_called()  # Tool should NOT have been executed

    @patch("supyagent.core.engine.discover_tools", return_value=[])
    def test_full_sandbox_init_chain(self, mock_discover, sessions_dir):
        """Test that sandbox init happens in correct order: session -> sandbox -> tools."""
        from supyagent.core.agent import Agent
        from supyagent.core.sandbox import SandboxManager
        from supyagent.core.session_manager import SessionManager

        config = _make_config(workspace="/tmp/ws", sandbox_enabled=True)

        with patch.object(SandboxManager, "_detect_runtime", return_value="podman"):
            agent = Agent(config, session_manager=SessionManager(base_dir=sessions_dir))

        # Session should exist before sandbox
        assert agent.session is not None
        # Sandbox should use session_id
        assert agent.sandbox_mgr is not None
        assert agent.session.meta.session_id[:12] in agent.sandbox_mgr.container_name
        # Tools should be loaded
        assert isinstance(agent.tools, list)
