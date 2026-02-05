"""
Comprehensive tests for default_tools/shell.py.

Covers: run_command, run_script, which, get_env
"""

import os
import tempfile
from pathlib import Path

import pytest

from supyagent.default_tools.shell import (
    RunCommandInput,
    RunScriptInput,
    WhichInput,
    GetEnvInput,
    run_command,
    run_script,
    which,
    get_env,
)


# =========================================================================
# run_command
# =========================================================================


class TestRunCommand:
    def test_simple_echo(self):
        result = run_command(RunCommandInput(command="echo hello"))
        assert result.ok is True
        assert result.stdout.strip() == "hello"
        assert result.exit_code == 0
        assert result.command == "echo hello"

    def test_command_with_pipe(self):
        result = run_command(RunCommandInput(command="echo 'hello world' | grep hello"))
        assert result.ok is True
        assert "hello" in result.stdout

    def test_command_exit_code_nonzero(self):
        result = run_command(RunCommandInput(command="false"))
        assert result.ok is False
        assert result.exit_code != 0

    def test_command_with_working_dir(self, tmp_path):
        result = run_command(
            RunCommandInput(command="pwd", working_dir=str(tmp_path))
        )
        assert result.ok is True
        assert str(tmp_path) in result.stdout.strip()

    def test_command_stderr(self):
        result = run_command(RunCommandInput(command="echo error >&2"))
        assert "error" in result.stderr

    def test_command_timeout(self):
        result = run_command(RunCommandInput(command="sleep 10", timeout=1))
        assert result.ok is False
        assert "timed out" in result.stderr.lower()
        assert result.exit_code == -1

    def test_command_with_env_var(self):
        result = run_command(RunCommandInput(command="echo $HOME"))
        assert result.ok is True
        assert result.stdout.strip() != ""

    def test_command_multiword_output(self):
        result = run_command(RunCommandInput(command="echo 'line1\nline2\nline3'"))
        assert result.ok is True

    def test_command_nonexistent(self):
        result = run_command(RunCommandInput(command="this_command_does_not_exist_xyz"))
        assert result.ok is False
        assert result.exit_code != 0

    def test_command_creates_file(self, tmp_path):
        filepath = tmp_path / "created.txt"
        result = run_command(
            RunCommandInput(command=f"echo test > {filepath}")
        )
        assert result.ok is True
        assert filepath.read_text().strip() == "test"


# =========================================================================
# run_script
# =========================================================================


class TestRunScript:
    def test_simple_script(self):
        result = run_script(RunScriptInput(script="echo hello\necho world"))
        assert result.ok is True
        assert "hello" in result.stdout
        assert "world" in result.stdout

    def test_script_with_variables(self):
        script = "NAME=test\necho $NAME"
        result = run_script(RunScriptInput(script=script))
        assert result.ok is True
        assert "test" in result.stdout

    def test_script_with_working_dir(self, tmp_path):
        result = run_script(
            RunScriptInput(script="pwd", working_dir=str(tmp_path))
        )
        assert result.ok is True
        assert str(tmp_path) in result.stdout

    def test_script_exit_code(self):
        result = run_script(RunScriptInput(script="exit 42"))
        assert result.ok is False
        assert result.exit_code == 42

    def test_script_timeout(self):
        result = run_script(RunScriptInput(script="sleep 10", timeout=1))
        assert result.ok is False
        assert "timed out" in result.stderr.lower()

    def test_multiline_script_with_logic(self, tmp_path):
        script = f"""
cd {tmp_path}
mkdir -p testdir
echo "hello" > testdir/file.txt
cat testdir/file.txt
"""
        result = run_script(RunScriptInput(script=script))
        assert result.ok is True
        assert "hello" in result.stdout
        assert (tmp_path / "testdir" / "file.txt").exists()

    def test_script_with_loop(self):
        script = "for i in 1 2 3; do echo $i; done"
        result = run_script(RunScriptInput(script=script))
        assert result.ok is True
        assert "1" in result.stdout
        assert "2" in result.stdout
        assert "3" in result.stdout

    def test_script_stderr_capture(self):
        result = run_script(RunScriptInput(script="echo err >&2"))
        assert "err" in result.stderr


# =========================================================================
# which
# =========================================================================


class TestWhich:
    def test_which_python(self):
        # python3 should exist on macOS
        result = which(WhichInput(command="python3"))
        assert result.ok is True
        assert result.path is not None
        assert "python3" in result.path

    def test_which_ls(self):
        result = which(WhichInput(command="ls"))
        assert result.ok is True
        assert result.path is not None

    def test_which_nonexistent(self):
        result = which(WhichInput(command="this_binary_does_not_exist_xyz"))
        assert result.ok is False
        assert "not found" in result.error.lower()

    def test_which_bash(self):
        result = which(WhichInput(command="bash"))
        assert result.ok is True
        assert "/bash" in result.path


# =========================================================================
# get_env
# =========================================================================


class TestGetEnv:
    def test_get_home(self):
        result = get_env(GetEnvInput(name="HOME"))
        assert result.ok is True
        assert result.value is not None
        assert "/" in result.value

    def test_get_path(self):
        result = get_env(GetEnvInput(name="PATH"))
        assert result.ok is True
        assert len(result.value) > 0

    def test_get_nonexistent_env(self):
        result = get_env(GetEnvInput(name="THIS_ENV_VAR_DOES_NOT_EXIST_XYZ"))
        assert result.ok is False
        assert "not set" in result.error.lower()

    def test_get_with_default(self):
        result = get_env(
            GetEnvInput(name="THIS_ENV_VAR_DOES_NOT_EXIST_XYZ", default="fallback")
        )
        assert result.ok is True
        assert result.value == "fallback"

    def test_get_env_with_set_var(self):
        os.environ["_TEST_SUPYAGENT_VAR"] = "test_value"
        try:
            result = get_env(GetEnvInput(name="_TEST_SUPYAGENT_VAR"))
            assert result.ok is True
            assert result.value == "test_value"
        finally:
            del os.environ["_TEST_SUPYAGENT_VAR"]
