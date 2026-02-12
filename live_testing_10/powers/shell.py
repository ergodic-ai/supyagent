# /// script
# dependencies = ["pydantic"]
# ///
"""
Shell command execution tools.

Allows agents to run bash/shell commands on the system.
Use with caution - consider restricting via agent tool permissions.
"""

import os
import shutil
import subprocess
from typing import Optional

from pydantic import BaseModel, Field


class RunCommandInput(BaseModel):
    """Input for run_command function."""

    command: str = Field(description="The shell command to execute")
    working_dir: Optional[str] = Field(
        default=None, description="Working directory for the command"
    )
    timeout: int = Field(
        default=60, description="Maximum seconds to wait for command"
    )
    max_output: int = Field(
        default=100000,
        description="Maximum bytes for stdout+stderr combined (0 = unlimited)",
    )


class RunCommandOutput(BaseModel):
    """Output for run_command function."""

    ok: bool
    stdout: str
    stderr: str
    exit_code: int
    command: str


def _truncate_output(text: str, max_bytes: int) -> str:
    """Truncate output preserving start, end, and error lines from middle."""
    if max_bytes <= 0 or len(text) <= max_bytes:
        return text

    error_patterns = ["error", "Error", "ERROR", "failed", "Failed", "FAILED",
                      "exception", "Exception", "Traceback", "panic", "PANIC"]

    lines = text.splitlines(keepends=True)
    keep_start = int(max_bytes * 0.4)
    keep_end = int(max_bytes * 0.4)

    prefix_lines, prefix_bytes = [], 0
    for line in lines:
        if prefix_bytes + len(line) > keep_start:
            break
        prefix_lines.append(line)
        prefix_bytes += len(line)

    suffix_lines, suffix_bytes = [], 0
    for line in reversed(lines):
        if suffix_bytes + len(line) > keep_end:
            break
        suffix_lines.insert(0, line)
        suffix_bytes += len(line)

    middle_start = len(prefix_lines)
    middle_end = len(lines) - len(suffix_lines)
    error_lines = []
    for i in range(middle_start, min(middle_end, len(lines))):
        if any(p in lines[i] for p in error_patterns):
            error_lines.append(lines[i])

    truncated_bytes = len(text) - prefix_bytes - suffix_bytes
    if error_lines:
        separator = f"\n[... truncated {truncated_bytes} bytes, key lines preserved ...]\n"
        separator += "".join(error_lines[:20])
        separator += "\n[...]\n"
    else:
        separator = f"\n[... truncated {truncated_bytes} bytes ...]\n"

    return "".join(prefix_lines) + separator + "".join(suffix_lines)


def run_command(input: RunCommandInput) -> RunCommandOutput:
    """
    Execute a shell command and return the result.

    Examples:
        >>> run_command({"command": "ls -la"})
        >>> run_command({"command": "echo 'hello' | grep hello"})
        >>> run_command({"command": "pwd", "working_dir": "/tmp"})
    """
    try:
        working_dir = input.working_dir
        if working_dir:
            working_dir = os.path.expanduser(working_dir)
            working_dir = os.path.expandvars(working_dir)

        result = subprocess.run(
            input.command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=input.timeout,
            cwd=working_dir,
        )

        stdout = _truncate_output(result.stdout, input.max_output)
        stderr = _truncate_output(result.stderr, input.max_output)

        return RunCommandOutput(
            ok=result.returncode == 0,
            stdout=stdout,
            stderr=stderr,
            exit_code=result.returncode,
            command=input.command,
        )

    except subprocess.TimeoutExpired:
        return RunCommandOutput(
            ok=False,
            stdout="",
            stderr=f"Command timed out after {input.timeout} seconds",
            exit_code=-1,
            command=input.command,
        )
    except Exception as e:
        return RunCommandOutput(
            ok=False,
            stdout="",
            stderr=str(e),
            exit_code=-1,
            command=input.command,
        )


class RunScriptInput(BaseModel):
    """Input for run_script function."""

    script: str = Field(description="Multi-line script content")
    interpreter: str = Field(
        default="/bin/bash", description="Script interpreter"
    )
    working_dir: Optional[str] = Field(
        default=None, description="Working directory"
    )
    timeout: int = Field(default=120, description="Max seconds to wait")
    max_output: int = Field(
        default=100000,
        description="Maximum bytes for stdout+stderr combined (0 = unlimited)",
    )


class RunScriptOutput(BaseModel):
    """Output for run_script function."""

    ok: bool
    stdout: str
    stderr: str
    exit_code: int


def run_script(input: RunScriptInput) -> RunScriptOutput:
    """
    Execute a multi-line shell script.

    Examples:
        >>> run_script({"script": "cd /tmp\\necho $(pwd)\\nls"})
    """
    try:
        working_dir = input.working_dir
        if working_dir:
            working_dir = os.path.expanduser(working_dir)

        result = subprocess.run(
            [input.interpreter],
            input=input.script,
            capture_output=True,
            text=True,
            timeout=input.timeout,
            cwd=working_dir,
        )

        stdout = _truncate_output(result.stdout, input.max_output)
        stderr = _truncate_output(result.stderr, input.max_output)

        return RunScriptOutput(
            ok=result.returncode == 0,
            stdout=stdout,
            stderr=stderr,
            exit_code=result.returncode,
        )

    except subprocess.TimeoutExpired:
        return RunScriptOutput(
            ok=False,
            stdout="",
            stderr=f"Script timed out after {input.timeout} seconds",
            exit_code=-1,
        )
    except Exception as e:
        return RunScriptOutput(
            ok=False,
            stdout="",
            stderr=str(e),
            exit_code=-1,
        )


class WhichInput(BaseModel):
    """Input for which function."""

    command: str = Field(description="Command name to look up")


class WhichOutput(BaseModel):
    """Output for which function."""

    ok: bool
    path: Optional[str] = None
    error: Optional[str] = None


def which(input: WhichInput) -> WhichOutput:
    """
    Find the path to an executable.

    Examples:
        >>> which({"command": "python"})
        >>> which({"command": "git"})
    """
    path = shutil.which(input.command)
    if path:
        return WhichOutput(ok=True, path=path)
    else:
        return WhichOutput(ok=False, error=f"Command not found: {input.command}")


class GetEnvInput(BaseModel):
    """Input for get_env function."""

    name: str = Field(description="Environment variable name")
    default: Optional[str] = Field(
        default=None, description="Default if not set"
    )


class GetEnvOutput(BaseModel):
    """Output for get_env function."""

    ok: bool
    value: Optional[str] = None
    error: Optional[str] = None


def get_env(input: GetEnvInput) -> GetEnvOutput:
    """
    Get an environment variable value.

    Examples:
        >>> get_env({"name": "HOME"})
        >>> get_env({"name": "MY_VAR", "default": "fallback"})
    """
    value = os.environ.get(input.name, input.default)
    if value is not None:
        return GetEnvOutput(ok=True, value=value)
    else:
        return GetEnvOutput(
            ok=False, error=f"Environment variable not set: {input.name}"
        )
