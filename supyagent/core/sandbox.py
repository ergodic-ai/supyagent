"""
Workspace sandbox for isolated tool execution via OCI containers.

Supports Podman (preferred) and Docker as container runtimes.
The SandboxManager maintains a persistent container per session —
all tool calls execute inside it via `{runtime} exec`.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from supyagent.models.agent_config import AgentConfig, SandboxConfig

logger = logging.getLogger(__name__)

# Known path-like argument keys for workspace validation (soft mode)
_PATH_KEYS = frozenset({
    "path", "file_path", "filepath", "working_dir", "directory", "dir",
    "source", "destination", "target", "filename", "folder",
})


class SandboxError(RuntimeError):
    """Raised when the sandbox cannot be started or used."""


class SandboxManager:
    """
    Manages a persistent OCI container for sandboxed tool execution.

    One instance per session. Shared across parent and all delegated child agents
    so they use the same container — no escape via delegation.
    """

    CONTAINER_WORKSPACE = "/workspace"

    def __init__(
        self,
        workspace: Path,
        config: SandboxConfig,
        session_id: str,
    ):
        self.workspace = workspace.expanduser().resolve()
        self.config = config
        self.session_id = session_id
        self._container_name: str | None = None
        self._runtime: str | None = None
        self._started = False

        # Check if inheriting container from parent process (subprocess delegation)
        existing_container = os.environ.get("SUPYAGENT_SANDBOX_CONTAINER")
        if existing_container:
            self._container_name = existing_container
            self._runtime = os.environ.get(
                "SUPYAGENT_SANDBOX_RUNTIME", self._detect_runtime()
            )
            self._started = True

    @property
    def container_name(self) -> str:
        if self._container_name is None:
            short_id = self.session_id[:12]
            self._container_name = f"supyagent-{short_id}"
        return self._container_name

    @property
    def runtime(self) -> str:
        if self._runtime is None:
            self._runtime = self._detect_runtime()
        return self._runtime

    def _detect_runtime(self) -> str:
        if self.config.runtime != "auto":
            if shutil.which(self.config.runtime):
                return self.config.runtime
            raise SandboxError(
                f"Container runtime '{self.config.runtime}' not found on PATH."
            )
        for cmd in ("podman", "docker"):
            if shutil.which(cmd):
                return cmd
        raise SandboxError(
            "No container runtime found. Install podman (https://podman.io) "
            "or docker, or set sandbox.enabled = false."
        )

    # ── Container lifecycle ─────────────────────────────────────────

    def ensure_started(self) -> None:
        """Start the container if not already running. Called lazily on first tool call."""
        if self._started and self._is_container_running():
            return
        self._start_container()
        self._started = True

    def _start_container(self) -> None:
        self._remove_if_exists()

        cmd = [
            self.runtime, "run",
            "--detach",
            "--name", self.container_name,
            # Mount workspace read-write
            "--volume", f"{self.workspace}:{self.CONTAINER_WORKSPACE}:rw",
            # Set working directory
            "--workdir", self.CONTAINER_WORKSPACE,
            # Resource limits
            "--memory", self.config.memory_limit,
            # Network mode
            "--network", self.config.network,
        ]

        # Match host user to avoid file permission issues
        uid = os.getuid()
        gid = os.getgid()
        cmd.extend(["--user", f"{uid}:{gid}"])

        # Extra mounts
        for mount in self.config.extra_mounts:
            host = Path(mount.host_path).expanduser().resolve()
            container_path = mount.container_path or f"/mnt/{host.name}"
            mode = "ro" if mount.readonly else "rw"
            cmd.extend(["--volume", f"{host}:{container_path}:{mode}"])

        # Extra environment variables
        for key, value in self.config.env.items():
            cmd.extend(["--env", f"{key}={value}"])

        # Image + keep-alive command
        cmd.extend([self.config.image, "tail", "-f", "/dev/null"])

        logger.info("Starting sandbox container: %s", self.container_name)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise SandboxError(
                f"Failed to start sandbox container: {result.stderr.strip()}"
            )

        # Auto-install uv + supypowers if not present
        self._ensure_toolchain()

        # Run user-defined setup commands
        for setup_cmd in self.config.setup_commands:
            logger.info("Sandbox setup: %s", setup_cmd)
            self._exec_in_container(["sh", "-c", setup_cmd], timeout=120)

    def _ensure_toolchain(self) -> None:
        """Install uv and supypowers inside the container if not already present."""
        # Check for uv
        uv_check = self._exec_in_container(["sh", "-c", "which uv"], timeout=10)
        if uv_check.returncode != 0:
            logger.info("Installing uv in sandbox container...")
            self._exec_in_container(
                ["sh", "-c", "pip install uv 2>/dev/null || pip3 install uv"],
                timeout=120,
            )
        # Check for supypowers
        sp_check = self._exec_in_container(["sh", "-c", "which supypowers"], timeout=10)
        if sp_check.returncode != 0:
            logger.info("Installing supypowers in sandbox container...")
            self._exec_in_container(["pip", "install", "supypowers"], timeout=120)

    def _is_container_running(self) -> bool:
        try:
            result = subprocess.run(
                [self.runtime, "inspect", "--format", "{{.State.Running}}", self.container_name],
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0 and "true" in result.stdout.lower()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _remove_if_exists(self) -> None:
        subprocess.run(
            [self.runtime, "rm", "-f", self.container_name],
            capture_output=True, timeout=15,
        )

    def _exec_in_container(
        self,
        cmd: list[str],
        env: dict[str, str] | None = None,
        timeout: float = 300,
    ) -> subprocess.CompletedProcess:
        exec_cmd = [self.runtime, "exec"]
        if env:
            for k, v in env.items():
                exec_cmd.extend(["--env", f"{k}={v}"])
        exec_cmd.append(self.container_name)
        exec_cmd.extend(cmd)
        return subprocess.run(exec_cmd, capture_output=True, text=True, timeout=timeout)

    # ── Command wrapping ────────────────────────────────────────────

    def wrap_command(
        self,
        cmd: list[str],
        env: dict[str, str] | None = None,
    ) -> list[str]:
        """
        Wrap a host command to execute inside the container.

        Transforms: ["supypowers", "run", "files:read_file", '{"path": "data/file.csv"}']
        Into: ["podman", "exec", "--env", "K=V", "supyagent-abc123", "supypowers", "run", ...]
        """
        self.ensure_started()

        wrapped = [self.runtime, "exec"]
        if env:
            for k, v in env.items():
                wrapped.extend(["--env", f"{k}={v}"])
        wrapped.append(self.container_name)
        wrapped.extend(cmd)
        return wrapped

    # ── Cleanup ─────────────────────────────────────────────────────

    def stop(self) -> None:
        """Stop and remove the container."""
        if not self._container_name:
            return
        logger.info("Stopping sandbox container: %s", self._container_name)
        try:
            subprocess.run(
                [self.runtime, "stop", "--time", "5", self._container_name],
                capture_output=True, timeout=15,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        try:
            subprocess.run(
                [self.runtime, "rm", "-f", self._container_name],
                capture_output=True, timeout=15,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        self._started = False

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass


class WorkspaceValidator:
    """
    Soft workspace boundary: validates that tool file paths stay within the workspace.
    Used when workspace is set but sandbox.enabled is false.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace.expanduser().resolve()

    def validate_path(self, path_str: str) -> str | None:
        """
        Check if a path is within the workspace.
        Returns None if valid, or an error message if outside.
        """
        try:
            resolved = Path(path_str).expanduser().resolve()
            resolved.relative_to(self.workspace)
            return None
        except ValueError:
            return (
                f"Path '{path_str}' is outside the workspace '{self.workspace}'. "
                "Use paths within the workspace directory."
            )

    def check_tool_args(self, tool_name: str, args: dict) -> str | None:
        """
        Scan tool arguments for path-like values and validate them.
        Returns None if all valid, or the first error message.
        """
        for key, value in args.items():
            if key.lower() in _PATH_KEYS and isinstance(value, str) and value:
                # Only validate absolute paths or home-relative paths
                if value.startswith("/") or value.startswith("~"):
                    error = self.validate_path(value)
                    if error:
                        return error
        return None


def create_sandbox_context_prompt(
    config: AgentConfig,
    sandbox_mgr: SandboxManager | None,
) -> str:
    """Generate system prompt addendum describing the sandbox environment."""
    if not config.workspace:
        return ""

    lines = ["\n\n---\n\n## Workspace"]

    if sandbox_mgr:
        lines.append(
            "You are running in a sandboxed environment. "
            "Your workspace is the current directory."
        )
        lines.append("- Use relative paths for all file operations")
        lines.append("- You cannot access files outside the workspace and mounted directories")

        if config.sandbox.extra_mounts:
            for mount in config.sandbox.extra_mounts:
                host = Path(mount.host_path).expanduser().resolve()
                container_path = mount.container_path or f"/mnt/{host.name}"
                mode = "read-only" if mount.readonly else "read-write"
                lines.append(f"- Additional directory: {container_path} ({mode})")

        if not config.sandbox.allow_shell:
            lines.append("- Shell/exec tools are disabled in this sandbox")
    else:
        # Soft mode (workspace set, sandbox off)
        lines.append(
            f"Your workspace is: {config.workspace}"
        )
        lines.append(
            "- Keep all file operations within the workspace directory"
        )
        lines.append("- Use relative paths when possible")

    return "\n".join(lines)
