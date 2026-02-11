"""
Workspace configuration for supyagent projects.

A workspace is a directory that has been initialized with `supyagent hello`.
Configuration lives in .supyagent/workspace.yaml and defines the workspace
profile, execution mode, heartbeat settings, and per-agent model overrides.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

WORKSPACE_DIR = ".supyagent"
WORKSPACE_CONFIG_FILE = "workspace.yaml"
GOALS_FILE = "GOALS.md"

GOALS_TEMPLATE = """\
# Goals

## User Goals
<!-- Define what you want to achieve. The agent reads this on every conversation. -->

{user_goals}

## Subgoals
<!-- Managed by the agent. Uses checkbox syntax: [ ] pending, [x] done, [~] skipped -->
"""


class HeartbeatConfig(BaseModel):
    enabled: bool = False
    interval: str = "5m"
    max_events_per_cycle: int = 10


class ExecutionConfig(BaseModel):
    mode: str = "yolo"  # "yolo" or "isolated"


class WorkspaceConfig(BaseModel):
    """Workspace-level configuration stored in .supyagent/workspace.yaml."""

    name: str = ""
    profile: str = "coding"  # coding | automation | full | custom

    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)

    # Per-agent model overrides: agent_name -> model_string_or_role
    models: dict[str, str] = Field(default_factory=dict)


def workspace_dir(root: Path | None = None) -> Path:
    """Get the .supyagent directory path for a workspace."""
    return (root or Path.cwd()) / WORKSPACE_DIR


def workspace_config_path(root: Path | None = None) -> Path:
    """Get the workspace.yaml path."""
    return workspace_dir(root) / WORKSPACE_CONFIG_FILE


def goals_path(root: Path | None = None) -> Path:
    """Get the GOALS.md path."""
    return (root or Path.cwd()) / GOALS_FILE


def is_workspace_initialized(root: Path | None = None) -> bool:
    """Check if the current directory (or given root) is an initialized workspace."""
    return workspace_config_path(root).exists()


def load_workspace(root: Path | None = None) -> WorkspaceConfig:
    """
    Load workspace configuration from .supyagent/workspace.yaml.

    Returns default config if file doesn't exist.
    """
    path = workspace_config_path(root)
    if not path.exists():
        return WorkspaceConfig()

    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return WorkspaceConfig(**data)
    except (yaml.YAMLError, OSError, ValueError):
        return WorkspaceConfig()


def save_workspace(config: WorkspaceConfig, root: Path | None = None) -> Path:
    """Save workspace configuration to .supyagent/workspace.yaml."""
    ws_dir = workspace_dir(root)
    ws_dir.mkdir(parents=True, exist_ok=True)

    path = workspace_config_path(root)
    data = config.model_dump(exclude_defaults=False)

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return path


def create_goals_file(user_goals: str = "", root: Path | None = None) -> Path:
    """
    Create GOALS.md in the workspace root.

    Args:
        user_goals: Initial user goals text. If empty, creates template with placeholder.
    """
    path = goals_path(root)
    if not user_goals.strip():
        user_goals = "- "

    content = GOALS_TEMPLATE.format(user_goals=user_goals)
    path.write_text(content)
    return path


def read_goals(root: Path | None = None) -> str:
    """Read GOALS.md content. Returns empty string if file doesn't exist."""
    path = goals_path(root)
    if path.exists():
        return path.read_text()
    return ""


def has_active_goals(root: Path | None = None) -> bool:
    """
    Check if GOALS.md contains actual user goals (not just the empty template).

    Extracts the "## User Goals" section and checks whether it has any
    non-comment, non-blank content beyond the default placeholder "- ".
    """
    content = read_goals(root)
    if not content.strip():
        return False

    # Extract content between "## User Goals" and the next "##" heading
    import re

    match = re.search(
        r"##\s*User Goals\s*\n(.*?)(?=\n##|\Z)", content, re.DOTALL
    )
    if not match:
        return False

    section = match.group(1)
    # Strip HTML comments and blank lines
    lines = []
    for line in section.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("<!--") and not stripped.endswith("-->"):
            lines.append(stripped)

    # Check if there's meaningful content (not just "- ")
    meaningful = [ln for ln in lines if ln != "-" and ln != "- "]
    return len(meaningful) > 0


def get_workspace_model(agent_name: str, root: Path | None = None) -> str | None:
    """
    Get the model override for an agent from workspace config.

    Returns the model string/role from workspace.models, or None if no override.
    The caller is responsible for resolving role names via ModelRegistry.
    """
    config = load_workspace(root)
    return config.models.get(agent_name)


def initialize_workspace(
    name: str | None = None,
    profile: str = "coding",
    execution_mode: str = "yolo",
    heartbeat_enabled: bool = False,
    heartbeat_interval: str = "5m",
    model_overrides: dict[str, str] | None = None,
    user_goals: str = "",
    root: Path | None = None,
) -> dict[str, Any]:
    """
    Initialize a complete workspace. Creates .supyagent/ directory,
    workspace.yaml, and GOALS.md.

    Returns dict with created file paths.
    """
    actual_root = root or Path.cwd()
    ws_name = name or actual_root.name

    config = WorkspaceConfig(
        name=ws_name,
        profile=profile,
        execution=ExecutionConfig(mode=execution_mode),
        heartbeat=HeartbeatConfig(
            enabled=heartbeat_enabled,
            interval=heartbeat_interval,
        ),
        models=model_overrides or {},
    )

    config_path = save_workspace(config, root)
    goals_file = create_goals_file(user_goals, root)

    return {
        "workspace_config": str(config_path),
        "goals_file": str(goals_file),
        "workspace_dir": str(workspace_dir(root)),
    }
