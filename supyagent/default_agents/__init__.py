"""
Default agent templates bundled with supyagent.

These templates are installed to the user's agents/ directory when running
`supyagent hello` or `supyagent new --role <role>`. Each template has a
strong, role-specific system prompt.
"""

import re
from pathlib import Path
from typing import Any

AGENTS_DIR = Path(__file__).parent
MODEL_PLACEHOLDER = "__MODEL_PLACEHOLDER__"
DEFAULT_MODEL = "anthropic/claude-sonnet-4-5-20250929"

# Role metadata for display and workspace profile composition
AGENT_ROLES: dict[str, dict[str, Any]] = {
    "assistant": {
        "description": "Intelligent task router that delegates to specialists",
        "delegates": ["planner", "coder"],
    },
    "coder": {
        "description": "Expert programmer — search-first, edit-minimal, test-after",
        "delegates": [],
    },
    "planner": {
        "description": "Deep-thinking planner that analyzes, plans, and orchestrates",
        "delegates": ["coder"],
    },
    "writer": {
        "description": "Content creation specialist for docs, articles, communications",
        "delegates": [],
    },
}

# Workspace profiles: which agents to install together
WORKSPACE_PROFILES: dict[str, list[str]] = {
    "coding": ["assistant", "coder", "planner"],
    "automation": ["assistant", "writer"],
    "full": ["assistant", "coder", "planner", "writer"],
}


def get_bundled_agents() -> list[Path]:
    """Get list of bundled agent YAML template files."""
    return sorted(f for f in AGENTS_DIR.glob("*.yaml") if f.name != "__init__.py")


def list_default_agents() -> list[dict[str, Any]]:
    """
    List available default agent templates with metadata.

    Returns list of dicts with keys: role, description, delegates, file.
    """
    result = []
    for role, meta in AGENT_ROLES.items():
        result.append({
            "role": role,
            "description": meta["description"],
            "delegates": meta["delegates"],
            "file": f"{role}.yaml",
        })
    return result


def get_agent_template(role: str) -> str:
    """
    Read a bundled agent template as a raw YAML string.

    Raises ValueError if role is not recognized.
    """
    if role not in AGENT_ROLES:
        available = ", ".join(AGENT_ROLES.keys())
        raise ValueError(f"Unknown agent role '{role}'. Available: {available}")

    template_path = AGENTS_DIR / f"{role}.yaml"
    return template_path.read_text()


def install_agent(
    role: str,
    target_dir: Path | str = "agents",
    *,
    model: str | None = None,
    name: str | None = None,
    standalone: bool = False,
    force: bool = False,
) -> Path | None:
    """
    Install a single agent template to the target directory.

    Args:
        role: Agent role (assistant, coder, planner, writer).
        target_dir: Directory to install to (default: agents/).
        model: Model string to use. Replaces __MODEL_PLACEHOLDER__.
        name: Override agent name (default: role name).
        standalone: If True, strip delegates for standalone use.
        force: If True, overwrite existing file.

    Returns:
        Path to the created file, or None if it already existed and force=False.
    """
    template = get_agent_template(role)
    effective_model = model or DEFAULT_MODEL
    effective_name = name or role

    # Substitute model placeholder
    content = template.replace(MODEL_PLACEHOLDER, effective_model)

    # Override name if different from role
    if effective_name != role:
        content = re.sub(r"^name: .+$", f"name: {effective_name}", content, count=1, flags=re.M)

    # Strip delegates for standalone mode
    if standalone:
        # Replace delegates list with empty list
        content = re.sub(
            r"^delegates:\n(?:\s+-\s+.+\n)*",
            "delegates: []\n",
            content,
            count=1,
            flags=re.M,
        )

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    dest = target / f"{effective_name}.yaml"
    if dest.exists() and not force:
        return None

    dest.write_text(content)
    return dest


def install_workspace_agents(
    profile: str,
    target_dir: Path | str = "agents",
    *,
    model: str | None = None,
    force: bool = False,
) -> list[Path]:
    """
    Install all agents for a workspace profile with delegation wired.

    Args:
        profile: Workspace profile name (coding, automation, full).
        target_dir: Directory to install to (default: agents/).
        model: Model string to use for all agents.
        force: If True, overwrite existing files.

    Returns:
        List of created file paths.

    Raises:
        ValueError: If profile is not recognized.
    """
    if profile not in WORKSPACE_PROFILES:
        available = ", ".join(WORKSPACE_PROFILES.keys())
        raise ValueError(f"Unknown workspace profile '{profile}'. Available: {available}")

    roles = WORKSPACE_PROFILES[profile]
    installed = []

    for role in roles:
        # When installing as part of a team, keep delegates intact (standalone=False)
        # but only if the delegate agents are also being installed
        role_delegates = AGENT_ROLES[role]["delegates"]
        has_all_delegates = all(d in roles for d in role_delegates)

        path = install_agent(
            role,
            target_dir,
            model=model,
            standalone=not has_all_delegates,
            force=force,
        )
        if path:
            installed.append(path)

    return installed


def install_default_agents(
    target_dir: Path | str = "agents",
    *,
    model: str | None = None,
    roles: list[str] | None = None,
) -> int:
    """
    Install default agents. Convenience wrapper.

    Args:
        target_dir: Directory to install to.
        model: Model string for all agents.
        roles: Specific roles to install, or None for the default team (coding profile).

    Returns:
        Number of agents installed.
    """
    if roles is None:
        paths = install_workspace_agents("coding", target_dir, model=model)
        return len(paths)

    count = 0
    for role in roles:
        path = install_agent(role, target_dir, model=model, standalone=True)
        if path:
            count += 1
    return count
