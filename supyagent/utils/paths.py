"""
Path utilities for supyagent.
"""

from pathlib import Path


def get_agents_dir() -> Path:
    """Get the agents directory."""
    return Path("agents")


def get_supyagent_dir() -> Path:
    """Get the .supyagent runtime directory."""
    path = Path(".supyagent")
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_sessions_dir() -> Path:
    """Get the sessions directory."""
    path = get_supyagent_dir() / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_credentials_dir() -> Path:
    """Get the credentials directory."""
    path = get_supyagent_dir() / "credentials"
    path.mkdir(parents=True, exist_ok=True)
    return path
