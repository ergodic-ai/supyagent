"""
Default supypowers tools bundled with supyagent.

These tools are copied to the user's project when running `supyagent init`.
"""

import shutil
from pathlib import Path

# Path to the bundled default tools
TOOLS_DIR = Path(__file__).parent


def get_bundled_tools() -> list[Path]:
    """Get list of bundled tool files."""
    return [f for f in TOOLS_DIR.glob("*.py") if f.name != "__init__.py"]


def install_default_tools(target_dir: Path | str = "powers") -> int:
    """
    Install default tools to a target directory.

    Args:
        target_dir: Directory to install tools to (default: powers/)

    Returns:
        Number of files installed
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    installed = 0
    for tool_file in get_bundled_tools():
        dest = target / tool_file.name
        if not dest.exists():
            shutil.copy(tool_file, dest)
            installed += 1

    # Create __init__.py if not exists
    init_file = target / "__init__.py"
    if not init_file.exists():
        init_file.write_text('"""Supypowers tools for this project."""\n')
        installed += 1

    return installed


def list_default_tools() -> list[dict]:
    """
    List available default tools.

    Returns:
        List of tool info dicts
    """
    tools = []
    for tool_file in get_bundled_tools():
        # Read first docstring
        content = tool_file.read_text()
        description = ""
        if '"""' in content:
            start = content.find('"""') + 3
            end = content.find('"""', start)
            if end > start:
                description = content[start:end].strip().split("\n")[0]

        tools.append(
            {
                "name": tool_file.stem,
                "file": tool_file.name,
                "description": description,
            }
        )

    return tools
