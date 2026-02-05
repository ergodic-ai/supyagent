# /// script
# dependencies = ["pydantic"]
# ///
"""
File and directory finding tools.

Find files by name, glob pattern, extension, or recently modified.
Provides structured output with metadata (size, modified time, type).
"""

import os
import time
from pathlib import Path
from typing import Optional, List

from pydantic import BaseModel, Field


# Directories to always skip
_SKIP_DIRS = {
    "__pycache__",
    "node_modules",
    ".git",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".egg-info",
    ".eggs",
}


# =============================================================================
# Find files by glob pattern
# =============================================================================


class FindFilesInput(BaseModel):
    """Input for find_files function."""

    pattern: str = Field(
        description="Glob pattern to match (e.g., '*.py', 'test_*.py', '**/*.ts')"
    )
    path: str = Field(default=".", description="Directory to search in")
    max_results: int = Field(
        default=200, description="Maximum number of results to return"
    )
    include_hidden: bool = Field(
        default=False, description="Include hidden files/directories"
    )
    file_type: str = Field(
        default="all",
        description="Filter by type: 'file', 'dir', or 'all'",
    )


class FileEntry(BaseModel):
    """A single file/directory entry."""

    path: str
    name: str
    type: str  # "file" or "dir"
    size: Optional[int] = None  # bytes, only for files
    extension: Optional[str] = None


class FindFilesOutput(BaseModel):
    """Output for find_files function."""

    ok: bool
    entries: List[FileEntry] = []
    total_found: int = 0
    truncated: bool = False
    error: Optional[str] = None


def find_files(input: FindFilesInput) -> FindFilesOutput:
    """
    Find files and directories matching a glob pattern.

    Automatically skips __pycache__, node_modules, .git, .venv, etc.

    Examples:
        >>> find_files({"pattern": "*.py", "path": "src"})
        >>> find_files({"pattern": "test_*.py"})
        >>> find_files({"pattern": "*.yaml", "path": "agents"})
        >>> find_files({"pattern": "Dockerfile*", "file_type": "file"})
    """
    try:
        root = Path(os.path.expanduser(input.path))
        if not root.exists():
            return FindFilesOutput(ok=False, error=f"Path not found: {input.path}")
        if not root.is_dir():
            return FindFilesOutput(ok=False, error=f"Not a directory: {input.path}")

        entries: List[FileEntry] = []
        truncated = False

        for item in _walk_and_match(root, input.pattern, input.include_hidden):
            if input.file_type == "file" and not item.is_file():
                continue
            if input.file_type == "dir" and not item.is_dir():
                continue

            try:
                rel = str(item.relative_to(root))
            except ValueError:
                rel = str(item)

            entry = FileEntry(
                path=rel,
                name=item.name,
                type="dir" if item.is_dir() else "file",
                size=item.stat().st_size if item.is_file() else None,
                extension=item.suffix if item.is_file() and item.suffix else None,
            )
            entries.append(entry)

            if len(entries) >= input.max_results:
                truncated = True
                break

        return FindFilesOutput(
            ok=True,
            entries=entries,
            total_found=len(entries),
            truncated=truncated,
        )

    except Exception as e:
        return FindFilesOutput(ok=False, error=str(e))


def _walk_and_match(root: Path, pattern: str, include_hidden: bool) -> list[Path]:
    """Walk directory tree and match files against pattern."""
    from fnmatch import fnmatch

    results = []

    def _walk(directory: Path):
        try:
            for entry in sorted(directory.iterdir()):
                if entry.is_dir():
                    if entry.name in _SKIP_DIRS:
                        continue
                    if not include_hidden and entry.name.startswith("."):
                        continue
                    # Check if directory name matches pattern
                    if fnmatch(entry.name, pattern):
                        results.append(entry)
                    _walk(entry)
                elif entry.is_file():
                    if not include_hidden and entry.name.startswith("."):
                        continue
                    if fnmatch(entry.name, pattern):
                        results.append(entry)
        except PermissionError:
            pass

    _walk(root)
    return results


# =============================================================================
# Find recently modified files
# =============================================================================


class RecentFilesInput(BaseModel):
    """Input for recent_files function."""

    path: str = Field(default=".", description="Directory to search in")
    minutes: int = Field(
        default=60,
        description="Find files modified within this many minutes",
    )
    glob: str = Field(default="*", description="File glob filter (e.g., '*.py')")
    max_results: int = Field(
        default=50, description="Maximum number of results"
    )


class RecentFileEntry(BaseModel):
    """A recently modified file entry."""

    path: str
    name: str
    size: int
    modified_seconds_ago: int
    extension: Optional[str] = None


class RecentFilesOutput(BaseModel):
    """Output for recent_files function."""

    ok: bool
    entries: List[RecentFileEntry] = []
    total_found: int = 0
    error: Optional[str] = None


def recent_files(input: RecentFilesInput) -> RecentFilesOutput:
    """
    Find files modified within a recent time window.

    Results are sorted by modification time (most recent first).

    Examples:
        >>> recent_files({"minutes": 30, "glob": "*.py"})
        >>> recent_files({"path": "src", "minutes": 10})
    """
    try:
        root = Path(os.path.expanduser(input.path))
        if not root.exists():
            return RecentFilesOutput(ok=False, error=f"Path not found: {input.path}")

        cutoff = time.time() - (input.minutes * 60)
        entries: List[RecentFileEntry] = []

        from fnmatch import fnmatch

        def _walk(directory: Path):
            try:
                for entry in directory.iterdir():
                    if entry.is_dir():
                        if entry.name in _SKIP_DIRS or entry.name.startswith("."):
                            continue
                        _walk(entry)
                    elif entry.is_file():
                        if entry.name.startswith("."):
                            continue
                        if not fnmatch(entry.name, input.glob):
                            continue
                        try:
                            mtime = entry.stat().st_mtime
                            if mtime >= cutoff:
                                rel = str(entry.relative_to(root))
                                entries.append(
                                    RecentFileEntry(
                                        path=rel,
                                        name=entry.name,
                                        size=entry.stat().st_size,
                                        modified_seconds_ago=int(time.time() - mtime),
                                        extension=entry.suffix if entry.suffix else None,
                                    )
                                )
                        except OSError:
                            pass
            except PermissionError:
                pass

        _walk(root)

        # Sort by most recently modified first
        entries.sort(key=lambda e: e.modified_seconds_ago)

        # Truncate
        entries = entries[: input.max_results]

        return RecentFilesOutput(
            ok=True,
            entries=entries,
            total_found=len(entries),
        )

    except Exception as e:
        return RecentFilesOutput(ok=False, error=str(e))


# =============================================================================
# Directory tree (structured view)
# =============================================================================


class TreeInput(BaseModel):
    """Input for directory_tree function."""

    path: str = Field(default=".", description="Directory to show tree for")
    max_depth: int = Field(
        default=3, description="Maximum depth to traverse (1-10)"
    )
    include_hidden: bool = Field(
        default=False, description="Include hidden files/directories"
    )
    include_files: bool = Field(
        default=True, description="Include files (set False for dirs only)"
    )


class TreeNode(BaseModel):
    """A node in the directory tree."""

    name: str
    type: str  # "file" or "dir"
    children: Optional[List["TreeNode"]] = None
    size: Optional[int] = None  # bytes, only for files


class TreeOutput(BaseModel):
    """Output for directory_tree function."""

    ok: bool
    tree: Optional[TreeNode] = None
    total_files: int = 0
    total_dirs: int = 0
    error: Optional[str] = None


def directory_tree(input: TreeInput) -> TreeOutput:
    """
    Get a structured directory tree view.

    Returns a nested tree structure that shows the project layout.
    Automatically skips __pycache__, node_modules, .git, etc.

    Examples:
        >>> directory_tree({"path": ".", "max_depth": 2})
        >>> directory_tree({"path": "src", "max_depth": 4, "include_files": True})
    """
    try:
        root = Path(os.path.expanduser(input.path))
        if not root.exists():
            return TreeOutput(ok=False, error=f"Path not found: {input.path}")
        if not root.is_dir():
            return TreeOutput(ok=False, error=f"Not a directory: {input.path}")

        max_depth = min(max(input.max_depth, 1), 10)
        file_count = 0
        dir_count = 0

        def _build(directory: Path, depth: int) -> TreeNode:
            nonlocal file_count, dir_count
            dir_count += 1

            children: List[TreeNode] = []

            if depth < max_depth:
                try:
                    for entry in sorted(directory.iterdir()):
                        if entry.is_dir():
                            if entry.name in _SKIP_DIRS:
                                continue
                            if not input.include_hidden and entry.name.startswith("."):
                                continue
                            children.append(_build(entry, depth + 1))
                        elif entry.is_file() and input.include_files:
                            if not input.include_hidden and entry.name.startswith("."):
                                continue
                            file_count += 1
                            children.append(
                                TreeNode(
                                    name=entry.name,
                                    type="file",
                                    size=entry.stat().st_size,
                                )
                            )
                except PermissionError:
                    pass

            return TreeNode(
                name=directory.name or str(directory),
                type="dir",
                children=children if children else None,
            )

        tree = _build(root, 0)

        return TreeOutput(
            ok=True,
            tree=tree,
            total_files=file_count,
            total_dirs=dir_count,
        )

    except Exception as e:
        return TreeOutput(ok=False, error=str(e))
