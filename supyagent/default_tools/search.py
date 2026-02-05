# /// script
# dependencies = ["pydantic"]
# ///
"""
Code search tools.

Search file contents by pattern (regex or literal) with structured output
including file paths, line numbers, and surrounding context.
"""

import os
import re
from pathlib import Path
from typing import Optional, List

from pydantic import BaseModel, Field


# =============================================================================
# Grep / Search in files
# =============================================================================


class SearchMatch(BaseModel):
    """A single search match."""

    file: str
    line_number: int
    line: str
    context_before: List[str] = []
    context_after: List[str] = []


class SearchInput(BaseModel):
    """Input for search function."""

    pattern: str = Field(description="Search pattern (regex or literal string)")
    path: str = Field(default=".", description="Directory or file to search in")
    glob: str = Field(
        default="*",
        description="File glob pattern to filter (e.g., '*.py', '*.ts')",
    )
    regex: bool = Field(
        default=False,
        description="Treat pattern as regex (default: literal string match)",
    )
    case_sensitive: bool = Field(
        default=True, description="Case-sensitive search"
    )
    context_lines: int = Field(
        default=0,
        description="Number of context lines before and after each match (0-5)",
    )
    max_results: int = Field(
        default=100,
        description="Maximum number of matches to return",
    )
    include_hidden: bool = Field(
        default=False,
        description="Include hidden files/directories (starting with .)",
    )


class SearchOutput(BaseModel):
    """Output for search function."""

    ok: bool
    matches: List[SearchMatch] = []
    total_matches: int = 0
    files_searched: int = 0
    truncated: bool = False
    error: Optional[str] = None


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

# Binary file extensions to skip
_BINARY_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".o", ".a", ".dylib", ".dll", ".exe",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".mp3", ".mp4", ".avi", ".mov", ".flac", ".wav",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".woff", ".woff2", ".ttf", ".eot",
    ".sqlite", ".db",
}


def _should_skip_dir(name: str, include_hidden: bool) -> bool:
    if name in _SKIP_DIRS:
        return True
    if not include_hidden and name.startswith("."):
        return True
    return False


def _is_binary(path: Path) -> bool:
    return path.suffix.lower() in _BINARY_EXTENSIONS


def search(input: SearchInput) -> SearchOutput:
    """
    Search for a pattern across files in a directory.

    Returns structured matches with file paths, line numbers, and optional
    context lines. Skips binary files, __pycache__, node_modules, .git, etc.

    Examples:
        >>> search({"pattern": "def main", "path": "src", "glob": "*.py"})
        >>> search({"pattern": "TODO", "path": ".", "context_lines": 2})
        >>> search({"pattern": "import.*json", "regex": True, "glob": "*.py"})
    """
    try:
        root = Path(os.path.expanduser(input.path))
        if not root.exists():
            return SearchOutput(ok=False, error=f"Path not found: {input.path}")

        # Compile the pattern
        flags = 0 if input.case_sensitive else re.IGNORECASE
        if input.regex:
            try:
                compiled = re.compile(input.pattern, flags)
            except re.error as e:
                return SearchOutput(ok=False, error=f"Invalid regex: {e}")
        else:
            escaped = re.escape(input.pattern)
            compiled = re.compile(escaped, flags)

        ctx = min(max(input.context_lines, 0), 5)
        matches: List[SearchMatch] = []
        files_searched = 0
        truncated = False

        # Collect files to search
        if root.is_file():
            files = [root]
        else:
            files = _collect_files(root, input.glob, input.include_hidden)

        for file_path in files:
            if _is_binary(file_path):
                continue

            try:
                text = file_path.read_text(errors="replace")
            except (PermissionError, OSError):
                continue

            files_searched += 1
            lines = text.splitlines()

            for i, line in enumerate(lines):
                if compiled.search(line):
                    # Gather context
                    before = lines[max(0, i - ctx) : i] if ctx > 0 else []
                    after = lines[i + 1 : i + 1 + ctx] if ctx > 0 else []

                    rel = _relative_path(file_path, root)
                    matches.append(
                        SearchMatch(
                            file=rel,
                            line_number=i + 1,
                            line=line,
                            context_before=before,
                            context_after=after,
                        )
                    )

                    if len(matches) >= input.max_results:
                        truncated = True
                        break

            if truncated:
                break

        return SearchOutput(
            ok=True,
            matches=matches,
            total_matches=len(matches),
            files_searched=files_searched,
            truncated=truncated,
        )

    except Exception as e:
        return SearchOutput(ok=False, error=str(e))


def _collect_files(root: Path, glob_pattern: str, include_hidden: bool) -> list[Path]:
    """Recursively collect files matching glob, skipping ignored dirs."""
    result = []
    try:
        for entry in sorted(root.iterdir()):
            if entry.is_dir():
                if _should_skip_dir(entry.name, include_hidden):
                    continue
                result.extend(_collect_files(entry, glob_pattern, include_hidden))
            elif entry.is_file():
                if not include_hidden and entry.name.startswith("."):
                    continue
                from fnmatch import fnmatch
                if fnmatch(entry.name, glob_pattern):
                    result.append(entry)
    except PermissionError:
        pass
    return result


def _relative_path(file_path: Path, root: Path) -> str:
    """Get relative path string, falling back to absolute."""
    try:
        return str(file_path.relative_to(root))
    except ValueError:
        return str(file_path)


# =============================================================================
# Count occurrences
# =============================================================================


class CountMatchesInput(BaseModel):
    """Input for count_matches function."""

    pattern: str = Field(description="Pattern to count (literal or regex)")
    path: str = Field(default=".", description="Directory or file to search")
    glob: str = Field(default="*", description="File glob filter")
    regex: bool = Field(default=False, description="Treat pattern as regex")
    case_sensitive: bool = Field(default=True, description="Case-sensitive search")


class FileMatchCount(BaseModel):
    """Match count for a single file."""

    file: str
    count: int


class CountMatchesOutput(BaseModel):
    """Output for count_matches function."""

    ok: bool
    total: int = 0
    by_file: List[FileMatchCount] = []
    files_searched: int = 0
    error: Optional[str] = None


def count_matches(input: CountMatchesInput) -> CountMatchesOutput:
    """
    Count occurrences of a pattern across files.

    Useful for getting an overview before doing a detailed search.

    Examples:
        >>> count_matches({"pattern": "TODO", "glob": "*.py"})
        >>> count_matches({"pattern": "console\\.log", "regex": True, "glob": "*.ts"})
    """
    try:
        root = Path(os.path.expanduser(input.path))
        if not root.exists():
            return CountMatchesOutput(ok=False, error=f"Path not found: {input.path}")

        flags = 0 if input.case_sensitive else re.IGNORECASE
        if input.regex:
            try:
                compiled = re.compile(input.pattern, flags)
            except re.error as e:
                return CountMatchesOutput(ok=False, error=f"Invalid regex: {e}")
        else:
            compiled = re.compile(re.escape(input.pattern), flags)

        by_file: List[FileMatchCount] = []
        total = 0
        files_searched = 0

        if root.is_file():
            files = [root]
        else:
            files = _collect_files(root, input.glob, include_hidden=False)

        for file_path in files:
            if _is_binary(file_path):
                continue
            try:
                text = file_path.read_text(errors="replace")
            except (PermissionError, OSError):
                continue

            files_searched += 1
            count = len(compiled.findall(text))
            if count > 0:
                rel = _relative_path(file_path, root)
                by_file.append(FileMatchCount(file=rel, count=count))
                total += count

        by_file.sort(key=lambda x: x.count, reverse=True)

        return CountMatchesOutput(
            ok=True,
            total=total,
            by_file=by_file,
            files_searched=files_searched,
        )

    except Exception as e:
        return CountMatchesOutput(ok=False, error=str(e))
