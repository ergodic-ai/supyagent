# /// script
# dependencies = ["pydantic"]
# ///
"""
File system operation tools.

Allows agents to read, write, and manage files and directories.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

# =============================================================================
# Read File
# =============================================================================


class ReadFileInput(BaseModel):
    """Input for read_file function."""

    path: str = Field(description="Path to the file to read")
    encoding: str = Field(default="utf-8", description="File encoding")


class ReadFileOutput(BaseModel):
    """Output for read_file function."""

    ok: bool
    content: Optional[str] = None
    size: Optional[int] = None
    path: Optional[str] = None
    error: Optional[str] = None


def read_file(input: ReadFileInput) -> ReadFileOutput:
    """
    Read the contents of a file.

    Examples:
        >>> read_file({"path": "README.md"})
        >>> read_file({"path": "data.txt", "encoding": "latin-1"})
    """
    try:
        path = os.path.expanduser(input.path)
        p = Path(path)

        if not p.exists():
            return ReadFileOutput(ok=False, error=f"File not found: {path}")

        if not p.is_file():
            return ReadFileOutput(ok=False, error=f"Not a file: {path}")

        size = p.stat().st_size
        if size > 10 * 1024 * 1024:
            return ReadFileOutput(
                ok=False,
                error=f"File too large ({size} bytes). Maximum is 10MB.",
            )

        content = p.read_text(encoding=input.encoding)
        return ReadFileOutput(
            ok=True,
            content=content,
            size=size,
            path=str(p.absolute()),
        )

    except UnicodeDecodeError:
        return ReadFileOutput(
            ok=False,
            error=f"Cannot decode file as {input.encoding}",
        )
    except PermissionError:
        return ReadFileOutput(ok=False, error=f"Permission denied: {input.path}")
    except Exception as e:
        return ReadFileOutput(ok=False, error=str(e))


# =============================================================================
# Read File Lines (specific line range)
# =============================================================================


class ReadFileLinesInput(BaseModel):
    """Input for read_file_lines function."""

    path: str = Field(description="Path to the file to read")
    start_line: int = Field(
        default=1, description="First line to read (1-based, inclusive)"
    )
    end_line: int = Field(
        default=-1,
        description="Last line to read (1-based, inclusive). -1 = end of file.",
    )
    encoding: str = Field(default="utf-8", description="File encoding")
    include_line_numbers: bool = Field(
        default=True,
        description="Prefix each line with its line number",
    )


class ReadFileLinesOutput(BaseModel):
    """Output for read_file_lines function."""

    ok: bool
    content: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    total_lines: int = 0
    path: Optional[str] = None
    error: Optional[str] = None


def read_file_lines(input: ReadFileLinesInput) -> ReadFileLinesOutput:
    """
    Read specific lines from a file.

    Much more efficient than read_file for large files. Returns lines with
    optional line numbers. Line numbers are 1-based and inclusive.

    Examples:
        >>> read_file_lines({"path": "main.py", "start_line": 1, "end_line": 50})
        >>> read_file_lines({"path": "main.py", "start_line": 100, "end_line": 150})
        >>> read_file_lines({"path": "big.log", "start_line": -1, "end_line": -1})  # last line
    """
    try:
        path = os.path.expanduser(input.path)
        p = Path(path)

        if not p.exists():
            return ReadFileLinesOutput(ok=False, error=f"File not found: {path}")

        if not p.is_file():
            return ReadFileLinesOutput(ok=False, error=f"Not a file: {path}")

        all_lines = p.read_text(encoding=input.encoding).splitlines()
        total_lines = len(all_lines)

        # Handle negative indices (like Python: -1 = last line)
        start = input.start_line
        end = input.end_line

        if start < 0:
            start = max(1, total_lines + start + 1)
        if end < 0:
            end = total_lines

        # Clamp
        start = max(1, min(start, total_lines))
        end = max(start, min(end, total_lines))

        # Extract lines (1-based to 0-based)
        selected = all_lines[start - 1 : end]

        if input.include_line_numbers:
            width = len(str(end))
            lines_out = [
                f"{str(start + i).rjust(width)}| {line}"
                for i, line in enumerate(selected)
            ]
        else:
            lines_out = selected

        content = "\n".join(lines_out)

        return ReadFileLinesOutput(
            ok=True,
            content=content,
            start_line=start,
            end_line=end,
            total_lines=total_lines,
            path=str(p.absolute()),
        )

    except UnicodeDecodeError:
        return ReadFileLinesOutput(
            ok=False, error=f"Cannot decode file as {input.encoding}"
        )
    except Exception as e:
        return ReadFileLinesOutput(ok=False, error=str(e))


# =============================================================================
# Write File
# =============================================================================


class WriteFileInput(BaseModel):
    """Input for write_file function."""

    path: str = Field(description="Path to the file to write")
    content: str = Field(description="Content to write")
    encoding: str = Field(default="utf-8", description="File encoding")
    create_dirs: bool = Field(
        default=True, description="Create parent directories if needed"
    )


class WriteFileOutput(BaseModel):
    """Output for write_file function."""

    ok: bool
    path: Optional[str] = None
    size: Optional[int] = None
    error: Optional[str] = None


def write_file(input: WriteFileInput) -> WriteFileOutput:
    """
    Write content to a file.

    Examples:
        >>> write_file({"path": "output.txt", "content": "Hello, world!"})
        >>> write_file({"path": "data/results.json", "content": '{"status": "ok"}'})
    """
    try:
        path = os.path.expanduser(input.path)
        p = Path(path)

        if input.create_dirs:
            p.parent.mkdir(parents=True, exist_ok=True)

        p.write_text(input.content, encoding=input.encoding)

        return WriteFileOutput(
            ok=True,
            path=str(p.absolute()),
            size=len(input.content.encode(input.encoding)),
        )

    except PermissionError:
        return WriteFileOutput(ok=False, error=f"Permission denied: {input.path}")
    except Exception as e:
        return WriteFileOutput(ok=False, error=str(e))


# =============================================================================
# List Directory
# =============================================================================


class ListDirectoryInput(BaseModel):
    """Input for list_directory function."""

    path: str = Field(default=".", description="Directory path")
    pattern: Optional[str] = Field(
        default=None, description="Optional glob pattern (e.g., '*.py')"
    )
    recursive: bool = Field(default=False, description="List recursively")


class FileInfo(BaseModel):
    """Information about a single file/directory."""

    name: str
    path: str
    type: str  # "file" or "directory"
    size: Optional[int] = None


class ListDirectoryOutput(BaseModel):
    """Output for list_directory function."""

    ok: bool
    items: List[FileInfo] = []
    count: int = 0
    error: Optional[str] = None


def list_directory(input: ListDirectoryInput) -> ListDirectoryOutput:
    """
    List files and directories.

    Examples:
        >>> list_directory({"path": "."})
        >>> list_directory({"path": "src", "pattern": "*.py"})
        >>> list_directory({"path": ".", "pattern": "*.md", "recursive": True})
    """
    try:
        path = os.path.expanduser(input.path)
        p = Path(path)

        if not p.exists():
            return ListDirectoryOutput(
                ok=False, error=f"Directory not found: {path}"
            )

        if not p.is_dir():
            return ListDirectoryOutput(ok=False, error=f"Not a directory: {path}")

        items = []

        if input.pattern:
            matches = p.rglob(input.pattern) if input.recursive else p.glob(input.pattern)
        else:
            matches = p.iterdir()

        for item in sorted(matches):
            try:
                stat = item.stat()
                items.append(
                    FileInfo(
                        name=item.name,
                        path=str(item),
                        type="directory" if item.is_dir() else "file",
                        size=stat.st_size if item.is_file() else None,
                    )
                )
            except (PermissionError, OSError):
                pass  # Skip inaccessible files

        return ListDirectoryOutput(ok=True, items=items, count=len(items))

    except Exception as e:
        return ListDirectoryOutput(ok=False, error=str(e))


# =============================================================================
# File Info
# =============================================================================


class FileInfoInput(BaseModel):
    """Input for file_info function."""

    path: str = Field(description="Path to get info for")


class FileInfoOutput(BaseModel):
    """Output for file_info function."""

    ok: bool
    path: Optional[str] = None
    name: Optional[str] = None
    type: Optional[str] = None
    size: Optional[int] = None
    extension: Optional[str] = None
    error: Optional[str] = None


def file_info(input: FileInfoInput) -> FileInfoOutput:
    """
    Get detailed information about a file or directory.

    Examples:
        >>> file_info({"path": "README.md"})
    """
    try:
        path = os.path.expanduser(input.path)
        p = Path(path)

        if not p.exists():
            return FileInfoOutput(ok=False, error=f"Path not found: {path}")

        stat = p.stat()

        return FileInfoOutput(
            ok=True,
            path=str(p.absolute()),
            name=p.name,
            type="directory" if p.is_dir() else "file",
            size=stat.st_size,
            extension=p.suffix if p.is_file() else None,
        )

    except Exception as e:
        return FileInfoOutput(ok=False, error=str(e))


# =============================================================================
# Delete File
# =============================================================================


class DeleteFileInput(BaseModel):
    """Input for delete_file function."""

    path: str = Field(description="Path to the file to delete")


class DeleteFileOutput(BaseModel):
    """Output for delete_file function."""

    ok: bool
    deleted: Optional[str] = None
    error: Optional[str] = None


def delete_file(input: DeleteFileInput) -> DeleteFileOutput:
    """
    Delete a file.

    Examples:
        >>> delete_file({"path": "temp.txt"})
    """
    try:
        path = os.path.expanduser(input.path)
        p = Path(path)

        if not p.exists():
            return DeleteFileOutput(ok=False, error=f"File not found: {path}")

        if p.is_dir():
            return DeleteFileOutput(
                ok=False, error=f"Use delete_directory for directories: {path}"
            )

        p.unlink()
        return DeleteFileOutput(ok=True, deleted=str(p))

    except Exception as e:
        return DeleteFileOutput(ok=False, error=str(e))


# =============================================================================
# Create Directory
# =============================================================================


class CreateDirectoryInput(BaseModel):
    """Input for create_directory function."""

    path: str = Field(description="Directory path to create")
    parents: bool = Field(
        default=True, description="Create parent directories if needed"
    )


class CreateDirectoryOutput(BaseModel):
    """Output for create_directory function."""

    ok: bool
    path: Optional[str] = None
    error: Optional[str] = None


def create_directory(input: CreateDirectoryInput) -> CreateDirectoryOutput:
    """
    Create a directory.

    Examples:
        >>> create_directory({"path": "new_folder"})
        >>> create_directory({"path": "a/b/c/deep"})
    """
    try:
        path = os.path.expanduser(input.path)
        p = Path(path)

        p.mkdir(parents=input.parents, exist_ok=True)

        return CreateDirectoryOutput(ok=True, path=str(p.absolute()))

    except Exception as e:
        return CreateDirectoryOutput(ok=False, error=str(e))


# =============================================================================
# Copy File
# =============================================================================


class CopyFileInput(BaseModel):
    """Input for copy_file function."""

    source: str = Field(description="Source file path")
    destination: str = Field(description="Destination path")


class CopyFileOutput(BaseModel):
    """Output for copy_file function."""

    ok: bool
    source: Optional[str] = None
    destination: Optional[str] = None
    error: Optional[str] = None


def copy_file(input: CopyFileInput) -> CopyFileOutput:
    """
    Copy a file.

    Examples:
        >>> copy_file({"source": "file.txt", "destination": "backup/file.txt"})
    """
    try:
        source = os.path.expanduser(input.source)
        destination = os.path.expanduser(input.destination)

        # Create destination directory if needed
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source, destination)

        return CopyFileOutput(ok=True, source=source, destination=destination)

    except Exception as e:
        return CopyFileOutput(ok=False, error=str(e))


# =============================================================================
# Move File
# =============================================================================


class MoveFileInput(BaseModel):
    """Input for move_file function."""

    source: str = Field(description="Source path")
    destination: str = Field(description="Destination path")


class MoveFileOutput(BaseModel):
    """Output for move_file function."""

    ok: bool
    source: Optional[str] = None
    destination: Optional[str] = None
    error: Optional[str] = None


def move_file(input: MoveFileInput) -> MoveFileOutput:
    """
    Move or rename a file or directory.

    Examples:
        >>> move_file({"source": "old.txt", "destination": "new.txt"})
    """
    try:
        source = os.path.expanduser(input.source)
        destination = os.path.expanduser(input.destination)

        shutil.move(source, destination)

        return MoveFileOutput(ok=True, source=source, destination=destination)

    except Exception as e:
        return MoveFileOutput(ok=False, error=str(e))
