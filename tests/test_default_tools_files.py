"""
Comprehensive tests for default_tools/files.py.

Covers: read_file, read_file_lines, write_file, list_directory,
        file_info, delete_file, create_directory, copy_file, move_file
"""

import os
import tempfile
from pathlib import Path

import pytest

from supyagent.default_tools.files import (
    ReadFileInput,
    ReadFileLinesInput,
    WriteFileInput,
    ListDirectoryInput,
    FileInfoInput,
    DeleteFileInput,
    CreateDirectoryInput,
    CopyFileInput,
    MoveFileInput,
    read_file,
    read_file_lines,
    write_file,
    list_directory,
    file_info,
    delete_file,
    create_directory,
    copy_file,
    move_file,
)


@pytest.fixture
def workspace(tmp_path):
    """Create a realistic workspace with files and directories."""
    # Files
    (tmp_path / "hello.txt").write_text("Hello, world!\n")
    (tmp_path / "data.json").write_text('{"key": "value"}\n')

    # Multi-line file
    lines = [f"Line {i}" for i in range(1, 101)]
    (tmp_path / "hundred_lines.txt").write_text("\n".join(lines) + "\n")

    # Nested directories
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("def main():\n    print('hello')\n")
    (src / "utils.py").write_text("def helper():\n    pass\n")

    sub = src / "sub"
    sub.mkdir()
    (sub / "deep.py").write_text("# deep file\n")

    # Empty file
    (tmp_path / "empty.txt").write_text("")

    return tmp_path


# =========================================================================
# read_file
# =========================================================================


class TestReadFile:
    def test_read_existing_file(self, workspace):
        result = read_file(ReadFileInput(path=str(workspace / "hello.txt")))
        assert result.ok is True
        assert result.content == "Hello, world!\n"
        assert result.size is not None
        assert result.size > 0
        assert result.path is not None

    def test_read_json_file(self, workspace):
        result = read_file(ReadFileInput(path=str(workspace / "data.json")))
        assert result.ok is True
        assert '"key"' in result.content

    def test_read_empty_file(self, workspace):
        result = read_file(ReadFileInput(path=str(workspace / "empty.txt")))
        assert result.ok is True
        assert result.content == ""
        assert result.size == 0

    def test_read_nonexistent_file(self, workspace):
        result = read_file(ReadFileInput(path=str(workspace / "nope.txt")))
        assert result.ok is False
        assert "not found" in result.error.lower()

    def test_read_directory_instead_of_file(self, workspace):
        result = read_file(ReadFileInput(path=str(workspace / "src")))
        assert result.ok is False
        assert "Not a file" in result.error

    def test_read_nested_file(self, workspace):
        result = read_file(ReadFileInput(path=str(workspace / "src" / "main.py")))
        assert result.ok is True
        assert "def main" in result.content

    def test_read_with_tilde_expansion(self):
        """Tilde paths should be expanded."""
        # Just verify it doesn't crash; ~ may or may not have readable files
        result = read_file(ReadFileInput(path="~/nonexistent_test_file_xyz"))
        assert result.ok is False  # File shouldn't exist


# =========================================================================
# read_file_lines
# =========================================================================


class TestReadFileLines:
    def test_read_first_ten_lines(self, workspace):
        result = read_file_lines(
            ReadFileLinesInput(path=str(workspace / "hundred_lines.txt"), start_line=1, end_line=10)
        )
        assert result.ok is True
        assert result.start_line == 1
        assert result.end_line == 10
        assert result.total_lines == 100
        # With line numbers enabled (default)
        assert " 1| Line 1" in result.content
        assert "10| Line 10" in result.content

    def test_read_middle_lines(self, workspace):
        result = read_file_lines(
            ReadFileLinesInput(path=str(workspace / "hundred_lines.txt"), start_line=50, end_line=55)
        )
        assert result.ok is True
        assert result.start_line == 50
        assert result.end_line == 55
        assert "Line 50" in result.content
        assert "Line 55" in result.content

    def test_read_last_line_negative_index(self, workspace):
        result = read_file_lines(
            ReadFileLinesInput(path=str(workspace / "hundred_lines.txt"), start_line=-1, end_line=-1)
        )
        assert result.ok is True
        assert "Line 100" in result.content

    def test_read_entire_file_default(self, workspace):
        result = read_file_lines(
            ReadFileLinesInput(path=str(workspace / "hundred_lines.txt"))
        )
        assert result.ok is True
        assert result.start_line == 1
        assert result.end_line == 100
        assert result.total_lines == 100

    def test_read_without_line_numbers(self, workspace):
        result = read_file_lines(
            ReadFileLinesInput(
                path=str(workspace / "hundred_lines.txt"),
                start_line=1,
                end_line=3,
                include_line_numbers=False,
            )
        )
        assert result.ok is True
        assert "|" not in result.content
        assert result.content == "Line 1\nLine 2\nLine 3"

    def test_read_beyond_file_end(self, workspace):
        result = read_file_lines(
            ReadFileLinesInput(
                path=str(workspace / "hundred_lines.txt"), start_line=95, end_line=200
            )
        )
        assert result.ok is True
        assert result.end_line == 100  # Clamped to actual end

    def test_read_nonexistent_file(self, workspace):
        result = read_file_lines(ReadFileLinesInput(path=str(workspace / "nope.txt")))
        assert result.ok is False
        assert "not found" in result.error.lower()

    def test_read_empty_file_lines(self, workspace):
        result = read_file_lines(ReadFileLinesInput(path=str(workspace / "empty.txt")))
        assert result.ok is True
        assert result.total_lines == 0


# =========================================================================
# write_file
# =========================================================================


class TestWriteFile:
    def test_write_new_file(self, workspace):
        path = str(workspace / "new.txt")
        result = write_file(WriteFileInput(path=path, content="New content!"))
        assert result.ok is True
        assert result.size == len("New content!")
        assert Path(path).read_text() == "New content!"

    def test_write_overwrites_existing(self, workspace):
        path = str(workspace / "hello.txt")
        result = write_file(WriteFileInput(path=path, content="Overwritten"))
        assert result.ok is True
        assert Path(path).read_text() == "Overwritten"

    def test_write_creates_parent_dirs(self, workspace):
        path = str(workspace / "a" / "b" / "c" / "deep.txt")
        result = write_file(WriteFileInput(path=path, content="Deep file"))
        assert result.ok is True
        assert Path(path).read_text() == "Deep file"

    def test_write_without_create_dirs(self, workspace):
        path = str(workspace / "nonexistent_dir" / "file.txt")
        result = write_file(WriteFileInput(path=path, content="test", create_dirs=False))
        assert result.ok is False

    def test_write_empty_content(self, workspace):
        path = str(workspace / "blank.txt")
        result = write_file(WriteFileInput(path=path, content=""))
        assert result.ok is True
        assert Path(path).read_text() == ""
        assert result.size == 0

    def test_write_unicode_content(self, workspace):
        path = str(workspace / "unicode.txt")
        content = "Hello 🌍\nСтрока на русском\n日本語テスト"
        result = write_file(WriteFileInput(path=path, content=content))
        assert result.ok is True
        assert Path(path).read_text() == content

    def test_write_multiline(self, workspace):
        path = str(workspace / "multi.txt")
        content = "line 1\nline 2\nline 3\n"
        result = write_file(WriteFileInput(path=path, content=content))
        assert result.ok is True
        assert Path(path).read_text() == content


# =========================================================================
# list_directory
# =========================================================================


class TestListDirectory:
    def test_list_top_level(self, workspace):
        result = list_directory(ListDirectoryInput(path=str(workspace)))
        assert result.ok is True
        assert result.count > 0
        names = {item.name for item in result.items}
        assert "hello.txt" in names
        assert "src" in names

    def test_list_with_pattern(self, workspace):
        result = list_directory(ListDirectoryInput(path=str(workspace), pattern="*.txt"))
        assert result.ok is True
        for item in result.items:
            assert item.name.endswith(".txt")

    def test_list_recursive(self, workspace):
        result = list_directory(
            ListDirectoryInput(path=str(workspace), pattern="*.py", recursive=True)
        )
        assert result.ok is True
        names = {item.name for item in result.items}
        assert "main.py" in names
        assert "deep.py" in names

    def test_list_nonexistent_dir(self, workspace):
        result = list_directory(ListDirectoryInput(path=str(workspace / "nope")))
        assert result.ok is False
        assert "not found" in result.error.lower()

    def test_list_file_not_dir(self, workspace):
        result = list_directory(ListDirectoryInput(path=str(workspace / "hello.txt")))
        assert result.ok is False
        assert "Not a directory" in result.error

    def test_list_subdirectory(self, workspace):
        result = list_directory(ListDirectoryInput(path=str(workspace / "src")))
        assert result.ok is True
        names = {item.name for item in result.items}
        assert "main.py" in names
        assert "utils.py" in names
        assert "sub" in names

    def test_items_have_type(self, workspace):
        result = list_directory(ListDirectoryInput(path=str(workspace)))
        assert result.ok is True
        types = {item.name: item.type for item in result.items}
        assert types.get("hello.txt") == "file"
        assert types.get("src") == "directory"

    def test_files_have_size(self, workspace):
        result = list_directory(ListDirectoryInput(path=str(workspace)))
        assert result.ok is True
        for item in result.items:
            if item.type == "file":
                assert item.size is not None and item.size >= 0


# =========================================================================
# file_info
# =========================================================================


class TestFileInfo:
    def test_file_info_on_file(self, workspace):
        result = file_info(FileInfoInput(path=str(workspace / "hello.txt")))
        assert result.ok is True
        assert result.name == "hello.txt"
        assert result.type == "file"
        assert result.size > 0
        assert result.extension == ".txt"

    def test_file_info_on_directory(self, workspace):
        result = file_info(FileInfoInput(path=str(workspace / "src")))
        assert result.ok is True
        assert result.name == "src"
        assert result.type == "directory"

    def test_file_info_nonexistent(self, workspace):
        result = file_info(FileInfoInput(path=str(workspace / "nope")))
        assert result.ok is False
        assert "not found" in result.error.lower()

    def test_file_info_python_file(self, workspace):
        result = file_info(FileInfoInput(path=str(workspace / "src" / "main.py")))
        assert result.ok is True
        assert result.extension == ".py"

    def test_file_info_no_extension(self, workspace):
        (workspace / "Makefile").write_text("all:\n\techo hello\n")
        result = file_info(FileInfoInput(path=str(workspace / "Makefile")))
        assert result.ok is True
        assert result.extension == ""  # no extension


# =========================================================================
# delete_file
# =========================================================================


class TestDeleteFile:
    def test_delete_existing_file(self, workspace):
        path = workspace / "hello.txt"
        assert path.exists()
        result = delete_file(DeleteFileInput(path=str(path)))
        assert result.ok is True
        assert not path.exists()

    def test_delete_nonexistent_file(self, workspace):
        result = delete_file(DeleteFileInput(path=str(workspace / "nope.txt")))
        assert result.ok is False
        assert "not found" in result.error.lower()

    def test_delete_directory_fails(self, workspace):
        result = delete_file(DeleteFileInput(path=str(workspace / "src")))
        assert result.ok is False
        assert "delete_directory" in result.error.lower()

    def test_delete_then_read_fails(self, workspace):
        path = workspace / "data.json"
        delete_file(DeleteFileInput(path=str(path)))
        result = read_file(ReadFileInput(path=str(path)))
        assert result.ok is False


# =========================================================================
# create_directory
# =========================================================================


class TestCreateDirectory:
    def test_create_simple_dir(self, workspace):
        path = workspace / "newdir"
        result = create_directory(CreateDirectoryInput(path=str(path)))
        assert result.ok is True
        assert path.is_dir()

    def test_create_nested_dirs(self, workspace):
        path = workspace / "a" / "b" / "c"
        result = create_directory(CreateDirectoryInput(path=str(path)))
        assert result.ok is True
        assert path.is_dir()

    def test_create_existing_dir_is_ok(self, workspace):
        result = create_directory(CreateDirectoryInput(path=str(workspace / "src")))
        assert result.ok is True

    def test_create_without_parents(self, workspace):
        path = workspace / "x" / "y" / "z"
        result = create_directory(CreateDirectoryInput(path=str(path), parents=False))
        assert result.ok is False  # parent x/ doesn't exist


# =========================================================================
# copy_file
# =========================================================================


class TestCopyFile:
    def test_copy_file_same_dir(self, workspace):
        result = copy_file(
            CopyFileInput(
                source=str(workspace / "hello.txt"),
                destination=str(workspace / "hello_copy.txt"),
            )
        )
        assert result.ok is True
        assert (workspace / "hello_copy.txt").read_text() == "Hello, world!\n"
        # Original still exists
        assert (workspace / "hello.txt").exists()

    def test_copy_to_new_subdir(self, workspace):
        result = copy_file(
            CopyFileInput(
                source=str(workspace / "hello.txt"),
                destination=str(workspace / "backup" / "hello.txt"),
            )
        )
        assert result.ok is True
        assert (workspace / "backup" / "hello.txt").read_text() == "Hello, world!\n"

    def test_copy_nonexistent_source(self, workspace):
        result = copy_file(
            CopyFileInput(
                source=str(workspace / "nope.txt"),
                destination=str(workspace / "dest.txt"),
            )
        )
        assert result.ok is False

    def test_copy_preserves_content(self, workspace):
        # Write binary-like content
        content = "Special chars: àéîõü\nTab\there\n"
        (workspace / "special.txt").write_text(content)
        result = copy_file(
            CopyFileInput(
                source=str(workspace / "special.txt"),
                destination=str(workspace / "special_copy.txt"),
            )
        )
        assert result.ok is True
        assert (workspace / "special_copy.txt").read_text() == content


# =========================================================================
# move_file
# =========================================================================


class TestMoveFile:
    def test_move_file_rename(self, workspace):
        result = move_file(
            MoveFileInput(
                source=str(workspace / "hello.txt"),
                destination=str(workspace / "renamed.txt"),
            )
        )
        assert result.ok is True
        assert not (workspace / "hello.txt").exists()
        assert (workspace / "renamed.txt").read_text() == "Hello, world!\n"

    def test_move_file_to_subdir(self, workspace):
        dest = workspace / "archive"
        dest.mkdir()
        result = move_file(
            MoveFileInput(
                source=str(workspace / "hello.txt"),
                destination=str(dest / "hello.txt"),
            )
        )
        assert result.ok is True
        assert not (workspace / "hello.txt").exists()
        assert (dest / "hello.txt").exists()

    def test_move_nonexistent_source(self, workspace):
        result = move_file(
            MoveFileInput(
                source=str(workspace / "nope.txt"),
                destination=str(workspace / "dest.txt"),
            )
        )
        assert result.ok is False

    def test_move_directory(self, workspace):
        result = move_file(
            MoveFileInput(
                source=str(workspace / "src"),
                destination=str(workspace / "source"),
            )
        )
        assert result.ok is True
        assert not (workspace / "src").exists()
        assert (workspace / "source" / "main.py").exists()


# =========================================================================
# Integration: write then read
# =========================================================================


class TestFileRoundTrips:
    def test_write_then_read(self, workspace):
        path = str(workspace / "roundtrip.txt")
        content = "Round trip content\nWith multiple lines\n"
        write_file(WriteFileInput(path=path, content=content))
        result = read_file(ReadFileInput(path=path))
        assert result.ok is True
        assert result.content == content

    def test_write_then_read_lines(self, workspace):
        path = str(workspace / "numbered.txt")
        lines = "\n".join(f"Item {i}" for i in range(1, 21)) + "\n"
        write_file(WriteFileInput(path=path, content=lines))
        result = read_file_lines(
            ReadFileLinesInput(path=path, start_line=5, end_line=8, include_line_numbers=False)
        )
        assert result.ok is True
        assert result.content == "Item 5\nItem 6\nItem 7\nItem 8"

    def test_copy_then_delete_original(self, workspace):
        src = str(workspace / "hello.txt")
        dst = str(workspace / "backup.txt")
        copy_file(CopyFileInput(source=src, destination=dst))
        delete_file(DeleteFileInput(path=src))
        assert not Path(src).exists()
        result = read_file(ReadFileInput(path=dst))
        assert result.ok is True
        assert result.content == "Hello, world!\n"
