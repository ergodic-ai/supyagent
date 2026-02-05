"""
Comprehensive tests for default_tools/edit.py.

Covers: edit_replace, insert_lines, replace_lines, multi_edit, regex_replace
"""

import os
from pathlib import Path

import pytest

from supyagent.default_tools.edit import (
    EditReplaceInput,
    InsertLinesInput,
    ReplaceLinesInput,
    MultiEditInput,
    SingleEdit,
    RegexReplaceInput,
    edit_replace,
    insert_lines,
    replace_lines,
    multi_edit,
    regex_replace,
)


@pytest.fixture
def sample_file(tmp_path):
    """Create a sample Python file for editing."""
    content = """import os
import sys

def main():
    x = 1
    y = 2
    print(x + y)

def helper():
    pass

if __name__ == "__main__":
    main()
"""
    path = tmp_path / "sample.py"
    path.write_text(content)
    return path


@pytest.fixture
def numbered_file(tmp_path):
    """Create a file with predictable numbered lines."""
    lines = [f"Line {i}" for i in range(1, 21)]
    path = tmp_path / "numbered.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


# =========================================================================
# edit_replace
# =========================================================================


class TestEditReplace:
    def test_simple_replace(self, sample_file):
        result = edit_replace(
            EditReplaceInput(file=str(sample_file), old_text="x = 1", new_text="x = 42")
        )
        assert result.ok is True
        assert result.replacements_made == 1
        content = sample_file.read_text()
        assert "x = 42" in content
        assert "x = 1" not in content

    def test_replace_multiline(self, sample_file):
        result = edit_replace(
            EditReplaceInput(
                file=str(sample_file),
                old_text="def helper():\n    pass",
                new_text="def helper():\n    return 'helped!'",
            )
        )
        assert result.ok is True
        assert "return 'helped!'" in sample_file.read_text()

    def test_replace_not_found(self, sample_file):
        result = edit_replace(
            EditReplaceInput(file=str(sample_file), old_text="DOES NOT EXIST", new_text="new")
        )
        assert result.ok is False
        assert "not found" in result.error.lower()

    def test_replace_all_occurrences(self, tmp_path):
        path = tmp_path / "repeats.txt"
        path.write_text("foo bar foo baz foo")
        result = edit_replace(
            EditReplaceInput(file=str(path), old_text="foo", new_text="qux", count=0)
        )
        assert result.ok is True
        assert result.replacements_made == 3
        assert path.read_text() == "qux bar qux baz qux"

    def test_replace_limited_count(self, tmp_path):
        path = tmp_path / "repeats.txt"
        path.write_text("foo bar foo baz foo")
        result = edit_replace(
            EditReplaceInput(file=str(path), old_text="foo", new_text="qux", count=2)
        )
        assert result.ok is True
        assert result.replacements_made == 2
        assert path.read_text() == "qux bar qux baz foo"

    def test_replace_preserves_indentation(self, sample_file):
        result = edit_replace(
            EditReplaceInput(
                file=str(sample_file),
                old_text="    print(x + y)",
                new_text="    result = x + y\n    print(result)",
            )
        )
        assert result.ok is True
        content = sample_file.read_text()
        assert "    result = x + y\n    print(result)" in content

    def test_replace_nonexistent_file(self, tmp_path):
        result = edit_replace(
            EditReplaceInput(file=str(tmp_path / "nope.py"), old_text="a", new_text="b")
        )
        assert result.ok is False
        assert "not found" in result.error.lower()

    def test_replace_on_directory(self, tmp_path):
        result = edit_replace(
            EditReplaceInput(file=str(tmp_path), old_text="a", new_text="b")
        )
        assert result.ok is False
        assert "Not a file" in result.error

    def test_replace_with_empty_string(self, sample_file):
        """Replace text with nothing (deletion)."""
        result = edit_replace(
            EditReplaceInput(file=str(sample_file), old_text="import sys\n", new_text="")
        )
        assert result.ok is True
        assert "import sys" not in sample_file.read_text()

    def test_replace_whitespace_sensitive(self, tmp_path):
        path = tmp_path / "ws.txt"
        path.write_text("  hello  \n  world  \n")
        result = edit_replace(
            EditReplaceInput(file=str(path), old_text="  hello  ", new_text="  hi  ")
        )
        assert result.ok is True
        assert "  hi  " in path.read_text()


# =========================================================================
# insert_lines
# =========================================================================


class TestInsertLines:
    def test_insert_at_beginning(self, numbered_file):
        result = insert_lines(
            InsertLinesInput(file=str(numbered_file), line_number=0, text="# Header")
        )
        assert result.ok is True
        assert result.lines_inserted == 1
        content = numbered_file.read_text()
        assert content.startswith("# Header\n")

    def test_insert_at_end(self, numbered_file):
        result = insert_lines(
            InsertLinesInput(file=str(numbered_file), line_number=-1, text="# Footer")
        )
        assert result.ok is True
        content = numbered_file.read_text()
        assert content.rstrip().endswith("# Footer")

    def test_insert_in_middle(self, numbered_file):
        result = insert_lines(
            InsertLinesInput(file=str(numbered_file), line_number=5, text="INSERTED")
        )
        assert result.ok is True
        lines = numbered_file.read_text().splitlines()
        assert lines[3] == "Line 4"
        assert lines[4] == "INSERTED"  # Inserted BEFORE line 5
        assert lines[5] == "Line 5"

    def test_insert_multiple_lines(self, numbered_file):
        result = insert_lines(
            InsertLinesInput(
                file=str(numbered_file), line_number=1, text="A\nB\nC"
            )
        )
        assert result.ok is True
        assert result.lines_inserted == 3
        lines = numbered_file.read_text().splitlines()
        assert lines[0] == "A"
        assert lines[1] == "B"
        assert lines[2] == "C"
        assert lines[3] == "Line 1"

    def test_insert_nonexistent_file(self, tmp_path):
        result = insert_lines(
            InsertLinesInput(file=str(tmp_path / "nope.txt"), line_number=1, text="test")
        )
        assert result.ok is False

    def test_insert_on_directory(self, tmp_path):
        result = insert_lines(
            InsertLinesInput(file=str(tmp_path), line_number=1, text="test")
        )
        assert result.ok is False

    def test_insert_preserves_line_count(self, numbered_file):
        original_lines = len(numbered_file.read_text().splitlines())
        insert_lines(
            InsertLinesInput(file=str(numbered_file), line_number=10, text="NEW")
        )
        new_lines = len(numbered_file.read_text().splitlines())
        assert new_lines == original_lines + 1


# =========================================================================
# replace_lines
# =========================================================================


class TestReplaceLines:
    def test_replace_single_line(self, numbered_file):
        result = replace_lines(
            ReplaceLinesInput(
                file=str(numbered_file), start_line=5, end_line=5, new_text="REPLACED"
            )
        )
        assert result.ok is True
        assert result.lines_removed == 1
        assert result.lines_added == 1
        lines = numbered_file.read_text().splitlines()
        assert lines[4] == "REPLACED"

    def test_replace_range_with_fewer_lines(self, numbered_file):
        result = replace_lines(
            ReplaceLinesInput(
                file=str(numbered_file), start_line=3, end_line=7, new_text="SINGLE"
            )
        )
        assert result.ok is True
        assert result.lines_removed == 5
        assert result.lines_added == 1
        lines = numbered_file.read_text().splitlines()
        assert lines[2] == "SINGLE"
        assert lines[3] == "Line 8"

    def test_replace_range_with_more_lines(self, numbered_file):
        result = replace_lines(
            ReplaceLinesInput(
                file=str(numbered_file),
                start_line=1,
                end_line=2,
                new_text="A\nB\nC\nD",
            )
        )
        assert result.ok is True
        assert result.lines_removed == 2
        assert result.lines_added == 4
        lines = numbered_file.read_text().splitlines()
        assert lines[:4] == ["A", "B", "C", "D"]
        assert lines[4] == "Line 3"

    def test_replace_beyond_file_end(self, numbered_file):
        result = replace_lines(
            ReplaceLinesInput(
                file=str(numbered_file), start_line=18, end_line=25, new_text="END"
            )
        )
        assert result.ok is True
        lines = numbered_file.read_text().splitlines()
        assert lines[-1] == "END"

    def test_replace_start_beyond_file(self, numbered_file):
        result = replace_lines(
            ReplaceLinesInput(
                file=str(numbered_file), start_line=999, end_line=1000, new_text="nope"
            )
        )
        assert result.ok is False
        assert "beyond" in result.error.lower()

    def test_replace_nonexistent_file(self, tmp_path):
        result = replace_lines(
            ReplaceLinesInput(
                file=str(tmp_path / "nope.txt"), start_line=1, end_line=1, new_text="x"
            )
        )
        assert result.ok is False

    def test_replace_first_line(self, numbered_file):
        result = replace_lines(
            ReplaceLinesInput(
                file=str(numbered_file), start_line=1, end_line=1, new_text="FIRST"
            )
        )
        assert result.ok is True
        assert numbered_file.read_text().startswith("FIRST\n")

    def test_replace_last_line(self, numbered_file):
        result = replace_lines(
            ReplaceLinesInput(
                file=str(numbered_file), start_line=20, end_line=20, new_text="LAST"
            )
        )
        assert result.ok is True
        lines = numbered_file.read_text().splitlines()
        assert lines[-1] == "LAST"


# =========================================================================
# multi_edit
# =========================================================================


class TestMultiEdit:
    def test_multiple_edits(self, sample_file):
        result = multi_edit(
            MultiEditInput(
                file=str(sample_file),
                edits=[
                    SingleEdit(old_text="x = 1", new_text="x = 10"),
                    SingleEdit(old_text="y = 2", new_text="y = 20"),
                ],
            )
        )
        assert result.ok is True
        assert result.applied == 2
        assert len(result.failed) == 0
        content = sample_file.read_text()
        assert "x = 10" in content
        assert "y = 20" in content

    def test_partial_failure(self, sample_file):
        result = multi_edit(
            MultiEditInput(
                file=str(sample_file),
                edits=[
                    SingleEdit(old_text="x = 1", new_text="x = 10"),
                    SingleEdit(old_text="DOES_NOT_EXIST", new_text="nope"),
                    SingleEdit(old_text="y = 2", new_text="y = 20"),
                ],
            )
        )
        assert result.ok is True  # Some edits succeeded
        assert result.applied == 2
        assert len(result.failed) == 1
        assert "Edit 2" in result.failed[0]

    def test_all_edits_fail(self, sample_file):
        result = multi_edit(
            MultiEditInput(
                file=str(sample_file),
                edits=[
                    SingleEdit(old_text="NOPE1", new_text="a"),
                    SingleEdit(old_text="NOPE2", new_text="b"),
                ],
            )
        )
        assert result.ok is False
        assert result.applied == 0
        assert len(result.failed) == 2

    def test_sequential_edits(self, tmp_path):
        """Second edit sees the result of the first."""
        path = tmp_path / "seq.txt"
        path.write_text("A -> B -> C")
        result = multi_edit(
            MultiEditInput(
                file=str(path),
                edits=[
                    SingleEdit(old_text="A", new_text="X"),
                    SingleEdit(old_text="X -> B", new_text="DONE"),  # depends on first edit
                ],
            )
        )
        assert result.ok is True
        assert result.applied == 2
        assert path.read_text() == "DONE -> C"

    def test_nonexistent_file(self, tmp_path):
        result = multi_edit(
            MultiEditInput(
                file=str(tmp_path / "nope.py"),
                edits=[SingleEdit(old_text="a", new_text="b")],
            )
        )
        assert result.ok is False
        assert "not found" in result.error.lower()

    def test_empty_edits_list(self, sample_file):
        result = multi_edit(
            MultiEditInput(file=str(sample_file), edits=[])
        )
        assert result.ok is False
        assert result.applied == 0

    def test_multi_edit_does_not_modify_on_failure(self, sample_file):
        """If all edits fail, file should not be modified."""
        original = sample_file.read_text()
        multi_edit(
            MultiEditInput(
                file=str(sample_file),
                edits=[SingleEdit(old_text="NOPE", new_text="x")],
            )
        )
        assert sample_file.read_text() == original


# =========================================================================
# regex_replace
# =========================================================================


class TestRegexReplace:
    def test_simple_regex(self, sample_file):
        result = regex_replace(
            RegexReplaceInput(
                file=str(sample_file),
                pattern=r"x = \d+",
                replacement="x = 99",
            )
        )
        assert result.ok is True
        assert result.replacements_made == 1
        assert "x = 99" in sample_file.read_text()

    def test_regex_with_capture_group(self, tmp_path):
        path = tmp_path / "funcs.py"
        path.write_text("def foo():\n    pass\ndef bar():\n    pass\n")
        result = regex_replace(
            RegexReplaceInput(
                file=str(path),
                pattern=r"def (\w+)\(\):",
                replacement=r"def \1(self):",
            )
        )
        assert result.ok is True
        assert result.replacements_made == 2
        content = path.read_text()
        assert "def foo(self):" in content
        assert "def bar(self):" in content

    def test_regex_replace_all(self, tmp_path):
        path = tmp_path / "test.txt"
        path.write_text("cat dog cat dog cat")
        result = regex_replace(
            RegexReplaceInput(file=str(path), pattern="cat", replacement="bird")
        )
        assert result.ok is True
        assert result.replacements_made == 3
        assert path.read_text() == "bird dog bird dog bird"

    def test_regex_replace_limited(self, tmp_path):
        path = tmp_path / "test.txt"
        path.write_text("cat dog cat dog cat")
        result = regex_replace(
            RegexReplaceInput(file=str(path), pattern="cat", replacement="bird", count=1)
        )
        assert result.ok is True
        assert result.replacements_made == 1
        assert path.read_text() == "bird dog cat dog cat"

    def test_regex_no_match(self, sample_file):
        result = regex_replace(
            RegexReplaceInput(
                file=str(sample_file), pattern=r"ZZZZZ\d+", replacement="nope"
            )
        )
        assert result.ok is False
        assert "not found" in result.error.lower()

    def test_invalid_regex(self, sample_file):
        result = regex_replace(
            RegexReplaceInput(
                file=str(sample_file), pattern=r"[invalid", replacement="x"
            )
        )
        assert result.ok is False
        assert "invalid regex" in result.error.lower()

    def test_regex_nonexistent_file(self, tmp_path):
        result = regex_replace(
            RegexReplaceInput(
                file=str(tmp_path / "nope.py"), pattern="x", replacement="y"
            )
        )
        assert result.ok is False

    def test_regex_version_bump(self, tmp_path):
        path = tmp_path / "config.py"
        path.write_text('VERSION = "1.2.3"\nNAME = "myapp"\n')
        result = regex_replace(
            RegexReplaceInput(
                file=str(path),
                pattern=r'VERSION = "[^"]+"',
                replacement='VERSION = "2.0.0"',
            )
        )
        assert result.ok is True
        assert 'VERSION = "2.0.0"' in path.read_text()

    def test_regex_multiline_pattern(self, tmp_path):
        path = tmp_path / "test.txt"
        path.write_text("aaa 123 bbb 456 ccc 789")
        result = regex_replace(
            RegexReplaceInput(
                file=str(path), pattern=r"\d+", replacement="NUM"
            )
        )
        assert result.ok is True
        assert result.replacements_made == 3
        assert path.read_text() == "aaa NUM bbb NUM ccc NUM"


# =========================================================================
# Integration: edit then verify
# =========================================================================


class TestEditIntegration:
    def test_edit_replace_then_read(self, sample_file):
        edit_replace(
            EditReplaceInput(file=str(sample_file), old_text="x = 1", new_text="x = 100")
        )
        content = sample_file.read_text()
        assert "x = 100" in content
        assert "x = 1\n" not in content  # But x = 100 contains "x = 1" prefix

    def test_insert_then_replace_lines(self, numbered_file):
        """Insert at top, then replace the inserted lines."""
        insert_lines(
            InsertLinesInput(file=str(numbered_file), line_number=0, text="HEADER1\nHEADER2")
        )
        lines = numbered_file.read_text().splitlines()
        assert lines[0] == "HEADER1"
        assert lines[1] == "HEADER2"

        replace_lines(
            ReplaceLinesInput(
                file=str(numbered_file), start_line=1, end_line=2, new_text="COMBINED_HEADER"
            )
        )
        lines = numbered_file.read_text().splitlines()
        assert lines[0] == "COMBINED_HEADER"
        assert lines[1] == "Line 1"

    def test_multi_edit_then_regex(self, sample_file):
        """Chain multi_edit with regex_replace."""
        multi_edit(
            MultiEditInput(
                file=str(sample_file),
                edits=[
                    SingleEdit(old_text="x = 1", new_text="x = 10"),
                    SingleEdit(old_text="y = 2", new_text="y = 20"),
                ],
            )
        )
        regex_replace(
            RegexReplaceInput(
                file=str(sample_file),
                pattern=r"(\w+) = (\d+)",
                replacement=r"\1: int = \2",
                count=2,  # Only first 2 matches
            )
        )
        content = sample_file.read_text()
        assert "x: int = 10" in content
        assert "y: int = 20" in content
