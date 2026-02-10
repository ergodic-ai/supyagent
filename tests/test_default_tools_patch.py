"""
Comprehensive tests for default_tools/patch.py.

Covers: apply_patch with Add, Update, Delete operations,
multi-file patches, dry run, atomic validation, and error cases.
"""

import pytest

from supyagent.default_tools.patch import (
    ApplyPatchInput,
    apply_patch,
)


@pytest.fixture
def sample_py(tmp_path):
    """Create a sample Python file for patching."""
    content = "import os\nimport sys\n\ndef main():\n    x = 1\n    y = 2\n    print(x + y)\n\ndef helper():\n    pass\n"
    path = tmp_path / "sample.py"
    path.write_text(content)
    return path


@pytest.fixture
def config_file(tmp_path):
    """Create a sample config file."""
    content = 'HOST = "localhost"\nPORT = 8080\nDEBUG = False\n'
    path = tmp_path / "config.py"
    path.write_text(content)
    return path


# =========================================================================
# Add File
# =========================================================================


class TestAddFile:
    def test_add_new_file(self, tmp_path):
        patch = "*** Add File: hello.py\n+print('hello world')\n+print('goodbye')\n"
        result = apply_patch(
            ApplyPatchInput(patch=patch, working_dir=str(tmp_path))
        )
        assert result.ok is True
        assert result.files_added == 1
        created = tmp_path / "hello.py"
        assert created.exists()
        content = created.read_text()
        assert "print('hello world')" in content
        assert "print('goodbye')" in content

    def test_add_file_already_exists(self, tmp_path, sample_py):
        patch = "*** Add File: sample.py\n+new content\n"
        result = apply_patch(
            ApplyPatchInput(patch=patch, working_dir=str(tmp_path))
        )
        assert result.ok is False
        assert "already exists" in result.error

    def test_add_file_nested_dirs(self, tmp_path):
        patch = "*** Add File: src/lib/utils.py\n+def helper():\n+    return True\n"
        result = apply_patch(
            ApplyPatchInput(patch=patch, working_dir=str(tmp_path))
        )
        assert result.ok is True
        created = tmp_path / "src" / "lib" / "utils.py"
        assert created.exists()
        assert "def helper():" in created.read_text()

    def test_add_empty_file(self, tmp_path):
        patch = "*** Add File: empty.txt\n"
        result = apply_patch(
            ApplyPatchInput(patch=patch, working_dir=str(tmp_path))
        )
        assert result.ok is True
        assert (tmp_path / "empty.txt").exists()


# =========================================================================
# Update File
# =========================================================================


class TestUpdateFile:
    def test_update_single_hunk(self, tmp_path, sample_py):
        patch = (
            "*** Update File: sample.py\n"
            "@@ def main():\n"
            "-    x = 1\n"
            "+    x = 42\n"
        )
        result = apply_patch(
            ApplyPatchInput(patch=patch, working_dir=str(tmp_path))
        )
        assert result.ok is True
        assert result.files_modified == 1
        content = sample_py.read_text()
        assert "x = 42" in content
        assert "x = 1" not in content

    def test_update_multiple_hunks(self, tmp_path, sample_py):
        patch = (
            "*** Update File: sample.py\n"
            "@@ def main():\n"
            "-    x = 1\n"
            "+    x = 10\n"
            "@@ def helper():\n"
            "-    pass\n"
            "+    return 'helped'\n"
        )
        result = apply_patch(
            ApplyPatchInput(patch=patch, working_dir=str(tmp_path))
        )
        assert result.ok is True
        content = sample_py.read_text()
        assert "x = 10" in content
        assert "return 'helped'" in content

    def test_update_context_not_found(self, tmp_path, sample_py):
        patch = (
            "*** Update File: sample.py\n"
            "@@ def nonexistent_function():\n"
            "-    old\n"
            "+    new\n"
        )
        result = apply_patch(
            ApplyPatchInput(patch=patch, working_dir=str(tmp_path))
        )
        assert result.ok is False
        assert "not found" in result.error

    def test_update_remove_lines_mismatch(self, tmp_path, sample_py):
        patch = (
            "*** Update File: sample.py\n"
            "@@ def main():\n"
            "-    wrong_content = 999\n"
            "+    new_content = 1\n"
        )
        result = apply_patch(
            ApplyPatchInput(patch=patch, working_dir=str(tmp_path))
        )
        assert result.ok is False
        assert "don't match" in result.error

    def test_update_nonexistent_file(self, tmp_path):
        patch = (
            "*** Update File: nonexistent.py\n"
            "@@ some context\n"
            "-old\n"
            "+new\n"
        )
        result = apply_patch(
            ApplyPatchInput(patch=patch, working_dir=str(tmp_path))
        )
        assert result.ok is False
        assert "not found" in result.error

    def test_update_add_lines(self, tmp_path, sample_py):
        """Add new lines after a context anchor (no remove lines)."""
        patch = (
            "*** Update File: sample.py\n"
            "@@ import sys\n"
            "+import json\n"
        )
        result = apply_patch(
            ApplyPatchInput(patch=patch, working_dir=str(tmp_path))
        )
        assert result.ok is True
        content = sample_py.read_text()
        assert "import json" in content

    def test_update_remove_lines(self, tmp_path, sample_py):
        """Remove lines with no replacement."""
        patch = (
            "*** Update File: sample.py\n"
            "@@ import os\n"
            "-import sys\n"
        )
        result = apply_patch(
            ApplyPatchInput(patch=patch, working_dir=str(tmp_path))
        )
        assert result.ok is True
        content = sample_py.read_text()
        assert "import sys" not in content
        assert "import os" in content

    def test_update_with_context_lines(self, tmp_path, config_file):
        """Unchanged context lines (prefixed with space) are preserved."""
        patch = (
            "*** Update File: config.py\n"
            '@@ HOST = "localhost"\n'
            " PORT = 8080\n"
            "-DEBUG = False\n"
            "+DEBUG = True\n"
        )
        result = apply_patch(
            ApplyPatchInput(patch=patch, working_dir=str(tmp_path))
        )
        assert result.ok is True
        content = config_file.read_text()
        assert "DEBUG = True" in content
        assert "PORT = 8080" in content


# =========================================================================
# Delete File
# =========================================================================


class TestDeleteFile:
    def test_delete_file(self, tmp_path, sample_py):
        patch = "*** Delete File: sample.py\n"
        result = apply_patch(
            ApplyPatchInput(patch=patch, working_dir=str(tmp_path))
        )
        assert result.ok is True
        assert result.files_deleted == 1
        assert not sample_py.exists()

    def test_delete_nonexistent(self, tmp_path):
        patch = "*** Delete File: nonexistent.py\n"
        result = apply_patch(
            ApplyPatchInput(patch=patch, working_dir=str(tmp_path))
        )
        assert result.ok is False
        assert "not found" in result.error


# =========================================================================
# Multi-file patches
# =========================================================================


class TestMultiFile:
    def test_add_update_delete(self, tmp_path, sample_py):
        """Combined operations across multiple files in one patch."""
        patch = (
            "*** Add File: new_module.py\n"
            "+def greet():\n"
            "+    return 'hello'\n"
            "\n"
            "*** Update File: sample.py\n"
            "@@ def main():\n"
            "-    x = 1\n"
            "+    x = 99\n"
        )
        result = apply_patch(
            ApplyPatchInput(patch=patch, working_dir=str(tmp_path))
        )
        assert result.ok is True
        assert result.files_added == 1
        assert result.files_modified == 1
        assert (tmp_path / "new_module.py").exists()
        assert "x = 99" in sample_py.read_text()

    def test_multi_file_update(self, tmp_path, sample_py, config_file):
        patch = (
            "*** Update File: sample.py\n"
            "@@ def main():\n"
            "-    x = 1\n"
            "+    x = 100\n"
            "\n"
            "*** Update File: config.py\n"
            '@@ HOST = "localhost"\n'
            "-PORT = 8080\n"
            "+PORT = 9090\n"
        )
        result = apply_patch(
            ApplyPatchInput(patch=patch, working_dir=str(tmp_path))
        )
        assert result.ok is True
        assert result.files_modified == 2
        assert "x = 100" in sample_py.read_text()
        assert "PORT = 9090" in config_file.read_text()


# =========================================================================
# Dry run
# =========================================================================


class TestDryRun:
    def test_dry_run_does_not_modify(self, tmp_path, sample_py):
        original = sample_py.read_text()
        patch = (
            "*** Update File: sample.py\n"
            "@@ def main():\n"
            "-    x = 1\n"
            "+    x = 999\n"
        )
        result = apply_patch(
            ApplyPatchInput(patch=patch, working_dir=str(tmp_path), dry_run=True)
        )
        assert result.ok is True
        assert result.files_modified == 1
        # File should NOT be changed
        assert sample_py.read_text() == original

    def test_dry_run_add_does_not_create(self, tmp_path):
        patch = "*** Add File: should_not_exist.py\n+content\n"
        result = apply_patch(
            ApplyPatchInput(patch=patch, working_dir=str(tmp_path), dry_run=True)
        )
        assert result.ok is True
        assert not (tmp_path / "should_not_exist.py").exists()


# =========================================================================
# Atomic rollback
# =========================================================================


class TestAtomicValidation:
    def test_second_op_fails_no_first_applied(self, tmp_path, sample_py):
        """If second operation fails validation, first should not be applied."""
        original = sample_py.read_text()
        patch = (
            "*** Update File: sample.py\n"
            "@@ def main():\n"
            "-    x = 1\n"
            "+    x = 999\n"
            "\n"
            "*** Update File: nonexistent.py\n"
            "@@ context\n"
            "-old\n"
            "+new\n"
        )
        result = apply_patch(
            ApplyPatchInput(patch=patch, working_dir=str(tmp_path))
        )
        assert result.ok is False
        # First file should NOT have been modified
        assert sample_py.read_text() == original


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    def test_empty_patch(self, tmp_path):
        result = apply_patch(
            ApplyPatchInput(patch="", working_dir=str(tmp_path))
        )
        assert result.ok is False
        assert "No operations" in result.error

    def test_no_hunks_in_update(self, tmp_path, sample_py):
        patch = "*** Update File: sample.py\n"
        result = apply_patch(
            ApplyPatchInput(patch=patch, working_dir=str(tmp_path))
        )
        assert result.ok is False
        assert "No hunks" in result.error or "no hunks" in result.error

    def test_results_contain_file_info(self, tmp_path, sample_py):
        patch = (
            "*** Update File: sample.py\n"
            "@@ def main():\n"
            "-    x = 1\n"
            "+    x = 2\n"
        )
        result = apply_patch(
            ApplyPatchInput(patch=patch, working_dir=str(tmp_path))
        )
        assert len(result.results) == 1
        assert result.results[0].file == "sample.py"
        assert result.results[0].operation == "update"
        assert result.results[0].ok is True
