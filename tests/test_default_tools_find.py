"""
Comprehensive tests for default_tools/find.py.

Covers: find_files, recent_files, directory_tree
"""


import pytest

from supyagent.default_tools.find import (
    FindFilesInput,
    RecentFilesInput,
    TreeInput,
    directory_tree,
    find_files,
    recent_files,
)


@pytest.fixture
def project(tmp_path):
    """Create a realistic project structure."""
    # Source files
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("# main\n")
    (src / "utils.py").write_text("# utils\n")
    (src / "config.yaml").write_text("key: value\n")

    # Nested
    sub = src / "sub"
    sub.mkdir()
    (sub / "deep.py").write_text("# deep\n")
    (sub / "data.json").write_text("{}\n")

    # Tests
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_main.py").write_text("# test\n")
    (tests / "test_utils.py").write_text("# test\n")

    # Root files
    (tmp_path / "README.md").write_text("# Readme\n")
    (tmp_path / "setup.py").write_text("# setup\n")
    (tmp_path / "Dockerfile").write_text("FROM python:3.12\n")

    # Hidden
    hidden = tmp_path / ".config"
    hidden.mkdir()
    (hidden / "settings.json").write_text("{}\n")

    # Directories that should be skipped
    cache = src / "__pycache__"
    cache.mkdir()
    (cache / "main.cpython-312.pyc").write_bytes(b"\x00")

    nm = tmp_path / "node_modules"
    nm.mkdir()
    (nm / "package.json").write_text("{}\n")

    return tmp_path


# =========================================================================
# find_files
# =========================================================================


class TestFindFiles:
    def test_find_all_python_files(self, project):
        result = find_files(FindFilesInput(pattern="*.py", path=str(project)))
        assert result.ok is True
        assert result.total_found >= 5  # main, utils, deep, test_main, test_utils, setup
        names = {e.name for e in result.entries}
        assert "main.py" in names
        assert "deep.py" in names

    def test_find_by_prefix(self, project):
        result = find_files(FindFilesInput(pattern="test_*", path=str(project)))
        assert result.ok is True
        for e in result.entries:
            assert e.name.startswith("test_")

    def test_find_yaml_files(self, project):
        result = find_files(FindFilesInput(pattern="*.yaml", path=str(project)))
        assert result.ok is True
        assert result.total_found >= 1
        assert any(e.name == "config.yaml" for e in result.entries)

    def test_find_files_only(self, project):
        result = find_files(
            FindFilesInput(pattern="*", path=str(project), file_type="file")
        )
        assert result.ok is True
        for e in result.entries:
            assert e.type == "file"

    def test_find_dirs_only(self, project):
        result = find_files(
            FindFilesInput(pattern="*", path=str(project), file_type="dir")
        )
        assert result.ok is True
        for e in result.entries:
            assert e.type == "dir"

    def test_find_skips_pycache(self, project):
        result = find_files(FindFilesInput(pattern="*.pyc", path=str(project)))
        assert result.ok is True
        assert result.total_found == 0  # __pycache__ is skipped

    def test_find_skips_node_modules(self, project):
        result = find_files(FindFilesInput(pattern="package.json", path=str(project)))
        assert result.ok is True
        assert result.total_found == 0

    def test_find_skips_hidden_by_default(self, project):
        result = find_files(FindFilesInput(pattern="settings.json", path=str(project)))
        assert result.ok is True
        assert result.total_found == 0

    def test_find_includes_hidden_when_requested(self, project):
        result = find_files(
            FindFilesInput(
                pattern="settings.json", path=str(project), include_hidden=True
            )
        )
        assert result.ok is True
        assert result.total_found >= 1

    def test_find_with_max_results(self, project):
        result = find_files(
            FindFilesInput(pattern="*.py", path=str(project), max_results=2)
        )
        assert result.ok is True
        assert result.total_found <= 2
        assert result.truncated is True

    def test_find_nonexistent_path(self, project):
        result = find_files(FindFilesInput(pattern="*", path=str(project / "nope")))
        assert result.ok is False
        assert "not found" in result.error.lower()

    def test_find_file_not_dir(self, project):
        result = find_files(
            FindFilesInput(pattern="*", path=str(project / "README.md"))
        )
        assert result.ok is False
        assert "Not a directory" in result.error

    def test_find_no_results(self, project):
        result = find_files(FindFilesInput(pattern="*.xyz", path=str(project)))
        assert result.ok is True
        assert result.total_found == 0
        assert result.truncated is False

    def test_find_entries_have_metadata(self, project):
        result = find_files(FindFilesInput(pattern="*.py", path=str(project)))
        assert result.ok is True
        for e in result.entries:
            assert e.name is not None
            assert e.path is not None
            assert e.type in ("file", "dir")
            if e.type == "file":
                assert e.size is not None and e.size >= 0
                assert e.extension == ".py"

    def test_find_dockerfile(self, project):
        result = find_files(FindFilesInput(pattern="Dockerfile*", path=str(project)))
        assert result.ok is True
        assert result.total_found >= 1

    def test_find_in_subdirectory(self, project):
        result = find_files(
            FindFilesInput(pattern="*.py", path=str(project / "src"))
        )
        assert result.ok is True
        names = {e.name for e in result.entries}
        assert "main.py" in names
        assert "deep.py" in names
        assert "test_main.py" not in names  # not in src/


# =========================================================================
# recent_files
# =========================================================================


class TestRecentFiles:
    def test_find_recent_files(self, project):
        result = recent_files(
            RecentFilesInput(path=str(project), minutes=60)
        )
        assert result.ok is True
        # All files were just created, so they should all be recent
        assert result.total_found > 0

    def test_recent_with_glob(self, project):
        result = recent_files(
            RecentFilesInput(path=str(project), minutes=60, glob="*.py")
        )
        assert result.ok is True
        for e in result.entries:
            assert e.name.endswith(".py")

    def test_recent_sorted_by_time(self, project):
        result = recent_files(
            RecentFilesInput(path=str(project), minutes=60)
        )
        assert result.ok is True
        if len(result.entries) >= 2:
            times = [e.modified_seconds_ago for e in result.entries]
            assert times == sorted(times)  # ascending = most recent first

    def test_recent_with_max_results(self, project):
        result = recent_files(
            RecentFilesInput(path=str(project), minutes=60, max_results=3)
        )
        assert result.ok is True
        assert result.total_found <= 3

    def test_recent_nonexistent_path(self, project):
        result = recent_files(RecentFilesInput(path=str(project / "nope")))
        assert result.ok is False

    def test_recent_entries_have_metadata(self, project):
        result = recent_files(
            RecentFilesInput(path=str(project), minutes=60)
        )
        assert result.ok is True
        for e in result.entries:
            assert e.name is not None
            assert e.path is not None
            assert e.size >= 0
            assert e.modified_seconds_ago >= 0

    def test_recent_very_short_window(self, project):
        """With a tiny window (0 minutes), nothing should match."""
        result = recent_files(
            RecentFilesInput(path=str(project), minutes=0)
        )
        assert result.ok is True
        assert result.total_found == 0

    def test_recent_skips_hidden_dirs(self, project):
        result = recent_files(
            RecentFilesInput(path=str(project), minutes=60)
        )
        assert result.ok is True
        for e in result.entries:
            assert not e.path.startswith(".")

    def test_recent_skips_pycache(self, project):
        result = recent_files(
            RecentFilesInput(path=str(project), minutes=60)
        )
        assert result.ok is True
        for e in result.entries:
            assert "__pycache__" not in e.path


# =========================================================================
# directory_tree
# =========================================================================


class TestDirectoryTree:
    def test_basic_tree(self, project):
        result = directory_tree(TreeInput(path=str(project)))
        assert result.ok is True
        assert result.tree is not None
        assert result.tree.type == "dir"
        assert result.total_files > 0
        assert result.total_dirs > 0

    def test_tree_root_name(self, project):
        result = directory_tree(TreeInput(path=str(project)))
        assert result.ok is True
        assert result.tree.name == project.name

    def test_tree_has_children(self, project):
        result = directory_tree(TreeInput(path=str(project), max_depth=1))
        assert result.ok is True
        assert result.tree.children is not None
        child_names = {c.name for c in result.tree.children}
        assert "src" in child_names
        assert "tests" in child_names
        assert "README.md" in child_names

    def test_tree_depth_limit(self, project):
        result = directory_tree(TreeInput(path=str(project), max_depth=1))
        assert result.ok is True
        # At depth 1, directories at depth 1 should have no children (not expanded)
        for child in result.tree.children:
            if child.type == "dir":
                # Children at depth 1 shouldn't be expanded further
                assert child.children is None

    def test_tree_deeper(self, project):
        result = directory_tree(TreeInput(path=str(project), max_depth=3))
        assert result.ok is True
        # Should find the deep.py
        found_deep = False

        def walk(node, depth=0):
            nonlocal found_deep
            if node.name == "deep.py":
                found_deep = True
            if node.children:
                for c in node.children:
                    walk(c, depth + 1)

        walk(result.tree)
        assert found_deep

    def test_tree_dirs_only(self, project):
        result = directory_tree(
            TreeInput(path=str(project), include_files=False)
        )
        assert result.ok is True
        assert result.total_files == 0

        def check_no_files(node):
            if node.children:
                for c in node.children:
                    assert c.type == "dir", f"Found file {c.name} when include_files=False"
                    check_no_files(c)

        check_no_files(result.tree)

    def test_tree_skips_pycache(self, project):
        result = directory_tree(TreeInput(path=str(project), max_depth=5))
        assert result.ok is True

        def check_no_pycache(node):
            assert node.name != "__pycache__"
            if node.children:
                for c in node.children:
                    check_no_pycache(c)

        check_no_pycache(result.tree)

    def test_tree_skips_node_modules(self, project):
        result = directory_tree(TreeInput(path=str(project), max_depth=5))
        assert result.ok is True

        def check_no_nm(node):
            assert node.name != "node_modules"
            if node.children:
                for c in node.children:
                    check_no_nm(c)

        check_no_nm(result.tree)

    def test_tree_skips_hidden_by_default(self, project):
        result = directory_tree(TreeInput(path=str(project), max_depth=5))
        assert result.ok is True

        def check_no_hidden(node):
            assert not node.name.startswith(".")
            if node.children:
                for c in node.children:
                    check_no_hidden(c)

        check_no_hidden(result.tree)

    def test_tree_includes_hidden_when_requested(self, project):
        result = directory_tree(
            TreeInput(path=str(project), max_depth=5, include_hidden=True)
        )
        assert result.ok is True
        found_hidden = False

        def walk(node):
            nonlocal found_hidden
            if node.name.startswith("."):
                found_hidden = True
            if node.children:
                for c in node.children:
                    walk(c)

        walk(result.tree)
        assert found_hidden

    def test_tree_nonexistent_path(self, project):
        result = directory_tree(TreeInput(path=str(project / "nope")))
        assert result.ok is False

    def test_tree_file_not_dir(self, project):
        result = directory_tree(TreeInput(path=str(project / "README.md")))
        assert result.ok is False

    def test_tree_max_depth_clamped(self, project):
        """max_depth is clamped to 1-10."""
        result = directory_tree(TreeInput(path=str(project), max_depth=0))
        assert result.ok is True  # Clamped to 1

        result = directory_tree(TreeInput(path=str(project), max_depth=100))
        assert result.ok is True  # Clamped to 10

    def test_tree_file_nodes_have_size(self, project):
        result = directory_tree(TreeInput(path=str(project), max_depth=3))
        assert result.ok is True

        def check_sizes(node):
            if node.type == "file":
                assert node.size is not None
                assert node.size >= 0
            if node.children:
                for c in node.children:
                    check_sizes(c)

        check_sizes(result.tree)

    def test_tree_empty_directory(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = directory_tree(TreeInput(path=str(empty)))
        assert result.ok is True
        assert result.tree is not None
        assert result.tree.children is None  # Empty dir
        assert result.total_files == 0

    def test_tree_subdirectory(self, project):
        result = directory_tree(TreeInput(path=str(project / "src")))
        assert result.ok is True
        assert result.tree.name == "src"
        child_names = {c.name for c in result.tree.children}
        assert "main.py" in child_names
        assert "sub" in child_names
