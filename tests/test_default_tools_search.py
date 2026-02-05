"""
Comprehensive tests for default_tools/search.py.

Covers: search, count_matches
"""

import os
from pathlib import Path

import pytest

from supyagent.default_tools.search import (
    SearchInput,
    CountMatchesInput,
    search,
    count_matches,
)


@pytest.fixture
def project(tmp_path):
    """Create a realistic project structure for searching."""
    # Python files
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text(
        "import os\nimport json\n\ndef main():\n    # TODO: implement\n    pass\n"
    )
    (src / "utils.py").write_text(
        "import os\n\ndef helper():\n    # TODO: refactor\n    return True\n\ndef other():\n    return False\n"
    )
    (src / "config.py").write_text(
        'VERSION = "1.0.0"\nDEBUG = True\nNAME = "myapp"\n'
    )

    # Nested
    sub = src / "sub"
    sub.mkdir()
    (sub / "deep.py").write_text("# deep module\nimport os\n\nclass Deep:\n    pass\n")

    # Test files
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_main.py").write_text(
        "def test_main():\n    assert True\n\ndef test_helper():\n    assert True\n"
    )

    # Config files
    (tmp_path / "config.yaml").write_text("key: value\nnested:\n  inner: 42\n")
    (tmp_path / "README.md").write_text("# My Project\n\nThis is a README.\n")

    # Hidden files (should be skipped by default)
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "secret.txt").write_text("secret stuff\n")

    # __pycache__ (should be skipped)
    cache = src / "__pycache__"
    cache.mkdir()
    (cache / "main.cpython-312.pyc").write_bytes(b"\x00\x00\x00\x00")

    return tmp_path


# =========================================================================
# search
# =========================================================================


class TestSearch:
    def test_literal_search(self, project):
        result = search(SearchInput(pattern="import os", path=str(project)))
        assert result.ok is True
        assert result.total_matches >= 3  # main.py, utils.py, deep.py
        assert result.files_searched > 0
        for m in result.matches:
            assert "import os" in m.line

    def test_search_with_glob_filter(self, project):
        result = search(
            SearchInput(pattern="import os", path=str(project), glob="*.py")
        )
        assert result.ok is True
        assert result.total_matches >= 3
        for m in result.matches:
            assert m.file.endswith(".py")

    def test_search_case_insensitive(self, project):
        result = search(
            SearchInput(pattern="todo", path=str(project), case_sensitive=False)
        )
        assert result.ok is True
        assert result.total_matches >= 2  # "# TODO" in main.py and utils.py

    def test_search_case_sensitive(self, project):
        result = search(
            SearchInput(pattern="todo", path=str(project), case_sensitive=True)
        )
        assert result.ok is True
        # "TODO" won't match lowercase "todo"
        assert result.total_matches == 0

    def test_search_regex(self, project):
        result = search(
            SearchInput(pattern=r"def \w+\(\):", path=str(project), regex=True)
        )
        assert result.ok is True
        assert result.total_matches >= 3  # main, helper, other

    def test_search_with_context(self, project):
        result = search(
            SearchInput(pattern="def main", path=str(project), context_lines=2)
        )
        assert result.ok is True
        assert result.total_matches >= 1
        m = result.matches[0]
        assert len(m.context_before) <= 2
        # There should be context_after lines too
        assert len(m.context_after) <= 2

    def test_search_max_results(self, project):
        result = search(
            SearchInput(pattern="import", path=str(project), max_results=2)
        )
        assert result.ok is True
        assert result.total_matches <= 2
        assert result.truncated is True

    def test_search_no_results(self, project):
        result = search(
            SearchInput(pattern="XYZZY_UNIQUE_STRING", path=str(project))
        )
        assert result.ok is True
        assert result.total_matches == 0
        assert result.truncated is False

    def test_search_nonexistent_path(self, project):
        result = search(SearchInput(pattern="test", path=str(project / "nope")))
        assert result.ok is False
        assert "not found" in result.error.lower()

    def test_search_single_file(self, project):
        result = search(
            SearchInput(pattern="VERSION", path=str(project / "src" / "config.py"))
        )
        assert result.ok is True
        assert result.total_matches == 1
        assert result.files_searched == 1

    def test_search_skips_pycache(self, project):
        """__pycache__ directories should be skipped."""
        result = search(SearchInput(pattern="", path=str(project)))
        for m in result.matches:
            assert "__pycache__" not in m.file

    def test_search_skips_hidden_by_default(self, project):
        result = search(SearchInput(pattern="secret", path=str(project)))
        assert result.ok is True
        assert result.total_matches == 0

    def test_search_includes_hidden_when_requested(self, project):
        result = search(
            SearchInput(pattern="secret", path=str(project), include_hidden=True)
        )
        assert result.ok is True
        assert result.total_matches >= 1

    def test_search_invalid_regex(self, project):
        result = search(
            SearchInput(pattern="[invalid", path=str(project), regex=True)
        )
        assert result.ok is False
        assert "invalid regex" in result.error.lower()

    def test_search_match_has_line_numbers(self, project):
        result = search(
            SearchInput(pattern="def main", path=str(project / "src"))
        )
        assert result.ok is True
        for m in result.matches:
            assert m.line_number >= 1

    def test_search_context_clamped(self, project):
        """Context lines should be clamped to 0-5."""
        result = search(
            SearchInput(pattern="def main", path=str(project), context_lines=10)
        )
        assert result.ok is True
        for m in result.matches:
            assert len(m.context_before) <= 5
            assert len(m.context_after) <= 5

    def test_search_special_regex_chars_literal(self, project):
        """Special regex chars should be escaped in literal mode."""
        # config.py has VERSION = "1.0.0" which contains dots
        result = search(
            SearchInput(pattern="1.0.0", path=str(project))
        )
        assert result.ok is True
        assert result.total_matches >= 1


# =========================================================================
# count_matches
# =========================================================================


class TestCountMatches:
    def test_count_literal(self, project):
        result = count_matches(
            CountMatchesInput(pattern="import os", path=str(project))
        )
        assert result.ok is True
        assert result.total >= 3
        assert result.files_searched > 0
        assert len(result.by_file) > 0

    def test_count_with_glob(self, project):
        result = count_matches(
            CountMatchesInput(pattern="import", path=str(project), glob="*.py")
        )
        assert result.ok is True
        assert result.total >= 3
        for f in result.by_file:
            assert f.file.endswith(".py")

    def test_count_case_insensitive(self, project):
        result = count_matches(
            CountMatchesInput(pattern="todo", path=str(project), case_sensitive=False)
        )
        assert result.ok is True
        assert result.total >= 2

    def test_count_regex(self, project):
        result = count_matches(
            CountMatchesInput(pattern=r"def \w+", path=str(project), regex=True)
        )
        assert result.ok is True
        assert result.total >= 3

    def test_count_no_matches(self, project):
        result = count_matches(
            CountMatchesInput(pattern="XYZZY_UNIQUE", path=str(project))
        )
        assert result.ok is True
        assert result.total == 0
        assert len(result.by_file) == 0

    def test_count_nonexistent_path(self, project):
        result = count_matches(
            CountMatchesInput(pattern="test", path=str(project / "nope"))
        )
        assert result.ok is False

    def test_count_sorted_by_count(self, project):
        result = count_matches(
            CountMatchesInput(pattern="import", path=str(project))
        )
        assert result.ok is True
        if len(result.by_file) >= 2:
            counts = [f.count for f in result.by_file]
            assert counts == sorted(counts, reverse=True)

    def test_count_single_file(self, project):
        result = count_matches(
            CountMatchesInput(
                pattern="import", path=str(project / "src" / "main.py")
            )
        )
        assert result.ok is True
        assert result.total == 2  # import os, import json

    def test_count_invalid_regex(self, project):
        result = count_matches(
            CountMatchesInput(pattern="[bad", path=str(project), regex=True)
        )
        assert result.ok is False
        assert "invalid regex" in result.error.lower()
