"""
Comprehensive tests for default_tools/git.py.

Covers: git_status, git_diff, git_commit, git_log, git_branch
"""

import subprocess

import pytest

from supyagent.default_tools.git import (
    GitBranchInput,
    GitCommitInput,
    GitDiffInput,
    GitLogInput,
    GitStatusInput,
    git_branch,
    git_commit,
    git_diff,
    git_log,
    git_status,
)


@pytest.fixture
def git_repo(tmp_path):
    """Create a real Git repository with an initial commit."""
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True
    )
    (tmp_path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"], cwd=tmp_path, capture_output=True
    )
    return tmp_path


# =========================================================================
# git_status
# =========================================================================


class TestGitStatus:
    def test_clean_repo(self, git_repo):
        result = git_status(GitStatusInput(working_dir=str(git_repo)))
        assert result.ok is True
        assert result.branch is not None
        assert result.staged == []
        assert result.unstaged == []
        assert result.untracked == []

    def test_modified_file(self, git_repo):
        (git_repo / "README.md").write_text("# Updated\n")
        result = git_status(GitStatusInput(working_dir=str(git_repo)))
        assert result.ok is True
        assert len(result.unstaged) == 1
        assert result.unstaged[0].file == "README.md"
        assert result.unstaged[0].status == "modified"

    def test_untracked_file(self, git_repo):
        (git_repo / "new_file.txt").write_text("hello\n")
        result = git_status(GitStatusInput(working_dir=str(git_repo)))
        assert result.ok is True
        assert "new_file.txt" in result.untracked

    def test_staged_file(self, git_repo):
        (git_repo / "README.md").write_text("# Staged change\n")
        subprocess.run(["git", "add", "README.md"], cwd=git_repo, capture_output=True)
        result = git_status(GitStatusInput(working_dir=str(git_repo)))
        assert result.ok is True
        assert len(result.staged) == 1
        assert result.staged[0].file == "README.md"
        assert result.staged[0].status == "modified"

    def test_added_file(self, git_repo):
        (git_repo / "new.txt").write_text("new\n")
        subprocess.run(["git", "add", "new.txt"], cwd=git_repo, capture_output=True)
        result = git_status(GitStatusInput(working_dir=str(git_repo)))
        assert result.ok is True
        assert len(result.staged) == 1
        assert result.staged[0].status == "added"

    def test_not_a_repo(self, tmp_path):
        result = git_status(GitStatusInput(working_dir=str(tmp_path)))
        assert result.ok is False
        assert result.error is not None


# =========================================================================
# git_diff
# =========================================================================


class TestGitDiff:
    def test_no_changes(self, git_repo):
        result = git_diff(GitDiffInput(working_dir=str(git_repo)))
        assert result.ok is True
        assert result.diff_text == ""
        assert result.files == []

    def test_unstaged_changes(self, git_repo):
        (git_repo / "README.md").write_text("# Changed\nNew line\n")
        result = git_diff(GitDiffInput(working_dir=str(git_repo)))
        assert result.ok is True
        assert "Changed" in result.diff_text
        assert len(result.files) == 1
        assert result.files[0].path == "README.md"
        assert result.files[0].additions > 0

    def test_staged_changes(self, git_repo):
        (git_repo / "README.md").write_text("# Staged\n")
        subprocess.run(["git", "add", "README.md"], cwd=git_repo, capture_output=True)
        result = git_diff(GitDiffInput(working_dir=str(git_repo), staged=True))
        assert result.ok is True
        assert "Staged" in result.diff_text
        assert len(result.files) == 1

    def test_diff_specific_file(self, git_repo):
        (git_repo / "README.md").write_text("# Changed\n")
        (git_repo / "other.txt").write_text("other\n")
        subprocess.run(["git", "add", "other.txt"], cwd=git_repo, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "add other"], cwd=git_repo, capture_output=True
        )
        (git_repo / "other.txt").write_text("changed other\n")
        result = git_diff(GitDiffInput(working_dir=str(git_repo), file="other.txt"))
        assert result.ok is True
        assert "other" in result.diff_text

    def test_diff_with_deletions(self, git_repo):
        (git_repo / "README.md").write_text("")
        result = git_diff(GitDiffInput(working_dir=str(git_repo)))
        assert result.ok is True
        assert len(result.files) == 1
        assert result.files[0].deletions > 0


# =========================================================================
# git_commit
# =========================================================================


class TestGitCommit:
    def test_commit_specific_files(self, git_repo):
        (git_repo / "new.txt").write_text("hello\n")
        result = git_commit(
            GitCommitInput(
                message="Add new file",
                files=["new.txt"],
                working_dir=str(git_repo),
            )
        )
        assert result.ok is True
        assert result.hash is not None
        assert len(result.hash) == 40
        assert result.message == "Add new file"
        assert result.files_changed >= 1

    def test_commit_all(self, git_repo):
        (git_repo / "README.md").write_text("# Updated\n")
        (git_repo / "another.txt").write_text("another\n")
        result = git_commit(
            GitCommitInput(
                message="Update all",
                all=True,
                working_dir=str(git_repo),
            )
        )
        assert result.ok is True
        assert result.hash is not None
        assert result.files_changed >= 1

    def test_commit_nothing_staged(self, git_repo):
        result = git_commit(
            GitCommitInput(
                message="Empty commit attempt",
                working_dir=str(git_repo),
            )
        )
        assert result.ok is False
        assert result.error is not None

    def test_commit_message_preserved(self, git_repo):
        (git_repo / "file.txt").write_text("content\n")
        git_commit(
            GitCommitInput(
                message="Test message here",
                files=["file.txt"],
                working_dir=str(git_repo),
            )
        )
        log_result = subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            cwd=git_repo,
            capture_output=True,
            text=True,
        )
        assert log_result.stdout.strip() == "Test message here"


# =========================================================================
# git_log
# =========================================================================


class TestGitLog:
    def test_log_initial_commit(self, git_repo):
        result = git_log(GitLogInput(working_dir=str(git_repo)))
        assert result.ok is True
        assert len(result.commits) == 1
        assert result.commits[0].message == "Initial commit"
        assert result.commits[0].author == "Test"
        assert len(result.commits[0].hash) == 40

    def test_log_multiple_commits(self, git_repo):
        for i in range(5):
            (git_repo / f"file{i}.txt").write_text(f"content {i}\n")
            subprocess.run(["git", "add", "."], cwd=git_repo, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", f"Commit {i}"],
                cwd=git_repo,
                capture_output=True,
            )

        result = git_log(GitLogInput(n=3, working_dir=str(git_repo)))
        assert result.ok is True
        assert len(result.commits) == 3
        # Most recent first
        assert result.commits[0].message == "Commit 4"

    def test_log_limit(self, git_repo):
        result = git_log(GitLogInput(n=1, working_dir=str(git_repo)))
        assert result.ok is True
        assert len(result.commits) == 1

    def test_log_has_date(self, git_repo):
        result = git_log(GitLogInput(working_dir=str(git_repo)))
        assert result.ok is True
        assert result.commits[0].date != ""

    def test_log_not_a_repo(self, tmp_path):
        result = git_log(GitLogInput(working_dir=str(tmp_path)))
        assert result.ok is False
        assert result.error is not None


# =========================================================================
# git_branch
# =========================================================================


class TestGitBranch:
    def test_list_branches(self, git_repo):
        result = git_branch(GitBranchInput(action="list", working_dir=str(git_repo)))
        assert result.ok is True
        assert len(result.branches) >= 1
        assert result.current is not None

    def test_create_branch(self, git_repo):
        result = git_branch(
            GitBranchInput(action="create", name="feature/test", working_dir=str(git_repo))
        )
        assert result.ok is True
        assert result.current == "feature/test"

        # Verify we are on the new branch
        list_result = git_branch(GitBranchInput(action="list", working_dir=str(git_repo)))
        assert list_result.current == "feature/test"

    def test_switch_branch(self, git_repo):
        # Create a branch first
        git_branch(
            GitBranchInput(action="create", name="other-branch", working_dir=str(git_repo))
        )
        # We're on other-branch now, switch back
        # First, figure out the original default branch name
        log_result = subprocess.run(
            ["git", "branch"], cwd=git_repo, capture_output=True, text=True
        )
        branches = [b.strip().lstrip("* ") for b in log_result.stdout.strip().splitlines()]
        original_branch = [b for b in branches if b != "other-branch"][0]

        result = git_branch(
            GitBranchInput(action="switch", name=original_branch, working_dir=str(git_repo))
        )
        assert result.ok is True
        assert result.current == original_branch

    def test_delete_branch(self, git_repo):
        # Create and switch back
        git_branch(
            GitBranchInput(action="create", name="to-delete", working_dir=str(git_repo))
        )
        # Switch away first (can't delete current branch)
        log_result = subprocess.run(
            ["git", "branch"], cwd=git_repo, capture_output=True, text=True
        )
        branches = [b.strip().lstrip("* ") for b in log_result.stdout.strip().splitlines()]
        other_branch = [b for b in branches if b != "to-delete"][0]
        git_branch(
            GitBranchInput(action="switch", name=other_branch, working_dir=str(git_repo))
        )

        result = git_branch(
            GitBranchInput(action="delete", name="to-delete", working_dir=str(git_repo))
        )
        assert result.ok is True

    def test_create_without_name(self, git_repo):
        result = git_branch(
            GitBranchInput(action="create", working_dir=str(git_repo))
        )
        assert result.ok is False
        assert "name" in result.error.lower()

    def test_switch_nonexistent(self, git_repo):
        result = git_branch(
            GitBranchInput(action="switch", name="does-not-exist", working_dir=str(git_repo))
        )
        assert result.ok is False
        assert result.error is not None

    def test_unknown_action(self, git_repo):
        result = git_branch(
            GitBranchInput(action="merge", working_dir=str(git_repo))
        )
        assert result.ok is False
        assert "Unknown action" in result.error
