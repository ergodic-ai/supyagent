# /// script
# dependencies = ["pydantic"]
# ///
"""
Git operations tools.

Allows agents to interact with Git repositories — check status, view diffs,
commit changes, browse history, and manage branches.
"""

import os
import subprocess
from typing import List, Optional

from pydantic import BaseModel, Field

# =============================================================================
# Git Status
# =============================================================================


class GitStatusInput(BaseModel):
    """Input for git_status function."""

    working_dir: str = Field(default=".", description="Path to the Git repository")


class FileStatus(BaseModel):
    """Status of a single file."""

    file: str
    status: str


class GitStatusOutput(BaseModel):
    """Output for git_status function."""

    ok: bool
    branch: Optional[str] = None
    staged: List[FileStatus] = []
    unstaged: List[FileStatus] = []
    untracked: List[str] = []
    error: Optional[str] = None


def git_status(input: GitStatusInput) -> GitStatusOutput:
    """
    Get the current status of a Git repository.

    Returns staged, unstaged, and untracked files along with the current branch.

    Examples:
        >>> git_status({"working_dir": "."})
        >>> git_status({"working_dir": "/path/to/repo"})
    """
    try:
        working_dir = os.path.expanduser(input.working_dir)

        # Get current branch
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        branch = branch_result.stdout.strip() if branch_result.returncode == 0 else None

        # Get porcelain status
        status_result = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if status_result.returncode != 0:
            return GitStatusOutput(
                ok=False,
                error=status_result.stderr.strip() or "git status failed",
            )

        staged = []
        unstaged = []
        untracked = []

        for line in status_result.stdout.splitlines():
            if len(line) < 4:
                continue

            x = line[0]  # staging area status
            y = line[1]  # working tree status
            filepath = line[3:]

            # Handle renames: "R  old -> new"
            if " -> " in filepath:
                filepath = filepath.split(" -> ")[-1]

            # Untracked files
            if x == "?" and y == "?":
                untracked.append(filepath)
                continue

            # Staged changes (index column)
            status_map = {"M": "modified", "A": "added", "D": "deleted", "R": "renamed"}
            if x in status_map:
                staged.append(FileStatus(file=filepath, status=status_map[x]))

            # Unstaged changes (working tree column)
            if y in status_map:
                unstaged.append(FileStatus(file=filepath, status=status_map[y]))

        return GitStatusOutput(
            ok=True,
            branch=branch,
            staged=staged,
            unstaged=unstaged,
            untracked=untracked,
        )

    except subprocess.TimeoutExpired:
        return GitStatusOutput(ok=False, error="git status timed out")
    except Exception as e:
        return GitStatusOutput(ok=False, error=str(e))


# =============================================================================
# Git Diff
# =============================================================================


class GitDiffInput(BaseModel):
    """Input for git_diff function."""

    working_dir: str = Field(default=".", description="Path to the Git repository")
    staged: bool = Field(default=False, description="Show staged (cached) changes instead of unstaged")
    file: Optional[str] = Field(default=None, description="Limit diff to a specific file")
    ref: Optional[str] = Field(
        default=None, description="Diff against a specific ref (branch, tag, commit)"
    )


class DiffFileStat(BaseModel):
    """Statistics for a single file in a diff."""

    path: str
    additions: int
    deletions: int


class GitDiffOutput(BaseModel):
    """Output for git_diff function."""

    ok: bool
    diff_text: str = ""
    files: List[DiffFileStat] = []
    error: Optional[str] = None


def git_diff(input: GitDiffInput) -> GitDiffOutput:
    """
    Show changes in a Git repository.

    Can show unstaged changes, staged changes, or diff against a ref.

    Examples:
        >>> git_diff({"working_dir": "."})
        >>> git_diff({"staged": True})
        >>> git_diff({"ref": "main"})
        >>> git_diff({"file": "src/main.py"})
    """
    try:
        working_dir = os.path.expanduser(input.working_dir)

        # Build diff command for raw output
        cmd = ["git", "diff"]
        if input.staged:
            cmd.append("--cached")
        if input.ref:
            cmd.append(input.ref)
        if input.file:
            cmd.extend(["--", input.file])

        diff_result = subprocess.run(
            cmd,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if diff_result.returncode != 0:
            return GitDiffOutput(
                ok=False,
                error=diff_result.stderr.strip() or "git diff failed",
            )

        diff_text = diff_result.stdout

        # Build stat command
        stat_cmd = ["git", "diff", "--numstat"]
        if input.staged:
            stat_cmd.append("--cached")
        if input.ref:
            stat_cmd.append(input.ref)
        if input.file:
            stat_cmd.extend(["--", input.file])

        stat_result = subprocess.run(
            stat_cmd,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )

        files = []
        if stat_result.returncode == 0:
            for line in stat_result.stdout.splitlines():
                parts = line.split("\t")
                if len(parts) == 3:
                    additions = int(parts[0]) if parts[0] != "-" else 0
                    deletions = int(parts[1]) if parts[1] != "-" else 0
                    files.append(
                        DiffFileStat(path=parts[2], additions=additions, deletions=deletions)
                    )

        return GitDiffOutput(ok=True, diff_text=diff_text, files=files)

    except subprocess.TimeoutExpired:
        return GitDiffOutput(ok=False, error="git diff timed out")
    except Exception as e:
        return GitDiffOutput(ok=False, error=str(e))


# =============================================================================
# Git Commit
# =============================================================================


class GitCommitInput(BaseModel):
    """Input for git_commit function."""

    message: str = Field(description="Commit message")
    files: Optional[List[str]] = Field(
        default=None, description="Specific files to stage and commit"
    )
    all: bool = Field(default=False, description="Stage all changes before committing (git add -A)")
    working_dir: str = Field(default=".", description="Path to the Git repository")


class GitCommitOutput(BaseModel):
    """Output for git_commit function."""

    ok: bool
    hash: Optional[str] = None
    message: Optional[str] = None
    files_changed: int = 0
    error: Optional[str] = None


def git_commit(input: GitCommitInput) -> GitCommitOutput:
    """
    Create a Git commit.

    Optionally stage specific files or all changes before committing.

    Examples:
        >>> git_commit({"message": "Fix bug in parser", "files": ["src/parser.py"]})
        >>> git_commit({"message": "Update all configs", "all": True})
    """
    try:
        working_dir = os.path.expanduser(input.working_dir)

        # Stage files if requested
        if input.files:
            add_result = subprocess.run(
                ["git", "add"] + input.files,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if add_result.returncode != 0:
                return GitCommitOutput(
                    ok=False,
                    error=add_result.stderr.strip() or "git add failed",
                )

        if input.all:
            add_result = subprocess.run(
                ["git", "add", "-A"],
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if add_result.returncode != 0:
                return GitCommitOutput(
                    ok=False,
                    error=add_result.stderr.strip() or "git add -A failed",
                )

        # Commit
        commit_result = subprocess.run(
            ["git", "commit", "-m", input.message],
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if commit_result.returncode != 0:
            return GitCommitOutput(
                ok=False,
                error=commit_result.stderr.strip() or commit_result.stdout.strip() or "git commit failed",
            )

        # Get the commit hash
        hash_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        commit_hash = hash_result.stdout.strip() if hash_result.returncode == 0 else None

        # Count files changed from commit output
        files_changed = 0
        stat_result = subprocess.run(
            ["git", "diff", "--stat", "HEAD~1", "HEAD"],
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if stat_result.returncode == 0:
            # Last line of --stat is summary like " 3 files changed, 10 insertions(+)"
            stat_lines = stat_result.stdout.strip().splitlines()
            if stat_lines:
                summary = stat_lines[-1]
                for part in summary.split(","):
                    part = part.strip()
                    if "file" in part and "changed" in part:
                        try:
                            files_changed = int(part.split()[0])
                        except (ValueError, IndexError):
                            pass

        return GitCommitOutput(
            ok=True,
            hash=commit_hash,
            message=input.message,
            files_changed=files_changed,
        )

    except subprocess.TimeoutExpired:
        return GitCommitOutput(ok=False, error="git commit timed out")
    except Exception as e:
        return GitCommitOutput(ok=False, error=str(e))


# =============================================================================
# Git Log
# =============================================================================


class GitLogInput(BaseModel):
    """Input for git_log function."""

    n: int = Field(default=10, description="Number of commits to show")
    ref: Optional[str] = Field(default=None, description="Branch, tag, or commit ref to start from")
    working_dir: str = Field(default=".", description="Path to the Git repository")


class CommitInfo(BaseModel):
    """Information about a single commit."""

    hash: str
    author: str
    date: str
    message: str


class GitLogOutput(BaseModel):
    """Output for git_log function."""

    ok: bool
    commits: List[CommitInfo] = []
    error: Optional[str] = None


def git_log(input: GitLogInput) -> GitLogOutput:
    """
    Show the commit history of a Git repository.

    Examples:
        >>> git_log({"n": 5})
        >>> git_log({"n": 20, "ref": "main"})
        >>> git_log({"working_dir": "/path/to/repo"})
    """
    try:
        working_dir = os.path.expanduser(input.working_dir)

        cmd = ["git", "log", f"-n{input.n}", "--format=%H%n%an%n%ai%n%s"]
        if input.ref:
            cmd.append(input.ref)

        result = subprocess.run(
            cmd,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return GitLogOutput(
                ok=False,
                error=result.stderr.strip() or "git log failed",
            )

        commits = []
        lines = result.stdout.strip().splitlines()

        # Each commit is 4 lines: hash, author, date, message
        for i in range(0, len(lines), 4):
            if i + 3 < len(lines):
                commits.append(
                    CommitInfo(
                        hash=lines[i],
                        author=lines[i + 1],
                        date=lines[i + 2],
                        message=lines[i + 3],
                    )
                )

        return GitLogOutput(ok=True, commits=commits)

    except subprocess.TimeoutExpired:
        return GitLogOutput(ok=False, error="git log timed out")
    except Exception as e:
        return GitLogOutput(ok=False, error=str(e))


# =============================================================================
# Git Branch
# =============================================================================


class GitBranchInput(BaseModel):
    """Input for git_branch function."""

    action: str = Field(
        default="list",
        description="Action to perform: list, create, switch, or delete",
    )
    name: Optional[str] = Field(default=None, description="Branch name (for create/switch/delete)")
    working_dir: str = Field(default=".", description="Path to the Git repository")


class BranchInfo(BaseModel):
    """Information about a single branch."""

    name: str
    is_current: bool = False
    is_remote: bool = False


class GitBranchOutput(BaseModel):
    """Output for git_branch function."""

    ok: bool
    branches: List[BranchInfo] = []
    current: Optional[str] = None
    error: Optional[str] = None


def git_branch(input: GitBranchInput) -> GitBranchOutput:
    """
    Manage Git branches — list, create, switch, or delete.

    Examples:
        >>> git_branch({"action": "list"})
        >>> git_branch({"action": "create", "name": "feature/new-thing"})
        >>> git_branch({"action": "switch", "name": "main"})
        >>> git_branch({"action": "delete", "name": "old-branch"})
    """
    try:
        working_dir = os.path.expanduser(input.working_dir)

        if input.action == "list":
            result = subprocess.run(
                ["git", "branch", "-a"],
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return GitBranchOutput(
                    ok=False,
                    error=result.stderr.strip() or "git branch failed",
                )

            branches = []
            current = None

            for line in result.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue

                is_current = line.startswith("* ")
                branch_name = line.lstrip("* ").strip()

                # Skip HEAD pointer lines like "remotes/origin/HEAD -> origin/main"
                if " -> " in branch_name:
                    continue

                is_remote = branch_name.startswith("remotes/")

                if is_current:
                    current = branch_name

                branches.append(
                    BranchInfo(
                        name=branch_name,
                        is_current=is_current,
                        is_remote=is_remote,
                    )
                )

            return GitBranchOutput(ok=True, branches=branches, current=current)

        elif input.action == "create":
            if not input.name:
                return GitBranchOutput(ok=False, error="Branch name is required for create")

            result = subprocess.run(
                ["git", "checkout", "-b", input.name],
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return GitBranchOutput(
                    ok=False,
                    error=result.stderr.strip() or "git checkout -b failed",
                )

            return GitBranchOutput(ok=True, current=input.name)

        elif input.action == "switch":
            if not input.name:
                return GitBranchOutput(ok=False, error="Branch name is required for switch")

            result = subprocess.run(
                ["git", "checkout", input.name],
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return GitBranchOutput(
                    ok=False,
                    error=result.stderr.strip() or "git checkout failed",
                )

            return GitBranchOutput(ok=True, current=input.name)

        elif input.action == "delete":
            if not input.name:
                return GitBranchOutput(ok=False, error="Branch name is required for delete")

            result = subprocess.run(
                ["git", "branch", "-d", input.name],
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return GitBranchOutput(
                    ok=False,
                    error=result.stderr.strip() or "git branch -d failed",
                )

            return GitBranchOutput(ok=True)

        else:
            return GitBranchOutput(
                ok=False,
                error=f"Unknown action: {input.action}. Use list, create, switch, or delete.",
            )

    except subprocess.TimeoutExpired:
        return GitBranchOutput(ok=False, error="git branch operation timed out")
    except Exception as e:
        return GitBranchOutput(ok=False, error=str(e))
