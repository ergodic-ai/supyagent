# /// script
# dependencies = ["pydantic"]
# ///
"""
Multi-file patch tool.

Apply context-anchored diffs across multiple files in a single atomic operation.
Inspired by the V4A diff format — uses context lines to locate changes rather
than fragile line numbers or exact string matching.
"""

import os
import re
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

# =============================================================================
# Models
# =============================================================================


class PatchResult(BaseModel):
    """Result for a single file operation within a patch."""

    file: str
    operation: str  # "add", "update", "delete"
    ok: bool
    error: Optional[str] = None


class ApplyPatchInput(BaseModel):
    """Input for apply_patch function."""

    patch: str = Field(
        description=(
            "Multi-file patch in V4A-style format. Supported operations:\n"
            "*** Add File: path/to/new.py\n"
            "+line1\n+line2\n\n"
            "*** Update File: path/to/existing.py\n"
            "@@ context line that exists in file\n"
            "-old line\n+new line\n\n"
            "*** Delete File: path/to/unwanted.py"
        )
    )
    dry_run: bool = Field(
        default=False,
        description="Validate the patch without applying changes",
    )
    working_dir: str = Field(
        default=".", description="Base directory for resolving relative paths"
    )


class ApplyPatchOutput(BaseModel):
    """Output for apply_patch function."""

    ok: bool
    results: List[PatchResult] = []
    files_modified: int = 0
    files_added: int = 0
    files_deleted: int = 0
    error: Optional[str] = None


# =============================================================================
# Patch parsing
# =============================================================================

_OP_ADD = re.compile(r"^\*\*\*\s+Add File:\s+(.+)$")
_OP_UPDATE = re.compile(r"^\*\*\*\s+Update File:\s+(.+)$")
_OP_DELETE = re.compile(r"^\*\*\*\s+Delete File:\s+(.+)$")
_HUNK_HEADER = re.compile(r"^@@\s+(.+?)(?:\s+@@)?$")


class _Hunk:
    """A single context-anchored hunk within an Update operation."""

    def __init__(self, context: str):
        self.context = context
        self.remove_lines: list[str] = []
        self.add_lines: list[str] = []


class _FileOp:
    """A parsed file operation."""

    def __init__(self, op: str, path: str):
        self.op = op  # "add", "update", "delete"
        self.path = path
        self.add_content: list[str] = []  # For Add operations
        self.hunks: list[_Hunk] = []  # For Update operations


def _parse_patch(patch_text: str) -> list[_FileOp]:
    """Parse a V4A-style patch string into structured operations."""
    ops: list[_FileOp] = []
    current_op: _FileOp | None = None
    current_hunk: _Hunk | None = None
    lines = patch_text.splitlines()

    for line in lines:
        # Check for new file operation headers
        m_add = _OP_ADD.match(line)
        m_update = _OP_UPDATE.match(line)
        m_delete = _OP_DELETE.match(line)

        if m_add:
            if current_hunk and current_op:
                current_op.hunks.append(current_hunk)
            current_hunk = None
            current_op = _FileOp("add", m_add.group(1).strip())
            ops.append(current_op)
            continue

        if m_update:
            if current_hunk and current_op:
                current_op.hunks.append(current_hunk)
            current_hunk = None
            current_op = _FileOp("update", m_update.group(1).strip())
            ops.append(current_op)
            continue

        if m_delete:
            if current_hunk and current_op:
                current_op.hunks.append(current_hunk)
            current_hunk = None
            current_op = _FileOp("delete", m_delete.group(1).strip())
            ops.append(current_op)
            continue

        if current_op is None:
            continue

        # Inside an Add operation — collect content lines
        if current_op.op == "add":
            if line.startswith("+"):
                current_op.add_content.append(line[1:])
            elif line.startswith(" "):
                current_op.add_content.append(line[1:])
            elif line == "":
                # Blank line in add block — could be content separator or empty line
                # Only add if we already have content (avoid leading blanks)
                if current_op.add_content:
                    current_op.add_content.append("")
            continue

        # Inside an Update operation
        if current_op.op == "update":
            m_hunk = _HUNK_HEADER.match(line)
            if m_hunk:
                if current_hunk:
                    current_op.hunks.append(current_hunk)
                current_hunk = _Hunk(m_hunk.group(1).strip())
                continue

            if current_hunk is not None:
                if line.startswith("-"):
                    current_hunk.remove_lines.append(line[1:])
                elif line.startswith("+"):
                    current_hunk.add_lines.append(line[1:])
                elif line.startswith(" "):
                    # Context line — treated as both remove and add (unchanged)
                    current_hunk.remove_lines.append(line[1:])
                    current_hunk.add_lines.append(line[1:])
            continue

    # Flush last hunk
    if current_hunk and current_op and current_op.op == "update":
        current_op.hunks.append(current_hunk)

    return ops


# =============================================================================
# Hunk application
# =============================================================================


def _find_context_and_apply_hunk(
    file_lines: list[str], hunk: _Hunk
) -> list[str] | None:
    """Find the context anchor in file_lines and apply the hunk.

    Returns the modified lines, or None if the context/remove lines don't match.
    """
    context = hunk.context

    # Search for the context line in the file
    candidates: list[int] = []
    for i, fline in enumerate(file_lines):
        if fline.rstrip("\n") == context or fline.strip() == context.strip():
            candidates.append(i)

    if not candidates:
        return None

    for ctx_idx in candidates:
        # The remove lines should appear immediately after the context line
        start = ctx_idx + 1
        end = start + len(hunk.remove_lines)

        if end > len(file_lines):
            continue

        # Verify remove lines match
        match = True
        for j, expected in enumerate(hunk.remove_lines):
            actual = file_lines[start + j].rstrip("\n")
            if actual != expected and actual.strip() != expected.strip():
                match = False
                break

        if match:
            # Apply: replace the remove lines with add lines
            new_lines = file_lines[:start] + [
                ln + "\n" for ln in hunk.add_lines
            ] + file_lines[end:]
            return new_lines

    return None


# =============================================================================
# Main function
# =============================================================================


def apply_patch(input: ApplyPatchInput) -> ApplyPatchOutput:
    """
    Apply a multi-file patch using context-anchored diffs.

    The patch format supports three operations:
    - **Add File**: Create a new file with the given content.
    - **Update File**: Apply context-anchored hunks to an existing file.
    - **Delete File**: Remove an existing file.

    All operations are validated before any changes are written (atomic).
    Use dry_run=True to validate without applying.

    Examples:
        >>> apply_patch({"patch": "*** Add File: hello.py\\n+print('hello')\\n"})
        >>> apply_patch({"patch": "*** Update File: main.py\\n@@ def main():\\n-    pass\\n+    print('hello')\\n"})
        >>> apply_patch({"patch": "*** Delete File: old_module.py"})
    """
    try:
        base = Path(os.path.expanduser(input.working_dir)).resolve()
        ops = _parse_patch(input.patch)

        if not ops:
            return ApplyPatchOutput(ok=False, error="No operations found in patch")

        results: list[PatchResult] = []
        # Staged changes: list of (path, new_content | None) where None = delete
        staged: list[tuple[Path, str | None]] = []

        # ── Phase 1: Validate all operations ──

        for op in ops:
            fpath = (base / op.path).resolve()

            if op.op == "add":
                if fpath.exists():
                    results.append(PatchResult(
                        file=op.path, operation="add", ok=False,
                        error=f"File already exists: {op.path}",
                    ))
                    return ApplyPatchOutput(
                        ok=False, results=results,
                        error=f"Validation failed: file already exists: {op.path}",
                    )
                content = "\n".join(op.add_content)
                if content and not content.endswith("\n"):
                    content += "\n"
                staged.append((fpath, content))
                results.append(PatchResult(file=op.path, operation="add", ok=True))

            elif op.op == "delete":
                if not fpath.exists():
                    results.append(PatchResult(
                        file=op.path, operation="delete", ok=False,
                        error=f"File not found: {op.path}",
                    ))
                    return ApplyPatchOutput(
                        ok=False, results=results,
                        error=f"Validation failed: file not found for delete: {op.path}",
                    )
                staged.append((fpath, None))
                results.append(PatchResult(file=op.path, operation="delete", ok=True))

            elif op.op == "update":
                if not fpath.exists():
                    results.append(PatchResult(
                        file=op.path, operation="update", ok=False,
                        error=f"File not found: {op.path}",
                    ))
                    return ApplyPatchOutput(
                        ok=False, results=results,
                        error=f"Validation failed: file not found for update: {op.path}",
                    )
                if not op.hunks:
                    results.append(PatchResult(
                        file=op.path, operation="update", ok=False,
                        error="No hunks in update operation",
                    ))
                    return ApplyPatchOutput(
                        ok=False, results=results,
                        error=f"Validation failed: no hunks for update: {op.path}",
                    )

                file_lines = fpath.read_text().splitlines(keepends=True)

                for hi, hunk in enumerate(op.hunks):
                    new_lines = _find_context_and_apply_hunk(file_lines, hunk)
                    if new_lines is None:
                        ctx_preview = hunk.context[:60]
                        results.append(PatchResult(
                            file=op.path, operation="update", ok=False,
                            error=(
                                f"Hunk {hi + 1} failed: context '{ctx_preview}' not found "
                                f"or remove lines don't match"
                            ),
                        ))
                        return ApplyPatchOutput(
                            ok=False, results=results,
                            error=(
                                f"Validation failed in {op.path} hunk {hi + 1}: "
                                f"context '{ctx_preview}' not found or lines don't match"
                            ),
                        )
                    file_lines = new_lines

                staged.append((fpath, "".join(file_lines)))
                results.append(PatchResult(file=op.path, operation="update", ok=True))

        # ── Phase 2: Apply all changes (or skip if dry_run) ──

        if input.dry_run:
            added = sum(1 for r in results if r.operation == "add")
            modified = sum(1 for r in results if r.operation == "update")
            deleted = sum(1 for r in results if r.operation == "delete")
            return ApplyPatchOutput(
                ok=True, results=results,
                files_added=added, files_modified=modified, files_deleted=deleted,
            )

        for fpath, content in staged:
            if content is None:
                fpath.unlink()
            else:
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(content)

        added = sum(1 for r in results if r.operation == "add")
        modified = sum(1 for r in results if r.operation == "update")
        deleted = sum(1 for r in results if r.operation == "delete")

        return ApplyPatchOutput(
            ok=True, results=results,
            files_added=added, files_modified=modified, files_deleted=deleted,
        )

    except Exception as e:
        return ApplyPatchOutput(ok=False, error=str(e))
