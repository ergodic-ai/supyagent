# /// script
# dependencies = ["pydantic"]
# ///
"""
File editing tools.

Surgical file editing using search-and-replace, line insertions, and
multi-edit patches — without requiring the agent to rewrite entire files.
"""

import os
import re
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

# =============================================================================
# Search and Replace in a file
# =============================================================================


class EditReplaceInput(BaseModel):
    """Input for edit_replace function."""

    file: str = Field(description="Path to the file to edit")
    old_text: str = Field(
        description="Exact text to find in the file (must match precisely, including whitespace)"
    )
    new_text: str = Field(description="Text to replace old_text with")
    count: int = Field(
        default=1,
        description="Maximum number of replacements (0 = all occurrences)",
    )


class EditReplaceOutput(BaseModel):
    """Output for edit_replace function."""

    ok: bool
    replacements_made: int = 0
    error: Optional[str] = None


def edit_replace(input: EditReplaceInput) -> EditReplaceOutput:
    """
    Replace exact text in a file. The old_text must match precisely.

    Use this instead of rewriting an entire file — it's safer and faster.
    Always read the file first to find the exact text to replace.

    Examples:
        >>> edit_replace({"file": "main.py", "old_text": "x = 1", "new_text": "x = 2"})
        >>> edit_replace({"file": "config.json", "old_text": '"debug": false', "new_text": '"debug": true'})
    """
    try:
        path = Path(os.path.expanduser(input.file))
        if not path.exists():
            return EditReplaceOutput(ok=False, error=f"File not found: {input.file}")
        if not path.is_file():
            return EditReplaceOutput(ok=False, error=f"Not a file: {input.file}")

        content = path.read_text()

        if input.old_text not in content:
            return EditReplaceOutput(
                ok=False,
                error="old_text not found in file. Make sure it matches exactly (including whitespace and indentation).",
            )

        # Count occurrences
        occurrences = content.count(input.old_text)

        if input.count == 0:
            # Replace all
            new_content = content.replace(input.old_text, input.new_text)
            replacements = occurrences
        else:
            new_content = content.replace(input.old_text, input.new_text, input.count)
            replacements = min(input.count, occurrences)

        path.write_text(new_content)

        return EditReplaceOutput(ok=True, replacements_made=replacements)

    except Exception as e:
        return EditReplaceOutput(ok=False, error=str(e))


# =============================================================================
# Insert lines at a position
# =============================================================================


class InsertLinesInput(BaseModel):
    """Input for insert_lines function."""

    file: str = Field(description="Path to the file to edit")
    line_number: int = Field(
        description="Line number to insert BEFORE (1-based). Use 0 to prepend, -1 to append."
    )
    text: str = Field(description="Text to insert (can be multiple lines)")


class InsertLinesOutput(BaseModel):
    """Output for insert_lines function."""

    ok: bool
    lines_inserted: int = 0
    error: Optional[str] = None


def insert_lines(input: InsertLinesInput) -> InsertLinesOutput:
    """
    Insert text at a specific line number in a file.

    Line numbers are 1-based. Use 0 to prepend at the start, -1 to append at the end.

    Examples:
        >>> insert_lines({"file": "main.py", "line_number": 1, "text": "import os\\nimport sys"})
        >>> insert_lines({"file": "main.py", "line_number": -1, "text": "# End of file"})
    """
    try:
        path = Path(os.path.expanduser(input.file))
        if not path.exists():
            return InsertLinesOutput(ok=False, error=f"File not found: {input.file}")
        if not path.is_file():
            return InsertLinesOutput(ok=False, error=f"Not a file: {input.file}")

        content = path.read_text()
        lines = content.splitlines(keepends=True)

        new_lines = input.text.splitlines(keepends=True)
        # Ensure the last line has a newline
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"

        line_count = len(new_lines)

        if input.line_number == 0:
            # Prepend
            lines = new_lines + lines
        elif input.line_number == -1:
            # Append
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            lines.extend(new_lines)
        else:
            idx = input.line_number - 1
            if idx < 0:
                idx = 0
            if idx > len(lines):
                # Pad with newlines if needed
                while len(lines) < idx:
                    lines.append("\n")
            lines = lines[:idx] + new_lines + lines[idx:]

        path.write_text("".join(lines))

        return InsertLinesOutput(ok=True, lines_inserted=line_count)

    except Exception as e:
        return InsertLinesOutput(ok=False, error=str(e))


# =============================================================================
# Replace lines by range
# =============================================================================


class ReplaceLinesInput(BaseModel):
    """Input for replace_lines function."""

    file: str = Field(description="Path to the file to edit")
    start_line: int = Field(description="First line to replace (1-based, inclusive)")
    end_line: int = Field(description="Last line to replace (1-based, inclusive)")
    new_text: str = Field(description="Replacement text (can be more or fewer lines)")


class ReplaceLinesOutput(BaseModel):
    """Output for replace_lines function."""

    ok: bool
    lines_removed: int = 0
    lines_added: int = 0
    error: Optional[str] = None


def replace_lines(input: ReplaceLinesInput) -> ReplaceLinesOutput:
    """
    Replace a range of lines in a file with new text.

    Line numbers are 1-based and inclusive. Read the file first to identify
    the exact line range to replace.

    Examples:
        >>> replace_lines({"file": "main.py", "start_line": 5, "end_line": 10, "new_text": "# replaced block"})
    """
    try:
        path = Path(os.path.expanduser(input.file))
        if not path.exists():
            return ReplaceLinesOutput(ok=False, error=f"File not found: {input.file}")
        if not path.is_file():
            return ReplaceLinesOutput(ok=False, error=f"Not a file: {input.file}")

        content = path.read_text()
        lines = content.splitlines(keepends=True)

        start = max(1, input.start_line) - 1  # Convert to 0-based
        end = min(len(lines), input.end_line)  # inclusive, but for slice it's exclusive

        if start >= len(lines):
            return ReplaceLinesOutput(
                ok=False,
                error=f"start_line {input.start_line} is beyond file length ({len(lines)} lines)",
            )

        removed = end - start

        new_lines = input.new_text.splitlines(keepends=True)
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"

        lines = lines[:start] + new_lines + lines[end:]
        path.write_text("".join(lines))

        return ReplaceLinesOutput(
            ok=True,
            lines_removed=removed,
            lines_added=len(new_lines),
        )

    except Exception as e:
        return ReplaceLinesOutput(ok=False, error=str(e))


# =============================================================================
# Multi-edit (batch replacements in one call)
# =============================================================================


class SingleEdit(BaseModel):
    """A single search-and-replace edit."""

    old_text: str = Field(description="Exact text to find")
    new_text: str = Field(description="Text to replace with")


class MultiEditInput(BaseModel):
    """Input for multi_edit function."""

    file: str = Field(description="Path to the file to edit")
    edits: List[SingleEdit] = Field(
        description="List of search-and-replace edits to apply in order"
    )


class MultiEditOutput(BaseModel):
    """Output for multi_edit function."""

    ok: bool
    applied: int = 0
    failed: List[str] = []
    error: Optional[str] = None


def multi_edit(input: MultiEditInput) -> MultiEditOutput:
    """
    Apply multiple search-and-replace edits to a single file in one operation.

    Edits are applied sequentially. If an edit fails (old_text not found),
    it is skipped and reported, but remaining edits still proceed.

    Examples:
        >>> multi_edit({"file": "main.py", "edits": [
        ...     {"old_text": "import os", "new_text": "import os\\nimport sys"},
        ...     {"old_text": "x = 1", "new_text": "x = 42"}
        ... ]})
    """
    try:
        path = Path(os.path.expanduser(input.file))
        if not path.exists():
            return MultiEditOutput(ok=False, error=f"File not found: {input.file}")
        if not path.is_file():
            return MultiEditOutput(ok=False, error=f"Not a file: {input.file}")

        content = path.read_text()
        applied = 0
        failed: List[str] = []

        for i, edit in enumerate(input.edits):
            if edit.old_text in content:
                content = content.replace(edit.old_text, edit.new_text, 1)
                applied += 1
            else:
                failed.append(
                    f"Edit {i + 1}: old_text not found: {edit.old_text[:80]}..."
                    if len(edit.old_text) > 80
                    else f"Edit {i + 1}: old_text not found: {edit.old_text}"
                )

        if applied > 0:
            path.write_text(content)

        return MultiEditOutput(
            ok=applied > 0,
            applied=applied,
            failed=failed,
        )

    except Exception as e:
        return MultiEditOutput(ok=False, error=str(e))


# =============================================================================
# Regex replace
# =============================================================================


class RegexReplaceInput(BaseModel):
    """Input for regex_replace function."""

    file: str = Field(description="Path to the file to edit")
    pattern: str = Field(description="Regex pattern to match")
    replacement: str = Field(
        description="Replacement string (can use \\1, \\2 for capture groups)"
    )
    count: int = Field(
        default=0, description="Max replacements (0 = all)"
    )


class RegexReplaceOutput(BaseModel):
    """Output for regex_replace function."""

    ok: bool
    replacements_made: int = 0
    error: Optional[str] = None


def regex_replace(input: RegexReplaceInput) -> RegexReplaceOutput:
    """
    Replace text in a file using a regex pattern.

    Supports capture groups in the replacement string (\\1, \\2, etc.).
    Use this for complex replacements that need pattern matching.

    Examples:
        >>> regex_replace({"file": "main.py", "pattern": "def (\\w+)\\(\\):", "replacement": "def \\1(self):"})
        >>> regex_replace({"file": "config.py", "pattern": "VERSION = \"[^\"]+\"", "replacement": "VERSION = \"2.0.0\""})
    """
    try:
        path = Path(os.path.expanduser(input.file))
        if not path.exists():
            return RegexReplaceOutput(ok=False, error=f"File not found: {input.file}")
        if not path.is_file():
            return RegexReplaceOutput(ok=False, error=f"Not a file: {input.file}")

        try:
            compiled = re.compile(input.pattern)
        except re.error as e:
            return RegexReplaceOutput(ok=False, error=f"Invalid regex: {e}")

        content = path.read_text()

        if input.count == 0:
            new_content, n = compiled.subn(input.replacement, content)
        else:
            new_content, n = compiled.subn(input.replacement, content, count=input.count)

        if n == 0:
            return RegexReplaceOutput(
                ok=False, error="Pattern not found in file"
            )

        path.write_text(new_content)
        return RegexReplaceOutput(ok=True, replacements_made=n)

    except Exception as e:
        return RegexReplaceOutput(ok=False, error=str(e))
