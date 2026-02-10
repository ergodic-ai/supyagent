"""Binary content materialization for CLI output.

Detects base64-encoded binary content in API responses and saves it
to temp files, replacing inline content with file paths that external
tools (like Claude Code's Read tool) can consume visually.
"""

from __future__ import annotations

import base64
import mimetypes
import re
import time
import uuid
from pathlib import Path
from typing import Any

# MIME prefixes that should stay inline as text (not materialized)
TEXT_MIME_PREFIXES = ("text/",)

# Minimum content length to consider for materialization
MIN_CONTENT_LENGTH = 256

# Max age for temp files before auto-cleanup (seconds)
TEMP_FILE_MAX_AGE_SECONDS = 3600

# Default temp directory
DEFAULT_TEMP_DIR = Path.home() / ".supyagent" / "tmp"

# Extension fallbacks for types mimetypes may not know
_EXT_FALLBACKS = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/svg+xml": ".svg",
}


def _is_base64(s: str) -> bool:
    """Check if a string is valid base64."""
    try:
        base64.b64decode(s, validate=True)
        return True
    except Exception:
        return False


def _sanitize_filename(name: str) -> str:
    """Sanitize a filename for safe filesystem use."""
    safe = re.sub(r"[^\w\s\-.]", "_", name).strip()[:100]
    return safe or "file"


def _get_extension(mime_type: str) -> str:
    """Get file extension from MIME type."""
    ext = mimetypes.guess_extension(mime_type)
    if ext:
        return ext
    return _EXT_FALLBACKS.get(mime_type, ".bin")


def _is_binary_node(node: dict) -> bool:
    """Check if a dict represents a binary content payload."""
    content = node.get("content")
    mime_type = node.get("mimeType")
    if not isinstance(content, str) or not isinstance(mime_type, str):
        return False
    if any(mime_type.startswith(p) for p in TEXT_MIME_PREFIXES):
        return False
    if len(content) < MIN_CONTENT_LENGTH:
        return False
    return _is_base64(content)


def _materialize_node(node: dict, temp_dir: Path) -> dict:
    """Save binary content to a temp file and return modified node."""
    temp_dir.mkdir(parents=True, exist_ok=True)

    content_b64 = node["content"]
    mime_type = node["mimeType"]
    raw_bytes = base64.b64decode(content_b64)

    ext = _get_extension(mime_type)
    name = node.get("name", "")
    if name:
        sanitized = _sanitize_filename(name)
        if not sanitized.lower().endswith(ext.lower()):
            sanitized = f"{sanitized}{ext}"
        filename = sanitized
    else:
        filename = f"{uuid.uuid4().hex[:12]}{ext}"

    file_path = temp_dir / filename
    if file_path.exists():
        stem = file_path.stem
        file_path = temp_dir / f"{stem}_{uuid.uuid4().hex[:6]}{ext}"

    file_path.write_bytes(raw_bytes)

    result = {k: v for k, v in node.items() if k != "content"}
    result["filePath"] = str(file_path)
    return result


def cleanup_temp_dir(temp_dir: Path | None = None) -> int:
    """Remove temp files older than TEMP_FILE_MAX_AGE_SECONDS.

    Returns the number of files removed.
    """
    d = temp_dir or DEFAULT_TEMP_DIR
    if not d.exists():
        return 0
    cutoff = time.time() - TEMP_FILE_MAX_AGE_SECONDS
    removed = 0
    for f in d.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)
            removed += 1
    return removed


def materialize_binary_content(
    data: Any,
    temp_dir: Path | None = None,
) -> Any:
    """Recursively walk a JSON-like structure and materialize binary content.

    Detects dicts with ``{"content": "<base64>", "mimeType": "<binary-type>"}``
    and replaces ``content`` with ``filePath`` pointing to a saved temp file.

    Args:
        data: The parsed JSON response (dict, list, or scalar).
        temp_dir: Directory for temp files. Defaults to ``~/.supyagent/tmp/``.

    Returns:
        The same structure with binary content replaced by file paths.
    """
    d = temp_dir or DEFAULT_TEMP_DIR

    if isinstance(data, dict):
        if _is_binary_node(data):
            return _materialize_node(data, d)
        return {k: materialize_binary_content(v, d) for k, v in data.items()}

    if isinstance(data, list):
        return [materialize_binary_content(item, d) for item in data]

    return data
