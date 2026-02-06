"""Media utilities for multimodal content handling.

Provides functions for encoding images, detecting images in tool results,
converting between in-memory (base64) and stored (media://) representations,
and extracting text from multimodal content.
"""

from __future__ import annotations

import base64
import hashlib
import mimetypes
import re
from pathlib import Path
from typing import Any

# Type alias for message content: plain string or OpenAI multimodal list
Content = str | list[dict[str, Any]]

# Supported image MIME types
SUPPORTED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}

# Image file extensions for auto-detection
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

# Reference pattern for stored media
MEDIA_REF_PATTERN = re.compile(r"^media://([a-zA-Z0-9_.\-]+)$")


def is_multimodal(content: Content | None) -> bool:
    """Check if content is in multimodal list format."""
    return isinstance(content, list)


def encode_image_to_base64(file_path: str | Path) -> str:
    """Read an image file and return a base64 data URL.

    Args:
        file_path: Path to the image file.

    Returns:
        Data URL string: ``data:image/png;base64,...``

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the MIME type is not a supported image type.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {file_path}")
    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
    if mime_type not in SUPPORTED_IMAGE_TYPES:
        raise ValueError(f"Unsupported image type: {mime_type}")
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def make_image_content_part(file_path: str | Path) -> dict[str, Any]:
    """Create an OpenAI-format image_url content part from a file path."""
    data_url = encode_image_to_base64(file_path)
    return {"type": "image_url", "image_url": {"url": data_url}}


def make_text_content_part(text: str) -> dict[str, Any]:
    """Create an OpenAI-format text content part."""
    return {"type": "text", "text": text}


def wrap_with_image(text: str, image_path: str | Path) -> list[dict[str, Any]]:
    """Create multimodal content with text and an image.

    Args:
        text: Text message (can be empty).
        image_path: Path to the image file.

    Returns:
        List of content parts in OpenAI multimodal format.
    """
    parts: list[dict[str, Any]] = []
    if text:
        parts.append(make_text_content_part(text))
    parts.append(make_image_content_part(image_path))
    return parts


def detect_images_in_tool_result(result: dict[str, Any]) -> list[str]:
    """Detect image file paths in a tool result dict.

    Uses two strategies:
    1. Explicit ``_images`` key with a list of file paths.
    2. Auto-detect screenshot-style results: ``ok=True`` with a ``path``
       key ending in a known image extension.

    Returns:
        List of existing image file paths found.
    """
    images: list[str] = []

    # Strategy 1: explicit _images key
    if "_images" in result:
        for img in result["_images"]:
            if isinstance(img, str) and Path(img).exists():
                images.append(img)

    # Strategy 2: auto-detect screenshot pattern (check top level and nested "data")
    def _check_path(container: dict[str, Any]) -> None:
        if container.get("ok") and "path" in container:
            path_val = container["path"]
            if isinstance(path_val, str):
                p = Path(path_val)
                if p.suffix.lower() in IMAGE_EXTENSIONS and p.exists():
                    if path_val not in images:
                        images.append(path_val)

    _check_path(result)
    if isinstance(result.get("data"), dict):
        _check_path(result["data"])

    return images


def content_to_text(content: Content | None) -> str:
    """Extract plain text from content, replacing images with ``[image]``."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                parts.append("[image]")
        elif isinstance(part, str):
            parts.append(part)
    return " ".join(parts)


def strip_images(content: Content) -> str:
    """Replace all image parts with ``[image]`` text. Alias for content_to_text."""
    return content_to_text(content)


def save_media_file(media_dir: Path, data_url: str) -> str:
    """Save a base64 data URL to the media directory.

    Args:
        media_dir: Directory to save the file in (created if needed).
        data_url: Base64 data URL (``data:image/png;base64,...``).

    Returns:
        The filename (not full path), based on content hash for deduplication.

    Raises:
        ValueError: If the data URL format is invalid.
    """
    media_dir.mkdir(parents=True, exist_ok=True)
    match = re.match(r"data:(image/[\w+]+);base64,(.+)", data_url, re.DOTALL)
    if not match:
        raise ValueError("Invalid data URL format")
    mime_type, b64_data = match.group(1), match.group(2)
    raw = base64.b64decode(b64_data)
    ext = mimetypes.guess_extension(mime_type) or ".png"
    content_hash = hashlib.sha256(raw).hexdigest()[:12]
    filename = f"{content_hash}{ext}"
    (media_dir / filename).write_bytes(raw)
    return filename


def content_to_storable(content: Content, media_dir: Path) -> Content:
    """Convert multimodal content for storage.

    Replaces inline base64 data URLs with ``media://filename`` references.
    Plain strings pass through unchanged.
    """
    if isinstance(content, str):
        return content
    stored_parts: list[dict[str, Any]] = []
    for part in content:
        if part.get("type") == "image_url":
            url = part["image_url"]["url"]
            if url.startswith("data:"):
                filename = save_media_file(media_dir, url)
                stored_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"media://{filename}"},
                })
            else:
                stored_parts.append(part)
        else:
            stored_parts.append(part)
    return stored_parts


def resolve_media_refs(content: Content, media_dir: Path) -> Content:
    """Resolve ``media://`` references back to base64 data URLs.

    Plain strings pass through unchanged. Missing files become placeholder text.
    """
    if isinstance(content, str):
        return content
    resolved: list[dict[str, Any]] = []
    for part in content:
        if part.get("type") == "image_url":
            url = part["image_url"]["url"]
            m = MEDIA_REF_PATTERN.match(url)
            if m:
                filename = m.group(1)
                file_path = media_dir / filename
                if file_path.exists():
                    data_url = encode_image_to_base64(file_path)
                    resolved.append({"type": "image_url", "image_url": {"url": data_url}})
                else:
                    resolved.append(make_text_content_part(f"[image missing: {filename}]"))
            else:
                resolved.append(part)
        else:
            resolved.append(part)
    return resolved
