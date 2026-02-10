"""Tests for supyagent.utils.binary — binary content materialization."""

from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest

from supyagent.utils.binary import (
    _is_binary_node,
    _sanitize_filename,
    cleanup_temp_dir,
    materialize_binary_content,
)

# Minimal valid PDF for testing (padded to exceed MIN_CONTENT_LENGTH of 256 base64 chars)
TINY_PDF = (
    b"%PDF-1.0\n1 0 obj\n<<>>\nendobj\nxref\n0 0\ntrailer\n<<>>\nstartxref\n0\n%%EOF"
    + b"\x00" * 150
)

# Minimal 1x1 PNG for testing (padded to exceed MIN_CONTENT_LENGTH of 256 base64 chars)
TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    + b"\x00" * 150
)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


# -- _is_binary_node ---------------------------------------------------------


class TestIsBinaryNode:
    def test_valid_pdf_node(self):
        node = {"content": _b64(TINY_PDF), "mimeType": "application/pdf"}
        assert _is_binary_node(node) is True

    def test_valid_png_node(self):
        node = {"content": _b64(TINY_PNG), "mimeType": "image/png"}
        assert _is_binary_node(node) is True

    def test_text_mime_rejected(self):
        node = {"content": _b64(b"a,b\nc,d" * 50), "mimeType": "text/csv"}
        assert _is_binary_node(node) is False

    def test_text_plain_rejected(self):
        node = {"content": _b64(b"hello world " * 30), "mimeType": "text/plain"}
        assert _is_binary_node(node) is False

    def test_short_content_rejected(self):
        node = {"content": "abc", "mimeType": "application/pdf"}
        assert _is_binary_node(node) is False

    def test_missing_content_key(self):
        assert _is_binary_node({"mimeType": "application/pdf"}) is False

    def test_missing_mimetype_key(self):
        assert _is_binary_node({"content": _b64(TINY_PDF)}) is False

    def test_non_string_content(self):
        assert _is_binary_node({"content": 123, "mimeType": "application/pdf"}) is False

    def test_invalid_base64_rejected(self):
        node = {"content": "not_valid!!!" * 30, "mimeType": "application/pdf"}
        assert _is_binary_node(node) is False


# -- _sanitize_filename -------------------------------------------------------


class TestSanitizeFilename:
    def test_basic(self):
        assert _sanitize_filename("Report.pdf") == "Report.pdf"

    def test_unsafe_chars(self):
        result = _sanitize_filename("file/with:bad*chars?.pdf")
        assert "/" not in result
        assert ":" not in result
        assert "*" not in result
        assert "?" not in result

    def test_long_name_truncated(self):
        result = _sanitize_filename("a" * 200)
        assert len(result) <= 100

    def test_empty_fallback(self):
        assert _sanitize_filename("") == "file"


# -- materialize_binary_content -----------------------------------------------


class TestMaterializeBinaryContent:
    def test_materializes_pdf(self, tmp_path):
        data = {
            "id": "f1",
            "name": "Report.pdf",
            "mimeType": "application/pdf",
            "size": len(TINY_PDF),
            "content": _b64(TINY_PDF),
        }
        result = materialize_binary_content(data, temp_dir=tmp_path)

        assert "content" not in result
        assert "filePath" in result
        assert result["id"] == "f1"
        assert result["name"] == "Report.pdf"
        fp = Path(result["filePath"])
        assert fp.exists()
        assert fp.read_bytes() == TINY_PDF
        assert fp.suffix == ".pdf"

    def test_materializes_png(self, tmp_path):
        data = {"name": "drawing.png", "mimeType": "image/png", "content": _b64(TINY_PNG)}
        result = materialize_binary_content(data, temp_dir=tmp_path)

        assert "filePath" in result
        fp = Path(result["filePath"])
        assert fp.exists()
        assert fp.read_bytes() == TINY_PNG

    def test_nested_in_ok_data(self, tmp_path):
        response = {
            "ok": True,
            "data": {
                "id": "f2",
                "name": "Slides.pdf",
                "mimeType": "application/pdf",
                "content": _b64(TINY_PDF),
            },
        }
        result = materialize_binary_content(response, temp_dir=tmp_path)

        assert result["ok"] is True
        assert "filePath" in result["data"]
        assert "content" not in result["data"]

    def test_non_binary_passthrough(self, tmp_path):
        data = {"messages": [{"id": "1", "body": "hello"}]}
        result = materialize_binary_content(data, temp_dir=tmp_path)
        assert result == data

    def test_list_with_binary(self, tmp_path):
        items = [
            {"name": "a.pdf", "mimeType": "application/pdf", "content": _b64(TINY_PDF)},
            {"name": "b.png", "mimeType": "image/png", "content": _b64(TINY_PNG)},
        ]
        result = materialize_binary_content(items, temp_dir=tmp_path)

        assert len(result) == 2
        for item in result:
            assert "filePath" in item
            assert "content" not in item

    def test_uuid_name_when_no_name(self, tmp_path):
        data = {"mimeType": "application/pdf", "content": _b64(TINY_PDF)}
        result = materialize_binary_content(data, temp_dir=tmp_path)

        fp = Path(result["filePath"])
        assert fp.suffix == ".pdf"
        assert fp.exists()

    def test_extension_added_if_missing(self, tmp_path):
        data = {"name": "MyDeck", "mimeType": "application/pdf", "content": _b64(TINY_PDF)}
        result = materialize_binary_content(data, temp_dir=tmp_path)

        fp = Path(result["filePath"])
        assert fp.name.startswith("MyDeck")
        assert fp.suffix == ".pdf"

    def test_no_double_extension(self, tmp_path):
        data = {"name": "Report.pdf", "mimeType": "application/pdf", "content": _b64(TINY_PDF)}
        result = materialize_binary_content(data, temp_dir=tmp_path)

        fp = Path(result["filePath"])
        assert not fp.name.endswith(".pdf.pdf")

    def test_collision_handled(self, tmp_path):
        data = {"name": "Report.pdf", "mimeType": "application/pdf", "content": _b64(TINY_PDF)}
        # Create the file first to force a collision
        (tmp_path / "Report.pdf").write_bytes(b"existing")

        result = materialize_binary_content(data, temp_dir=tmp_path)
        fp = Path(result["filePath"])
        assert fp.exists()
        assert fp.name != "Report.pdf"  # should have UUID suffix
        assert fp.read_bytes() == TINY_PDF

    def test_scalar_passthrough(self, tmp_path):
        assert materialize_binary_content("hello", temp_dir=tmp_path) == "hello"
        assert materialize_binary_content(42, temp_dir=tmp_path) == 42
        assert materialize_binary_content(None, temp_dir=tmp_path) is None


# -- cleanup_temp_dir ----------------------------------------------------------


class TestCleanupTempDir:
    def test_removes_old_files(self, tmp_path):
        old_file = tmp_path / "old.pdf"
        old_file.write_bytes(b"old")
        os.utime(old_file, (0, 0))

        removed = cleanup_temp_dir(tmp_path)
        assert removed == 1
        assert not old_file.exists()

    def test_preserves_recent_files(self, tmp_path):
        new_file = tmp_path / "new.pdf"
        new_file.write_bytes(b"new")

        removed = cleanup_temp_dir(tmp_path)
        assert removed == 0
        assert new_file.exists()

    def test_nonexistent_dir(self, tmp_path):
        assert cleanup_temp_dir(tmp_path / "nope") == 0

    def test_mixed_old_and_new(self, tmp_path):
        old = tmp_path / "old.pdf"
        old.write_bytes(b"old")
        os.utime(old, (0, 0))

        new = tmp_path / "new.pdf"
        new.write_bytes(b"new")

        removed = cleanup_temp_dir(tmp_path)
        assert removed == 1
        assert not old.exists()
        assert new.exists()
