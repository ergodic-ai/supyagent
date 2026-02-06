"""Unit tests for supyagent.utils.media."""

import base64
from pathlib import Path

import pytest

from supyagent.utils.media import (
    Content,
    content_to_storable,
    content_to_text,
    detect_images_in_tool_result,
    encode_image_to_base64,
    is_multimodal,
    make_image_content_part,
    make_text_content_part,
    resolve_media_refs,
    save_media_file,
    strip_images,
    wrap_with_image,
)

# Minimal 1x1 red PNG (67 bytes)
TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.fixture
def png_file(tmp_path):
    """Create a temporary PNG file."""
    p = tmp_path / "test.png"
    p.write_bytes(TINY_PNG)
    return p


@pytest.fixture
def jpg_file(tmp_path):
    """Create a temporary JPEG file (fake but valid extension)."""
    p = tmp_path / "test.jpg"
    # JPEG magic bytes + minimal data
    p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20 + b"\xff\xd9")
    return p


# ========== is_multimodal ==========


class TestIsMultimodal:
    def test_string(self):
        assert is_multimodal("hello") is False

    def test_none(self):
        assert is_multimodal(None) is False

    def test_list(self):
        assert is_multimodal([{"type": "text", "text": "hi"}]) is True

    def test_empty_list(self):
        assert is_multimodal([]) is True


# ========== encode_image_to_base64 ==========


class TestEncodeImageToBase64:
    def test_valid_png(self, png_file):
        result = encode_image_to_base64(png_file)
        assert result.startswith("data:image/png;base64,")
        # Decode the base64 part and verify it matches
        b64_part = result.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        assert decoded == TINY_PNG

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            encode_image_to_base64("/nonexistent/path.png")

    def test_unsupported_type(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("not an image")
        with pytest.raises(ValueError, match="Unsupported image type"):
            encode_image_to_base64(txt)


# ========== make_image_content_part ==========


class TestMakeImageContentPart:
    def test_format(self, png_file):
        part = make_image_content_part(png_file)
        assert part["type"] == "image_url"
        assert "image_url" in part
        assert part["image_url"]["url"].startswith("data:image/png;base64,")


# ========== make_text_content_part ==========


class TestMakeTextContentPart:
    def test_format(self):
        part = make_text_content_part("hello world")
        assert part == {"type": "text", "text": "hello world"}

    def test_empty(self):
        part = make_text_content_part("")
        assert part == {"type": "text", "text": ""}


# ========== wrap_with_image ==========


class TestWrapWithImage:
    def test_text_and_image(self, png_file):
        result = wrap_with_image("describe this", png_file)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "describe this"
        assert result[1]["type"] == "image_url"

    def test_no_text(self, png_file):
        result = wrap_with_image("", png_file)
        assert len(result) == 1
        assert result[0]["type"] == "image_url"


# ========== detect_images_in_tool_result ==========


class TestDetectImagesInToolResult:
    def test_screenshot_pattern(self, png_file):
        result = {"ok": True, "path": str(png_file), "width": 100, "height": 100}
        images = detect_images_in_tool_result(result)
        assert images == [str(png_file)]

    def test_explicit_images_key(self, png_file):
        result = {"ok": True, "data": "stuff", "_images": [str(png_file)]}
        images = detect_images_in_tool_result(result)
        assert images == [str(png_file)]

    def test_no_images(self):
        result = {"ok": True, "data": "just text"}
        assert detect_images_in_tool_result(result) == []

    def test_nonexistent_path(self):
        result = {"ok": True, "path": "/nonexistent/screenshot.png"}
        assert detect_images_in_tool_result(result) == []

    def test_non_image_extension(self, tmp_path):
        txt = tmp_path / "output.txt"
        txt.write_text("data")
        result = {"ok": True, "path": str(txt)}
        assert detect_images_in_tool_result(result) == []

    def test_failed_result_not_detected(self, png_file):
        result = {"ok": False, "path": str(png_file)}
        assert detect_images_in_tool_result(result) == []

    def test_nested_data_path(self, png_file):
        """Detect image path nested under result['data'] (supypowers format)."""
        result = {
            "ok": True,
            "data": {"ok": True, "path": str(png_file), "width": 1280, "height": 720},
            "tool_name": "browser__screenshot",
        }
        images = detect_images_in_tool_result(result)
        assert images == [str(png_file)]

    def test_no_duplicate_from_both_strategies(self, png_file):
        result = {"ok": True, "path": str(png_file), "_images": [str(png_file)]}
        images = detect_images_in_tool_result(result)
        assert len(images) == 1


# ========== content_to_text ==========


class TestContentToText:
    def test_string_passthrough(self):
        assert content_to_text("hello") == "hello"

    def test_none(self):
        assert content_to_text(None) == ""

    def test_multimodal(self):
        content = [
            {"type": "text", "text": "Look at this:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        result = content_to_text(content)
        assert result == "Look at this: [image]"

    def test_text_only_list(self):
        content = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
        assert content_to_text(content) == "hello world"


# ========== strip_images ==========


class TestStripImages:
    def test_same_as_content_to_text(self):
        content = [
            {"type": "text", "text": "text"},
            {"type": "image_url", "image_url": {"url": "x"}},
        ]
        assert strip_images(content) == content_to_text(content)


# ========== save_media_file ==========


class TestSaveMediaFile:
    def test_saves_file(self, tmp_path):
        b64 = base64.b64encode(TINY_PNG).decode()
        data_url = f"data:image/png;base64,{b64}"
        filename = save_media_file(tmp_path / "media", data_url)
        assert filename.endswith(".png")
        assert (tmp_path / "media" / filename).exists()
        assert (tmp_path / "media" / filename).read_bytes() == TINY_PNG

    def test_deduplication(self, tmp_path):
        b64 = base64.b64encode(TINY_PNG).decode()
        data_url = f"data:image/png;base64,{b64}"
        name1 = save_media_file(tmp_path, data_url)
        name2 = save_media_file(tmp_path, data_url)
        assert name1 == name2

    def test_invalid_data_url(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid data URL"):
            save_media_file(tmp_path, "not-a-data-url")


# ========== content_to_storable + resolve_media_refs ==========


class TestStorageRoundtrip:
    def test_string_passthrough(self, tmp_path):
        assert content_to_storable("hello", tmp_path) == "hello"
        assert resolve_media_refs("hello", tmp_path) == "hello"

    def test_roundtrip(self, png_file, tmp_path):
        media_dir = tmp_path / "media"
        data_url = encode_image_to_base64(png_file)

        # Original multimodal content
        original = [
            {"type": "text", "text": "screenshot result"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]

        # Store: base64 → media:// ref
        stored = content_to_storable(original, media_dir)
        assert isinstance(stored, list)
        assert stored[0] == {"type": "text", "text": "screenshot result"}
        assert stored[1]["image_url"]["url"].startswith("media://")

        # Resolve: media:// ref → base64
        resolved = resolve_media_refs(stored, media_dir)
        assert isinstance(resolved, list)
        assert resolved[0] == original[0]
        assert resolved[1]["image_url"]["url"] == data_url

    def test_resolve_missing_file(self, tmp_path):
        content = [
            {"type": "image_url", "image_url": {"url": "media://missing.png"}},
        ]
        resolved = resolve_media_refs(content, tmp_path)
        assert resolved[0]["type"] == "text"
        assert "image missing" in resolved[0]["text"]

    def test_external_url_preserved(self, tmp_path):
        content = [
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        stored = content_to_storable(content, tmp_path)
        assert stored[0]["image_url"]["url"] == "https://example.com/img.png"
