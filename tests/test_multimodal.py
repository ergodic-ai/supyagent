"""Integration tests for multimodal support across the pipeline.

Tests that multimodal content (images + text) flows correctly through:
- Token counting
- Context management (emergency truncate, summary formatting)
- Session management (title extraction, media dir cleanup)
- Engine (image detection in tool results)
- Agent (multimodal input, persistence, reconstruction)
"""

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from supyagent.core.context_manager import ContextManager
from supyagent.core.session_manager import SessionManager
from supyagent.core.tokens import count_message_tokens, count_messages_tokens
from supyagent.models.session import Message, Session, SessionMeta
from supyagent.utils.media import (
    Content,
    content_to_storable,
    encode_image_to_base64,
    make_image_content_part,
    make_text_content_part,
    resolve_media_refs,
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
    p = tmp_path / "screenshot.png"
    p.write_bytes(TINY_PNG)
    return p


@pytest.fixture
def data_url():
    """Get a base64 data URL for the tiny PNG."""
    b64 = base64.b64encode(TINY_PNG).decode()
    return f"data:image/png;base64,{b64}"


@pytest.fixture
def multimodal_content(data_url):
    """Create a multimodal content list with text and image."""
    return [
        {"type": "text", "text": "Here is a screenshot"},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]


# =============================================================================
# Token Counting
# =============================================================================


class TestMultimodalTokenCounting:
    def test_text_only_message(self):
        """Standard text message counting still works."""
        msg = {"role": "user", "content": "Hello world"}
        tokens = count_message_tokens(msg)
        assert tokens > 4  # base overhead + text

    def test_multimodal_message_includes_image_estimate(self, multimodal_content):
        """Multimodal message should include ~765 tokens per image."""
        msg = {"role": "user", "content": multimodal_content}
        tokens = count_message_tokens(msg)
        # Should include: 4 base + text tokens + 765 image estimate
        assert tokens >= 765

    def test_multiple_images(self, data_url):
        """Multiple images should each add ~765 tokens."""
        content = [
            {"type": "text", "text": "Two images"},
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        msg = {"role": "user", "content": content}
        tokens = count_message_tokens(msg)
        assert tokens >= 765 * 2

    def test_text_only_list_no_image_overhead(self):
        """List with only text parts should not have image overhead."""
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"},
        ]
        msg = {"role": "user", "content": content}
        tokens = count_message_tokens(msg)
        assert tokens < 100  # No image tokens, just text + overhead

    def test_tool_calls_list_not_affected(self):
        """tool_calls list should use the existing str() fallback."""
        msg = {
            "role": "assistant",
            "content": "Calling tool",
            "tool_calls": [{"id": "call_1", "function": {"name": "test", "arguments": "{}"}}],
        }
        tokens = count_message_tokens(msg)
        assert tokens > 4  # Should count something

    def test_messages_total_with_multimodal(self, multimodal_content):
        """count_messages_tokens should handle a mix of text and multimodal messages."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": multimodal_content},
            {"role": "assistant", "content": "I see the screenshot."},
        ]
        total = count_messages_tokens(messages)
        assert total >= 765  # At least the image token estimate


# =============================================================================
# Context Manager
# =============================================================================


class TestMultimodalContextManager:
    def test_emergency_truncate_strips_images(self, multimodal_content):
        """Emergency truncate should replace multimodal content with text."""
        cm = ContextManager(model="default", min_recent_messages=2)

        # Need enough messages so the multimodal one falls in the truncatable middle
        # protected_start=2, protected_end=2, so indices 2..N-2 are truncatable
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "summary placeholder"},
            {"role": "tool", "content": multimodal_content},  # index 2 = truncatable middle
            {"role": "assistant", "content": "middle message"},
            {"role": "user", "content": "recent message 1"},
            {"role": "assistant", "content": "recent message 2"},
        ]

        result = cm._emergency_truncate(messages, target_tokens=200)

        # The multimodal tool message should have been converted to text
        found_stripped = False
        for msg in result:
            content = msg.get("content", "")
            if isinstance(content, str) and "[image]" in content:
                found_stripped = True
                break

        # Either the image was stripped to text, or the message was dropped entirely
        has_multimodal = any(isinstance(m.get("content"), list) for m in result)
        assert found_stripped or not has_multimodal, (
            "Multimodal content should be stripped to text during emergency truncate"
        )

    def test_format_messages_for_summary_multimodal(self, multimodal_content):
        """Summary formatter should convert multimodal content to text."""
        cm = ContextManager(model="default")

        messages = [
            {"role": "user", "content": multimodal_content},
            {"role": "assistant", "content": "I see the image."},
        ]

        text = cm._format_messages_for_summary(messages)
        assert "Here is a screenshot" in text
        assert "[image]" in text
        # Should not contain raw base64
        assert "data:image" not in text


# =============================================================================
# Session Manager
# =============================================================================


class TestMultimodalSessionManager:
    def test_title_from_multimodal_message(self, tmp_path, multimodal_content):
        """Title should be extracted from text part of multimodal content."""
        sm = SessionManager(base_dir=tmp_path)
        session = sm.create_session("test_agent", "test/model")

        # Send a multimodal user message
        msg = Message(type="user", content=multimodal_content)
        sm.append_message(session, msg)

        assert session.meta.title is not None
        assert "Here is a screenshot" in session.meta.title
        # Should not contain base64 or image_url
        assert "data:image" not in session.meta.title

    def test_media_dir_cleanup_on_delete(self, tmp_path, png_file):
        """Deleting a session should also delete its media directory."""
        sm = SessionManager(base_dir=tmp_path)
        session = sm.create_session("test_agent", "test/model")
        sid = session.meta.session_id

        # Create a media directory with a file
        media_dir = tmp_path / "test_agent" / f"{sid}_media"
        media_dir.mkdir(parents=True)
        (media_dir / "abc123.png").write_bytes(TINY_PNG)

        assert media_dir.exists()

        # Delete the session
        sm.delete_session("test_agent", sid)

        # Media dir should be gone
        assert not media_dir.exists()

    def test_multimodal_message_persists(self, tmp_path, multimodal_content):
        """Multimodal content should be serializable and loadable."""
        sm = SessionManager(base_dir=tmp_path)
        session = sm.create_session("test_agent", "test/model")

        msg = Message(type="user", content=multimodal_content)
        sm.append_message(session, msg)

        # Reload session
        loaded = sm.load_session("test_agent", session.meta.session_id)
        assert loaded is not None
        assert len(loaded.messages) == 1
        assert isinstance(loaded.messages[0].content, list)
        assert loaded.messages[0].content[0]["type"] == "text"

    def test_string_content_backward_compat(self, tmp_path):
        """Plain string messages should still work as before."""
        sm = SessionManager(base_dir=tmp_path)
        session = sm.create_session("test_agent", "test/model")

        msg = Message(type="user", content="Hello world")
        sm.append_message(session, msg)

        loaded = sm.load_session("test_agent", session.meta.session_id)
        assert loaded is not None
        assert loaded.messages[0].content == "Hello world"


# =============================================================================
# Media Storage Roundtrip
# =============================================================================


class TestMediaStorageRoundtrip:
    def test_store_and_resolve(self, tmp_path, data_url):
        """content_to_storable → resolve_media_refs should roundtrip."""
        media_dir = tmp_path / "media"
        original = [
            {"type": "text", "text": "screenshot result"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]

        # Store
        stored = content_to_storable(original, media_dir)
        assert isinstance(stored, list)
        assert stored[1]["image_url"]["url"].startswith("media://")

        # Resolve
        resolved = resolve_media_refs(stored, media_dir)
        assert resolved[1]["image_url"]["url"] == data_url

    def test_string_passthrough(self, tmp_path):
        """Plain strings pass through both directions unchanged."""
        assert content_to_storable("hello", tmp_path) == "hello"
        assert resolve_media_refs("hello", tmp_path) == "hello"

    def test_missing_file_becomes_placeholder(self, tmp_path):
        """Missing media ref becomes a text placeholder."""
        content = [
            {"type": "image_url", "image_url": {"url": "media://gone.png"}},
        ]
        resolved = resolve_media_refs(content, tmp_path)
        assert resolved[0]["type"] == "text"
        assert "missing" in resolved[0]["text"]

    def test_external_url_preserved(self, tmp_path):
        """https:// URLs are not converted to media refs."""
        content = [
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        stored = content_to_storable(content, tmp_path)
        assert stored[0]["image_url"]["url"] == "https://example.com/img.png"


# =============================================================================
# Engine Image Detection
# =============================================================================


class TestEngineImageDetection:
    """Test that the engine correctly builds multimodal tool messages."""

    def test_tool_result_with_image_path(self, png_file):
        """Tool result containing an image path should produce multimodal content."""
        from supyagent.utils.media import detect_images_in_tool_result

        result = {"ok": True, "path": str(png_file), "width": 100, "height": 100}
        images = detect_images_in_tool_result(result)
        assert len(images) == 1
        assert images[0] == str(png_file)

    def test_tool_result_without_image(self):
        """Tool result without images should produce plain JSON string."""
        from supyagent.utils.media import detect_images_in_tool_result

        result = {"ok": True, "data": "just text"}
        images = detect_images_in_tool_result(result)
        assert images == []

    def test_multimodal_tool_content_structure(self, png_file):
        """When images detected, tool content should be a list with text + image parts."""
        from supyagent.utils.media import detect_images_in_tool_result

        result = {"ok": True, "path": str(png_file)}
        images = detect_images_in_tool_result(result)

        # Simulate what engine.py does
        if images:
            tool_content = [make_text_content_part(json.dumps(result))]
            for img_path in images:
                tool_content.append(make_image_content_part(img_path))
        else:
            tool_content = json.dumps(result)

        assert isinstance(tool_content, list)
        assert len(tool_content) == 2
        assert tool_content[0]["type"] == "text"
        assert tool_content[1]["type"] == "image_url"
        assert tool_content[1]["image_url"]["url"].startswith("data:image/png;base64,")


# =============================================================================
# Message Model
# =============================================================================


class TestMultimodalMessage:
    def test_message_with_multimodal_content(self, multimodal_content):
        """Message model should accept list content."""
        msg = Message(type="user", content=multimodal_content)
        assert isinstance(msg.content, list)
        assert msg.content[0]["type"] == "text"

    def test_message_serialization_roundtrip(self, multimodal_content):
        """Multimodal message should survive JSON serialization."""
        msg = Message(type="user", content=multimodal_content)
        json_str = msg.model_dump_json()
        data = json.loads(json_str)
        restored = Message(**data)
        assert isinstance(restored.content, list)
        assert restored.content[0]["text"] == "Here is a screenshot"

    def test_message_with_none_content(self):
        """Message with None content still works."""
        msg = Message(type="assistant", content=None)
        assert msg.content is None

    def test_message_with_string_content(self):
        """Message with plain string still works."""
        msg = Message(type="user", content="Hello")
        assert msg.content == "Hello"
