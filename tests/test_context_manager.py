"""
Tests for context management functionality.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from supyagent.core.context_manager import ContextManager
from supyagent.core.tokens import (
    count_message_tokens,
    count_messages_tokens,
    count_tokens,
    get_context_limit,
    get_encoding,
)
from supyagent.models.agent_config import AgentConfig, ContextSettings
from supyagent.models.context import ContextSummary


# =============================================================================
# Token Counting Tests
# =============================================================================


class TestGetEncoding:
    """Tests for get_encoding function."""

    def test_gpt4_encoding(self):
        encoding = get_encoding("gpt-4")
        assert encoding.name == "cl100k_base"

    def test_claude_encoding(self):
        encoding = get_encoding("claude-3-5-sonnet")
        assert encoding.name == "cl100k_base"

    def test_default_encoding(self):
        encoding = get_encoding("some-unknown-model")
        assert encoding.name == "cl100k_base"


class TestCountTokens:
    """Tests for count_tokens function."""

    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_simple_string(self):
        tokens = count_tokens("Hello, world!")
        assert tokens > 0
        assert tokens < 10  # Should be around 4 tokens

    def test_longer_text(self):
        text = "This is a longer piece of text that should have more tokens."
        tokens = count_tokens(text)
        assert tokens > 10


class TestCountMessageTokens:
    """Tests for count_message_tokens function."""

    def test_simple_message(self):
        message = {"role": "user", "content": "Hello"}
        tokens = count_message_tokens(message)
        assert tokens > 4  # At least base overhead

    def test_message_with_tool_calls(self):
        message = {
            "role": "assistant",
            "content": "Let me help",
            "tool_calls": [{"id": "1", "function": {"name": "test"}}],
        }
        tokens = count_message_tokens(message)
        assert tokens > 10


class TestCountMessagesTokens:
    """Tests for count_messages_tokens function."""

    def test_empty_list(self):
        tokens = count_messages_tokens([])
        assert tokens == 3  # Just conversation overhead

    def test_multiple_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        tokens = count_messages_tokens(messages)
        assert tokens > 15


class TestGetContextLimit:
    """Tests for get_context_limit function."""

    def test_gpt4o_limit(self):
        assert get_context_limit("gpt-4o") == 128000

    def test_claude_limit(self):
        assert get_context_limit("claude-3-5-sonnet") == 200000

    def test_kimi_limit(self):
        assert get_context_limit("openrouter/moonshotai/kimi-k2.5") == 128000

    def test_unknown_model_default(self):
        assert get_context_limit("unknown-model-xyz") == 128000


# =============================================================================
# ContextSummary Tests
# =============================================================================


class TestContextSummary:
    """Tests for ContextSummary dataclass."""

    def test_basic_creation(self):
        summary = ContextSummary(
            content="This is a summary",
            messages_summarized=10,
            first_message_idx=0,
            last_message_idx=9,
        )
        assert summary.content == "This is a summary"
        assert summary.messages_summarized == 10
        assert summary.token_count == 0  # Default

    def test_to_dict(self):
        summary = ContextSummary(
            content="Summary text",
            messages_summarized=5,
            first_message_idx=0,
            last_message_idx=4,
            token_count=100,
        )
        d = summary.to_dict()
        assert d["content"] == "Summary text"
        assert d["messages_summarized"] == 5
        assert d["token_count"] == 100
        assert "created_at" in d

    def test_from_dict(self):
        data = {
            "content": "Restored summary",
            "messages_summarized": 15,
            "first_message_idx": 0,
            "last_message_idx": 14,
            "created_at": "2024-01-15T10:30:00",
            "token_count": 200,
        }
        summary = ContextSummary.from_dict(data)
        assert summary.content == "Restored summary"
        assert summary.messages_summarized == 15
        assert summary.token_count == 200

    def test_to_message(self):
        summary = ContextSummary(
            content="Key points discussed",
            messages_summarized=20,
            first_message_idx=0,
            last_message_idx=19,
        )
        msg = summary.to_message()
        assert msg["role"] == "system"
        assert "20 previous messages" in msg["content"]
        assert "Key points discussed" in msg["content"]


# =============================================================================
# ContextSettings Tests
# =============================================================================


class TestContextSettings:
    """Tests for ContextSettings model."""

    def test_default_values(self):
        settings = ContextSettings()
        assert settings.auto_summarize is True
        assert settings.max_messages_before_summary == 30
        assert settings.max_tokens_before_summary == 128_000
        assert settings.min_recent_messages == 6
        assert settings.response_reserve == 4096

    def test_custom_values(self):
        settings = ContextSettings(
            max_messages_before_summary=15,
            max_tokens_before_summary=64000,
        )
        assert settings.max_messages_before_summary == 15
        assert settings.max_tokens_before_summary == 64000


# =============================================================================
# ContextManager Tests
# =============================================================================


class TestContextManagerInit:
    """Tests for ContextManager initialization."""

    def test_basic_init(self):
        mgr = ContextManager(model="gpt-4")
        assert mgr.model == "gpt-4"
        assert mgr.max_messages_before_summary == 30
        assert mgr.max_tokens_before_summary == 128_000
        assert mgr.summary is None

    def test_custom_thresholds(self):
        mgr = ContextManager(
            model="gpt-4",
            max_messages_before_summary=10,
            max_tokens_before_summary=50000,
        )
        assert mgr.max_messages_before_summary == 10
        assert mgr.max_tokens_before_summary == 50000

    def test_loads_existing_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            summary_data = {
                "content": "Previous summary",
                "messages_summarized": 10,
                "first_message_idx": 0,
                "last_message_idx": 9,
                "created_at": "2024-01-15T10:30:00",
                "token_count": 150,
            }
            summary_path.write_text(json.dumps(summary_data))

            mgr = ContextManager(model="gpt-4", summary_storage_path=summary_path)
            assert mgr.summary is not None
            assert mgr.summary.content == "Previous summary"


class TestContextManagerShouldSummarize:
    """Tests for should_summarize method."""

    def test_not_enough_messages(self):
        mgr = ContextManager(model="gpt-4", min_recent_messages=6)
        messages = [{"role": "user", "content": "Hi"}] * 5
        assert mgr.should_summarize(messages) is False

    def test_triggers_on_message_count(self):
        mgr = ContextManager(
            model="gpt-4",
            max_messages_before_summary=10,
            max_tokens_before_summary=1_000_000,  # High so won't trigger
        )
        messages = [{"role": "user", "content": "Message"}] * 15
        assert mgr.should_summarize(messages) is True

    def test_triggers_on_token_count(self):
        mgr = ContextManager(
            model="gpt-4",
            max_messages_before_summary=1000,  # High so won't trigger
            max_tokens_before_summary=100,  # Low so will trigger
        )
        messages = [{"role": "user", "content": "This is a fairly long message " * 10}] * 15
        assert mgr.should_summarize(messages) is True

    def test_no_trigger_below_thresholds(self):
        mgr = ContextManager(
            model="gpt-4",
            max_messages_before_summary=100,
            max_tokens_before_summary=1_000_000,
        )
        messages = [{"role": "user", "content": "Short"}] * 15
        assert mgr.should_summarize(messages) is False


class TestContextManagerBuildMessages:
    """Tests for build_messages_for_llm method."""

    def test_basic_build(self):
        mgr = ContextManager(model="gpt-4")
        system_prompt = "You are helpful"
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        result = mgr.build_messages_for_llm(system_prompt, messages)

        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful"
        assert len(result) == 3  # system + 2 messages

    def test_includes_summary_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            summary_data = {
                "content": "Previously discussed X",
                "messages_summarized": 5,
                "first_message_idx": 0,
                "last_message_idx": 4,
                "created_at": "2024-01-15T10:30:00",
                "token_count": 50,
            }
            summary_path.write_text(json.dumps(summary_data))

            mgr = ContextManager(model="gpt-4", summary_storage_path=summary_path)
            messages = [{"role": "user", "content": f"Msg {i}"} for i in range(10)]

            result = mgr.build_messages_for_llm("System", messages)

            # Should have system + summary + recent messages
            assert result[0]["role"] == "system"
            assert result[1]["role"] == "system"  # Summary is injected as system
            assert "5 previous messages" in result[1]["content"]

    def test_respects_min_recent_messages(self):
        mgr = ContextManager(
            model="gpt-4",
            min_recent_messages=3,
            response_reserve=100000,  # Large reserve to force truncation
        )
        messages = [{"role": "user", "content": f"Message {i}"} for i in range(10)]

        result = mgr.build_messages_for_llm("System", messages)

        # Should have at least min_recent_messages
        non_system = [m for m in result if m["role"] != "system"]
        assert len(non_system) >= 3


class TestContextManagerGetTriggerStatus:
    """Tests for get_trigger_status method."""

    def test_status_without_summary(self):
        mgr = ContextManager(
            model="gpt-4",
            max_messages_before_summary=30,
            max_tokens_before_summary=128000,
        )
        messages = [{"role": "user", "content": "Hi"}] * 15

        status = mgr.get_trigger_status(messages)

        assert status["messages_since_summary"] == 15
        assert status["messages_threshold"] == 30
        assert status["messages_percent"] == 50.0
        assert status["tokens_threshold"] == 128000
        assert "will_trigger" in status

    def test_status_with_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            summary_data = {
                "content": "Summary",
                "messages_summarized": 10,
                "first_message_idx": 0,
                "last_message_idx": 9,
                "created_at": "2024-01-15T10:30:00",
                "token_count": 50,
            }
            summary_path.write_text(json.dumps(summary_data))

            mgr = ContextManager(
                model="gpt-4",
                summary_storage_path=summary_path,
                max_messages_before_summary=30,
            )
            messages = [{"role": "user", "content": "Hi"}] * 20

            status = mgr.get_trigger_status(messages)

            # Messages since summary = 20 - 9 - 1 = 10
            assert status["messages_since_summary"] == 10


class TestContextManagerGenerateSummary:
    """Tests for generate_summary method."""

    def test_generate_summary_no_llm(self):
        mgr = ContextManager(model="gpt-4")
        messages = [{"role": "user", "content": "Hi"}] * 15

        with pytest.raises(ValueError, match="LLM client required"):
            mgr.generate_summary(messages)

    def test_generate_summary_empty_messages(self):
        mock_llm = MagicMock()
        mgr = ContextManager(model="gpt-4", llm=mock_llm, min_recent_messages=6)

        with pytest.raises(ValueError, match="No messages to summarize"):
            mgr.generate_summary([], up_to_idx=0)

    def test_generate_summary_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"

            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content="Summary: User asked about X."))
            ]
            mock_llm = MagicMock()
            mock_llm.chat.return_value = mock_response

            mgr = ContextManager(
                model="gpt-4",
                llm=mock_llm,
                summary_storage_path=summary_path,
                min_recent_messages=2,
            )
            messages = [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."},
                {"role": "user", "content": "Tell me more"},
                {"role": "assistant", "content": "It's widely used."},
                {"role": "user", "content": "Thanks"},
            ]

            summary = mgr.generate_summary(messages, up_to_idx=2)

            assert summary.content == "Summary: User asked about X."
            assert summary.messages_summarized == 3
            assert summary_path.exists()  # Should be persisted


class TestContextManagerFormatMessages:
    """Tests for _format_messages_for_summary method."""

    def test_formats_user_assistant(self):
        mgr = ContextManager(model="gpt-4")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = mgr._format_messages_for_summary(messages)

        assert "USER: Hello" in result
        assert "ASSISTANT: Hi there!" in result

    def test_truncates_tool_results(self):
        mgr = ContextManager(model="gpt-4")
        long_result = "x" * 600  # Over 500 chars
        messages = [{"role": "tool", "content": long_result}]

        result = mgr._format_messages_for_summary(messages)

        assert "[Tool Result]:" in result
        assert "..." in result  # Should be truncated

    def test_skips_system_messages(self):
        mgr = ContextManager(model="gpt-4")
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]

        result = mgr._format_messages_for_summary(messages)

        assert "SYSTEM" not in result
        assert "USER: Hi" in result


# =============================================================================
# Integration with AgentConfig
# =============================================================================


class TestAgentConfigContextSettings:
    """Tests for context settings in AgentConfig."""

    def test_default_context_settings(self):
        config = AgentConfig(
            name="test",
            model={"provider": "gpt-4"},
            system_prompt="Test prompt",
        )
        assert config.context.auto_summarize is True
        assert config.context.max_messages_before_summary == 30
        assert config.context.max_tokens_before_summary == 128_000

    def test_custom_context_settings(self):
        config = AgentConfig(
            name="test",
            model={"provider": "gpt-4"},
            system_prompt="Test prompt",
            context={
                "max_messages_before_summary": 20,
                "max_tokens_before_summary": 64000,
            },
        )
        assert config.context.max_messages_before_summary == 20
        assert config.context.max_tokens_before_summary == 64000
