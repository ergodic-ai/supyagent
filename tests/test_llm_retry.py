"""
Tests for LLM retry logic.
"""

from unittest.mock import MagicMock, patch

import pytest

from supyagent.core.llm import LLMClient


class TestLLMRetry:
    """Tests for LLMClient retry logic."""

    @patch("supyagent.core.llm.completion")
    def test_success_no_retry(self, mock_completion):
        """Successful call should not retry."""
        mock_response = MagicMock()
        mock_completion.return_value = mock_response

        client = LLMClient("test-model", max_retries=3, retry_delay=0.01)
        result = client.chat([{"role": "user", "content": "test"}])

        assert result == mock_response
        assert mock_completion.call_count == 1

    @patch("supyagent.core.llm.completion")
    def test_retry_on_rate_limit(self, mock_completion):
        """Should retry on RateLimitError."""
        from litellm.exceptions import RateLimitError

        mock_response = MagicMock()
        mock_completion.side_effect = [
            RateLimitError("Rate limited", "model", "provider"),
            RateLimitError("Rate limited", "model", "provider"),
            mock_response,
        ]

        client = LLMClient("test-model", max_retries=3, retry_delay=0.01)
        result = client.chat([{"role": "user", "content": "test"}])

        assert result == mock_response
        assert mock_completion.call_count == 3

    @patch("supyagent.core.llm.completion")
    def test_retry_on_service_unavailable(self, mock_completion):
        """Should retry on ServiceUnavailableError."""
        from litellm.exceptions import ServiceUnavailableError

        mock_response = MagicMock()
        mock_completion.side_effect = [
            ServiceUnavailableError("Unavailable", "model", "provider"),
            mock_response,
        ]

        client = LLMClient("test-model", max_retries=3, retry_delay=0.01)
        result = client.chat([{"role": "user", "content": "test"}])

        assert result == mock_response
        assert mock_completion.call_count == 2

    @patch("supyagent.core.llm.completion")
    def test_retry_on_connection_error(self, mock_completion):
        """Should retry on APIConnectionError."""
        from litellm.exceptions import APIConnectionError

        mock_response = MagicMock()
        mock_completion.side_effect = [
            APIConnectionError("Connection failed", "model", "provider"),
            mock_response,
        ]

        client = LLMClient("test-model", max_retries=3, retry_delay=0.01)
        result = client.chat([{"role": "user", "content": "test"}])

        assert result == mock_response
        assert mock_completion.call_count == 2

    @patch("supyagent.core.llm.completion")
    def test_raises_after_max_retries(self, mock_completion):
        """Should raise after exhausting retries."""
        from litellm.exceptions import RateLimitError

        mock_completion.side_effect = RateLimitError(
            "Rate limited", "model", "provider"
        )

        client = LLMClient("test-model", max_retries=2, retry_delay=0.01)

        with pytest.raises(RateLimitError):
            client.chat([{"role": "user", "content": "test"}])

        # 1 initial + 2 retries = 3 attempts
        assert mock_completion.call_count == 3

    @patch("supyagent.core.llm.completion")
    def test_no_retry_on_non_transient_error(self, mock_completion):
        """Should not retry on non-transient errors like bad request."""
        mock_completion.side_effect = ValueError("Invalid model")

        client = LLMClient("test-model", max_retries=3, retry_delay=0.01)

        with pytest.raises(ValueError, match="Invalid model"):
            client.chat([{"role": "user", "content": "test"}])

        assert mock_completion.call_count == 1

    @patch("supyagent.core.llm.completion")
    def test_zero_retries(self, mock_completion):
        """With max_retries=0, should only try once."""
        from litellm.exceptions import RateLimitError

        mock_completion.side_effect = RateLimitError(
            "Rate limited", "model", "provider"
        )

        client = LLMClient("test-model", max_retries=0, retry_delay=0.01)

        with pytest.raises(RateLimitError):
            client.chat([{"role": "user", "content": "test"}])

        assert mock_completion.call_count == 1

    @patch("supyagent.core.llm.completion")
    def test_passes_tools_and_stream(self, mock_completion):
        """Should pass tools and stream kwargs correctly."""
        mock_response = MagicMock()
        mock_completion.return_value = mock_response

        client = LLMClient("test-model")
        tools = [{"type": "function", "function": {"name": "test"}}]
        client.chat(
            [{"role": "user", "content": "test"}],
            tools=tools,
            stream=True,
        )

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == "auto"
        assert call_kwargs["stream"] is True
