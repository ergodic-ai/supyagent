"""
LiteLLM wrapper for unified LLM access.
"""

import logging
import time
from typing import Any

import litellm
from litellm import completion
from litellm.exceptions import (
    APIConnectionError,
    RateLimitError,
    ServiceUnavailableError,
)
from litellm.types.utils import ModelResponse

# Suppress LiteLLM debug messages (e.g., "Provider List: ...")
litellm.suppress_debug_info = True

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Wrapper around LiteLLM for consistent LLM access.

    Supports any provider that LiteLLM supports:
    - OpenAI: openai/gpt-4o
    - Anthropic: anthropic/claude-3-5-sonnet-20241022
    - Ollama: ollama/llama3
    - And 100+ more...
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
    ):
        """
        Initialize the LLM client.

        Args:
            model: LiteLLM model identifier (e.g., 'anthropic/claude-3-5-sonnet-20241022')
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            max_retries: Maximum number of retries on transient errors
            retry_delay: Initial delay between retries (seconds)
            retry_backoff: Exponential backoff multiplier
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> ModelResponse:
        """
        Send a chat completion request with automatic retry on transient errors.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions (OpenAI format)
            stream: Whether to stream the response

        Returns:
            LiteLLM ModelResponse
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        last_error: Exception | None = None
        delay = self.retry_delay

        for attempt in range(self.max_retries + 1):
            try:
                return completion(**kwargs)
            except (RateLimitError, ServiceUnavailableError, APIConnectionError) as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt + 1,
                        self.max_retries + 1,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= self.retry_backoff
                else:
                    raise

        raise last_error  # Should not reach here

    def change_model(self, model: str) -> None:
        """Change the model being used."""
        self.model = model
