"""
LiteLLM wrapper for unified LLM access.
"""

from typing import Any

import litellm
from litellm import completion
from litellm.types.utils import ModelResponse

# Suppress LiteLLM debug messages (e.g., "Provider List: ...")
litellm.suppress_debug_info = True


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
    ):
        """
        Initialize the LLM client.

        Args:
            model: LiteLLM model identifier (e.g., 'anthropic/claude-3-5-sonnet-20241022')
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> ModelResponse:
        """
        Send a chat completion request.

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

        return completion(**kwargs)

    def change_model(self, model: str) -> None:
        """Change the model being used."""
        self.model = model
