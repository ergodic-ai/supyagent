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
    APIError,
    AuthenticationError,
    BadRequestError,
    BudgetExceededError,
    ContextWindowExceededError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
)
from litellm.types.utils import ModelResponse

# Suppress LiteLLM debug messages (e.g., "Provider List: ...")
litellm.suppress_debug_info = True

logger = logging.getLogger(__name__)


# Map of provider prefixes to their expected API key env vars
_PROVIDER_KEY_HINTS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "google": "GOOGLE_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "azure": "AZURE_API_KEY",
    "cohere": "COHERE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "together": "TOGETHER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def _extract_error_message(error: Exception) -> str:
    """Extract the most useful part of a LiteLLM error message."""
    import re

    msg = str(error)
    # Try to extract JSON message from OpenRouter-style errors
    match = re.search(r'"message"\s*:\s*"([^"]+)"', msg)
    if match:
        return match.group(1)
    # Truncate very long messages
    if len(msg) > 200:
        return msg[:200] + "..."
    return msg


class LLMError(Exception):
    """User-friendly LLM error with actionable guidance."""

    def __init__(self, message: str, original: Exception | None = None):
        self.original = original
        super().__init__(message)


def _friendly_llm_error(model: str, error: Exception) -> LLMError:
    """Convert a LiteLLM exception to a user-friendly error message."""
    provider = model.split("/")[0].lower() if "/" in model else model.split("/")[0].lower()

    if isinstance(error, AuthenticationError):
        key_name = _PROVIDER_KEY_HINTS.get(provider, f"{provider.upper()}_API_KEY")
        return LLMError(
            f"Authentication failed for '{model}'. "
            f"Check that {key_name} is set correctly.\n"
            f"  Run: supyagent config set {key_name}",
            original=error,
        )

    if isinstance(error, NotFoundError):
        return LLMError(
            f"Model '{model}' not found. Check the model name and provider.\n"
            f"  LiteLLM format: provider/model (e.g., openai/gpt-4o, anthropic/claude-3-5-sonnet-20241022)\n"
            f"  See: https://docs.litellm.ai/docs/providers",
            original=error,
        )

    if isinstance(error, RateLimitError):
        return LLMError(
            f"Rate limit exceeded for '{model}'. Wait a moment and try again.\n"
            f"  If this persists, check your API plan limits.",
            original=error,
        )

    if isinstance(error, BudgetExceededError):
        return LLMError(
            f"API budget/credits exhausted for '{model}'. "
            f"Add credits at your provider's dashboard.",
            original=error,
        )

    if isinstance(error, ContextWindowExceededError):
        return LLMError(
            f"Context too large for '{model}'. "
            f"Try reducing conversation history or using a model with a larger context window.",
            original=error,
        )

    if isinstance(error, BadRequestError):
        return LLMError(
            f"Model '{model}' rejected the request. "
            f"This may indicate incompatible tool schemas or invalid parameters.\n"
            f"  Details: {_extract_error_message(error)}",
            original=error,
        )

    if isinstance(error, APIError):
        msg = str(error)
        if any(kw in msg.lower() for kw in ["402", "credits", "insufficient"]):
            return LLMError(
                f"Credits exhausted for '{model}'. Add more at your provider's dashboard.\n"
                f"  {_extract_error_message(error)}",
                original=error,
            )
        return LLMError(
            f"API error from {provider}: {_extract_error_message(error)}",
            original=error,
        )

    if isinstance(error, APIConnectionError):
        return LLMError(
            f"Cannot connect to {provider} API. Check your internet connection.\n"
            f"  If using a custom endpoint, verify the URL is correct.",
            original=error,
        )

    if isinstance(error, ServiceUnavailableError):
        return LLMError(
            f"The {provider} API is temporarily unavailable. Try again in a moment.",
            original=error,
        )

    # Generic fallback
    return LLMError(f"LLM error ({type(error).__name__}): {error}", original=error)


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
        max_tokens: int | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
        fallback_models: list[str] | None = None,
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
            fallback_models: Fallback model identifiers tried when primary exhausts retries
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff
        self.fallback_models = fallback_models or []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> ModelResponse:
        """
        Send a chat completion request with automatic retry and model failover.

        Tries the primary model with retries, then each fallback model in order.
        Non-transient errors (auth, model not found) raise immediately without failover.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions (OpenAI format)
            stream: Whether to stream the response

        Returns:
            LiteLLM ModelResponse
        """
        models = [self.model] + self.fallback_models

        last_error: Exception | None = None
        for i, model in enumerate(models):
            if i > 0:
                logger.warning("Falling back to model: %s", model)

            result, error = self._try_model(model, messages, tools, stream)
            if result is not None:
                return result
            last_error = error

        # All models exhausted
        raise _friendly_llm_error(self.model, last_error)

    def _try_model(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        stream: bool,
    ) -> tuple[ModelResponse | None, Exception | None]:
        """
        Try a single model with retries.

        Returns:
            (response, None) on success, or (None, last_error) on transient failure.
            Raises LLMError immediately on non-transient errors (auth, model not found).
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": stream,
        }

        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        last_error: Exception | None = None
        delay = self.retry_delay

        for attempt in range(self.max_retries + 1):
            try:
                return completion(**kwargs), None
            except (RateLimitError, ServiceUnavailableError) as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(
                        "LLM call failed (attempt %d/%d, model %s): %s. Retrying in %.1fs...",
                        attempt + 1,
                        self.max_retries + 1,
                        model,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= self.retry_backoff
            except APIConnectionError as e:
                # Sub-classify: credit/budget errors masquerading as connection errors
                msg = str(e).lower()
                if any(kw in msg for kw in ["402", "credits", "insufficient", "budget"]):
                    raise _friendly_llm_error(model, e) from e
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(
                        "LLM call failed (attempt %d/%d, model %s): %s. Retrying in %.1fs...",
                        attempt + 1,
                        self.max_retries + 1,
                        model,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= self.retry_backoff
            except APIError as e:
                # Sub-classify: credit/budget errors are non-transient
                msg = str(e).lower()
                if any(kw in msg for kw in ["402", "credits", "insufficient", "budget"]):
                    raise _friendly_llm_error(model, e) from e
                # Other API errors: retry
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(
                        "LLM call failed (attempt %d/%d, model %s): %s. Retrying in %.1fs...",
                        attempt + 1,
                        self.max_retries + 1,
                        model,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= self.retry_backoff
            except (
                AuthenticationError,
                NotFoundError,
                BudgetExceededError,
                BadRequestError,
                ContextWindowExceededError,
            ) as e:
                # Non-transient: raise immediately, no failover
                raise _friendly_llm_error(model, e) from e

        return None, last_error

    def change_model(self, model: str) -> None:
        """Change the model being used."""
        self.model = model
