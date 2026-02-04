"""
Token counting utilities for context management.

Uses tiktoken for accurate token estimation. For non-OpenAI models,
we use cl100k_base as an approximation.
"""

from typing import Any

import tiktoken

# Model to encoding mapping (approximate for non-OpenAI models)
MODEL_ENCODINGS = {
    "gpt-4": "cl100k_base",
    "gpt-3.5": "cl100k_base",
    "claude": "cl100k_base",  # Approximation
    "default": "cl100k_base",
}


def get_encoding(model: str) -> tiktoken.Encoding:
    """Get the tiktoken encoding for a model."""
    for prefix, encoding in MODEL_ENCODINGS.items():
        if prefix in model.lower():
            return tiktoken.get_encoding(encoding)
    return tiktoken.get_encoding(MODEL_ENCODINGS["default"])


def count_tokens(text: str, model: str = "default") -> int:
    """Count tokens in a text string."""
    if not text:
        return 0
    encoding = get_encoding(model)
    return len(encoding.encode(text))


def count_message_tokens(message: dict[str, Any], model: str = "default") -> int:
    """
    Count tokens in a message dict.

    Accounts for message overhead (role, formatting).
    """
    encoding = get_encoding(model)
    tokens = 4  # Base overhead per message

    for key, value in message.items():
        if isinstance(value, str):
            tokens += len(encoding.encode(value))
        elif isinstance(value, list):  # tool_calls
            tokens += len(encoding.encode(str(value)))

    return tokens


def count_messages_tokens(messages: list[dict[str, Any]], model: str = "default") -> int:
    """Count total tokens across all messages."""
    total = 3  # Conversation overhead
    for msg in messages:
        total += count_message_tokens(msg, model)
    return total


# Context window limits (conservative estimates)
CONTEXT_LIMITS = {
    "gpt-4o": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16384,
    "claude-3-5-sonnet": 200000,
    "claude-3-opus": 200000,
    "claude-3-haiku": 200000,
    "kimi": 128000,  # Moonshot AI
    "deepseek": 64000,
    "default": 128000,  # Reasonable default for modern models
}


def get_context_limit(model: str) -> int:
    """Get the context window limit for a model."""
    model_lower = model.lower()
    for prefix, limit in CONTEXT_LIMITS.items():
        if prefix in model_lower:
            return limit
    return CONTEXT_LIMITS["default"]
