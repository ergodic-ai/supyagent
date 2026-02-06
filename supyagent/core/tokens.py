"""
Token counting utilities for context management.

Uses tiktoken for accurate token estimation. For non-OpenAI models,
we use cl100k_base as an approximation.
"""

import json
import logging
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)

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
        elif isinstance(value, list) and key == "content":
            # Multimodal content: list of text/image parts
            for part in value:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        tokens += len(encoding.encode(part.get("text", "")))
                    elif part.get("type") == "image_url":
                        # OpenAI low-detail estimate: 85 base + 170 * 4 tiles
                        tokens += 765
                else:
                    tokens += len(encoding.encode(str(part)))
        elif isinstance(value, list):  # tool_calls etc.
            tokens += len(encoding.encode(str(value)))

    return tokens


def count_messages_tokens(messages: list[dict[str, Any]], model: str = "default") -> int:
    """Count total tokens across all messages."""
    total = 3  # Conversation overhead
    for msg in messages:
        total += count_message_tokens(msg, model)
    return total


def count_tools_tokens(tools: list[dict[str, Any]], model: str = "default") -> int:
    """
    Count tokens consumed by tool definitions.

    Tool definitions are sent as a separate parameter but consume context tokens.
    We serialize them to JSON and count -- this is an approximation,
    as the actual token count depends on the provider's serialization.
    """
    if not tools:
        return 0
    tools_text = json.dumps(tools)
    return count_tokens(tools_text, model)


# Context window limits (conservative fallback estimates)
CONTEXT_LIMITS = {
    "gpt-4o": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16384,
    "claude-3-5-sonnet": 200000,
    "claude-3-opus": 200000,
    "claude-3-haiku": 200000,
    "claude-4": 200000,
    "gemini-2": 1048576,
    "gemini-3": 1048576,
    "gemini-1.5": 1048576,
    "gemini-pro": 32768,
    "llama": 128000,
    "mistral": 128000,
    "qwen": 128000,
    "kimi": 128000,  # Moonshot AI
    "deepseek": 64000,
    "default": 128000,  # Reasonable default for modern models
}


def get_context_limit(model: str) -> int:
    """
    Get the context window limit for a model.

    Strategy:
    1. Try LiteLLM's model cost data (most up-to-date)
    2. Fall back to our hardcoded map
    3. Fall back to conservative default
    """
    # Strategy 1: LiteLLM's model info
    try:
        import litellm

        model_info = litellm.get_model_info(model)
        if model_info and "max_input_tokens" in model_info:
            return model_info["max_input_tokens"]
    except Exception:
        pass

    # Strategy 2: Our hardcoded map (prefix matching)
    model_lower = model.lower()
    for prefix, limit in CONTEXT_LIMITS.items():
        if prefix in model_lower:
            return limit

    # Strategy 3: Conservative default
    logger.warning(
        "Unknown model '%s' for context limit. Using default %d. "
        "Consider adding it to CONTEXT_LIMITS or verifying the model name.",
        model,
        CONTEXT_LIMITS["default"],
    )
    return CONTEXT_LIMITS["default"]
