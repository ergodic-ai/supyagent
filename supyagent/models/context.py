"""
Context summary model for persisting conversation summaries.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> datetime:
    """Get current UTC time in a timezone-aware manner."""
    return datetime.now(timezone.utc)


@dataclass
class ContextSummary:
    """
    A compressed summary of older conversation history.

    This is stored alongside the session and used by the ContextManager
    to provide context to the LLM without exceeding token limits.
    """

    content: str  # The summary text
    messages_summarized: int  # Number of messages this summarizes
    first_message_idx: int  # Index of first summarized message
    last_message_idx: int  # Index of last summarized message
    created_at: datetime = field(default_factory=_utc_now)
    token_count: int = 0  # Tokens in the summary

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "content": self.content,
            "messages_summarized": self.messages_summarized,
            "first_message_idx": self.first_message_idx,
            "last_message_idx": self.last_message_idx,
            "created_at": self.created_at.isoformat(),
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextSummary":
        """Create from dictionary (JSON deserialization)."""
        return cls(
            content=data["content"],
            messages_summarized=data["messages_summarized"],
            first_message_idx=data["first_message_idx"],
            last_message_idx=data["last_message_idx"],
            created_at=datetime.fromisoformat(data["created_at"]),
            token_count=data.get("token_count", 0),
        )

    def to_message(self) -> dict[str, str]:
        """Convert to a system message for injection into the conversation."""
        return {
            "role": "system",
            "content": (
                f"[Context Summary - {self.messages_summarized} previous messages]\n\n"
                f"{self.content}"
            ),
        }
