"""Abstract base for AI SDK stream protocol encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class StreamEncoder(ABC):
    """Encodes agent events into Vercel AI SDK wire format bytes."""

    @abstractmethod
    def message_start(self, message_id: str) -> bytes:
        """Encode a message start event."""
        ...

    @abstractmethod
    def text_delta(self, text: str) -> bytes:
        """Encode a text chunk."""
        ...

    @abstractmethod
    def tool_call(
        self, tool_call_id: str, tool_name: str, args: dict[str, Any]
    ) -> bytes:
        """Encode a tool call event."""
        ...

    @abstractmethod
    def tool_result(self, tool_call_id: str, result: Any) -> bytes:
        """Encode a tool result event."""
        ...

    @abstractmethod
    def step_finish(
        self, finish_reason: str = "stop", usage: dict[str, int] | None = None
    ) -> bytes:
        """Encode a step finish event."""
        ...

    @abstractmethod
    def message_finish(
        self, finish_reason: str = "stop", usage: dict[str, int] | None = None
    ) -> bytes:
        """Encode the final message finish event."""
        ...

    @abstractmethod
    def error(self, message: str) -> bytes:
        """Encode an error."""
        ...

    @abstractmethod
    def content_type(self) -> str:
        """Return the Content-Type header value."""
        ...

    @abstractmethod
    def extra_headers(self) -> dict[str, str]:
        """Return protocol-specific headers."""
        ...
