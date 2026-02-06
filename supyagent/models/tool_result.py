"""
Structured result model for tool execution.

Replaces raw {"ok": ..., "error": ...} dicts with a typed Pydantic model
that includes execution metadata (duration, error classification, process ID).
"""

import json
import time
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Structured result from any tool execution."""

    ok: bool = Field(..., description="Whether the tool execution succeeded")
    data: Any = Field(None, description="Result data (if ok=True)")
    error: str | None = Field(None, description="Error message (if ok=False)")
    error_type: str | None = Field(None, description="Error classification")

    # Execution metadata
    tool_name: str | None = Field(None, description="Full tool name (script__func)")
    duration_ms: int | None = Field(None, description="Execution time in milliseconds")
    process_id: str | None = Field(
        None, description="Supervisor process ID (if async)"
    )

    def to_llm_content(self) -> str:
        """Serialize for passing back to LLM as tool result content."""
        if self.ok:
            if isinstance(self.data, str):
                return self.data
            return json.dumps(self.data) if self.data is not None else ""
        else:
            return json.dumps({"error": self.error, "error_type": self.error_type})

    def to_dict(self) -> dict[str, Any]:
        """Full dict including metadata."""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ToolResult":
        """Create from a legacy dict format."""
        return cls(
            ok=d.get("ok", False),
            data=d.get("data"),
            error=d.get("error"),
            error_type=d.get("error_type"),
            tool_name=d.get("tool_name"),
            duration_ms=d.get("duration_ms"),
            process_id=d.get("process_id"),
        )

    @classmethod
    def success(cls, data: Any = None, **kwargs: Any) -> "ToolResult":
        """Create a success result."""
        return cls(ok=True, data=data, **kwargs)

    @classmethod
    def fail(
        cls,
        error: str,
        error_type: str | None = None,
        **kwargs: Any,
    ) -> "ToolResult":
        """Create a failure result."""
        return cls(ok=False, error=error, error_type=error_type, **kwargs)


@contextmanager
def timed_execution():
    """Context manager that yields a dict where 'duration_ms' will be set on exit."""
    timing: dict[str, int] = {}
    start = time.monotonic()
    try:
        yield timing
    finally:
        elapsed = time.monotonic() - start
        timing["duration_ms"] = int(elapsed * 1000)
