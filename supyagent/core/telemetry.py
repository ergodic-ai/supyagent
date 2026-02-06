"""
Structured telemetry for agent execution.

Tracks agent turns, tool calls, LLM calls, and errors as structured events.
Events are stored locally as JSONL for inspection and can optionally
be exported to external systems.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Span:
    """A timed span of execution (e.g., a tool call or LLM call)."""

    def __init__(
        self,
        name: str,
        span_type: str,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.name = name
        self.span_type = span_type
        self.parent_id = parent_id
        self.metadata = metadata or {}
        self.start_time = time.monotonic()
        self.start_ts = datetime.now(UTC)
        self.end_time: float | None = None
        self.end_ts: datetime | None = None
        self.status: str = "running"
        self.error: str | None = None
        self.result_metadata: dict[str, Any] = {}

    def finish(
        self,
        status: str = "ok",
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mark this span as complete."""
        self.end_time = time.monotonic()
        self.end_ts = datetime.now(UTC)
        self.status = status
        self.error = error
        if metadata:
            self.result_metadata.update(metadata)

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        end = self.end_time or time.monotonic()
        return (end - self.start_time) * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        d: dict[str, Any] = {
            "name": self.name,
            "type": self.span_type,
            "status": self.status,
            "started_at": self.start_ts.isoformat(),
            "duration_ms": round(self.duration_ms, 1),
        }
        if self.parent_id:
            d["parent_id"] = self.parent_id
        if self.metadata:
            d["metadata"] = self.metadata
        if self.result_metadata:
            d["result"] = self.result_metadata
        if self.error:
            d["error"] = self.error
        if self.end_ts:
            d["ended_at"] = self.end_ts.isoformat()
        return d


class TelemetryCollector:
    """
    Collects and stores telemetry events from agent execution.

    Events are stored as JSONL in .supyagent/telemetry/<agent>/<session>.jsonl.
    """

    def __init__(
        self,
        agent_name: str,
        session_id: str | None = None,
        base_dir: Path | None = None,
        enabled: bool = True,
    ):
        self.agent_name = agent_name
        self.session_id = session_id or "default"
        self.enabled = enabled
        self.spans: list[Span] = []
        self._current_turn: Span | None = None

        if base_dir is None:
            base_dir = Path(".supyagent/telemetry")
        self._base_dir = base_dir

    def _log_path(self) -> Path:
        """Get the path to the telemetry log file."""
        d = self._base_dir / self.agent_name
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{self.session_id}.jsonl"

    def _write_event(self, event: dict[str, Any]) -> None:
        """Append an event to the telemetry log."""
        if not self.enabled:
            return
        try:
            with open(self._log_path(), "a") as f:
                f.write(json.dumps(event) + "\n")
        except OSError:
            pass

    def start_turn(self, turn_number: int) -> Span:
        """Start tracking a new agent turn."""
        span = Span(
            name=f"turn_{turn_number}",
            span_type="turn",
            metadata={"turn": turn_number, "agent": self.agent_name},
        )
        self._current_turn = span
        self.spans.append(span)
        return span

    def end_turn(
        self,
        span: Span,
        status: str = "ok",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """End a turn span and persist it."""
        span.finish(status=status, metadata=metadata)
        self._write_event(span.to_dict())
        self._current_turn = None

    def track_llm_call(
        self,
        model: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        duration_ms: float | None = None,
        stream: bool = False,
    ) -> None:
        """Record an LLM API call."""
        event: dict[str, Any] = {
            "type": "llm_call",
            "model": model,
            "agent": self.agent_name,
            "stream": stream,
            "ts": datetime.now(UTC).isoformat(),
        }
        if input_tokens is not None:
            event["input_tokens"] = input_tokens
        if output_tokens is not None:
            event["output_tokens"] = output_tokens
        if duration_ms is not None:
            event["duration_ms"] = round(duration_ms, 1)
        self._write_event(event)

    def track_tool_call(
        self,
        tool_name: str,
        duration_ms: float,
        ok: bool,
        error: str | None = None,
        is_service: bool = False,
    ) -> None:
        """Record a tool execution."""
        event: dict[str, Any] = {
            "type": "tool_call",
            "tool": tool_name,
            "agent": self.agent_name,
            "ok": ok,
            "duration_ms": round(duration_ms, 1),
            "is_service": is_service,
            "ts": datetime.now(UTC).isoformat(),
        }
        if error:
            event["error"] = error
        self._write_event(event)

    def track_error(
        self,
        error_type: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record an error event."""
        event: dict[str, Any] = {
            "type": "error",
            "error_type": error_type,
            "message": message,
            "agent": self.agent_name,
            "ts": datetime.now(UTC).isoformat(),
        }
        if context:
            event["context"] = context
        self._write_event(event)

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of collected telemetry for this session."""
        log_path = self._log_path()
        if not log_path.exists():
            return {"turns": 0, "tool_calls": 0, "llm_calls": 0, "errors": 0}

        turns = 0
        tool_calls = 0
        llm_calls = 0
        errors = 0
        total_llm_tokens = 0
        total_tool_time = 0.0

        try:
            with open(log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    event = json.loads(line)
                    etype = event.get("type")
                    if etype == "turn":
                        turns += 1
                    elif etype == "tool_call":
                        tool_calls += 1
                        total_tool_time += event.get("duration_ms", 0)
                    elif etype == "llm_call":
                        llm_calls += 1
                        total_llm_tokens += event.get("input_tokens", 0) + event.get("output_tokens", 0)
                    elif etype == "error":
                        errors += 1
        except (json.JSONDecodeError, OSError):
            pass

        return {
            "turns": turns,
            "tool_calls": tool_calls,
            "llm_calls": llm_calls,
            "errors": errors,
            "total_tokens": total_llm_tokens,
            "total_tool_time_ms": round(total_tool_time, 1),
        }
