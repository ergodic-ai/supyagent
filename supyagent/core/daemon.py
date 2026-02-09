"""
Daemon runner for poll-based event processing.

Extends BaseAgentEngine for agents that wake on a fixed interval,
check the inbox for unread events, process them via the LLM, and sleep.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from supyagent.core.credentials import CredentialManager
from supyagent.core.engine import BaseAgentEngine, MaxIterationsError
from supyagent.core.tools import MEMORY_TOOLS
from supyagent.models.agent_config import AgentConfig, get_full_system_prompt

if TYPE_CHECKING:
    from supyagent.core.sandbox import SandboxManager

logger = logging.getLogger(__name__)


class DaemonConfigError(RuntimeError):
    """Raised when daemon configuration is invalid or missing requirements."""


@dataclass
class DaemonCycleResult:
    """Result of a single daemon cycle."""

    cycle: int
    events_found: int
    events_processed: int
    elapsed_seconds: float
    response: str
    skipped: bool = False
    total_unread: int = 0
    error: str | None = None


def parse_interval(interval_str: str) -> float:
    """
    Parse an interval string like '30s', '5m', '1h' to seconds.

    Raises:
        ValueError: If the format is invalid.
    """
    match = re.match(r"^(\d+(?:\.\d+)?)\s*(s|m|h)$", interval_str.strip().lower())
    if not match:
        raise ValueError(
            f"Invalid interval format: '{interval_str}'. "
            "Use format like '30s', '5m', '1h'."
        )
    value = float(match.group(1))
    unit = match.group(2)
    multipliers = {"s": 1, "m": 60, "h": 3600}
    return value * multipliers[unit]


class DaemonRunner(BaseAgentEngine):
    """
    Runs agents in daemon mode — poll-based event processing.

    Wakes on a fixed wall-clock interval, checks the inbox for unread events,
    processes them via the LLM, and goes back to sleep. Uses memory for
    cross-cycle context. Does not prompt for credentials.
    """

    def __init__(
        self,
        config: AgentConfig,
        credential_manager: CredentialManager | None = None,
        sandbox_mgr: SandboxManager | None = None,
    ):
        super().__init__(config)
        self.credential_manager = credential_manager or CredentialManager()
        self._run_secrets: dict[str, str] = {}
        self._shutdown_event = threading.Event()
        self._cycle_count: int = 0

        # Parse schedule
        self._interval_seconds = parse_interval(config.schedule.interval)
        self._max_events = config.schedule.max_events_per_cycle
        self._scheduled_prompt = config.schedule.prompt

        # Service client is required for inbox access
        if not self._service_client:
            raise DaemonConfigError(
                "Daemon mode requires service connection. "
                "Run 'supyagent connect' first."
            )

        # Set up delegation
        if config.delegates:
            from supyagent.core.registry import AgentRegistry

            self._setup_delegation(registry=AgentRegistry())

        # Initialize sandbox / workspace validator
        if sandbox_mgr:
            self.sandbox_mgr = sandbox_mgr
        elif config.workspace and config.sandbox.enabled:
            from supyagent.core.sandbox import SandboxManager as _SandboxManager

            session_id = str(uuid.uuid4())
            self.sandbox_mgr = _SandboxManager(
                Path(config.workspace), config.sandbox, session_id
            )
        elif config.workspace:
            from supyagent.core.sandbox import WorkspaceValidator

            self.workspace_validator = WorkspaceValidator(Path(config.workspace))

        # Load tools (must be after sandbox init)
        self.tools = self._load_tools()

        # Initialize memory system (like Agent, unlike ExecutionRunner)
        if config.memory.enabled:
            from supyagent.core.memory import MemoryManager

            self.memory_mgr = MemoryManager(
                agent_name=config.name,
                llm=self.llm,
                extraction_threshold=config.memory.extraction_threshold,
                retrieval_limit=config.memory.retrieval_limit,
            )

    def _load_tools(self) -> list[dict[str, Any]]:
        """Load base tools + memory tools. No request_credential."""
        tools = self._load_base_tools()
        if self.config.memory.enabled:
            tools.extend(MEMORY_TOOLS)
            for t in MEMORY_TOOLS:
                name = t.get("function", {}).get("name", "")
                self._tool_sources[name] = "native"
        return tools

    def _get_secrets(self) -> dict[str, str]:
        """Get secrets for the current run."""
        return self._run_secrets

    def _system_prompt_kwargs(self) -> dict[str, Any]:
        """Add is_daemon flag to system prompt kwargs."""
        kwargs = super()._system_prompt_kwargs()
        kwargs["is_daemon"] = True
        return kwargs

    def _dispatch_tool_call(self, tool_call: Any) -> dict[str, Any]:
        """Reject credential requests, delegate rest to base."""
        if tool_call.function.name == "request_credential":
            return {
                "ok": False,
                "error": (
                    "Credential prompting not available in daemon mode. "
                    "Pre-configure secrets via 'supyagent config set' or --secrets flag."
                ),
            }
        return super()._dispatch_tool_call(tool_call)

    def _build_cycle_prompt(self, events: list[dict[str, Any]]) -> str:
        """Build the user message for a single daemon cycle."""
        parts: list[str] = []

        if events:
            parts.append(f"You have {len(events)} new inbox event(s) to process:\n")
            for i, event in enumerate(events, 1):
                parts.append(f"--- Event {i} ---")
                parts.append(f"ID: {event.get('id', 'unknown')}")
                parts.append(f"Provider: {event.get('provider', 'unknown')}")
                parts.append(f"Type: {event.get('event_type', 'unknown')}")
                parts.append(f"Received: {event.get('created_at', 'unknown')}")
                summary = event.get("summary")
                if summary:
                    parts.append(f"Summary: {summary}")
                data = event.get("data") or event.get("payload")
                if data:
                    parts.append(f"Data: {json.dumps(data, indent=2, default=str)}")
                parts.append("")

            parts.append(
                "Process each event appropriately using your available tools. "
                "After processing an event, archive it. "
                "Summarize what you did for each event."
            )

        if self._scheduled_prompt:
            if events:
                parts.append("\nAdditionally, perform this scheduled task:")
            parts.append(self._scheduled_prompt)

        return "\n".join(parts)

    def run_once(
        self,
        secrets: dict[str, str] | None = None,
    ) -> DaemonCycleResult:
        """
        Run a single daemon cycle.

        Fetches unread inbox events, builds a prompt, runs the LLM-tool loop,
        and extracts memories. Returns the cycle result.
        """
        # Merge secrets
        self._run_secrets = self.credential_manager.get_all_for_tools(self.config.name)
        if secrets:
            self._run_secrets.update(secrets)

        for key, value in self._run_secrets.items():
            os.environ[key] = value

        start = time.monotonic()

        # Fetch unread events
        inbox_result = self._service_client.inbox_list(
            status="unread", limit=self._max_events
        )
        events = inbox_result.get("events", [])
        total_unread = inbox_result.get("total", 0)

        has_work = bool(events) or bool(self._scheduled_prompt)

        if not has_work:
            elapsed = time.monotonic() - start
            return DaemonCycleResult(
                cycle=self._cycle_count,
                events_found=0,
                events_processed=0,
                elapsed_seconds=elapsed,
                response="",
                skipped=True,
                total_unread=total_unread,
            )

        self._cycle_count += 1

        # Build fresh messages for this cycle
        user_content = self._build_cycle_prompt(events)

        # Inject memory context into system prompt
        system_prompt = get_full_system_prompt(
            self.config, **self._system_prompt_kwargs()
        )
        if self.memory_mgr and self.config.memory.enabled:
            memory_context = self.memory_mgr.get_memory_context(
                user_content[:500], limit=self.config.memory.retrieval_limit
            )
            if memory_context:
                system_prompt = system_prompt + "\n\n" + memory_context

        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        max_iterations = self.config.limits.get("max_tool_calls_per_turn", 20)

        error_msg = None
        try:
            response = self._run_loop(max_iterations)
        except MaxIterationsError:
            response = ""
            last = self.messages[-1] if self.messages else {}
            if last.get("role") == "assistant":
                response = last.get("content", "") or ""
            error_msg = "Max tool iterations exceeded"
        except Exception as e:
            response = ""
            error_msg = str(e)

        # Memory extraction after cycle
        if self.memory_mgr and self.config.memory.auto_extract:
            recent = [m for m in self.messages if m.get("role") in ("user", "assistant")][-4:]
            if self.memory_mgr.has_memory_signal(recent):
                self.memory_mgr.mark_pending(recent)
            self.memory_mgr.flush_pending(
                f"daemon-cycle-{self._cycle_count}", force=True
            )

        elapsed = time.monotonic() - start
        return DaemonCycleResult(
            cycle=self._cycle_count,
            events_found=len(events),
            events_processed=len(events),
            elapsed_seconds=elapsed,
            response=response,
            skipped=False,
            total_unread=total_unread,
            error=error_msg,
        )

    def run(
        self,
        secrets: dict[str, str] | None = None,
        on_cycle: Callable[[DaemonCycleResult], None] | None = None,
    ) -> None:
        """
        Run the daemon loop. Blocks until shutdown() is called.

        Args:
            secrets: Pre-provided credentials
            on_cycle: Callback after each cycle with results
        """
        logger.info(
            "Daemon starting: agent=%s interval=%ss max_events=%d",
            self.config.name,
            self._interval_seconds,
            self._max_events,
        )

        while not self._shutdown_event.is_set():
            cycle_start = time.monotonic()

            try:
                result = self.run_once(secrets=secrets)
                if on_cycle:
                    on_cycle(result)
            except Exception as e:
                logger.error("Daemon cycle error: %s", e)

            # Wall-clock sleep: interval minus elapsed time
            elapsed = time.monotonic() - cycle_start
            sleep_time = max(0.0, self._interval_seconds - elapsed)

            if elapsed > self._interval_seconds:
                logger.warning(
                    "Cycle took %.1fs (interval is %.1fs) — starting next cycle immediately",
                    elapsed,
                    self._interval_seconds,
                )

            if sleep_time > 0:
                self._shutdown_event.wait(timeout=sleep_time)

        logger.info(
            "Daemon shutting down: %d cycles completed", self._cycle_count
        )
        self._cleanup()

    def shutdown(self) -> None:
        """Signal the daemon to stop after current cycle."""
        self._shutdown_event.set()

    def _cleanup(self) -> None:
        """Clean up resources on shutdown."""
        self._kill_managed_processes()
        if self.memory_mgr:
            self.memory_mgr.flush_pending(
                f"daemon-shutdown-{self._cycle_count}", force=True
            )
        if self.sandbox_mgr:
            self.sandbox_mgr.stop()
