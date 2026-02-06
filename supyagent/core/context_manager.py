"""
Context Manager for handling conversation context within token limits.

The ContextManager ensures that:
1. All messages are persisted to disk (full history)
2. Messages sent to the LLM stay within context window limits
3. Older messages are summarized to preserve context
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from supyagent.core.tokens import (
    count_messages_tokens,
    count_tokens,
    count_tools_tokens,
    get_context_limit,
)
from supyagent.models.context import ContextSummary

if TYPE_CHECKING:
    from supyagent.core.llm import LLMClient

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manages conversation context to stay within token limits.

    Strategy:
    1. Keep ALL messages in session storage (full persistence)
    2. When building messages for LLM:
       - Always include system prompt
       - Include context summary (if exists)
       - Include recent N messages that fit in budget
    3. Trigger summarization when EITHER:
       - N messages since last summary (max_messages_before_summary)
       - K total tokens exceeded (max_tokens_before_summary)
    """

    def __init__(
        self,
        model: str,
        llm: LLMClient | None = None,
        summary_storage_path: Path | None = None,
        # Per-agent configurable thresholds
        max_messages_before_summary: int = 30,  # N
        max_tokens_before_summary: int = 128_000,  # K
        min_recent_messages: int = 6,
        response_reserve: int = 4096,
    ):
        """
        Initialize the context manager.

        Args:
            model: Model identifier for token counting
            llm: LLM client for generating summaries
            summary_storage_path: Path to store/load summaries
            max_messages_before_summary: Trigger after N messages
            max_tokens_before_summary: Trigger when tokens exceed K
            min_recent_messages: Always include at least this many recent messages
            response_reserve: Reserve this many tokens for the response
        """
        self.model = model
        self.llm = llm
        self.context_limit = get_context_limit(model)
        self.summary_storage_path = summary_storage_path
        self._summary: ContextSummary | None = None

        # Per-agent thresholds
        self.max_messages_before_summary = max_messages_before_summary  # N
        self.max_tokens_before_summary = max_tokens_before_summary  # K
        self.min_recent_messages = min_recent_messages
        self.response_reserve = response_reserve

        # Load existing summary if available
        if summary_storage_path and summary_storage_path.exists():
            self._load_summary()

    def _load_summary(self) -> None:
        """Load summary from disk."""
        if not self.summary_storage_path:
            return
        try:
            with open(self.summary_storage_path) as f:
                data = json.load(f)
                self._summary = ContextSummary.from_dict(data)
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            self._summary = None

    def _save_summary(self) -> None:
        """Save summary to disk."""
        if self._summary and self.summary_storage_path:
            self.summary_storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.summary_storage_path, "w") as f:
                json.dump(self._summary.to_dict(), f, indent=2)

    def build_messages_for_llm(
        self,
        system_prompt: str,
        all_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Build the message list to send to the LLM.

        Args:
            system_prompt: The system prompt
            all_messages: All conversation messages (excluding system)
            tools: Optional tool definitions (for token budget accounting)

        Returns:
            Messages list optimized to fit context window
        """
        available_tokens = self.context_limit - self.response_reserve

        # Subtract tool definition tokens from budget
        if tools:
            tools_tokens = count_tools_tokens(tools, self.model)
            available_tokens -= tools_tokens

        # Start with system prompt
        messages = [{"role": "system", "content": system_prompt}]
        system_tokens = count_messages_tokens(messages, self.model)
        available_tokens -= system_tokens

        # Add summary if exists and applicable
        summary_covers_idx = -1

        if self._summary and self._summary.last_message_idx < len(all_messages):
            summary_msg = self._summary.to_message()
            summary_tokens = count_messages_tokens([summary_msg], self.model)

            if summary_tokens < available_tokens * 0.3:  # Summary shouldn't exceed 30%
                messages.append(summary_msg)
                available_tokens -= summary_tokens
                summary_covers_idx = self._summary.last_message_idx

        # Add recent messages (from newest to oldest until budget exhausted)
        recent_messages: list[dict[str, Any]] = []
        start_idx = summary_covers_idx + 1  # Start after summarized messages

        for i in range(len(all_messages) - 1, start_idx - 1, -1):
            msg = all_messages[i]
            msg_tokens = count_messages_tokens([msg], self.model)

            if msg_tokens <= available_tokens or len(recent_messages) < self.min_recent_messages:
                recent_messages.insert(0, msg)
                available_tokens -= msg_tokens
            else:
                break

        messages.extend(recent_messages)

        # PANIC MODE: Final safety check against context overflow
        total_tokens = count_messages_tokens(messages, self.model)
        if tools:
            total_tokens += count_tools_tokens(tools, self.model)

        target = self.context_limit - self.response_reserve
        if total_tokens > target:
            logger.warning(
                "Context overflow detected (%d tokens, limit %d). "
                "Truncating to fit.",
                total_tokens,
                self.context_limit,
            )
            tools_budget = count_tools_tokens(tools, self.model) if tools else 0
            messages = self._emergency_truncate(messages, target - tools_budget)

        return messages

    def _emergency_truncate(
        self,
        messages: list[dict[str, Any]],
        target_tokens: int,
    ) -> list[dict[str, Any]]:
        """
        Last-resort truncation when context is still too large.

        Strategy:
        1. Keep system prompt (index 0)
        2. Keep summary message (index 1, if exists)
        3. Keep the last min_recent_messages
        4. Truncate large tool results in the middle
        5. Drop oldest non-protected messages
        """
        protected_start = 2 if len(messages) > 2 else 1
        protected_end = min(self.min_recent_messages, len(messages) - protected_start)

        # Try truncating large content in middle messages
        for i in range(protected_start, len(messages) - protected_end):
            msg = messages[i]
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 2000:
                messages[i] = {
                    **msg,
                    "content": content[:1000] + "\n...[truncated]...\n" + content[-500:],
                }

        # Check if that's enough
        current = count_messages_tokens(messages, self.model)
        if current <= target_tokens:
            return messages

        # Still too big — drop middle messages one by one
        while current > target_tokens and len(messages) > protected_start + protected_end:
            messages.pop(protected_start)
            current = count_messages_tokens(messages, self.model)

        return messages

    def should_summarize(self, all_messages: list[dict[str, Any]]) -> bool:
        """
        Determine if we should generate/update the summary.

        Triggers (whichever comes first):
        - N messages since last summary (max_messages_before_summary)
        - K total tokens exceeded (max_tokens_before_summary)
        """
        # Need minimum messages to summarize meaningfully
        min_messages_to_summarize = self.min_recent_messages + 4
        if len(all_messages) < min_messages_to_summarize:
            return False

        # Calculate messages since last summary
        if self._summary:
            messages_since_summary = len(all_messages) - self._summary.last_message_idx - 1
        else:
            messages_since_summary = len(all_messages)

        # TRIGGER 1: N messages threshold
        if messages_since_summary >= self.max_messages_before_summary:
            return True

        # TRIGGER 2: K tokens threshold
        total_tokens = count_messages_tokens(all_messages, self.model)
        if total_tokens >= self.max_tokens_before_summary:
            return True

        return False

    def get_trigger_status(self, all_messages: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Get current status relative to summarization triggers.
        Useful for /context command display.
        """
        if self._summary:
            messages_since_summary = len(all_messages) - self._summary.last_message_idx - 1
        else:
            messages_since_summary = len(all_messages)

        total_tokens = count_messages_tokens(all_messages, self.model)

        # Calculate percentages (capped at 100)
        msg_pct = min(100, messages_since_summary / self.max_messages_before_summary * 100)
        tok_pct = min(100, total_tokens / self.max_tokens_before_summary * 100)

        return {
            "messages_since_summary": messages_since_summary,
            "messages_threshold": self.max_messages_before_summary,
            "messages_percent": msg_pct,
            "total_tokens": total_tokens,
            "tokens_threshold": self.max_tokens_before_summary,
            "tokens_percent": tok_pct,
            "will_trigger": self.should_summarize(all_messages),
        }

    def generate_summary(
        self,
        all_messages: list[dict[str, Any]],
        up_to_idx: int | None = None,
    ) -> ContextSummary:
        """
        Generate a summary of messages.

        Args:
            all_messages: All conversation messages
            up_to_idx: Summarize up to this index (default: len - min_recent)

        Returns:
            ContextSummary object
        """
        if not self.llm:
            raise ValueError("LLM client required for summarization")

        if up_to_idx is None:
            up_to_idx = max(0, len(all_messages) - self.min_recent_messages - 1)

        # Get messages to summarize
        messages_to_summarize = all_messages[: up_to_idx + 1]

        if not messages_to_summarize:
            raise ValueError("No messages to summarize")

        # Build summarization prompt
        conversation_text = self._format_messages_for_summary(messages_to_summarize)

        summary_prompt = f"""Summarize this conversation concisely. Focus on:
1. Key topics discussed
2. Important decisions or conclusions
3. Any tasks completed or pending
4. Relevant context for continuing the conversation

Keep the summary under 500 words.

Conversation:
{conversation_text}

Summary:"""

        # Check if summarization prompt fits in context
        prompt_tokens = count_tokens(summary_prompt, self.model)
        if prompt_tokens > self.context_limit * 0.8:
            # Truncate conversation text to fit
            logger.warning(
                "Summarization prompt too large (%d tokens, limit %d). Truncating.",
                prompt_tokens,
                self.context_limit,
            )
            max_chars = int(self.context_limit * 2)  # rough chars-to-tokens ratio
            conversation_text = (
                conversation_text[:max_chars]
                + "\n... [truncated for summarization]"
            )
            summary_prompt = f"""Summarize this conversation concisely. Focus on:
1. Key topics discussed
2. Important decisions or conclusions
3. Any tasks completed or pending
4. Relevant context for continuing the conversation

Keep the summary under 500 words.

Conversation:
{conversation_text}

Summary:"""

        # Generate summary
        response = self.llm.chat([{"role": "user", "content": summary_prompt}])

        summary_content = response.choices[0].message.content or ""

        # Create summary object
        self._summary = ContextSummary(
            content=summary_content,
            messages_summarized=len(messages_to_summarize),
            first_message_idx=0,
            last_message_idx=up_to_idx,
            token_count=count_tokens(summary_content, self.model),
        )

        # Persist
        self._save_summary()

        return self._summary

    def _format_messages_for_summary(self, messages: list[dict[str, Any]]) -> str:
        """Format messages into readable text for summarization."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")

            if role == "TOOL":
                # Truncate tool results
                content = content[:500] + "..." if len(content) > 500 else content
                lines.append(f"[Tool Result]: {content}")
            elif role == "ASSISTANT" and msg.get("tool_calls"):
                tools = [tc.get("function", {}).get("name", "?") for tc in msg["tool_calls"]]
                lines.append(f"ASSISTANT: [Called tools: {', '.join(tools)}]")
                if content:
                    lines.append(f"ASSISTANT: {content}")
            elif role == "SYSTEM":
                # Skip system messages in summary
                continue
            else:
                lines.append(f"{role}: {content}")

        return "\n\n".join(lines)

    @property
    def summary(self) -> ContextSummary | None:
        """Get the current summary, if any."""
        return self._summary
