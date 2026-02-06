"""
Context passing for multi-agent delegation.

Enables rich context to be passed from parent to child agents,
including conversation summaries and relevant facts.
"""

from dataclasses import dataclass, field
from typing import Any

from supyagent.core.llm import LLMClient


@dataclass
class DelegationContext:
    """
    Context passed from parent to child agent.

    Attributes:
        parent_agent: Name of the parent agent
        parent_task: The task the parent is working on
        conversation_summary: Summary of relevant conversation history
        relevant_facts: Key facts relevant to the delegation
        shared_data: Arbitrary data to pass to the child
    """

    parent_agent: str
    parent_task: str
    conversation_summary: str | None = None
    relevant_facts: list[str] = field(default_factory=list)
    shared_data: dict[str, Any] = field(default_factory=dict)
    depth: int = 0

    def to_prompt(self) -> str:
        """
        Convert context to a prompt prefix for the child agent.

        Returns:
            A formatted string to prepend to the child's task
        """
        parts = [
            f"You are being called by the '{self.parent_agent}' agent.",
            f"Parent's current task: {self.parent_task}",
        ]

        if self.conversation_summary:
            parts.append(f"\nConversation context:\n{self.conversation_summary}")

        if self.relevant_facts:
            parts.append("\nRelevant information:")
            for fact in self.relevant_facts:
                parts.append(f"- {fact}")

        if self.shared_data:
            parts.append("\nShared data:")
            for key, value in self.shared_data.items():
                parts.append(f"- {key}: {value}")

        return "\n".join(parts)


def summarize_conversation(
    messages: list[dict[str, Any]],
    llm: LLMClient,
    max_messages: int = 10,
) -> str | None:
    """
    Use LLM to create a summary of the conversation for context passing.

    Args:
        messages: List of conversation messages
        llm: LLM client for generating summary
        max_messages: Maximum number of recent messages to include

    Returns:
        A concise summary of the conversation, or None if no messages
    """
    # Extract just user and assistant messages
    conversation = []
    for msg in messages[-max_messages:]:
        role = msg.get("role")
        if role in ("user", "assistant"):
            role_label = "User" if role == "user" else "Assistant"
            content = msg.get("content", "")
            if content:
                # Truncate long messages
                if len(content) > 300:
                    content = content[:300] + "..."
                conversation.append(f"{role_label}: {content}")

    if not conversation:
        return None

    summary_prompt = f"""Summarize this conversation in 2-3 sentences, focusing on the main task and key decisions:

{chr(10).join(conversation)}

Summary:"""

    try:
        response = llm.chat([{"role": "user", "content": summary_prompt}])
        return response.choices[0].message.content
    except Exception:
        # If summarization fails, return None
        return None


def extract_relevant_facts(
    messages: list[dict[str, Any]],
    task: str,
    llm: LLMClient,
) -> list[str]:
    """
    Extract facts from the conversation relevant to a specific task.

    Args:
        messages: Conversation messages
        task: The task to extract relevant facts for
        llm: LLM client

    Returns:
        List of relevant facts
    """
    # Extract content from recent messages
    content_parts = []
    for msg in messages[-5:]:
        if msg.get("content"):
            content_parts.append(msg["content"])

    if not content_parts:
        return []

    context = "\n---\n".join(content_parts)

    prompt = f"""Given this conversation context and the upcoming task, list 2-4 key facts that would be relevant.

Conversation context:
{context}

Upcoming task: {task}

List the relevant facts (one per line, start each with "- "):"""

    try:
        response = llm.chat([{"role": "user", "content": prompt}])
        content = response.choices[0].message.content or ""

        # Parse bullet points
        facts = []
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                facts.append(line[2:])
            elif line.startswith("• "):
                facts.append(line[2:])

        return facts[:4]  # Limit to 4 facts
    except Exception:
        return []
