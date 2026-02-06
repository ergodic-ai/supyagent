"""
Core Agent class for interactive mode.

Extends BaseAgentEngine with session persistence, credential prompting,
and context-managed conversation history.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from supyagent.core.credentials import CredentialManager
from supyagent.core.context_manager import ContextManager
from supyagent.core.engine import BaseAgentEngine, MaxIterationsError
from supyagent.core.session_manager import SessionManager
from supyagent.core.tools import REQUEST_CREDENTIAL_TOOL, is_credential_request
from supyagent.models.agent_config import AgentConfig, get_full_system_prompt
from supyagent.models.session import Message, Session
from supyagent.utils.media import Content, content_to_storable, resolve_media_refs

if TYPE_CHECKING:
    from supyagent.core.delegation import DelegationManager
    from supyagent.core.registry import AgentRegistry


class Agent(BaseAgentEngine):
    """
    An LLM agent that can use supypowers tools.

    The agent maintains a conversation history and handles the loop of:
    1. Send user message to LLM
    2. If LLM wants to call tools, execute them
    3. Feed tool results back to LLM
    4. Repeat until LLM gives final response

    Agents can also delegate tasks to other agents if configured with delegates.
    """

    def __init__(
        self,
        config: AgentConfig,
        session: Session | None = None,
        session_manager: SessionManager | None = None,
        credential_manager: CredentialManager | None = None,
        registry: "AgentRegistry | None" = None,
        parent_instance_id: str | None = None,
    ):
        super().__init__(config)

        # Initialize session management
        self.session_manager = session_manager or SessionManager()

        # Initialize credential management
        self.credential_manager = credential_manager or CredentialManager()

        # Set up multi-agent delegation
        self._setup_delegation(registry, parent_instance_id)
        if not self.registry:
            self.registry = registry

        # Load available tools (base + credential request tool)
        self.tools = self._load_tools()

        # Initialize session
        if session:
            self.session = session
            self.messages = self._reconstruct_messages(session)
        else:
            self.session = self.session_manager.create_session(
                config.name, config.model.provider
            )
            self.messages = [
                {"role": "system", "content": get_full_system_prompt(config)}
            ]

        # Initialize context manager for handling long conversations
        summary_path = None
        if self.session:
            summary_path = Path(
                f".supyagent/sessions/{config.name}/{self.session.meta.session_id}_summary.json"
            )

        ctx = config.context
        self.context_manager = ContextManager(
            model=config.model.provider,
            llm=self.llm,
            summary_storage_path=summary_path,
            max_messages_before_summary=ctx.max_messages_before_summary,
            max_tokens_before_summary=ctx.max_tokens_before_summary,
            min_recent_messages=ctx.min_recent_messages,
            response_reserve=ctx.response_reserve,
        )

    def _load_tools(self) -> list[dict[str, Any]]:
        """Load base tools plus the credential request tool."""
        tools = self._load_base_tools()
        tools.append(REQUEST_CREDENTIAL_TOOL)
        return tools

    def _get_secrets(self) -> dict[str, str]:
        """Get secrets from credential manager."""
        return self.credential_manager.get_all_for_tools(self.config.name)

    def _build_messages_for_llm(self) -> list[dict[str, Any]]:
        """Apply context trimming via ContextManager."""
        system_prompt = get_full_system_prompt(self.config)
        conversation_messages = [m for m in self.messages if m.get("role") != "system"]
        return self.context_manager.build_messages_for_llm(
            system_prompt,
            conversation_messages,
            tools=self.tools,
        )

    def _dispatch_tool_call(self, tool_call: Any) -> dict[str, Any]:
        """Handle credential requests, then delegate to base dispatch."""
        if is_credential_request(tool_call):
            return self._handle_credential_request(tool_call)
        return super()._dispatch_tool_call(tool_call)

    def _get_media_dir(self) -> Path:
        """Get the media directory for the current session."""
        return Path(
            f".supyagent/sessions/{self.config.name}/{self.session.meta.session_id}_media"
        )

    def _reconstruct_messages(self, session: Session) -> list[dict[str, Any]]:
        """Reconstruct LLM message format from session history."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": get_full_system_prompt(self.config)}
        ]
        media_dir = self._get_media_dir()

        for msg in session.messages:
            content = msg.content
            # Resolve media:// refs back to base64 for multimodal content
            if isinstance(content, list):
                content = resolve_media_refs(content, media_dir)

            if msg.type == "user":
                messages.append({"role": "user", "content": content})
            elif msg.type == "assistant":
                m: dict[str, Any] = {"role": "assistant", "content": content}
                if msg.tool_calls:
                    m["tool_calls"] = msg.tool_calls
                messages.append(m)
            elif msg.type == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": content,
                })

        return messages

    def _check_and_summarize(self) -> None:
        """Check if summarization is needed and trigger it if so."""
        if not self.config.context.auto_summarize:
            return

        conversation_messages = [m for m in self.messages if m.get("role") != "system"]

        if self.context_manager.should_summarize(conversation_messages):
            try:
                self.context_manager.generate_summary(conversation_messages)
            except Exception:
                pass

    def _storable(self, content: Content) -> Content:
        """Convert multimodal content to storable format (media:// refs)."""
        if isinstance(content, list):
            return content_to_storable(content, self._get_media_dir())
        return content

    def send_message(self, content: Content) -> str:
        """
        Send a user message and get the agent's response.

        Handles the full tool-use loop with session persistence.
        """
        # Record user message
        user_msg = Message(type="user", content=self._storable(content))
        self.session_manager.append_message(self.session, user_msg)
        self.messages.append({"role": "user", "content": content})

        max_iterations = self.config.limits.get("max_tool_calls_per_turn", 20)
        media_dir = self._get_media_dir()

        # Persistence hooks
        def on_message(msg_content: str, tool_calls: list[dict[str, Any]]) -> None:
            asst_record = Message(
                type="assistant",
                content=msg_content,
                tool_calls=tool_calls if tool_calls else None,
            )
            self.session_manager.append_message(self.session, asst_record)

        def on_tool_result(
            tool_call_id: str, name: str, result: dict[str, Any]
        ) -> None:
            # Get the actual content from the last message (may be multimodal)
            last_msg = self.messages[-1]
            storable_content = self._storable(last_msg.get("content", json.dumps(result)))
            tool_msg = Message(
                type="tool_result",
                tool_call_id=tool_call_id,
                name=name,
                content=storable_content,
            )
            self.session_manager.append_message(self.session, tool_msg)

        try:
            result = self._run_loop(
                max_iterations,
                on_message=on_message,
                on_tool_result=on_tool_result,
            )
        except MaxIterationsError:
            result = self.messages[-1].get("content", "") or ""

        self._check_and_summarize()
        return result

    def send_message_stream(self, content: Content):
        """
        Send a user message and stream the agent's response.

        Yields events as tuples: (event_type, data)
        - ("text", str): Text chunk from the response
        - ("reasoning", str): Reasoning/thinking content
        - ("tool_start", {"name": str, "arguments": str}): Tool execution starting
        - ("tool_end", {"name": str, "result": dict}): Tool execution completed
        - ("done", str): Final complete response
        """
        # Record user message
        user_msg = Message(type="user", content=self._storable(content))
        self.session_manager.append_message(self.session, user_msg)
        self.messages.append({"role": "user", "content": content})

        max_iterations = self.config.limits.get("max_tool_calls_per_turn", 20)

        for event_type, data in self._run_loop_stream(max_iterations):
            if event_type == "_message":
                # Persist assistant message to session
                msg_content, tool_calls = data
                asst_record = Message(
                    type="assistant",
                    content=msg_content,
                    tool_calls=tool_calls,
                )
                self.session_manager.append_message(self.session, asst_record)
            elif event_type == "_tool_result":
                # Persist tool result to session (with media refs for images)
                tool_call_id, name, result = data
                last_msg = self.messages[-1]
                storable_content = self._storable(last_msg.get("content", json.dumps(result)))
                tool_msg = Message(
                    type="tool_result",
                    tool_call_id=tool_call_id,
                    name=name,
                    content=storable_content,
                )
                self.session_manager.append_message(self.session, tool_msg)
            elif event_type == "done":
                self._check_and_summarize()
                yield (event_type, data)
            else:
                # Pass through: text, reasoning, tool_start, tool_end
                yield (event_type, data)

    def _handle_credential_request(self, tool_call: Any) -> dict[str, Any]:
        """Handle a credential request from the LLM."""
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return {"ok": False, "error": "Invalid credential request arguments"}

        name = args.get("name", "")
        description = args.get("description", "")
        service = args.get("service")

        if not name:
            return {"ok": False, "error": "Credential name is required"}

        # Check if we already have it
        existing = self.credential_manager.get(self.config.name, name)
        if existing:
            return {
                "ok": True,
                "message": f"Credential {name} is already available",
            }

        # Prompt user for the credential
        result = self.credential_manager.prompt_for_credential(
            name=name,
            description=description,
            service=service,
        )

        if result is None:
            return {
                "ok": False,
                "error": f"User declined to provide credential {name}",
            }

        value, persist = result
        self.credential_manager.set(self.config.name, name, value, persist=persist)

        return {
            "ok": True,
            "message": f"Credential {name} has been provided and is now available",
        }

    def clear_history(self) -> None:
        """Clear conversation history and start a new session."""
        self.session = self.session_manager.create_session(
            self.config.name, self.config.model.provider
        )
        self.messages = [
            {"role": "system", "content": get_full_system_prompt(self.config)}
        ]
