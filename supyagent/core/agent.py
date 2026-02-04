"""
Core Agent class.

The Agent handles the conversation loop, tool execution, and message management.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from supyagent.core.context_manager import ContextManager
from supyagent.core.credentials import CredentialManager
from supyagent.core.llm import LLMClient
from supyagent.core.session_manager import SessionManager
from supyagent.core.tools import (
    REQUEST_CREDENTIAL_TOOL,
    discover_tools,
    execute_tool,
    filter_tools,
    is_credential_request,
    supypowers_to_openai_tools,
)
from supyagent.models.agent_config import AgentConfig, get_full_system_prompt
from supyagent.models.session import Message, Session

if TYPE_CHECKING:
    from supyagent.core.delegation import DelegationManager
    from supyagent.core.registry import AgentRegistry


class Agent:
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
        """
        Initialize an agent from configuration.

        Args:
            config: Agent configuration
            session: Optional existing session to resume
            session_manager: Optional session manager (creates one if not provided)
            credential_manager: Optional credential manager (creates one if not provided)
            registry: Optional agent registry for multi-agent support
            parent_instance_id: Instance ID of parent agent (if this is a sub-agent)
        """
        self.config = config

        # Initialize LLM client
        self.llm = LLMClient(
            model=config.model.provider,
            temperature=config.model.temperature,
            max_tokens=config.model.max_tokens,
        )

        # Initialize session management
        self.session_manager = session_manager or SessionManager()

        # Initialize credential management
        self.credential_manager = credential_manager or CredentialManager()

        # Initialize multi-agent support
        self.instance_id: str | None = None
        self.delegation_mgr: "DelegationManager | None" = None

        if config.delegates:
            # Lazy import to avoid circular dependency
            from supyagent.core.delegation import DelegationManager
            from supyagent.core.registry import AgentRegistry

            self.registry = registry or AgentRegistry()
            self.delegation_mgr = DelegationManager(
                self.registry,
                self,
                grandparent_instance_id=parent_instance_id,
            )
            self.instance_id = self.delegation_mgr.parent_id
        else:
            self.registry = registry

        # Load available tools (including delegation tools)
        self.tools = self._load_tools()

        # Initialize session
        if session:
            self.session = session
            self.messages = self._reconstruct_messages(session)
        else:
            self.session = self.session_manager.create_session(
                config.name, config.model.provider
            )
            self.messages: list[dict[str, Any]] = [
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
        """
        Discover and filter available tools.

        Returns:
            List of tools in OpenAI function calling format
        """
        tools: list[dict[str, Any]] = []

        # Discover tools from supypowers
        sp_tools = discover_tools()

        # Convert to OpenAI format
        openai_tools = supypowers_to_openai_tools(sp_tools)

        # Filter by permissions
        filtered = filter_tools(openai_tools, self.config.tools)
        tools.extend(filtered)

        # Always add the credential request tool
        tools.append(REQUEST_CREDENTIAL_TOOL)

        # Add delegation tools if this agent can delegate
        if self.delegation_mgr:
            tools.extend(self.delegation_mgr.get_delegation_tools())

        return tools

    def _reconstruct_messages(self, session: Session) -> list[dict[str, Any]]:
        """
        Reconstruct LLM message format from session history.

        Args:
            session: Session with message history

        Returns:
            List of messages in OpenAI format
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": get_full_system_prompt(self.config)}
        ]

        for msg in session.messages:
            if msg.type == "user":
                messages.append({"role": "user", "content": msg.content})
            elif msg.type == "assistant":
                m: dict[str, Any] = {"role": "assistant", "content": msg.content}
                if msg.tool_calls:
                    m["tool_calls"] = msg.tool_calls
                messages.append(m)
            elif msg.type == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })

        return messages

    def _get_messages_for_llm(self) -> list[dict[str, Any]]:
        """
        Get messages to send to LLM, managed for context limits.

        Uses the ContextManager to build an optimized message list that:
        - Includes the system prompt
        - Includes a summary of older messages (if available)
        - Includes recent messages that fit in the context window
        """
        system_prompt = get_full_system_prompt(self.config)

        # Filter out system messages from history (we add it fresh)
        conversation_messages = [m for m in self.messages if m.get("role") != "system"]

        return self.context_manager.build_messages_for_llm(
            system_prompt,
            conversation_messages,
        )

    def _check_and_summarize(self) -> None:
        """Check if summarization is needed and trigger it if so."""
        if not self.config.context.auto_summarize:
            return

        conversation_messages = [m for m in self.messages if m.get("role") != "system"]

        if self.context_manager.should_summarize(conversation_messages):
            try:
                self.context_manager.generate_summary(conversation_messages)
            except Exception:
                # Don't fail the conversation if summarization fails
                pass

    def send_message(self, content: str) -> str:
        """
        Send a user message and get the agent's response.

        This handles the full tool-use loop:
        1. Add user message
        2. Get LLM response
        3. If tools requested, execute them and continue
        4. Return final text response

        Args:
            content: User's message

        Returns:
            Agent's final text response
        """
        # Record user message
        user_msg = Message(type="user", content=content)
        self.session_manager.append_message(self.session, user_msg)
        self.messages.append({"role": "user", "content": content})

        # Maximum iterations to prevent infinite loops
        max_iterations = self.config.limits.get("max_tool_calls_per_turn", 20)
        iterations = 0

        while iterations < max_iterations:
            iterations += 1

            # Get context-managed messages for LLM
            messages_for_llm = self._get_messages_for_llm()

            # Call LLM
            response = self.llm.chat(
                messages=messages_for_llm,
                tools=self.tools if self.tools else None,
            )

            assistant_message = response.choices[0].message

            # Build message dict for history
            msg_dict: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_message.content,
            }

            # Build tool_calls list for session storage
            tool_calls_for_session: list[dict[str, Any]] = []

            if assistant_message.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_message.tool_calls
                ]
                tool_calls_for_session = msg_dict["tool_calls"]

            # Record assistant message to session
            asst_record = Message(
                type="assistant",
                content=assistant_message.content,
                tool_calls=tool_calls_for_session if tool_calls_for_session else None,
            )
            self.session_manager.append_message(self.session, asst_record)

            # Add to LLM message history
            self.messages.append(msg_dict)

            # If no tool calls, we're done
            if not assistant_message.tool_calls:
                return assistant_message.content or ""

            # Execute each tool call
            for tool_call in assistant_message.tool_calls:
                # Handle credential requests specially
                if is_credential_request(tool_call):
                    result = self._handle_credential_request(tool_call)
                else:
                    result = self._execute_tool_call(tool_call)

                # Record tool result to session
                tool_msg = Message(
                    type="tool_result",
                    tool_call_id=tool_call.id,
                    name=tool_call.function.name,
                    content=json.dumps(result),
                )
                self.session_manager.append_message(self.session, tool_msg)

                # Add to LLM message history
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })

        # If we hit max iterations, return what we have
        result = self.messages[-1].get("content", "") or ""

        # Check if we should summarize after this exchange
        self._check_and_summarize()

        return result

    def send_message_stream(self, content: str):
        """
        Send a user message and stream the agent's response.

        Yields events as tuples: (event_type, data)
        - ("text", str): Text chunk from the response
        - ("tool_start", {"name": str, "arguments": str}): Tool execution starting
        - ("tool_end", {"name": str, "result": dict}): Tool execution completed
        - ("done", str): Final complete response

        Args:
            content: User's message

        Yields:
            Tuple of (event_type, data)
        """
        from typing import Generator

        # Record user message
        user_msg = Message(type="user", content=content)
        self.session_manager.append_message(self.session, user_msg)
        self.messages.append({"role": "user", "content": content})

        # Maximum iterations to prevent infinite loops
        max_iterations = self.config.limits.get("max_tool_calls_per_turn", 20)
        iterations = 0
        final_content = ""

        while iterations < max_iterations:
            iterations += 1

            # Get context-managed messages for LLM
            messages_for_llm = self._get_messages_for_llm()

            # Call LLM with streaming
            response = self.llm.chat(
                messages=messages_for_llm,
                tools=self.tools if self.tools else None,
                stream=True,
            )

            # Collect streamed content and tool calls
            collected_content = ""
            collected_tool_calls: list[dict] = []

            for chunk in response:
                delta = chunk.choices[0].delta

                # Check for reasoning/thinking content (some models like Claude provide this)
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    yield ("reasoning", delta.reasoning_content)

                # Stream text content
                if delta.content:
                    collected_content += delta.content
                    yield ("text", delta.content)

                # Collect tool calls from stream
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        while len(collected_tool_calls) <= tc.index:
                            collected_tool_calls.append({
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            })
                        if tc.id:
                            collected_tool_calls[tc.index]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                collected_tool_calls[tc.index]["function"]["name"] = tc.function.name
                            if tc.function.arguments:
                                collected_tool_calls[tc.index]["function"]["arguments"] += tc.function.arguments

            # Build message dict for history
            msg_dict: dict[str, Any] = {
                "role": "assistant",
                "content": collected_content,
            }
            if collected_tool_calls:
                msg_dict["tool_calls"] = collected_tool_calls

            # Record assistant message to session
            asst_record = Message(
                type="assistant",
                content=collected_content,
                tool_calls=collected_tool_calls if collected_tool_calls else None,
            )
            self.session_manager.append_message(self.session, asst_record)

            # Add to LLM message history
            self.messages.append(msg_dict)

            # If no tool calls, we're done
            if not collected_tool_calls:
                final_content = collected_content
                break

            # Execute each tool call
            for tc_dict in collected_tool_calls:
                tool_name = tc_dict["function"]["name"]
                tool_args = tc_dict["function"]["arguments"]
                tool_id = tc_dict["id"]

                yield ("tool_start", {"name": tool_name, "arguments": tool_args})

                # Create a simple object to match the tool_call interface
                class ToolCallObj:
                    def __init__(self, id, name, arguments):
                        self.id = id
                        self.function = type('obj', (object,), {'name': name, 'arguments': arguments})()

                tc_obj = ToolCallObj(tool_id, tool_name, tool_args)

                # Handle credential requests specially
                if tool_name == "request_credential":
                    result = self._handle_credential_request(tc_obj)
                else:
                    result = self._execute_tool_call(tc_obj)

                yield ("tool_end", {"name": tool_name, "result": result})

                # Record tool result to session
                tool_msg = Message(
                    type="tool_result",
                    tool_call_id=tool_id,
                    name=tool_name,
                    content=json.dumps(result),
                )
                self.session_manager.append_message(self.session, tool_msg)

                # Add to LLM message history
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": json.dumps(result),
                })

        # Check if we should summarize after this exchange
        self._check_and_summarize()

        yield ("done", final_content)

    def _handle_credential_request(self, tool_call: Any) -> dict[str, Any]:
        """
        Handle a credential request from the LLM.

        Args:
            tool_call: The tool call requesting credentials

        Returns:
            Result dict indicating success or failure
        """
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

    def _execute_tool_call(self, tool_call: Any) -> dict[str, Any]:
        """
        Execute a single tool call.

        Args:
            tool_call: Tool call from LLM response

        Returns:
            Result dict from tool execution
        """
        name = tool_call.function.name

        # Check if this is a delegation tool
        if self.delegation_mgr and self.delegation_mgr.is_delegation_tool(name):
            return self.delegation_mgr.execute_delegation(tool_call)

        # Parse arguments
        arguments_str = tool_call.function.arguments
        try:
            args = json.loads(arguments_str)
        except json.JSONDecodeError:
            return {"ok": False, "error": f"Invalid JSON arguments: {arguments_str}"}

        # Parse script__function format
        if "__" not in name:
            return {"ok": False, "error": f"Invalid tool name format: {name}"}

        script, func = name.split("__", 1)

        # Get credentials for tool execution
        secrets = self.credential_manager.get_all_for_tools(self.config.name)

        # Execute via supypowers
        return execute_tool(script, func, args, secrets=secrets)

    def get_available_tools(self) -> list[str]:
        """
        Get list of available tool names.

        Returns:
            List of tool names
        """
        return [
            tool.get("function", {}).get("name", "unknown")
            for tool in self.tools
        ]

    def clear_history(self) -> None:
        """Clear conversation history and start a new session."""
        # Create a new session
        self.session = self.session_manager.create_session(
            self.config.name, self.config.model.provider
        )
        self.messages = [
            {"role": "system", "content": get_full_system_prompt(self.config)}
        ]
