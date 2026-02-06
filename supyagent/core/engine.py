"""
Base agent engine shared by Agent (interactive) and ExecutionRunner (execution).

Provides the core LLM-tool loop, tool discovery, tool dispatch, and delegation setup.
Subclasses add their own concerns (session persistence, credential prompting, output formatting).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generator

from supyagent.core.llm import LLMClient
from supyagent.core.models import ToolCallObj
from supyagent.core.tools import (
    discover_tools,
    execute_tool,
    filter_tools,
    supypowers_to_openai_tools,
)
from supyagent.models.agent_config import AgentConfig, get_full_system_prompt

if TYPE_CHECKING:
    from supyagent.core.delegation import DelegationManager
    from supyagent.core.registry import AgentRegistry


class MaxIterationsError(Exception):
    """Raised when the LLM-tool loop exceeds max_iterations."""

    def __init__(self, iterations: int):
        self.iterations = iterations
        super().__init__(f"Max tool iterations ({iterations}) exceeded")


class BaseAgentEngine(ABC):
    """
    Shared agent loop for both interactive and execution modes.

    Subclasses implement:
    - _get_secrets() → secrets for tool execution
    - _build_messages_for_llm() (optional override for context trimming)
    """

    def __init__(self, config: AgentConfig):
        self.config = config

        # Initialize LLM client
        self.llm = LLMClient(
            model=config.model.provider,
            temperature=config.model.temperature,
            max_tokens=config.model.max_tokens,
        )

        self.tools: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []

        # Delegation support
        self.delegation_mgr: DelegationManager | None = None
        self.instance_id: str | None = None
        self.registry: AgentRegistry | None = None

    def _setup_delegation(
        self,
        registry: "AgentRegistry | None" = None,
        parent_instance_id: str | None = None,
    ) -> None:
        """Set up delegation if this agent has delegates."""
        if self.config.delegates:
            from supyagent.core.delegation import DelegationManager
            from supyagent.core.registry import AgentRegistry

            self.registry = registry or AgentRegistry()
            self.delegation_mgr = DelegationManager(
                self.registry, self, grandparent_instance_id=parent_instance_id
            )
            self.instance_id = self.delegation_mgr.parent_id

    def _load_base_tools(self) -> list[dict[str, Any]]:
        """Load supypowers tools filtered by config permissions, plus delegation and process tools."""
        tools: list[dict[str, Any]] = []

        # Discover supypowers tools
        sp_tools = discover_tools()
        openai_tools = supypowers_to_openai_tools(sp_tools)
        filtered = filter_tools(openai_tools, self.config.tools)
        tools.extend(filtered)

        # Delegation tools
        if self.delegation_mgr:
            tools.extend(self.delegation_mgr.get_delegation_tools())

        # Process management tools
        from supyagent.core.process_tools import get_process_management_tools

        tools.extend(get_process_management_tools())

        return tools

    def _build_messages_for_llm(self) -> list[dict[str, Any]]:
        """
        Get messages to send to LLM.

        Default: return self.messages as-is.
        Agent overrides this to apply context trimming.
        """
        return list(self.messages)

    def _dispatch_tool_call(self, tool_call: Any) -> dict[str, Any]:
        """
        Route a tool call to the correct handler.

        Handles delegation tools, process tools, and supypowers tools.
        Subclasses can override to add credential handling.
        """
        name = tool_call.function.name

        # 1. Delegation tools
        if self.delegation_mgr and self.delegation_mgr.is_delegation_tool(name):
            return self.delegation_mgr.execute_delegation(tool_call)

        # 2. Process management tools
        from supyagent.core.process_tools import execute_process_tool, is_process_tool

        if is_process_tool(name):
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                return {"ok": False, "error": "Invalid JSON arguments"}
            return execute_process_tool(name, args)

        # 3. Supypowers tools (through supervisor)
        return self._execute_supypowers_tool(tool_call)

    def _execute_supypowers_tool(self, tool_call: Any) -> dict[str, Any]:
        """Execute a supypowers tool through the supervisor."""
        name = tool_call.function.name

        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return {"ok": False, "error": f"Invalid JSON arguments: {tool_call.function.arguments}"}

        if "__" not in name:
            return {"ok": False, "error": f"Invalid tool name format: {name}"}

        script, func = name.split("__", 1)
        secrets = self._get_secrets()

        # Resolve execution settings from agent config
        timeout, force_background, force_sync = self.config.supervisor.resolve_tool_settings(name)

        return execute_tool(
            script,
            func,
            args,
            secrets=secrets,
            timeout=timeout if not force_sync else None,
            background=force_background,
            use_supervisor=not force_sync,
        )

    @abstractmethod
    def _get_secrets(self) -> dict[str, str]:
        """Get secrets for tool execution."""
        ...

    def _run_loop(
        self,
        max_iterations: int,
        on_message: Any | None = None,
        on_tool_start: Any | None = None,
        on_tool_result: Any | None = None,
    ) -> str:
        """
        Core non-streaming LLM-tool loop. Returns final text content.

        Args:
            max_iterations: Maximum number of loop iterations
            on_message: Callback(content, tool_calls_list) after each assistant message
            on_tool_start: Callback(tool_call_id, name, arguments) before tool execution
            on_tool_result: Callback(tool_call_id, name, result) after tool execution

        Returns:
            Final text content from the LLM

        Raises:
            MaxIterationsError: If max iterations exceeded
        """
        iterations = 0
        content = ""

        while iterations < max_iterations:
            iterations += 1

            messages_for_llm = self._build_messages_for_llm()

            response = self.llm.chat(
                messages=messages_for_llm,
                tools=self.tools if self.tools else None,
            )

            msg = response.choices[0].message
            content = msg.content or ""
            tool_calls: list[dict[str, Any]] = []

            if msg.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]

            # Add to full message history
            msg_dict: dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                msg_dict["tool_calls"] = tool_calls
            self.messages.append(msg_dict)

            if on_message:
                on_message(content, tool_calls)

            # No tool calls → done
            if not tool_calls:
                return content

            # Execute each tool call
            for tc in tool_calls:
                tc_obj = ToolCallObj(tc["id"], tc["function"]["name"], tc["function"]["arguments"])

                if on_tool_start:
                    on_tool_start(tc["id"], tc["function"]["name"], tc["function"]["arguments"])

                result = self._dispatch_tool_call(tc_obj)

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result),
                })

                if on_tool_result:
                    on_tool_result(tc["id"], tc["function"]["name"], result)

        # Max iterations hit
        raise MaxIterationsError(max_iterations)

    def _run_loop_stream(
        self,
        max_iterations: int,
    ) -> Generator[tuple[str, Any], None, None]:
        """
        Core streaming LLM-tool loop. Yields (event_type, data) tuples.

        Event types:
        - ("text", str): Text chunk from the response
        - ("reasoning", str): Reasoning/thinking content
        - ("tool_start", {"name": str, "arguments": str}): Tool execution starting
        - ("tool_end", {"name": str, "result": dict}): Tool execution completed
        - ("_message", (content, tool_calls)): Internal hook for subclass persistence
        - ("_tool_result", (tool_call_id, name, result)): Internal hook for subclass persistence
        - ("done", str): Final complete response
        """
        iterations = 0
        final_content = ""

        while iterations < max_iterations:
            iterations += 1

            messages_for_llm = self._build_messages_for_llm()

            response = self.llm.chat(
                messages=messages_for_llm,
                tools=self.tools if self.tools else None,
                stream=True,
            )

            collected_content = ""
            collected_tool_calls: list[dict[str, Any]] = []

            for chunk in response:
                delta = chunk.choices[0].delta

                # Reasoning/thinking content (some models like Claude provide this)
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    yield ("reasoning", delta.reasoning_content)

                # Text content
                if delta.content:
                    collected_content += delta.content
                    yield ("text", delta.content)

                # Tool calls from stream
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        while len(collected_tool_calls) <= tc.index:
                            collected_tool_calls.append({
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            })
                        if tc.id:
                            collected_tool_calls[tc.index]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                collected_tool_calls[tc.index]["function"]["name"] = tc.function.name
                            if tc.function.arguments:
                                collected_tool_calls[tc.index]["function"][
                                    "arguments"
                                ] += tc.function.arguments

            # Add to full message history
            msg_dict: dict[str, Any] = {"role": "assistant", "content": collected_content}
            if collected_tool_calls:
                msg_dict["tool_calls"] = collected_tool_calls
            self.messages.append(msg_dict)

            # Internal hook for subclass persistence
            yield ("_message", (collected_content, collected_tool_calls if collected_tool_calls else None))

            # No tool calls → done
            if not collected_tool_calls:
                final_content = collected_content
                break

            # Execute each tool call
            for tc_dict in collected_tool_calls:
                tool_name = tc_dict["function"]["name"]
                tool_args = tc_dict["function"]["arguments"]
                tool_id = tc_dict["id"]

                yield ("tool_start", {"name": tool_name, "arguments": tool_args})

                tc_obj = ToolCallObj(tool_id, tool_name, tool_args)
                result = self._dispatch_tool_call(tc_obj)

                yield ("tool_end", {"name": tool_name, "result": result})

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": json.dumps(result),
                })

                # Internal hook for subclass persistence
                yield ("_tool_result", (tool_id, tool_name, result))

        yield ("done", final_content)

    def get_available_tools(self) -> list[str]:
        """Get list of available tool names."""
        return [tool.get("function", {}).get("name", "unknown") for tool in self.tools]
