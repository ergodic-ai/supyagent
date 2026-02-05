"""
Execution runner for non-interactive agent execution.

Execution agents are stateless, input→output pipelines designed for automation.
"""

import json
import os
import re
from typing import Any, Callable

from supyagent.core.credentials import CredentialManager
from supyagent.core.llm import LLMClient
from supyagent.core.tools import (
    discover_tools,
    execute_tool,
    filter_tools,
    is_credential_request,
    supypowers_to_openai_tools,
)
from supyagent.models.agent_config import AgentConfig, get_full_system_prompt

# Type for progress callback: (event_type, data)
# event_type: "tool_start", "tool_end", "thinking", "streaming"
ProgressCallback = Callable[[str, dict[str, Any]], None]


class ExecutionRunnerDelegationHelper:
    """Helper class to make ExecutionRunner work with DelegationManager."""
    
    def __init__(self, config: AgentConfig, llm: LLMClient, messages: list):
        self.config = config
        self.llm = llm
        self.messages = messages
        self.instance_id = None


class ExecutionRunner:
    """
    Runs agents in non-interactive execution mode.

    Key differences from interactive mode:
    - No session persistence
    - No credential prompting (must be pre-provided)
    - Single input → output execution
    - Designed for automation and pipelines
    """

    def __init__(
        self,
        config: AgentConfig,
        credential_manager: CredentialManager | None = None,
    ):
        """
        Initialize the execution runner.

        Args:
            config: Agent configuration
            credential_manager: Optional credential manager for stored credentials
        """
        self.config = config
        self.credential_manager = credential_manager or CredentialManager()

        # Initialize LLM client
        self.llm = LLMClient(
            model=config.model.provider,
            temperature=config.model.temperature,
            max_tokens=config.model.max_tokens,
        )

        # Set up delegation if this agent has delegates
        self.delegation_mgr = None
        if config.delegates:
            from supyagent.core.delegation import DelegationManager
            from supyagent.core.registry import AgentRegistry
            
            self.registry = AgentRegistry()
            # Create a helper object that mimics enough of Agent for DelegationManager
            self._delegation_helper = ExecutionRunnerDelegationHelper(config, self.llm, [])
            self.delegation_mgr = DelegationManager(self.registry, self._delegation_helper)

        # Load available tools (excluding credential request tool for execution mode)
        self.tools = self._load_tools()

    def _load_tools(self) -> list[dict[str, Any]]:
        """
        Load tools for execution mode.

        Note: Does NOT include request_credential tool since
        execution mode cannot prompt for credentials.
        """
        tools: list[dict[str, Any]] = []
        
        # If tools allowed, load supypowers tools
        if self.config.tools.allow:
            sp_tools = discover_tools()
            openai_tools = supypowers_to_openai_tools(sp_tools)
            filtered = filter_tools(openai_tools, self.config.tools)
            tools.extend(filtered)

        # Add delegation tools if available
        if self.delegation_mgr:
            tools.extend(self.delegation_mgr.get_delegation_tools())

        # Add process management tools
        from supyagent.core.process_tools import get_process_management_tools
        tools.extend(get_process_management_tools())

        return tools

    def run(
        self,
        task: str | dict[str, Any],
        secrets: dict[str, str] | None = None,
        output_format: str = "raw",
        on_progress: ProgressCallback | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """
        Execute a task and return the result.

        Args:
            task: Task description (string) or structured input (dict)
            secrets: Pre-provided credentials (KEY=VALUE)
            output_format: "raw" | "json" | "markdown"
            on_progress: Optional callback for progress updates
            stream: Whether to stream the response

        Returns:
            {"ok": True, "data": ...} or {"ok": False, "error": ...}
        """
        # Merge secrets: stored credentials + provided secrets
        all_secrets = self.credential_manager.get_all_for_tools(self.config.name)
        if secrets:
            all_secrets.update(secrets)

        # Inject secrets into environment for this execution
        for key, value in all_secrets.items():
            os.environ[key] = value

        try:
            # Build the prompt
            if isinstance(task, dict):
                user_content = self._format_structured_input(task)
            else:
                user_content = str(task)

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": get_full_system_prompt(self.config)},
                {"role": "user", "content": user_content},
            ]

            # Run with tool loop (max iterations for safety)
            max_iterations = self.config.limits.get("max_tool_calls_per_turn", 20)
            iterations = 0

            while iterations < max_iterations:
                iterations += 1

                # Handle streaming vs non-streaming
                if stream and on_progress:
                    # Stream the response
                    response = self.llm.chat(
                        messages=messages,
                        tools=self.tools if self.tools else None,
                        stream=True,
                    )
                    
                    # Collect streamed content
                    collected_content = ""
                    collected_tool_calls: list[dict] = []
                    
                    for chunk in response:
                        delta = chunk.choices[0].delta
                        
                        # Check for reasoning/thinking content (some models like Claude provide this)
                        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                            on_progress("reasoning", {"content": delta.reasoning_content})
                        
                        # Stream text content
                        if delta.content:
                            collected_content += delta.content
                            on_progress("streaming", {"content": delta.content})
                        
                        # Collect tool calls from stream
                        if delta.tool_calls:
                            for tc in delta.tool_calls:
                                # Initialize or extend tool call
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
                    
                    # Build assistant message from collected stream
                    msg_dict: dict[str, Any] = {
                        "role": "assistant",
                        "content": collected_content,
                    }
                    if collected_tool_calls:
                        msg_dict["tool_calls"] = collected_tool_calls
                    
                    messages.append(msg_dict)
                    
                    # Check if done (no tool calls)
                    if not collected_tool_calls:
                        return self._format_output(collected_content, output_format)
                    
                    # Execute collected tool calls
                    for tool_call_dict in collected_tool_calls:
                        tool_name = tool_call_dict["function"]["name"]
                        tool_args = tool_call_dict["function"]["arguments"]
                        tool_id = tool_call_dict["id"]
                        
                        # Notify progress
                        on_progress("tool_start", {
                            "name": tool_name,
                            "arguments": tool_args,
                        })
                        
                        # Create a simple object to match the tool_call interface
                        class ToolCallObj:
                            def __init__(self, id, name, arguments):
                                self.id = id
                                self.function = type('obj', (object,), {'name': name, 'arguments': arguments})()
                        
                        tc_obj = ToolCallObj(tool_id, tool_name, tool_args)
                        
                        # Check for credential requests
                        if tool_name == "request_credential":
                            try:
                                args = json.loads(tool_args)
                                cred_name = args.get("name", "unknown")
                            except json.JSONDecodeError:
                                cred_name = "unknown"
                            return {
                                "ok": False,
                                "error": f"Credential '{cred_name}' required but not provided. "
                                f"Pass it via --secrets {cred_name}=<value>",
                            }
                        
                        result = self._execute_tool(tc_obj, all_secrets)
                        
                        on_progress("tool_end", {
                            "name": tool_name,
                            "result": result,
                        })
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": json.dumps(result),
                        })
                else:
                    # Non-streaming mode
                    response = self.llm.chat(
                        messages=messages,
                        tools=self.tools if self.tools else None,
                    )
                    assistant_msg = response.choices[0].message

                    # Build message for history
                    msg_dict: dict[str, Any] = {
                        "role": "assistant",
                        "content": assistant_msg.content,
                    }

                    if assistant_msg.tool_calls:
                        msg_dict["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in assistant_msg.tool_calls
                        ]

                    messages.append(msg_dict)

                    # If no tool calls, we're done
                    if not assistant_msg.tool_calls:
                        return self._format_output(assistant_msg.content or "", output_format)

                    # Execute tools
                    for tool_call in assistant_msg.tool_calls:
                        # Notify progress if callback provided
                        if on_progress:
                            on_progress("tool_start", {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            })
                        
                        # Credential requests fail in execution mode
                        if is_credential_request(tool_call):
                            try:
                                args = json.loads(tool_call.function.arguments)
                                cred_name = args.get("name", "unknown")
                            except json.JSONDecodeError:
                                cred_name = "unknown"

                            return {
                                "ok": False,
                                "error": f"Credential '{cred_name}' required but not provided. "
                                f"Pass it via --secrets {cred_name}=<value>",
                            }

                        result = self._execute_tool(tool_call, all_secrets)
                        
                        # Notify progress if callback provided
                        if on_progress:
                            on_progress("tool_end", {
                                "name": tool_call.function.name,
                                "result": result,
                            })
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result),
                        })

            return {"ok": False, "error": "Max tool iterations exceeded"}

        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _format_structured_input(self, task: dict[str, Any]) -> str:
        """Format a structured input dict into a prompt."""
        return json.dumps(task, indent=2)

    def _format_output(self, content: str, output_format: str) -> dict[str, Any]:
        """Format the output according to requested format."""
        if output_format == "json":
            # Try to parse as JSON
            try:
                data = json.loads(content)
                return {"ok": True, "data": data}
            except json.JSONDecodeError:
                pass

            # Try to extract JSON from markdown code block
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
            if match:
                try:
                    data = json.loads(match.group(1))
                    return {"ok": True, "data": data}
                except json.JSONDecodeError:
                    pass

            # Return as-is
            return {"ok": True, "data": content}

        elif output_format == "markdown":
            return {"ok": True, "data": content, "format": "markdown"}

        else:  # raw
            return {"ok": True, "data": content}

    def _execute_tool(
        self,
        tool_call: Any,
        secrets: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Execute a tool call."""
        name = tool_call.function.name
        arguments_str = tool_call.function.arguments

        # Check for delegation tools
        if self.delegation_mgr and self.delegation_mgr.is_delegation_tool(name):
            return self.delegation_mgr.execute_delegation(tool_call)

        # Check for process management tools
        from supyagent.core.process_tools import is_process_tool, execute_process_tool
        if is_process_tool(name):
            try:
                args = json.loads(arguments_str)
            except json.JSONDecodeError:
                return {"ok": False, "error": "Invalid JSON arguments"}
            return execute_process_tool(name, args)

        try:
            args = json.loads(arguments_str)
        except json.JSONDecodeError:
            return {"ok": False, "error": f"Invalid JSON arguments: {arguments_str}"}

        if "__" not in name:
            return {"ok": False, "error": f"Invalid tool name format: {name}"}

        script, func = name.split("__", 1)

        # Check supervisor settings for this tool
        sup_settings = self.config.supervisor
        timeout = sup_settings.default_timeout
        use_supervisor = True  # Always use supervisor for timeout/background support

        # Check if tool matches force_background or force_sync patterns
        import fnmatch
        force_background = any(
            fnmatch.fnmatch(name, pattern) 
            for pattern in sup_settings.force_background_patterns
        )
        force_sync = any(
            fnmatch.fnmatch(name, pattern) 
            for pattern in sup_settings.force_sync_patterns
        )

        # Per-tool settings override
        for pattern, tool_settings in sup_settings.tool_settings.items():
            if fnmatch.fnmatch(name, pattern):
                timeout = tool_settings.timeout
                if tool_settings.mode == "background":
                    force_background = True
                elif tool_settings.mode == "sync":
                    force_sync = True
                break

        return execute_tool(
            script, func, args, 
            secrets=secrets,
            timeout=timeout if not force_sync else None,
            background=force_background,
            use_supervisor=use_supervisor and not force_sync,
        )
