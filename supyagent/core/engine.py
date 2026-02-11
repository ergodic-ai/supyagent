"""
Base agent engine shared by Agent (interactive) and ExecutionRunner (execution).

Provides the core LLM-tool loop, tool discovery, tool dispatch, and delegation setup.
Subclasses add their own concerns (session persistence, credential prompting, output formatting).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Generator

logger = logging.getLogger(__name__)

from supyagent.core.llm import LLMClient
from supyagent.core.models import ToolCallObj
from supyagent.core.tools import (
    discover_tools,
    execute_tool,
    filter_tools,
    is_memory_tool,
    supypowers_to_openai_tools,
)
from supyagent.models.agent_config import AgentConfig
from supyagent.utils.media import (
    detect_images_in_tool_result,
    make_image_content_part,
    make_text_content_part,
)

if TYPE_CHECKING:
    from supyagent.core.browser_session import BrowserSessionManager
    from supyagent.core.delegation import DelegationManager
    from supyagent.core.memory import MemoryManager
    from supyagent.core.registry import AgentRegistry
    from supyagent.core.sandbox import SandboxManager, WorkspaceValidator
    from supyagent.core.telemetry import TelemetryCollector


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
            max_retries=config.model.max_retries,
            retry_delay=config.model.retry_delay,
            retry_backoff=config.model.retry_backoff,
            fallback_models=config.model.fallback,
            cache=config.model.cache,
        )

        self.tools: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []
        self.supypowers_available: bool = False

        # Circuit breaker: track consecutive tool failures per turn
        self._tool_failure_counts: dict[str, int] = {}
        self._tool_failure_threshold: int = config.limits.get(
            "circuit_breaker_threshold", 3
        )

        # Repetition detection: track recent tool calls to catch loops
        self._recent_tool_calls: list[tuple[str, str]] = []  # (name, args_hash)
        self._repetition_threshold: int = config.limits.get("repetition_threshold", 4)

        # Lock for thread-safe access to circuit breaker / repetition state
        self._dispatch_lock = threading.Lock()

        # Memory system
        self.memory_mgr: MemoryManager | None = None

        # Sandbox / workspace isolation
        self.sandbox_mgr: SandboxManager | None = None
        self.workspace_validator: WorkspaceValidator | None = None

        # Delegation support
        self.delegation_mgr: DelegationManager | None = None
        self.instance_id: str | None = None
        self.registry: AgentRegistry | None = None

        # Service integration: HTTP-based tools from supyagent_service
        self._service_client = None
        self._service_tool_metadata: dict[str, dict[str, Any]] = {}
        self._tool_sources: dict[str, str] = {}  # tool_name -> source label
        if config.service.enabled:
            from supyagent.core.service import get_service_client

            self._service_client = get_service_client()

        # Persistent browser session (lazy-initialized on first browser tool call)
        self._browser_session: "BrowserSessionManager | None" = None

        # Telemetry
        self.telemetry: "TelemetryCollector | None" = None

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
        """Load supypowers + service tools filtered by config permissions, plus delegation and process tools."""
        tools: list[dict[str, Any]] = []
        self._tool_sources.clear()

        # Discover supypowers tools (inside sandbox container if active)
        sp_tools = discover_tools(sandbox=self.sandbox_mgr)
        self.supypowers_available = len(sp_tools) > 0

        # Build script path lookup: tool_name -> script_path
        sp_script_paths: dict[str, str] = {}
        for entry in sp_tools:
            script_path = entry.get("script", "")
            script_name = os.path.splitext(os.path.basename(script_path))[0]
            for func in entry.get("functions", []):
                tool_name = f"{script_name}__{func.get('name', '')}"
                sp_script_paths[tool_name] = script_path

        openai_tools = supypowers_to_openai_tools(sp_tools)
        filtered = filter_tools(openai_tools, self.config.tools)

        # Gate shell/exec tools when sandbox disallows them
        if self.config.sandbox.enabled and not self.config.sandbox.allow_shell:
            filtered = [
                t for t in filtered
                if not t.get("function", {}).get("name", "").startswith("shell__")
            ]

        for t in filtered:
            name = t.get("function", {}).get("name", "")
            self._tool_sources[name] = sp_script_paths.get(name, "supypower")
        tools.extend(filtered)

        # Discover service tools (HTTP-based third-party integrations)
        if self._service_client:
            service_tools = self._service_client.discover_tools()
            if service_tools:
                # Store metadata for dispatch routing, then filter
                for tool in service_tools:
                    name = tool.get("function", {}).get("name", "")
                    if name:
                        self._service_tool_metadata[name] = tool.get("metadata", {})

                filtered_service = filter_tools(service_tools, self.config.tools)

                # Strip metadata before sending to LLM (not part of OpenAI format)
                for tool in filtered_service:
                    tool.pop("metadata", None)
                    name = tool.get("function", {}).get("name", "")
                    self._tool_sources[name] = "remote"
                tools.extend(filtered_service)

        # Delegation tools
        if self.delegation_mgr:
            delegation_tools = self.delegation_mgr.get_delegation_tools()
            for t in delegation_tools:
                name = t.get("function", {}).get("name", "")
                self._tool_sources[name] = "native"
            tools.extend(delegation_tools)

        # Process management tools
        from supyagent.core.process_tools import get_process_management_tools

        process_tools = get_process_management_tools()
        for t in process_tools:
            name = t.get("function", {}).get("name", "")
            self._tool_sources[name] = "native"
        tools.extend(process_tools)

        return tools

    def reload_tools(self) -> int:
        """
        Reload tools from supypowers and service.

        Useful for picking up new/changed tools without restarting.
        Returns the new tool count.
        """
        self._service_tool_metadata.clear()
        self.tools = self._load_base_tools()
        return len(self.tools)

    def _system_prompt_kwargs(self) -> dict[str, Any]:
        """Get kwargs for get_full_system_prompt based on engine state."""
        from supyagent.core.sandbox import create_sandbox_context_prompt
        from supyagent.core.workspace import read_goals

        return {
            "supypowers_available": self.supypowers_available,
            "has_service": self._service_client is not None,
            "sandbox_context": create_sandbox_context_prompt(self.config, self.sandbox_mgr),
            "goals_content": read_goals(),
        }

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
        Includes circuit breaker logic for repeated failures.
        Subclasses can override to add credential handling.
        """
        import time as _time

        name = tool_call.function.name

        # Thread-safe check of repetition and circuit breaker
        args_hash = hashlib.md5(tool_call.function.arguments.encode()).hexdigest()
        call_sig = (name, args_hash)
        with self._dispatch_lock:
            repeat_count = self._recent_tool_calls.count(call_sig)
            if repeat_count >= self._repetition_threshold:
                return {
                    "ok": False,
                    "error": (
                        f"Repetitive tool call detected: '{name}' called with identical "
                        f"arguments {repeat_count + 1} times. Try a different approach."
                    ),
                    "error_type": "repetition_detected",
                }

            if self._tool_failure_counts.get(name, 0) >= self._tool_failure_threshold:
                return {
                    "ok": False,
                    "error": (
                        f"Tool '{name}' has failed {self._tool_failure_threshold} times "
                        "consecutively. Please try a different approach or tool."
                    ),
                    "error_type": "circuit_breaker",
                }

        start = _time.monotonic()
        is_service = name in self._service_tool_metadata

        # 1. Delegation tools
        if self.delegation_mgr and self.delegation_mgr.is_delegation_tool(name):
            result = self.delegation_mgr.execute_delegation(tool_call)
        elif self.memory_mgr and is_memory_tool(name):
            # 1b. Memory tools
            result = self._execute_memory_tool(tool_call)
        elif self._is_process_tool(name):
            # 2. Process management tools
            from supyagent.core.process_tools import execute_process_tool

            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                with self._dispatch_lock:
                    self._tool_failure_counts[name] = (
                        self._tool_failure_counts.get(name, 0) + 1
                    )
                return {
                    "ok": False,
                    "error": "Invalid JSON arguments",
                    "error_type": "invalid_args",
                }
            result = execute_process_tool(name, args)
        elif is_service:
            # 3. Service tools (HTTP routing to supyagent_service)
            result = self._execute_service_tool(tool_call)
        elif self._is_browser_tool(name):
            # 4. Browser tools (persistent daemon)
            result = self._execute_browser_tool(tool_call)
        else:
            # 5. Supypowers tools (through supervisor)
            result = self._execute_supypowers_tool(tool_call)

        elapsed_ms = (_time.monotonic() - start) * 1000

        # Thread-safe tracking of repetition and failures
        with self._dispatch_lock:
            self._recent_tool_calls.append(call_sig)
            if not result.get("ok", False):
                self._tool_failure_counts[name] = (
                    self._tool_failure_counts.get(name, 0) + 1
                )
            else:
                self._tool_failure_counts[name] = 0  # Reset on success

        # Track telemetry
        if self.telemetry:
            self.telemetry.track_tool_call(
                tool_name=name,
                duration_ms=elapsed_ms,
                ok=result.get("ok", False),
                error=result.get("error") if not result.get("ok") else None,
                is_service=is_service,
            )

        return result

    @staticmethod
    def _is_process_tool(name: str) -> bool:
        """Check if a tool name is a process management tool."""
        from supyagent.core.process_tools import is_process_tool

        return is_process_tool(name)

    def _execute_supypowers_tool(self, tool_call: Any) -> dict[str, Any]:
        """Execute a supypowers tool through the supervisor."""
        name = tool_call.function.name

        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return {"ok": False, "error": f"Invalid JSON arguments: {tool_call.function.arguments}", "error_type": "invalid_args"}

        if "__" not in name:
            return {"ok": False, "error": f"Invalid tool name format: {name}", "error_type": "tool_not_found"}

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
            sandbox=self.sandbox_mgr,
            workspace_validator=self.workspace_validator,
        )

    def _execute_service_tool(self, tool_call: Any) -> dict[str, Any]:
        """Execute a service tool via HTTP through the ServiceClient."""
        name = tool_call.function.name
        metadata = self._service_tool_metadata.get(name, {})

        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return {
                "ok": False,
                "error": f"Invalid JSON arguments: {tool_call.function.arguments}",
                "error_type": "invalid_args",
            }

        if not self._service_client:
            return {
                "ok": False,
                "error": "Service client not available. Run 'supyagent connect' to authenticate.",
            }

        return self._service_client.execute_tool(name, args, metadata)

    @staticmethod
    def _is_browser_tool(name: str) -> bool:
        """Check if a tool name is a browser tool that should use the persistent daemon."""
        return name.startswith("browser__")

    def _execute_browser_tool(self, tool_call: Any) -> dict[str, Any]:
        """Execute a browser tool via the persistent browser daemon."""
        name = tool_call.function.name

        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return {
                "ok": False,
                "error": f"Invalid JSON arguments: {tool_call.function.arguments}",
                "error_type": "invalid_args",
            }

        # Extract command from tool name (browser__browse -> browse)
        command = name.split("__", 1)[1] if "__" in name else name

        # Lazy-initialize the browser session manager
        if self._browser_session is None:
            from supyagent.core.browser_session import BrowserSessionManager

            self._browser_session = BrowserSessionManager()

        return self._browser_session.execute(command, args)

    def _execute_memory_tool(self, tool_call: Any) -> dict[str, Any]:
        """Execute a memory tool (search, write, recall)."""
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return {"ok": False, "error": "Invalid JSON arguments", "error_type": "invalid_args"}

        if not self.memory_mgr:
            return {"ok": False, "error": "Memory system not available"}

        if name == "memory_search":
            query = args.get("query", "")
            limit = args.get("limit", 10)
            results = self.memory_mgr.search(query, limit=limit)
            formatted = []
            for r in results:
                if r["type"] == "entity":
                    entity = r["entity"]
                    edges = r.get("edges", [])
                    entry = {
                        "name": entity.name,
                        "type": entity.entity_type,
                        "summary": entity.summary,
                        "facts": [e.fact for e in edges],
                    }
                    formatted.append(entry)
                elif r["type"] == "edge":
                    formatted.append({"fact": r["edge"].fact, "relationship": r["edge"].relationship})
            return {"ok": True, "data": {"results": formatted, "count": len(formatted)}}

        elif name == "memory_write":
            from supyagent.core.memory import _now, _short_id
            from supyagent.models.memory import EntityEdge, EntityNode

            subject_name = args.get("subject", "")
            fact = args.get("fact", "")
            if not subject_name or not fact:
                return {"ok": False, "error": "subject and fact are required"}

            subject_type = args.get("subject_type", "Concept")
            now = _now()

            # Resolve or create subject entity
            subject = self.memory_mgr.get_entity_by_name(subject_name)
            if not subject:
                subject = EntityNode(
                    id=_short_id(),
                    name=subject_name,
                    entity_type=subject_type,
                    created_at=now,
                    updated_at=now,
                )
                self.memory_mgr.add_entity(subject)

            # If object entity specified, create edge
            object_name = args.get("object")
            if object_name:
                object_type = args.get("object_type", "Concept")
                obj = self.memory_mgr.get_entity_by_name(object_name)
                if not obj:
                    obj = EntityNode(
                        id=_short_id(),
                        name=object_name,
                        entity_type=object_type,
                        created_at=now,
                        updated_at=now,
                    )
                    self.memory_mgr.add_entity(obj)

                relationship = args.get("relationship", "relates_to")
                edge = EntityEdge(
                    id=_short_id(),
                    source_id=subject.id,
                    target_id=obj.id,
                    relationship=relationship,
                    fact=fact,
                    valid_from=now,
                    created_at=now,
                )
                self.memory_mgr.add_edge(edge)
            else:
                # Store as a self-referencing fact
                edge = EntityEdge(
                    id=_short_id(),
                    source_id=subject.id,
                    target_id=subject.id,
                    relationship=args.get("relationship", "relates_to"),
                    fact=fact,
                    valid_from=now,
                    created_at=now,
                )
                self.memory_mgr.add_edge(edge)

            return {"ok": True, "data": {"message": f"Stored: {fact}"}}

        elif name == "memory_recall":
            entity_name = args.get("entity_name", "")
            if not entity_name:
                return {"ok": False, "error": "entity_name is required"}

            entity = self.memory_mgr.get_entity_by_name(entity_name)
            if not entity:
                return {"ok": True, "data": {"message": f"No memories found for '{entity_name}'"}}

            edges = self.memory_mgr.get_edges_for_entity(entity.id)
            return {
                "ok": True,
                "data": {
                    "entity": {
                        "name": entity.name,
                        "type": entity.entity_type,
                        "summary": entity.summary,
                        "properties": entity.properties,
                    },
                    "facts": [
                        {"fact": e.fact, "relationship": e.relationship, "confidence": e.confidence}
                        for e in edges
                    ],
                },
            }

        return {"ok": False, "error": f"Unknown memory tool: {name}"}

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
        # Reset circuit breaker and repetition tracking for new turn
        self._tool_failure_counts.clear()
        self._recent_tool_calls.clear()

        iterations = 0
        content = ""

        try:
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

                # Fire all on_tool_start callbacks
                if on_tool_start:
                    for tc in tool_calls:
                        on_tool_start(
                            tc["id"], tc["function"]["name"], tc["function"]["arguments"]
                        )

                # Execute tool calls in parallel
                def _execute_single(tc: dict[str, Any]) -> tuple[dict, dict, str | list]:
                    tc_obj = ToolCallObj(
                        tc["id"], tc["function"]["name"], tc["function"]["arguments"]
                    )
                    result = self._dispatch_tool_call(tc_obj)
                    images = detect_images_in_tool_result(result)
                    if images:
                        tc_content: str | list[dict[str, Any]] = [
                            make_text_content_part(json.dumps(result))
                        ]
                        for img_path in images:
                            try:
                                tc_content.append(make_image_content_part(img_path))
                            except (FileNotFoundError, ValueError):
                                pass
                    else:
                        tc_content = json.dumps(result)
                    return tc, result, tc_content

                max_workers = min(len(tool_calls), 5)
                results_ordered: list[tuple[dict, dict, str | list]] = []

                if len(tool_calls) == 1:
                    # Fast path: no thread overhead for single tool call
                    results_ordered.append(_execute_single(tool_calls[0]))
                else:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {
                            executor.submit(_execute_single, tc): i
                            for i, tc in enumerate(tool_calls)
                        }
                        indexed: dict[int, tuple] = {}
                        for future in futures:
                            idx = futures[future]
                            indexed[idx] = future.result()
                        results_ordered = [indexed[i] for i in range(len(tool_calls))]

                # Append results in original order and fire callbacks
                for tc, result, tool_content in results_ordered:
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_content,
                    })
                    if on_tool_result:
                        on_tool_result(tc["id"], tc["function"]["name"], result)

        except KeyboardInterrupt:
            logger.info("Interrupted by user during tool loop")
            self._kill_managed_processes()
            raise

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
        # Reset circuit breaker and repetition tracking for new turn
        self._tool_failure_counts.clear()
        self._recent_tool_calls.clear()

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

            # Yield all tool_start events
            for tc_dict in collected_tool_calls:
                yield (
                    "tool_start",
                    {
                        "id": tc_dict["id"],
                        "name": tc_dict["function"]["name"],
                        "arguments": tc_dict["function"]["arguments"],
                    },
                )

            # Execute tool calls in parallel
            def _exec_stream(tcd: dict[str, Any]) -> tuple[dict, dict, str | list]:
                tc_obj = ToolCallObj(
                    tcd["id"], tcd["function"]["name"], tcd["function"]["arguments"]
                )
                res = self._dispatch_tool_call(tc_obj)
                imgs = detect_images_in_tool_result(res)
                if imgs:
                    tc_cont: str | list[dict[str, Any]] = [
                        make_text_content_part(json.dumps(res))
                    ]
                    for ip in imgs:
                        try:
                            tc_cont.append(make_image_content_part(ip))
                        except (FileNotFoundError, ValueError):
                            pass
                else:
                    tc_cont = json.dumps(res)
                return tcd, res, tc_cont

            max_w = min(len(collected_tool_calls), 5)
            stream_results: list[tuple[dict, dict, str | list]] = []

            if len(collected_tool_calls) == 1:
                stream_results.append(_exec_stream(collected_tool_calls[0]))
            else:
                with ThreadPoolExecutor(max_workers=max_w) as executor:
                    futs = {
                        executor.submit(_exec_stream, tcd): i
                        for i, tcd in enumerate(collected_tool_calls)
                    }
                    idx_map: dict[int, tuple] = {}
                    for fut in futs:
                        idx_map[futs[fut]] = fut.result()
                    stream_results = [idx_map[i] for i in range(len(collected_tool_calls))]

            # Yield tool_end events and append messages in order
            for tc_dict, result, tool_content in stream_results:
                tool_id = tc_dict["id"]
                tool_name = tc_dict["function"]["name"]

                yield ("tool_end", {"id": tool_id, "name": tool_name, "result": result})

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": tool_content,
                })

                # Internal hook for subclass persistence
                yield ("_tool_result", (tool_id, tool_name, result))

        yield ("done", final_content)

    def _kill_managed_processes(self) -> None:
        """Kill all running managed processes (cleanup on interrupt)."""
        try:
            from supyagent.core.supervisor import get_supervisor, run_supervisor_coroutine

            supervisor = get_supervisor()
            running = supervisor.list_processes(include_completed=False)
            for proc in running:
                try:
                    run_supervisor_coroutine(supervisor.kill(proc["process_id"]))
                except Exception:
                    pass
        except Exception:
            pass

    def get_available_tools(self) -> list[str]:
        """Get list of available tool names."""
        return [tool.get("function", {}).get("name", "unknown") for tool in self.tools]

    def get_tools_detail(self) -> list[dict[str, str]]:
        """Get detailed tool info for display: name, description, source."""
        details = []
        for tool in self.tools:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            source = self._tool_sources.get(name, "native")
            details.append({"name": name, "description": desc, "source": source})
        return details
