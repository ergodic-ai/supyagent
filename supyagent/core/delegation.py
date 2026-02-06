"""
Delegation Manager for agent-to-agent task delegation.

Enables agents to invoke other agents (delegates) to perform subtasks,
supporting multi-agent orchestration patterns.

Supports two execution modes:
- subprocess: Child agents run in isolated processes (default, safer)
- in_process: Child agents run in the same process (faster, shared memory)
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from supyagent.core.context import DelegationContext, summarize_conversation
from supyagent.core.registry import AgentRegistry
from supyagent.models.agent_config import AgentNotFoundError, load_agent_config

if TYPE_CHECKING:
    from supyagent.core.agent import Agent


class DelegationManager:
    """
    Manages agent-to-agent delegation.

    Provides tools that allow a parent agent to delegate tasks to
    child agents (delegates), with proper context passing.

    Supports both subprocess (isolated) and in-process execution modes.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        parent_agent: "Agent",
        grandparent_instance_id: str | None = None,
    ):
        """
        Initialize the delegation manager.

        Args:
            registry: Agent registry for tracking instances
            parent_agent: The parent agent that will do the delegating
            grandparent_instance_id: Instance ID of the agent that created this one (if any)
        """
        self.registry = registry
        self.parent = parent_agent

        # Always register this agent, optionally with a grandparent
        self.parent_id = registry.register(parent_agent, parent_id=grandparent_instance_id)

    def get_delegation_tools(self) -> list[dict[str, Any]]:
        """
        Generate tool schemas for each delegatable agent.

        Returns:
            List of OpenAI-format tool definitions
        """
        tools: list[dict[str, Any]] = []

        # Get delegate configurations
        for delegate_name in self.parent.config.delegates:
            try:
                delegate_config = load_agent_config(delegate_name)
            except AgentNotFoundError:
                continue

            tool = {
                "type": "function",
                "function": {
                    "name": f"delegate_to_{delegate_name}",
                    "description": (
                        f"Delegate a task to the {delegate_name} agent. "
                        f"{delegate_config.description}"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "The task to delegate to this agent",
                            },
                            "context": {
                                "type": "string",
                                "description": (
                                    "Optional context from the current conversation "
                                    "to pass along"
                                ),
                            },
                            "mode": {
                                "type": "string",
                                "enum": ["subprocess", "in_process"],
                                "default": "subprocess",
                                "description": (
                                    "Execution mode: 'subprocess' for isolation (default), "
                                    "'in_process' for faster execution"
                                ),
                            },
                            "background": {
                                "type": "boolean",
                                "default": False,
                                "description": (
                                    "If true, returns immediately without waiting. "
                                    "Use for long-running tasks you don't need to wait for."
                                ),
                            },
                            "timeout": {
                                "type": "integer",
                                "default": 300,
                                "description": "Max seconds to wait before backgrounding (subprocess mode)",
                            },
                        },
                        "required": ["task"],
                    },
                },
            }
            tools.append(tool)

        # Add generic spawn tool
        if self.parent.config.delegates:
            tools.append({
                "type": "function",
                "function": {
                    "name": "spawn_agent",
                    "description": (
                        "Create and run a new agent instance for a specific task. "
                        f"Available agents: {', '.join(self.parent.config.delegates)}"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "agent_type": {
                                "type": "string",
                                "description": "The type of agent to spawn",
                                "enum": self.parent.config.delegates,
                            },
                            "task": {
                                "type": "string",
                                "description": "The task for the agent to perform",
                            },
                            "background": {
                                "type": "boolean",
                                "default": False,
                                "description": "Run in background without waiting",
                            },
                        },
                        "required": ["agent_type", "task"],
                    },
                },
            })

        return tools

    def is_delegation_tool(self, tool_name: str) -> bool:
        """Check if a tool name is a delegation tool."""
        return (
            tool_name.startswith("delegate_to_")
            or tool_name == "spawn_agent"
        )

    def execute_delegation(self, tool_call: Any) -> dict[str, Any]:
        """
        Execute a delegation tool call.

        Args:
            tool_call: The tool call from the LLM

        Returns:
            Result dict with ok/data or ok/error
        """
        name = tool_call.function.name

        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return {"ok": False, "error": "Invalid JSON in tool arguments"}

        if name == "spawn_agent":
            return self._delegate_task(
                args.get("agent_type", ""),
                args.get("task", ""),
                mode="subprocess",
                background=args.get("background", False),
            )

        if name.startswith("delegate_to_"):
            agent_name = name[len("delegate_to_"):]
            return self._delegate_task(
                agent_name,
                args.get("task", ""),
                extra_context=args.get("context"),
                mode=args.get("mode", "subprocess"),
                background=args.get("background", False),
                timeout=args.get("timeout", 300),
            )

        return {"ok": False, "error": f"Unknown delegation tool: {name}"}

    def _build_context(
        self,
        task: str,
        extra_context: str | None = None,
    ) -> DelegationContext:
        """Build context to pass to a delegate."""
        # Get conversation summary if we have enough messages to justify
        # the extra LLM call. For short conversations (≤ 4 non-system messages),
        # skip summarization to avoid adding unnecessary latency.
        summary = None
        if hasattr(self.parent, "messages") and self.parent.messages:
            non_system = [
                m for m in self.parent.messages if m.get("role") != "system"
            ]
            if len(non_system) > 4:
                summary = summarize_conversation(
                    self.parent.messages,
                    self.parent.llm,
                )

        context = DelegationContext(
            parent_agent=self.parent.config.name,
            parent_task=task,
            conversation_summary=summary,
        )

        if extra_context:
            context.relevant_facts.append(extra_context)

        return context

    def _delegate_task(
        self,
        agent_name: str,
        task: str,
        extra_context: str | None = None,
        mode: str = "subprocess",
        background: bool = False,
        timeout: float = 300,
    ) -> dict[str, Any]:
        """
        Delegate a task to another agent.

        Args:
            agent_name: Name of the agent to delegate to
            task: The task to perform
            extra_context: Optional additional context
            mode: "subprocess" (isolated) or "in_process" (shared memory)
            background: Return immediately without waiting (subprocess mode only)
            timeout: Seconds before auto-backgrounding (subprocess mode)

        Returns:
            Result dict
        """
        # Verify it's in the delegates list
        if agent_name not in self.parent.config.delegates:
            return {
                "ok": False,
                "error": f"Agent '{agent_name}' is not in the delegates list",
            }

        # Load the delegate config
        try:
            config = load_agent_config(agent_name)
        except AgentNotFoundError:
            return {"ok": False, "error": f"Agent '{agent_name}' not found"}

        # Check delegation depth
        parent_depth = self.registry.get_depth(self.parent_id)
        if parent_depth >= AgentRegistry.MAX_DEPTH:
            return {
                "ok": False,
                "error": (
                    f"Maximum delegation depth ({AgentRegistry.MAX_DEPTH}) reached. "
                    "Cannot delegate further."
                ),
            }

        # Build context
        context = self._build_context(task, extra_context)
        full_task = f"{context.to_prompt()}\n\n---\n\nYour task:\n{task}"

        if mode == "subprocess":
            return self._delegate_subprocess(
                agent_name, full_task, context, background, timeout
            )
        else:
            return self._delegate_in_process(agent_name, config, full_task)

    def _delegate_subprocess(
        self,
        agent_name: str,
        full_task: str,
        context: DelegationContext,
        background: bool,
        timeout: float,
    ) -> dict[str, Any]:
        """Delegate via subprocess using the ProcessSupervisor."""
        from supyagent.core.supervisor import get_supervisor, run_supervisor_coroutine

        # Calculate child depth: current cross-process depth + in-process depth + 1
        current_depth = int(os.environ.get("SUPYAGENT_DELEGATION_DEPTH", "0"))
        child_depth = current_depth + self.registry.get_depth(self.parent_id) + 1

        cmd = [
            "supyagent", "exec", agent_name,
            "--task", full_task,
            "--context", json.dumps({
                "parent_agent": context.parent_agent,
                "parent_task": context.parent_task,
                "conversation_summary": context.conversation_summary,
                "relevant_facts": context.relevant_facts,
            }),
            "--depth", str(child_depth),
            "--output", "json",
        ]

        supervisor = get_supervisor()

        try:
            return run_supervisor_coroutine(
                supervisor.execute(
                    cmd,
                    process_type="agent",
                    tool_name=f"agent__{agent_name}",
                    timeout=timeout,
                    force_background=background,
                    metadata={"agent_name": agent_name, "task": full_task[:200]},
                )
            )
        except Exception as e:
            return {"ok": False, "error": f"Subprocess delegation failed: {e}"}

    def _delegate_in_process(
        self,
        agent_name: str,
        config,
        full_task: str,
    ) -> dict[str, Any]:
        """Delegate in-process (original behavior)."""
        try:
            if config.type == "execution":
                from supyagent.core.executor import ExecutionRunner

                runner = ExecutionRunner(config)
                return runner.run(full_task, output_format="json")
            else:
                from supyagent.core.agent import Agent

                sub_agent = Agent(
                    config,
                    registry=self.registry,
                    parent_instance_id=self.parent_id,
                )

                response = sub_agent.send_message(full_task)

                # Mark as completed
                if sub_agent.instance_id:
                    self.registry.mark_completed(sub_agent.instance_id)

                return {"ok": True, "data": response}

        except Exception as e:
            return {"ok": False, "error": f"Delegation failed: {str(e)}"}

    def _spawn_agent(
        self,
        agent_type: str,
        task: str,
    ) -> dict[str, Any]:
        """
        Spawn a new agent instance.

        Args:
            agent_type: Type of agent to spawn
            task: Initial task for the agent

        Returns:
            Result dict
        """
        if agent_type not in self.parent.config.delegates:
            return {
                "ok": False,
                "error": (
                    f"Cannot spawn '{agent_type}' - not in delegates list. "
                    f"Available: {', '.join(self.parent.config.delegates)}"
                ),
            }

        return self._delegate_task(agent_type, task)
