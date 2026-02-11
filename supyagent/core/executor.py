"""
Execution runner for non-interactive agent execution.

Extends BaseAgentEngine for stateless, input->output pipelines designed for automation.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from supyagent.core.credentials import CredentialManager
from supyagent.core.engine import BaseAgentEngine, MaxIterationsError
from supyagent.models.agent_config import AgentConfig, get_full_system_prompt

if TYPE_CHECKING:
    from supyagent.core.sandbox import SandboxManager

# Type for progress callback: (event_type, data)
# event_type: "tool_start", "tool_end", "thinking", "streaming", "reasoning"
ProgressCallback = Callable[[str, dict[str, Any]], None]


class CredentialRequiredError(Exception):
    """Raised when a credential is needed but not available in execution mode."""

    pass


class ExecutionRunner(BaseAgentEngine):
    """
    Runs agents in non-interactive execution mode.

    Key differences from interactive mode:
    - No session persistence
    - No credential prompting (must be pre-provided)
    - Single input -> output execution
    - Designed for automation and pipelines
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
        self._goal_driven: bool = False

        # Set up delegation if this agent has delegates
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

        # Load available tools (no credential request tool in execution mode)
        # Must be after sandbox init so discover_tools runs inside container
        self.tools = self._load_tools()

    def _load_tools(self) -> list[dict[str, Any]]:
        """
        Load tools for execution mode.

        Note: Does NOT include request_credential tool since
        execution mode cannot prompt for credentials.
        """
        if not self.config.tools.allow:
            # No tools allowed, only process management tools
            from supyagent.core.process_tools import get_process_management_tools

            tools: list[dict[str, Any]] = []
            if self.delegation_mgr:
                tools.extend(self.delegation_mgr.get_delegation_tools())
            tools.extend(get_process_management_tools())
            return tools

        return self._load_base_tools()

    def _get_secrets(self) -> dict[str, str]:
        """Get secrets for the current run."""
        return self._run_secrets

    def _dispatch_tool_call(self, tool_call: Any) -> dict[str, Any]:
        """Fail on credential requests, delegate rest to base."""
        if tool_call.function.name == "request_credential":
            try:
                args = json.loads(tool_call.function.arguments)
                cred_name = args.get("name", "unknown")
            except json.JSONDecodeError:
                cred_name = "unknown"
            raise CredentialRequiredError(
                f"Credential '{cred_name}' required but not provided.\n\n"
                f"Fix: supyagent run <agent> \"task\" --secrets {cred_name}=<value>\n"
                f" or: supyagent run <agent> \"task\" --secrets .env\n\n"
                f"You can also store it globally:\n"
                f"  supyagent config set {cred_name}"
            )
        return super()._dispatch_tool_call(tool_call)

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
        self._run_secrets = self.credential_manager.get_all_for_tools(self.config.name)
        if secrets:
            self._run_secrets.update(secrets)

        # Inject secrets into environment for this execution
        for key, value in self._run_secrets.items():
            os.environ[key] = value

        tool_calls_completed = 0

        try:
            # Build the prompt
            if isinstance(task, dict):
                user_content = self._format_structured_input(task)
            else:
                user_content = str(task)

            self.messages = [
                {"role": "system", "content": get_full_system_prompt(self.config, **self._system_prompt_kwargs())},
                {"role": "user", "content": user_content},
            ]

            max_iterations = self.config.limits.get("max_tool_calls_per_turn", 100)

            if stream and on_progress:
                return self._run_streaming(max_iterations, output_format, on_progress)

            # Non-streaming with optional progress callbacks
            on_tool_start_cb = None
            on_tool_result_cb = None
            if on_progress:

                def on_tool_start_cb(
                    tool_call_id: str, name: str, arguments: str
                ) -> None:
                    on_progress("tool_start", {"name": name, "arguments": arguments})

                def on_tool_result_cb(
                    tool_call_id: str, name: str, result: dict[str, Any]
                ) -> None:
                    nonlocal tool_calls_completed
                    tool_calls_completed += 1
                    on_progress("tool_end", {"name": name, "result": result})

            else:
                # Still track tool calls even without progress callback
                def on_tool_result_cb(
                    tool_call_id: str, name: str, result: dict[str, Any]
                ) -> None:
                    nonlocal tool_calls_completed
                    tool_calls_completed += 1

            content = self._run_loop(
                max_iterations,
                on_tool_start=on_tool_start_cb,
                on_tool_result=on_tool_result_cb,
            )
            return self._format_output(content, output_format)

        except CredentialRequiredError as e:
            return {"ok": False, "error": str(e)}
        except MaxIterationsError:
            if tool_calls_completed > 0:
                return {
                    "ok": False,
                    "error": "Max tool iterations exceeded",
                    "partial": True,
                    "tool_calls_completed": tool_calls_completed,
                }
            return {"ok": False, "error": "Max tool iterations exceeded"}
        except Exception as e:
            if tool_calls_completed > 0:
                return {
                    "ok": False,
                    "error": str(e),
                    "partial": True,
                    "tool_calls_completed": tool_calls_completed,
                }
            return {"ok": False, "error": str(e)}

    def _run_streaming(
        self,
        max_iterations: int,
        output_format: str,
        on_progress: ProgressCallback,
    ) -> dict[str, Any]:
        """Run with streaming, translating engine events to progress callbacks."""
        final_content = ""

        try:
            for event_type, data in self._run_loop_stream(max_iterations):
                if event_type == "text":
                    on_progress("streaming", {"content": data})
                elif event_type == "reasoning":
                    on_progress("reasoning", {"content": data})
                elif event_type == "tool_start":
                    on_progress("tool_start", data)
                elif event_type == "tool_end":
                    on_progress("tool_end", data)
                elif event_type == "done":
                    final_content = data
                # Skip _message and _tool_result (no persistence in execution mode)
        except CredentialRequiredError as e:
            return {"ok": False, "error": str(e)}

        return self._format_output(final_content, output_format)

    def _format_structured_input(self, task: dict[str, Any]) -> str:
        """Format a structured input dict into a prompt."""
        return json.dumps(task, indent=2)

    def _format_goal_driven_message(self, task: str) -> str:
        """Format a goal-driven user message, optionally incorporating the task."""
        if task.strip():
            return (
                "Start working toward the workspace goals. "
                f"Your immediate focus: {task}\n\n"
                "Read GOALS.md, break down what needs to happen into subgoals, "
                "and begin executing. Keep working until the goals are met or you are blocked."
            )
        return (
            "Start working toward the workspace goals. "
            "Read GOALS.md, break down what needs to happen into subgoals, "
            "and begin executing. Keep working until all goals are met or you are blocked."
        )

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
