"""
Workflow orchestrator for multi-agent task execution.

Runs YAML-defined workflows where steps map to agents, with
dependency resolution and variable passing between steps.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from supyagent.core.config import load_config
from supyagent.core.executor import ExecutionRunner
from supyagent.models.agent_config import AgentNotFoundError, load_agent_config

logger = logging.getLogger(__name__)


class WorkflowStep:
    """A single step in a workflow."""

    def __init__(
        self,
        agent: str,
        task: str,
        output: str | None = None,
        depends_on: list[str] | None = None,
    ):
        self.agent = agent
        self.task = task
        self.output = output
        self.depends_on = depends_on or []

    def resolve_task(self, outputs: dict[str, str]) -> str:
        """Replace {{variable}} placeholders in the task with outputs from prior steps."""
        resolved = self.task
        for key, value in outputs.items():
            resolved = resolved.replace(f"{{{{{key}}}}}", value)
        return resolved


class Workflow:
    """A multi-step agent workflow defined in YAML."""

    def __init__(self, name: str, steps: list[WorkflowStep]):
        self.name = name
        self.steps = steps

    @classmethod
    def from_file(cls, path: Path) -> Workflow:
        """Load a workflow from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        if not data or not isinstance(data, dict):
            raise ValueError(f"Invalid workflow file: {path}")

        name = data.get("name", path.stem)
        raw_steps = data.get("steps", [])

        if not raw_steps:
            raise ValueError(f"Workflow '{name}' has no steps")

        steps = []
        for i, step_data in enumerate(raw_steps):
            if not isinstance(step_data, dict):
                raise ValueError(f"Step {i} must be a dict")
            if "agent" not in step_data or "task" not in step_data:
                raise ValueError(f"Step {i} missing 'agent' or 'task'")

            steps.append(WorkflowStep(
                agent=step_data["agent"],
                task=step_data["task"],
                output=step_data.get("output"),
                depends_on=step_data.get("depends_on", []),
            ))

        return cls(name=name, steps=steps)

    def validate(self) -> list[str]:
        """Validate the workflow. Returns list of issues."""
        issues: list[str] = []

        # Check agents exist
        for i, step in enumerate(self.steps):
            try:
                load_agent_config(step.agent)
            except AgentNotFoundError:
                issues.append(f"Step {i}: agent '{step.agent}' not found")

        # Check dependencies reference valid outputs
        defined_outputs = {s.output for s in self.steps if s.output}
        for i, step in enumerate(self.steps):
            for dep in step.depends_on:
                if dep not in defined_outputs:
                    issues.append(
                        f"Step {i}: depends_on '{dep}' not defined by any step"
                    )

        # Check for template variables that reference valid outputs
        for i, step in enumerate(self.steps):
            placeholders = re.findall(r"\{\{(\w+)\}\}", step.task)
            for ph in placeholders:
                if ph not in defined_outputs:
                    issues.append(
                        f"Step {i}: task references '{{{{{ph}}}}}' but "
                        f"no step outputs '{ph}'"
                    )

        return issues


def run_workflow(
    workflow: Workflow,
    on_step_start: Any | None = None,
    on_step_end: Any | None = None,
) -> dict[str, Any]:
    """
    Execute a workflow sequentially.

    Args:
        workflow: The workflow to execute
        on_step_start: Callback(step_index, agent_name, task)
        on_step_end: Callback(step_index, agent_name, result)

    Returns:
        Dict with outputs from each step
    """
    # Load global config (API keys)
    load_config()

    outputs: dict[str, str] = {}
    results: list[dict[str, Any]] = []

    for i, step in enumerate(workflow.steps):
        # Check dependencies are satisfied
        for dep in step.depends_on:
            if dep not in outputs:
                return {
                    "ok": False,
                    "error": f"Step {i} ({step.agent}): dependency '{dep}' not satisfied",
                    "results": results,
                }

        # Resolve task template
        resolved_task = step.resolve_task(outputs)

        if on_step_start:
            on_step_start(i, step.agent, resolved_task)

        # Load and run agent
        try:
            config = load_agent_config(step.agent)
            runner = ExecutionRunner(config)
            result = runner.run(resolved_task, output_format="json")
        except Exception as e:
            result = {"ok": False, "error": str(e)}

        results.append({
            "step": i,
            "agent": step.agent,
            "task": resolved_task,
            "result": result,
        })

        if on_step_end:
            on_step_end(i, step.agent, result)

        # Store output if step defines one
        if step.output and result.get("ok"):
            output_data = result.get("data", "")
            if isinstance(output_data, dict):
                output_data = json.dumps(output_data)
            outputs[step.output] = str(output_data)

        # Stop on failure
        if not result.get("ok"):
            return {
                "ok": False,
                "error": f"Step {i} ({step.agent}) failed: {result.get('error', 'unknown')}",
                "results": results,
                "outputs": outputs,
            }

    return {"ok": True, "results": results, "outputs": outputs}
