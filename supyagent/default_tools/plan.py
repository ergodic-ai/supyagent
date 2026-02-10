# /// script
# dependencies = ["pydantic"]
# ///
"""
Planning tools.

Allows agents to create, track, and update structured plans with steps.
Plans are stored as JSON files in .supyagent/plans/ within the working directory.
"""

import json
import os
import uuid
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

# =============================================================================
# Create Plan
# =============================================================================


class PlanStep(BaseModel):
    """A single step in a plan."""

    description: str = Field(description="Description of this step")
    status: str = Field(default="pending", description="Status: pending, in_progress, completed, skipped")
    notes: str = Field(default="", description="Optional notes for this step")


class CreatePlanInput(BaseModel):
    """Input for create_plan function."""

    title: str = Field(description="Title of the plan")
    steps: List[PlanStep] = Field(description="List of plan steps")
    working_dir: str = Field(default=".", description="Working directory for storing the plan")


class CreatePlanOutput(BaseModel):
    """Output for create_plan function."""

    ok: bool
    plan_id: Optional[str] = None
    title: Optional[str] = None
    steps_count: int = 0
    error: Optional[str] = None


def create_plan(input: CreatePlanInput) -> CreatePlanOutput:
    """
    Create a new plan with a list of steps.

    Plans are stored as JSON files in .supyagent/plans/.

    Examples:
        >>> create_plan({"title": "Deploy to production", "steps": [
        ...     {"description": "Run tests"},
        ...     {"description": "Build Docker image"},
        ...     {"description": "Push to registry"},
        ...     {"description": "Deploy to cluster"}
        ... ]})
    """
    try:
        working_dir = os.path.expanduser(input.working_dir)
        plans_dir = Path(working_dir) / ".supyagent" / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)

        plan_id = uuid.uuid4().hex[:8]

        plan_data = {
            "plan_id": plan_id,
            "title": input.title,
            "steps": [step.model_dump() for step in input.steps],
        }

        plan_path = plans_dir / f"{plan_id}.json"
        plan_path.write_text(json.dumps(plan_data, indent=2))

        return CreatePlanOutput(
            ok=True,
            plan_id=plan_id,
            title=input.title,
            steps_count=len(input.steps),
        )

    except Exception as e:
        return CreatePlanOutput(ok=False, error=str(e))


# =============================================================================
# Update Plan
# =============================================================================


class UpdatePlanInput(BaseModel):
    """Input for update_plan function."""

    plan_id: str = Field(description="ID of the plan to update")
    step_index: int = Field(description="Zero-based index of the step to update")
    status: str = Field(
        default="completed",
        description="New status: pending, in_progress, completed, skipped",
    )
    notes: Optional[str] = Field(default=None, description="Optional notes to add to the step")
    working_dir: str = Field(default=".", description="Working directory where the plan is stored")


class UpdatePlanOutput(BaseModel):
    """Output for update_plan function."""

    ok: bool
    plan_id: Optional[str] = None
    step_description: Optional[str] = None
    new_status: Optional[str] = None
    error: Optional[str] = None


def update_plan(input: UpdatePlanInput) -> UpdatePlanOutput:
    """
    Update the status or notes of a specific step in a plan.

    Examples:
        >>> update_plan({"plan_id": "abc12345", "step_index": 0, "status": "completed"})
        >>> update_plan({"plan_id": "abc12345", "step_index": 1, "status": "in_progress", "notes": "Started"})
    """
    try:
        working_dir = os.path.expanduser(input.working_dir)
        plan_path = Path(working_dir) / ".supyagent" / "plans" / f"{input.plan_id}.json"

        if not plan_path.exists():
            return UpdatePlanOutput(
                ok=False, error=f"Plan not found: {input.plan_id}"
            )

        plan_data = json.loads(plan_path.read_text())

        steps = plan_data.get("steps", [])
        if input.step_index < 0 or input.step_index >= len(steps):
            return UpdatePlanOutput(
                ok=False,
                error=f"Step index {input.step_index} out of range (0-{len(steps) - 1})",
            )

        step = steps[input.step_index]
        step["status"] = input.status
        if input.notes is not None:
            step["notes"] = input.notes

        plan_path.write_text(json.dumps(plan_data, indent=2))

        return UpdatePlanOutput(
            ok=True,
            plan_id=input.plan_id,
            step_description=step["description"],
            new_status=input.status,
        )

    except Exception as e:
        return UpdatePlanOutput(ok=False, error=str(e))


# =============================================================================
# Get Plan
# =============================================================================


class GetPlanInput(BaseModel):
    """Input for get_plan function."""

    plan_id: Optional[str] = Field(
        default=None, description="ID of the plan to retrieve. If omitted, returns the most recent plan."
    )
    working_dir: str = Field(default=".", description="Working directory where plans are stored")


class PlanStepStatus(BaseModel):
    """A step with its current status."""

    index: int
    description: str
    status: str
    notes: str = ""


class GetPlanOutput(BaseModel):
    """Output for get_plan function."""

    ok: bool
    plan_id: Optional[str] = None
    title: Optional[str] = None
    steps: List[PlanStepStatus] = []
    completed_count: int = 0
    total_count: int = 0
    error: Optional[str] = None


def get_plan(input: GetPlanInput) -> GetPlanOutput:
    """
    Retrieve a plan by ID, or get the most recent plan if no ID is given.

    Examples:
        >>> get_plan({"plan_id": "abc12345"})
        >>> get_plan({})  # gets most recent plan
    """
    try:
        working_dir = os.path.expanduser(input.working_dir)
        plans_dir = Path(working_dir) / ".supyagent" / "plans"

        if input.plan_id:
            plan_path = plans_dir / f"{input.plan_id}.json"
            if not plan_path.exists():
                return GetPlanOutput(
                    ok=False, error=f"Plan not found: {input.plan_id}"
                )
        else:
            if not plans_dir.exists():
                return GetPlanOutput(ok=False, error="No plans directory found")
            # Get most recent plan by modification time
            plan_files = list(plans_dir.glob("*.json"))
            if not plan_files:
                return GetPlanOutput(ok=False, error="No plans found")

            plan_path = max(plan_files, key=lambda p: p.stat().st_mtime)

        plan_data = json.loads(plan_path.read_text())

        steps = []
        completed_count = 0
        raw_steps = plan_data.get("steps", [])

        for i, step in enumerate(raw_steps):
            status = step.get("status", "pending")
            if status == "completed":
                completed_count += 1
            steps.append(
                PlanStepStatus(
                    index=i,
                    description=step.get("description", ""),
                    status=status,
                    notes=step.get("notes", ""),
                )
            )

        return GetPlanOutput(
            ok=True,
            plan_id=plan_data.get("plan_id"),
            title=plan_data.get("title"),
            steps=steps,
            completed_count=completed_count,
            total_count=len(raw_steps),
        )

    except Exception as e:
        return GetPlanOutput(ok=False, error=str(e))
