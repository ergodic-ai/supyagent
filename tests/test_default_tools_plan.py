"""
Comprehensive tests for default_tools/plan.py.

Covers: create_plan, update_plan, get_plan
"""

import json
import time

import pytest

from supyagent.default_tools.plan import (
    CreatePlanInput,
    GetPlanInput,
    PlanStep,
    UpdatePlanInput,
    create_plan,
    get_plan,
    update_plan,
)


@pytest.fixture
def workspace(tmp_path):
    """Provide a clean working directory for plan tests."""
    return tmp_path


# =========================================================================
# create_plan
# =========================================================================


class TestCreatePlan:
    def test_create_basic_plan(self, workspace):
        result = create_plan(
            CreatePlanInput(
                title="Deploy app",
                steps=[
                    PlanStep(description="Run tests"),
                    PlanStep(description="Build image"),
                    PlanStep(description="Deploy"),
                ],
                working_dir=str(workspace),
            )
        )
        assert result.ok is True
        assert result.plan_id is not None
        assert len(result.plan_id) == 8
        assert result.title == "Deploy app"
        assert result.steps_count == 3

    def test_create_plan_writes_json(self, workspace):
        result = create_plan(
            CreatePlanInput(
                title="Test plan",
                steps=[PlanStep(description="Step 1")],
                working_dir=str(workspace),
            )
        )
        plan_path = workspace / ".supyagent" / "plans" / f"{result.plan_id}.json"
        assert plan_path.exists()

        data = json.loads(plan_path.read_text())
        assert data["title"] == "Test plan"
        assert len(data["steps"]) == 1
        assert data["steps"][0]["description"] == "Step 1"
        assert data["steps"][0]["status"] == "pending"

    def test_create_plan_with_initial_status(self, workspace):
        result = create_plan(
            CreatePlanInput(
                title="Partial plan",
                steps=[
                    PlanStep(description="Already done", status="completed"),
                    PlanStep(description="Not yet"),
                ],
                working_dir=str(workspace),
            )
        )
        assert result.ok is True

        plan_path = workspace / ".supyagent" / "plans" / f"{result.plan_id}.json"
        data = json.loads(plan_path.read_text())
        assert data["steps"][0]["status"] == "completed"
        assert data["steps"][1]["status"] == "pending"

    def test_create_plan_empty_steps(self, workspace):
        result = create_plan(
            CreatePlanInput(
                title="Empty plan",
                steps=[],
                working_dir=str(workspace),
            )
        )
        assert result.ok is True
        assert result.steps_count == 0

    def test_create_multiple_plans(self, workspace):
        ids = set()
        for i in range(3):
            result = create_plan(
                CreatePlanInput(
                    title=f"Plan {i}",
                    steps=[PlanStep(description=f"Step for plan {i}")],
                    working_dir=str(workspace),
                )
            )
            assert result.ok is True
            ids.add(result.plan_id)

        # All IDs should be unique
        assert len(ids) == 3


# =========================================================================
# update_plan
# =========================================================================


class TestUpdatePlan:
    def test_update_step_status(self, workspace):
        create_result = create_plan(
            CreatePlanInput(
                title="Update test",
                steps=[
                    PlanStep(description="First step"),
                    PlanStep(description="Second step"),
                ],
                working_dir=str(workspace),
            )
        )

        result = update_plan(
            UpdatePlanInput(
                plan_id=create_result.plan_id,
                step_index=0,
                status="completed",
                working_dir=str(workspace),
            )
        )
        assert result.ok is True
        assert result.plan_id == create_result.plan_id
        assert result.step_description == "First step"
        assert result.new_status == "completed"

    def test_update_step_with_notes(self, workspace):
        create_result = create_plan(
            CreatePlanInput(
                title="Notes test",
                steps=[PlanStep(description="Do something")],
                working_dir=str(workspace),
            )
        )

        result = update_plan(
            UpdatePlanInput(
                plan_id=create_result.plan_id,
                step_index=0,
                status="in_progress",
                notes="Started working on this",
                working_dir=str(workspace),
            )
        )
        assert result.ok is True

        # Verify notes were saved
        plan_path = workspace / ".supyagent" / "plans" / f"{create_result.plan_id}.json"
        data = json.loads(plan_path.read_text())
        assert data["steps"][0]["notes"] == "Started working on this"
        assert data["steps"][0]["status"] == "in_progress"

    def test_update_nonexistent_plan(self, workspace):
        result = update_plan(
            UpdatePlanInput(
                plan_id="nonexist",
                step_index=0,
                status="completed",
                working_dir=str(workspace),
            )
        )
        assert result.ok is False
        assert "not found" in result.error.lower()

    def test_update_step_out_of_range(self, workspace):
        create_result = create_plan(
            CreatePlanInput(
                title="Range test",
                steps=[PlanStep(description="Only step")],
                working_dir=str(workspace),
            )
        )

        result = update_plan(
            UpdatePlanInput(
                plan_id=create_result.plan_id,
                step_index=5,
                status="completed",
                working_dir=str(workspace),
            )
        )
        assert result.ok is False
        assert "out of range" in result.error.lower()

    def test_update_persists_changes(self, workspace):
        create_result = create_plan(
            CreatePlanInput(
                title="Persist test",
                steps=[
                    PlanStep(description="Step A"),
                    PlanStep(description="Step B"),
                ],
                working_dir=str(workspace),
            )
        )

        # Update step 0
        update_plan(
            UpdatePlanInput(
                plan_id=create_result.plan_id,
                step_index=0,
                status="completed",
                working_dir=str(workspace),
            )
        )

        # Update step 1
        update_plan(
            UpdatePlanInput(
                plan_id=create_result.plan_id,
                step_index=1,
                status="in_progress",
                working_dir=str(workspace),
            )
        )

        # Read back and verify both updates persisted
        get_result = get_plan(
            GetPlanInput(plan_id=create_result.plan_id, working_dir=str(workspace))
        )
        assert get_result.ok is True
        assert get_result.steps[0].status == "completed"
        assert get_result.steps[1].status == "in_progress"


# =========================================================================
# get_plan
# =========================================================================


class TestGetPlan:
    def test_get_plan_by_id(self, workspace):
        create_result = create_plan(
            CreatePlanInput(
                title="Retrievable plan",
                steps=[
                    PlanStep(description="Step 1"),
                    PlanStep(description="Step 2"),
                    PlanStep(description="Step 3"),
                ],
                working_dir=str(workspace),
            )
        )

        result = get_plan(
            GetPlanInput(plan_id=create_result.plan_id, working_dir=str(workspace))
        )
        assert result.ok is True
        assert result.plan_id == create_result.plan_id
        assert result.title == "Retrievable plan"
        assert len(result.steps) == 3
        assert result.total_count == 3
        assert result.completed_count == 0

    def test_get_plan_with_completed_steps(self, workspace):
        create_result = create_plan(
            CreatePlanInput(
                title="Progress plan",
                steps=[
                    PlanStep(description="Done", status="completed"),
                    PlanStep(description="Also done", status="completed"),
                    PlanStep(description="Not done"),
                ],
                working_dir=str(workspace),
            )
        )

        result = get_plan(
            GetPlanInput(plan_id=create_result.plan_id, working_dir=str(workspace))
        )
        assert result.ok is True
        assert result.completed_count == 2
        assert result.total_count == 3

    def test_get_most_recent_plan(self, workspace):
        # Create first plan
        create_plan(
            CreatePlanInput(
                title="Old plan",
                steps=[PlanStep(description="Old step")],
                working_dir=str(workspace),
            )
        )

        # Small delay to ensure different mtime
        time.sleep(0.05)

        # Create second plan
        second = create_plan(
            CreatePlanInput(
                title="New plan",
                steps=[PlanStep(description="New step")],
                working_dir=str(workspace),
            )
        )

        # Get without specifying plan_id
        result = get_plan(GetPlanInput(working_dir=str(workspace)))
        assert result.ok is True
        assert result.plan_id == second.plan_id
        assert result.title == "New plan"

    def test_get_nonexistent_plan(self, workspace):
        result = get_plan(
            GetPlanInput(plan_id="nonexist", working_dir=str(workspace))
        )
        assert result.ok is False
        assert "not found" in result.error.lower()

    def test_get_plan_no_plans_dir(self, workspace):
        result = get_plan(GetPlanInput(working_dir=str(workspace)))
        assert result.ok is False
        assert "no plans" in result.error.lower()

    def test_get_plan_empty_plans_dir(self, workspace):
        (workspace / ".supyagent" / "plans").mkdir(parents=True)
        result = get_plan(GetPlanInput(working_dir=str(workspace)))
        assert result.ok is False
        assert "no plans" in result.error.lower()

    def test_get_plan_step_indices(self, workspace):
        create_result = create_plan(
            CreatePlanInput(
                title="Indexed plan",
                steps=[
                    PlanStep(description="Zero"),
                    PlanStep(description="One"),
                    PlanStep(description="Two"),
                ],
                working_dir=str(workspace),
            )
        )

        result = get_plan(
            GetPlanInput(plan_id=create_result.plan_id, working_dir=str(workspace))
        )
        assert result.ok is True
        assert result.steps[0].index == 0
        assert result.steps[1].index == 1
        assert result.steps[2].index == 2
        assert result.steps[0].description == "Zero"
        assert result.steps[2].description == "Two"
