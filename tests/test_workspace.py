"""
Tests for supyagent.core.workspace.
"""


import pytest

from supyagent.core.workspace import (
    ExecutionConfig,
    HeartbeatConfig,
    WorkspaceConfig,
    create_goals_file,
    get_workspace_model,
    goals_path,
    initialize_workspace,
    is_workspace_initialized,
    load_workspace,
    read_goals,
    save_workspace,
    workspace_config_path,
    workspace_dir,
)


@pytest.fixture
def ws_root(tmp_path):
    """Provide a temp directory as workspace root."""
    return tmp_path


class TestWorkspaceConfig:
    """Test WorkspaceConfig model defaults and serialization."""

    def test_default_values(self):
        c = WorkspaceConfig()
        assert c.name == ""
        assert c.profile == "coding"
        assert c.execution.mode == "yolo"
        assert c.heartbeat.enabled is False
        assert c.heartbeat.interval == "5m"
        assert c.heartbeat.max_events_per_cycle == 10
        assert c.models == {}

    def test_custom_values(self):
        c = WorkspaceConfig(
            name="myproject",
            profile="full",
            execution=ExecutionConfig(mode="isolated"),
            heartbeat=HeartbeatConfig(enabled=True, interval="10m"),
            models={"coder": "fast"},
        )
        assert c.name == "myproject"
        assert c.profile == "full"
        assert c.execution.mode == "isolated"
        assert c.heartbeat.enabled is True
        assert c.heartbeat.interval == "10m"
        assert c.models == {"coder": "fast"}


class TestWorkspacePaths:
    """Test path helper functions."""

    def test_workspace_dir(self, ws_root):
        assert workspace_dir(ws_root) == ws_root / ".supyagent"

    def test_workspace_config_path(self, ws_root):
        assert workspace_config_path(ws_root) == ws_root / ".supyagent" / "workspace.yaml"

    def test_goals_path(self, ws_root):
        assert goals_path(ws_root) == ws_root / "GOALS.md"


class TestWorkspaceInitialization:
    """Test workspace initialization detection."""

    def test_not_initialized(self, ws_root):
        assert is_workspace_initialized(ws_root) is False

    def test_initialized_after_save(self, ws_root):
        save_workspace(WorkspaceConfig(), ws_root)
        assert is_workspace_initialized(ws_root) is True


class TestSaveLoadWorkspace:
    """Test saving and loading workspace config."""

    def test_save_creates_dir(self, ws_root):
        config = WorkspaceConfig(name="test")
        path = save_workspace(config, ws_root)
        assert path.exists()
        assert (ws_root / ".supyagent").is_dir()

    def test_roundtrip(self, ws_root):
        original = WorkspaceConfig(
            name="roundtrip",
            profile="automation",
            execution=ExecutionConfig(mode="isolated"),
            heartbeat=HeartbeatConfig(enabled=True, interval="2m", max_events_per_cycle=5),
            models={"assistant": "smart", "coder": "fast"},
        )
        save_workspace(original, ws_root)
        loaded = load_workspace(ws_root)

        assert loaded.name == "roundtrip"
        assert loaded.profile == "automation"
        assert loaded.execution.mode == "isolated"
        assert loaded.heartbeat.enabled is True
        assert loaded.heartbeat.interval == "2m"
        assert loaded.heartbeat.max_events_per_cycle == 5
        assert loaded.models == {"assistant": "smart", "coder": "fast"}

    def test_load_nonexistent_returns_default(self, ws_root):
        config = load_workspace(ws_root)
        assert config.name == ""
        assert config.profile == "coding"

    def test_load_corrupt_returns_default(self, ws_root):
        ws = ws_root / ".supyagent"
        ws.mkdir(parents=True)
        (ws / "workspace.yaml").write_text("not: [valid: yaml: {{{")
        config = load_workspace(ws_root)
        assert config.profile == "coding"


class TestGoalsFile:
    """Test GOALS.md creation and reading."""

    def test_create_goals_empty(self, ws_root):
        path = create_goals_file("", ws_root)
        assert path.exists()
        content = path.read_text()
        assert "## User Goals" in content
        assert "## Subgoals" in content
        assert "- " in content  # placeholder bullet

    def test_create_goals_with_content(self, ws_root):
        path = create_goals_file("- Build a CLI tool\n- Add tests", ws_root)
        content = path.read_text()
        assert "Build a CLI tool" in content
        assert "Add tests" in content

    def test_read_goals_exists(self, ws_root):
        create_goals_file("- My goal", ws_root)
        content = read_goals(ws_root)
        assert "My goal" in content

    def test_read_goals_missing(self, ws_root):
        assert read_goals(ws_root) == ""


class TestGetWorkspaceModel:
    """Test per-agent model resolution from workspace config."""

    def test_model_override_exists(self, ws_root):
        config = WorkspaceConfig(models={"coder": "fast"})
        save_workspace(config, ws_root)
        assert get_workspace_model("coder", ws_root) == "fast"

    def test_model_override_missing(self, ws_root):
        config = WorkspaceConfig(models={"coder": "fast"})
        save_workspace(config, ws_root)
        assert get_workspace_model("writer", ws_root) is None

    def test_no_workspace_config(self, ws_root):
        assert get_workspace_model("coder", ws_root) is None


class TestInitializeWorkspace:
    """Test the full workspace initialization convenience function."""

    def test_basic_init(self, ws_root):
        result = initialize_workspace(
            name="test-project",
            profile="coding",
            root=ws_root,
        )

        assert "workspace_config" in result
        assert "goals_file" in result
        assert "workspace_dir" in result

        # Check workspace.yaml was created
        config = load_workspace(ws_root)
        assert config.name == "test-project"
        assert config.profile == "coding"
        assert config.execution.mode == "yolo"
        assert config.heartbeat.enabled is False

        # Check GOALS.md was created
        assert (ws_root / "GOALS.md").exists()

    def test_init_with_all_options(self, ws_root):
        initialize_workspace(
            name="full-project",
            profile="full",
            execution_mode="isolated",
            heartbeat_enabled=True,
            heartbeat_interval="10m",
            model_overrides={"coder": "fast"},
            user_goals="- Ship the product",
            root=ws_root,
        )

        config = load_workspace(ws_root)
        assert config.name == "full-project"
        assert config.profile == "full"
        assert config.execution.mode == "isolated"
        assert config.heartbeat.enabled is True
        assert config.heartbeat.interval == "10m"
        assert config.models == {"coder": "fast"}

        goals = read_goals(ws_root)
        assert "Ship the product" in goals

    def test_init_defaults_name_to_dir_name(self, ws_root):
        initialize_workspace(root=ws_root)
        config = load_workspace(ws_root)
        assert config.name == ws_root.name
