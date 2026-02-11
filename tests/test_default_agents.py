"""
Tests for supyagent.default_agents package.
"""


import pytest
import yaml

from supyagent.default_agents import (
    AGENT_ROLES,
    DEFAULT_MODEL,
    MODEL_PLACEHOLDER,
    WORKSPACE_PROFILES,
    get_agent_template,
    get_bundled_agents,
    install_agent,
    install_default_agents,
    install_workspace_agents,
    list_default_agents,
)


class TestAgentRolesMetadata:
    """Test the AGENT_ROLES and WORKSPACE_PROFILES constants."""

    def test_all_roles_present(self):
        assert set(AGENT_ROLES.keys()) == {"assistant", "coder", "planner", "writer"}

    def test_roles_have_required_fields(self):
        for role, meta in AGENT_ROLES.items():
            assert "description" in meta, f"{role} missing description"
            assert "delegates" in meta, f"{role} missing delegates"
            assert isinstance(meta["delegates"], list)

    def test_workspace_profiles(self):
        assert "coding" in WORKSPACE_PROFILES
        assert "automation" in WORKSPACE_PROFILES
        assert "full" in WORKSPACE_PROFILES

    def test_coding_profile_agents(self):
        assert WORKSPACE_PROFILES["coding"] == ["assistant", "coder", "planner"]

    def test_full_profile_has_all(self):
        assert set(WORKSPACE_PROFILES["full"]) == set(AGENT_ROLES.keys())


class TestBundledAgents:
    """Test template file discovery."""

    def test_get_bundled_agents(self):
        agents = get_bundled_agents()
        assert len(agents) == len(AGENT_ROLES)
        names = {a.stem for a in agents}
        assert names == set(AGENT_ROLES.keys())

    def test_all_yamls_exist(self):
        for role in AGENT_ROLES:
            path = get_bundled_agents()[0].parent / f"{role}.yaml"
            assert path.exists(), f"{role}.yaml not found"


class TestGetAgentTemplate:
    """Test reading raw template content."""

    def test_read_valid_role(self):
        for role in AGENT_ROLES:
            content = get_agent_template(role)
            assert MODEL_PLACEHOLDER in content
            assert f"name: {role}" in content

    def test_read_invalid_role(self):
        with pytest.raises(ValueError, match="Unknown agent role"):
            get_agent_template("nonexistent")


class TestListDefaultAgents:
    """Test the listing function."""

    def test_list_returns_all(self):
        agents = list_default_agents()
        assert len(agents) == len(AGENT_ROLES)
        roles = {a["role"] for a in agents}
        assert roles == set(AGENT_ROLES.keys())

    def test_list_has_required_keys(self):
        for agent in list_default_agents():
            assert "role" in agent
            assert "description" in agent
            assert "delegates" in agent
            assert "file" in agent


class TestInstallAgent:
    """Test installing individual agent templates."""

    def test_install_basic(self, tmp_path):
        dest = install_agent("coder", tmp_path)
        assert dest is not None
        assert dest.exists()
        assert dest.name == "coder.yaml"

        content = dest.read_text()
        assert MODEL_PLACEHOLDER not in content
        assert DEFAULT_MODEL in content
        assert "name: coder" in content

    def test_install_custom_model(self, tmp_path):
        dest = install_agent("coder", tmp_path, model="gpt-4o")
        content = dest.read_text()
        assert "gpt-4o" in content
        assert MODEL_PLACEHOLDER not in content

    def test_install_custom_name(self, tmp_path):
        dest = install_agent("coder", tmp_path, name="my-coder")
        assert dest.name == "my-coder.yaml"
        content = dest.read_text()
        assert "name: my-coder" in content

    def test_install_standalone_strips_delegates(self, tmp_path):
        dest = install_agent("assistant", tmp_path, standalone=True)
        content = dest.read_text()
        parsed = yaml.safe_load(content)
        assert parsed["delegates"] == []

    def test_install_non_standalone_keeps_delegates(self, tmp_path):
        dest = install_agent("assistant", tmp_path, standalone=False)
        content = dest.read_text()
        parsed = yaml.safe_load(content)
        assert len(parsed["delegates"]) > 0

    def test_install_no_overwrite(self, tmp_path):
        install_agent("coder", tmp_path)
        result = install_agent("coder", tmp_path)
        assert result is None

    def test_install_force_overwrite(self, tmp_path):
        install_agent("coder", tmp_path)
        result = install_agent("coder", tmp_path, force=True)
        assert result is not None

    def test_install_creates_target_dir(self, tmp_path):
        target = tmp_path / "nested" / "agents"
        dest = install_agent("coder", target)
        assert dest.exists()
        assert target.is_dir()

    def test_installed_yaml_is_valid(self, tmp_path):
        """All installed agents should produce valid YAML."""
        for role in AGENT_ROLES:
            dest = install_agent(role, tmp_path / role)
            content = dest.read_text()
            parsed = yaml.safe_load(content)
            assert parsed["name"] == role
            assert "system_prompt" in parsed or "model" in parsed


class TestInstallWorkspaceAgents:
    """Test profile-based installation."""

    def test_coding_profile(self, tmp_path):
        paths = install_workspace_agents("coding", tmp_path)
        names = {p.stem for p in paths}
        assert names == {"assistant", "coder", "planner"}

    def test_automation_profile(self, tmp_path):
        paths = install_workspace_agents("automation", tmp_path)
        names = {p.stem for p in paths}
        assert names == {"assistant", "writer"}

    def test_full_profile(self, tmp_path):
        paths = install_workspace_agents("full", tmp_path)
        names = {p.stem for p in paths}
        assert names == set(AGENT_ROLES.keys())

    def test_invalid_profile(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown workspace profile"):
            install_workspace_agents("invalid", tmp_path)

    def test_coding_delegation_preserved(self, tmp_path):
        """In coding profile, assistant delegates to planner and coder."""
        install_workspace_agents("coding", tmp_path)
        assistant = yaml.safe_load((tmp_path / "assistant.yaml").read_text())
        assert "planner" in assistant["delegates"]
        assert "coder" in assistant["delegates"]

        planner = yaml.safe_load((tmp_path / "planner.yaml").read_text())
        assert "coder" in planner["delegates"]

    def test_automation_delegates_stripped(self, tmp_path):
        """In automation profile, assistant has writer but no planner/coder."""
        install_workspace_agents("automation", tmp_path)
        assistant = yaml.safe_load((tmp_path / "assistant.yaml").read_text())
        # Assistant delegates to planner and coder by default, but those aren't
        # in automation profile, so they should be stripped
        delegates = assistant["delegates"]
        assert "planner" not in delegates
        assert "coder" not in delegates

    def test_custom_model(self, tmp_path):
        paths = install_workspace_agents("coding", tmp_path, model="gpt-4o")
        for p in paths:
            content = p.read_text()
            assert "gpt-4o" in content
            assert MODEL_PLACEHOLDER not in content

    def test_no_overwrite_by_default(self, tmp_path):
        first = install_workspace_agents("coding", tmp_path)
        second = install_workspace_agents("coding", tmp_path)
        assert len(first) == 3
        assert len(second) == 0

    def test_force_overwrite(self, tmp_path):
        install_workspace_agents("coding", tmp_path)
        second = install_workspace_agents("coding", tmp_path, force=True)
        assert len(second) == 3


class TestInstallDefaultAgents:
    """Test the convenience wrapper."""

    def test_default_installs_coding(self, tmp_path):
        count = install_default_agents(tmp_path)
        assert count == 3

    def test_specific_roles(self, tmp_path):
        count = install_default_agents(tmp_path, roles=["coder", "writer"])
        assert count == 2
        assert (tmp_path / "coder.yaml").exists()
        assert (tmp_path / "writer.yaml").exists()

    def test_specific_roles_standalone(self, tmp_path):
        """Individual role installs should be standalone (no delegates)."""
        install_default_agents(tmp_path, roles=["planner"])
        parsed = yaml.safe_load((tmp_path / "planner.yaml").read_text())
        assert parsed["delegates"] == []
