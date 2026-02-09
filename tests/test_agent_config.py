"""
Tests for agent configuration loading and validation.
"""

import pytest
from pydantic import ValidationError

from supyagent.models.agent_config import (
    AgentConfig,
    AgentConfigError,
    AgentNotFoundError,
    ModelConfig,
    ToolPermissions,
    load_agent_config,
)


class TestModelConfig:
    """Tests for ModelConfig validation."""

    def test_valid_model_config(self):
        """Test creating a valid model config."""
        config = ModelConfig(
            provider="anthropic/claude-3-5-sonnet-20241022",
            temperature=0.7,
            max_tokens=4096,
        )
        assert config.provider == "anthropic/claude-3-5-sonnet-20241022"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096

    def test_model_config_defaults(self):
        """Test model config defaults."""
        config = ModelConfig(provider="openai/gpt-4")
        assert config.temperature == 0.7
        assert config.max_tokens is None

    def test_temperature_bounds(self):
        """Test temperature validation."""
        # Valid temperatures
        ModelConfig(provider="test", temperature=0)
        ModelConfig(provider="test", temperature=2)

        # Invalid temperatures
        with pytest.raises(ValidationError):
            ModelConfig(provider="test", temperature=-0.1)
        with pytest.raises(ValidationError):
            ModelConfig(provider="test", temperature=2.1)

    def test_max_tokens_optional(self):
        """Test max_tokens accepts None (provider default) and positive values."""
        config = ModelConfig(provider="test")
        assert config.max_tokens is None
        config = ModelConfig(provider="test", max_tokens=8192)
        assert config.max_tokens == 8192


class TestToolPermissions:
    """Tests for ToolPermissions."""

    def test_empty_permissions(self):
        """Test default empty permissions."""
        perms = ToolPermissions()
        assert perms.allow == []
        assert perms.deny == []

    def test_permissions_with_patterns(self):
        """Test permissions with patterns."""
        perms = ToolPermissions(
            allow=["web_search:*", "summarize:summarize"],
            deny=["dangerous:*"],
        )
        assert "web_search:*" in perms.allow
        assert "dangerous:*" in perms.deny


class TestAgentConfig:
    """Tests for AgentConfig validation."""

    def test_valid_agent_config(self, sample_agent_config):
        """Test creating a valid agent config."""
        assert sample_agent_config.name == "test-agent"
        assert sample_agent_config.type == "interactive"

    def test_agent_config_defaults(self):
        """Test agent config defaults."""
        config = AgentConfig(
            name="test",
            model=ModelConfig(provider="test/model"),
            system_prompt="You are a test.",
        )
        assert config.description == ""
        assert config.version == "1.0"
        assert config.type == "interactive"
        assert config.delegates == []

    def test_agent_name_validation(self):
        """Test agent name must be non-empty and not too long."""
        # Empty name
        with pytest.raises(ValidationError):
            AgentConfig(
                name="",
                model=ModelConfig(provider="test"),
                system_prompt="Test",
            )

        # Name too long
        with pytest.raises(ValidationError):
            AgentConfig(
                name="a" * 51,
                model=ModelConfig(provider="test"),
                system_prompt="Test",
            )

    def test_agent_type_validation(self):
        """Test agent type must be valid."""
        # Valid types
        AgentConfig(
            name="test",
            type="interactive",
            model=ModelConfig(provider="test"),
            system_prompt="Test",
        )
        AgentConfig(
            name="test",
            type="execution",
            model=ModelConfig(provider="test"),
            system_prompt="Test",
        )

        # Invalid type
        with pytest.raises(ValidationError):
            AgentConfig(
                name="test",
                type="invalid",  # type: ignore
                model=ModelConfig(provider="test"),
                system_prompt="Test",
            )

    def test_system_prompt_required(self):
        """Test system prompt is required and non-empty."""
        with pytest.raises(ValidationError):
            AgentConfig(
                name="test",
                model=ModelConfig(provider="test"),
                system_prompt="",
            )


class TestLoadAgentConfig:
    """Tests for loading agent configs from YAML."""

    def test_load_valid_config(self, agents_dir, sample_agent_yaml):
        """Test loading a valid agent config from YAML."""
        # Write sample YAML
        agent_file = agents_dir / "test-agent.yaml"
        agent_file.write_text(sample_agent_yaml)

        # Load it
        config = load_agent_config("test-agent", agents_dir)

        assert config.name == "test-agent"
        assert config.description == "A test agent"
        assert config.model.provider == "anthropic/claude-3-5-sonnet-20241022"
        assert config.model.temperature == 0.7

    def test_load_missing_config(self, agents_dir):
        """Test loading a non-existent agent raises error."""
        with pytest.raises(AgentNotFoundError) as exc_info:
            load_agent_config("nonexistent", agents_dir)

        assert "nonexistent" in str(exc_info.value)

    def test_load_invalid_yaml(self, agents_dir):
        """Test loading invalid YAML raises error."""
        agent_file = agents_dir / "invalid.yaml"
        agent_file.write_text("name: test\n  invalid: yaml: here")

        with pytest.raises(Exception):
            load_agent_config("invalid", agents_dir)

    def test_load_missing_required_fields(self, agents_dir):
        """Test loading YAML with missing required fields raises AgentConfigError."""
        agent_file = agents_dir / "incomplete.yaml"
        agent_file.write_text("""
name: incomplete
# Missing model and system_prompt
""")

        with pytest.raises(AgentConfigError) as exc_info:
            load_agent_config("incomplete", agents_dir)

        # Should contain friendly error messages
        assert "model is required" in str(exc_info.value)
        assert "system_prompt is required" in str(exc_info.value)
