"""
Tests for supyagent.core.model_registry.
"""


import pytest

from supyagent.core.model_registry import ModelRegistry


@pytest.fixture
def registry(tmp_path):
    """Create a ModelRegistry backed by a temp directory."""
    return ModelRegistry(base_dir=tmp_path)


class TestModelRegistryBasic:
    """Test basic registration, listing, and removal."""

    def test_empty_registry(self, registry):
        assert registry.list_models() == []
        assert registry.get_default() is None

    def test_add_model(self, registry):
        registry.add("anthropic/claude-sonnet-4-5-20250929")
        assert "anthropic/claude-sonnet-4-5-20250929" in registry.list_models()

    def test_add_duplicate(self, registry):
        registry.add("gpt-4o")
        registry.add("gpt-4o")
        assert registry.list_models().count("gpt-4o") == 1

    def test_remove_model(self, registry):
        registry.add("gpt-4o")
        assert registry.remove("gpt-4o") is True
        assert "gpt-4o" not in registry.list_models()

    def test_remove_nonexistent(self, registry):
        assert registry.remove("not-here") is False

    def test_is_registered(self, registry):
        registry.add("gpt-4o")
        assert registry.is_registered("gpt-4o") is True
        assert registry.is_registered("gpt-3") is False


class TestModelRegistryDefault:
    """Test default model management."""

    def test_set_default(self, registry):
        registry.set_default("gpt-4o")
        assert registry.get_default() == "gpt-4o"

    def test_set_default_auto_registers(self, registry):
        registry.set_default("gpt-4o")
        assert registry.is_registered("gpt-4o") is True

    def test_remove_default_falls_back(self, registry):
        registry.add("model-a")
        registry.set_default("model-b")
        registry.remove("model-b")
        assert registry.get_default() == "model-a"

    def test_remove_last_model_clears_default(self, registry):
        registry.set_default("only-model")
        registry.remove("only-model")
        assert registry.get_default() is None


class TestModelRegistryRoles:
    """Test role assignment and resolution."""

    def test_assign_role(self, registry):
        registry.add("fast-model")
        registry.assign_role("fast", "fast-model")
        assert registry.get_role("fast") == "fast-model"

    def test_assign_role_auto_registers(self, registry):
        registry.assign_role("smart", "smart-model")
        assert registry.is_registered("smart-model") is True

    def test_get_role_falls_back_to_default(self, registry):
        registry.set_default("default-model")
        assert registry.get_role("fast") == "default-model"

    def test_get_role_unset_no_default(self, registry):
        assert registry.get_role("fast") is None

    def test_list_roles(self, registry):
        registry.assign_role("fast", "model-a")
        registry.assign_role("smart", "model-b")
        roles = registry.list_roles()
        assert roles == {"fast": "model-a", "smart": "model-b"}

    def test_unassign_role(self, registry):
        registry.assign_role("fast", "model-a")
        assert registry.unassign_role("fast") is True
        assert "fast" not in registry.list_roles()

    def test_unassign_nonexistent_role(self, registry):
        assert registry.unassign_role("nope") is False

    def test_remove_model_cleans_roles(self, registry):
        registry.assign_role("fast", "model-a")
        registry.remove("model-a")
        assert registry.list_roles() == {}


class TestModelRegistryResolve:
    """Test model/role resolution."""

    def test_resolve_role(self, registry):
        registry.assign_role("fast", "model-a")
        assert registry.resolve("fast") == "model-a"

    def test_resolve_literal(self, registry):
        assert registry.resolve("gpt-4o") == "gpt-4o"

    def test_resolve_unknown_role_treated_as_literal(self, registry):
        assert registry.resolve("custom-model") == "custom-model"


class TestModelRegistryProviderDetection:
    """Test provider/key detection from model strings."""

    @pytest.mark.parametrize(
        "model,expected_key",
        [
            ("anthropic/claude-sonnet-4-5", "ANTHROPIC_API_KEY"),
            ("claude-3-opus", "ANTHROPIC_API_KEY"),
            ("gpt-4o", "OPENAI_API_KEY"),
            ("o3-mini", "OPENAI_API_KEY"),
            ("google/gemini-2.5-flash", "GOOGLE_API_KEY"),
            ("gemini/gemini-2.5-pro", "GOOGLE_API_KEY"),
            ("deepseek/deepseek-chat", "DEEPSEEK_API_KEY"),
            ("openrouter/meta-llama/llama-4", "OPENROUTER_API_KEY"),
            ("groq/llama3-70b", "GROQ_API_KEY"),
        ],
    )
    def test_detect_provider_key(self, model, expected_key):
        assert ModelRegistry.detect_provider_key(model) == expected_key

    def test_detect_unknown_provider(self):
        assert ModelRegistry.detect_provider_key("ollama/llama3") is None

    def test_detect_provider_name(self):
        name = ModelRegistry.detect_provider_name("anthropic/claude-sonnet-4-5")
        assert name is not None
        assert "anthropic" in name.lower() or "Anthropic" in name


class TestModelRegistryPersistence:
    """Test that data persists across instances."""

    def test_persists_to_yaml(self, tmp_path):
        r1 = ModelRegistry(base_dir=tmp_path)
        r1.set_default("gpt-4o")
        r1.assign_role("fast", "gpt-4o-mini")

        # New instance should load saved data
        r2 = ModelRegistry(base_dir=tmp_path)
        assert r2.get_default() == "gpt-4o"
        assert r2.get_role("fast") == "gpt-4o-mini"
        assert set(r2.list_models()) == {"gpt-4o", "gpt-4o-mini"}

    def test_models_yaml_created(self, tmp_path):
        r = ModelRegistry(base_dir=tmp_path)
        r.add("test-model")
        assert (tmp_path / "models.yaml").exists()


class TestModelRegistrySummary:
    """Test summary output."""

    def test_summary(self, registry):
        registry.set_default("model-a")
        registry.assign_role("fast", "model-b")
        s = registry.summary()
        assert s["default"] == "model-a"
        assert s["roles"] == {"fast": "model-b"}
        assert "model-a" in s["registered"]
        assert "model-b" in s["registered"]
