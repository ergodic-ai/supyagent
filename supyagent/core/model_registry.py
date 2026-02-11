"""
Model registry for managing LLM models and role-based assignments.

Stores model configuration in ~/.supyagent/models.yaml that is shared
across all agents and projects. Supports role-based model assignment
(fast/smart/reasoning/cheap) so agents can reference roles instead of
specific model strings.
"""

import os
from pathlib import Path
from typing import Any

import yaml

from supyagent.core.config import KNOWN_LLM_KEYS, get_config_manager

# Map model string prefixes to API key environment variable names
MODEL_PREFIX_TO_KEY = {
    "anthropic/": "ANTHROPIC_API_KEY",
    "claude-": "ANTHROPIC_API_KEY",
    "gpt-": "OPENAI_API_KEY",
    "o1-": "OPENAI_API_KEY",
    "o3-": "OPENAI_API_KEY",
    "o4-": "OPENAI_API_KEY",
    "google/": "GOOGLE_API_KEY",
    "gemini/": "GOOGLE_API_KEY",
    "deepseek/": "DEEPSEEK_API_KEY",
    "openrouter/": "OPENROUTER_API_KEY",
    "groq/": "GROQ_API_KEY",
    "mistral/": "MISTRAL_API_KEY",
    "together/": "TOGETHER_API_KEY",
    "fireworks/": "FIREWORKS_API_KEY",
    "cohere/": "COHERE_API_KEY",
    "replicate/": "REPLICATE_API_KEY",
    "huggingface/": "HUGGINGFACE_API_KEY",
    "perplexity/": "PERPLEXITY_API_KEY",
    "azure/": "AZURE_API_KEY",
}

# Standard role names
STANDARD_ROLES = ("fast", "smart", "reasoning", "cheap")


class ModelRegistry:
    """
    Manages registered LLM models and role-based assignments.

    Configuration stored in ~/.supyagent/models.yaml:
        default: anthropic/claude-sonnet-4-5-20250929
        roles:
          fast: google/gemini-3-flash-preview
          smart: anthropic/claude-sonnet-4-5-20250929
        registered:
          - anthropic/claude-sonnet-4-5-20250929
          - google/gemini-3-flash-preview
    """

    def __init__(self, base_dir: Path | None = None):
        if base_dir is None:
            base_dir = Path.home() / ".supyagent"
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Any] | None = None

    def _models_path(self) -> Path:
        return self.base_dir / "models.yaml"

    def _load(self) -> dict[str, Any]:
        if self._cache is not None:
            return self._cache

        path = self._models_path()
        if not path.exists():
            self._cache = {"default": None, "roles": {}, "registered": []}
            return self._cache

        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            self._cache = {
                "default": data.get("default"),
                "roles": data.get("roles") or {},
                "registered": data.get("registered") or [],
            }
            return self._cache
        except (yaml.YAMLError, OSError):
            self._cache = {"default": None, "roles": {}, "registered": []}
            return self._cache

    def _save(self) -> None:
        data = self._load()
        path = self._models_path()
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    # ── Registration ──────────────────────────────────────────────────

    def add(self, model_string: str) -> None:
        """Register a model. Does not check for API keys — use check_api_key() for that."""
        data = self._load()
        if model_string not in data["registered"]:
            data["registered"].append(model_string)
            self._save()

    def remove(self, model_string: str) -> bool:
        """Unregister a model. Returns True if it was registered."""
        data = self._load()
        if model_string in data["registered"]:
            data["registered"].remove(model_string)
            # Clean up role assignments pointing to this model
            for role, assigned in list(data["roles"].items()):
                if assigned == model_string:
                    del data["roles"][role]
            # Clear default if it was this model
            if data["default"] == model_string:
                data["default"] = data["registered"][0] if data["registered"] else None
            self._save()
            return True
        return False

    def list_models(self) -> list[str]:
        """List all registered model strings."""
        return list(self._load()["registered"])

    def is_registered(self, model_string: str) -> bool:
        return model_string in self._load()["registered"]

    # ── Default ───────────────────────────────────────────────────────

    def set_default(self, model_string: str) -> None:
        """Set the default model. Auto-registers if not already registered."""
        data = self._load()
        if model_string not in data["registered"]:
            data["registered"].append(model_string)
        data["default"] = model_string
        self._save()

    def get_default(self) -> str | None:
        """Get the default model string, or None if not set."""
        return self._load()["default"]

    # ── Roles ─────────────────────────────────────────────────────────

    def assign_role(self, role: str, model_string: str) -> None:
        """Assign a model to a role (e.g. fast, smart, reasoning, cheap)."""
        data = self._load()
        if model_string not in data["registered"]:
            data["registered"].append(model_string)
        data["roles"][role] = model_string
        self._save()

    def get_role(self, role: str) -> str | None:
        """Get the model assigned to a role, falling back to default."""
        data = self._load()
        model = data["roles"].get(role)
        if model:
            return model
        return data["default"]

    def list_roles(self) -> dict[str, str]:
        """Return all role assignments."""
        return dict(self._load()["roles"])

    def unassign_role(self, role: str) -> bool:
        """Remove a role assignment. Returns True if it existed."""
        data = self._load()
        if role in data["roles"]:
            del data["roles"][role]
            self._save()
            return True
        return False

    # ── Resolution ────────────────────────────────────────────────────

    def resolve(self, model_or_role: str) -> str | None:
        """
        Resolve a model string or role name to a concrete model string.

        If model_or_role matches a known role name, returns the role's model.
        Otherwise treats it as a literal model string and returns it as-is.
        """
        data = self._load()
        # Check if it's a role name
        if model_or_role in data["roles"]:
            return data["roles"][model_or_role]
        # It's a literal model string
        return model_or_role

    # ── Provider Detection ────────────────────────────────────────────

    @staticmethod
    def detect_provider_key(model_string: str) -> str | None:
        """
        Detect the API key environment variable needed for a model string.

        Returns the key name (e.g. 'ANTHROPIC_API_KEY') or None if unknown.
        """
        model_lower = model_string.lower()
        for prefix, key_name in MODEL_PREFIX_TO_KEY.items():
            if model_lower.startswith(prefix):
                return key_name
        return None

    @staticmethod
    def detect_provider_name(model_string: str) -> str | None:
        """Detect the human-readable provider name from a model string."""
        key = ModelRegistry.detect_provider_key(model_string)
        if key:
            return KNOWN_LLM_KEYS.get(key, key)
        return None

    def check_api_key(self, model_string: str) -> bool:
        """
        Check if the API key for a model's provider is configured.

        Returns True if the key exists (in config or environment), False otherwise.
        """
        key_name = self.detect_provider_key(model_string)
        if key_name is None:
            # Unknown provider — assume it's fine (could be local/custom)
            return True
        config_mgr = get_config_manager()
        return config_mgr.get(key_name) is not None or os.environ.get(key_name) is not None

    def ensure_api_key(self, model_string: str) -> bool:
        """
        Check if API key exists for this model's provider.
        If not, interactively prompt the user to provide it.

        Returns True if key is available (existed or was just set).
        """
        key_name = self.detect_provider_key(model_string)
        if key_name is None:
            return True

        config_mgr = get_config_manager()
        if config_mgr.get(key_name) is not None:
            return True

        # Prompt for the key
        import getpass

        provider_name = KNOWN_LLM_KEYS.get(key_name, key_name)
        print(f"\n  Model '{model_string}' requires {provider_name}.")
        value = getpass.getpass(f"  Enter {key_name}: ")
        if value.strip():
            config_mgr.set(key_name, value.strip())
            return True
        return False

    # ── Summary ───────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return a summary dict for display purposes."""
        data = self._load()
        return {
            "default": data["default"],
            "roles": dict(data["roles"]),
            "registered": list(data["registered"]),
        }


# Global instance
_model_registry: ModelRegistry | None = None


def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry
