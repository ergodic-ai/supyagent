"""
Tests for supyagent.server.ui.routes.models — Model Registry Manager UI API.
"""

from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from supyagent.core.config import ConfigManager
from supyagent.core.model_registry import ModelRegistry
from supyagent.server.ui import create_ui_app


@pytest.fixture
def temp_config(tmp_path):
    """Create a temp ConfigManager."""
    return ConfigManager(base_dir=tmp_path / "config")


@pytest.fixture
def temp_registry(tmp_path):
    """Create a temp ModelRegistry."""
    return ModelRegistry(base_dir=tmp_path / "registry")


@pytest.fixture
def client(temp_config, temp_registry):
    """Create a test client with mocked config and registry."""
    done_data = {}

    def on_done(data):
        done_data.update(data)

    app = create_ui_app(mode="models", done_callback=on_done)

    # Patch at the source modules since routes use deferred imports
    with (
        patch(
            "supyagent.core.model_registry.get_model_registry",
            return_value=temp_registry,
        ),
        patch(
            "supyagent.core.config.get_config_manager",
            return_value=temp_config,
        ),
    ):
        yield TestClient(app), temp_registry, temp_config, done_data


class TestModelsPage:
    def test_get_models_page(self, client):
        test_client, *_ = client
        resp = test_client.get("/models")
        assert resp.status_code == 200
        assert "Model Registry" in resp.text


class TestModelsState:
    def test_empty_state(self, client):
        test_client, *_ = client
        resp = test_client.get("/api/models/state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["default"] is None
        assert data["models"] == []
        assert data["roles"] == {}
        assert "standard_roles" in data
        assert "providers" in data

    def test_state_with_models(self, client):
        test_client, registry, config_mgr, _ = client
        registry.set_default("gpt-4o")
        registry.assign_role("fast", "gpt-4o-mini")
        registry.add("gpt-4o-mini")

        resp = test_client.get("/api/models/state")
        data = resp.json()
        assert data["default"] == "gpt-4o"
        assert len(data["models"]) == 2
        model_names = [m["model"] for m in data["models"]]
        assert "gpt-4o" in model_names
        assert "gpt-4o-mini" in model_names


class TestModelsCRUD:
    def test_add_model(self, client):
        test_client, registry, *_ = client
        resp = test_client.post(
            "/api/models/add", json={"model": "anthropic/claude-sonnet-4-5"}
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert registry.is_registered("anthropic/claude-sonnet-4-5")

    def test_add_model_returns_key_info(self, client):
        test_client, registry, *_ = client
        resp = test_client.post(
            "/api/models/add", json={"model": "anthropic/claude-sonnet-4-5"}
        )
        data = resp.json()
        assert data["ok"] is True
        assert data["detected_key"] == "ANTHROPIC_API_KEY"
        assert data["has_key"] is False  # no key stored in temp config
        assert "missing_keys" in data

    def test_add_unknown_provider_returns_null_key(self, client):
        test_client, registry, *_ = client
        resp = test_client.post(
            "/api/models/add", json={"model": "mycompany/custom-model"}
        )
        data = resp.json()
        assert data["ok"] is True
        assert data["detected_key"] is None

    def test_add_model_with_key(self, client):
        test_client, registry, config_mgr, _ = client
        resp = test_client.post(
            "/api/models/add",
            json={
                "model": "mycompany/custom-model",
                "key_name": "MYCOMPANY_API_KEY",
                "key_value": "sk-custom-123",
            },
        )
        data = resp.json()
        assert data["ok"] is True
        assert registry.is_registered("mycompany/custom-model")
        assert config_mgr.get("MYCOMPANY_API_KEY") == "sk-custom-123"

    def test_add_duplicate(self, client):
        test_client, registry, *_ = client
        registry.add("gpt-4o")
        resp = test_client.post("/api/models/add", json={"model": "gpt-4o"})
        assert resp.json()["ok"] is True
        assert resp.json()["message"] == "Already registered"

    def test_remove_model(self, client):
        test_client, registry, *_ = client
        registry.add("gpt-4o")
        resp = test_client.post("/api/models/remove", json={"model": "gpt-4o"})
        assert resp.json()["ok"] is True
        assert not registry.is_registered("gpt-4o")

    def test_set_default(self, client):
        test_client, registry, *_ = client
        resp = test_client.post("/api/models/default", json={"model": "gpt-4o"})
        assert resp.json()["ok"] is True
        assert registry.get_default() == "gpt-4o"

    def test_assign_role(self, client):
        test_client, registry, *_ = client
        registry.add("fast-model")
        resp = test_client.post(
            "/api/models/assign-role",
            json={"role": "fast", "model": "fast-model"},
        )
        assert resp.json()["ok"] is True
        assert registry.get_role("fast") == "fast-model"

    def test_unassign_role(self, client):
        test_client, registry, *_ = client
        registry.assign_role("fast", "model-a")
        resp = test_client.post(
            "/api/models/unassign-role", json={"role": "fast"}
        )
        assert resp.json()["ok"] is True
        assert "fast" not in registry.list_roles()


class TestModelsVerify:
    def test_verify_missing_keys(self, client):
        test_client, registry, *_ = client
        registry.add("anthropic/claude-sonnet-4-5")
        with patch(
            "supyagent.server.ui.routes.models.litellm",
            create=True,
        ) as mock_litellm:
            mock_litellm.validate_environment.return_value = {
                "keys_in_environment": False,
                "missing_keys": ["ANTHROPIC_API_KEY"],
            }
            # Patch the import inside the function
            with patch(
                "supyagent.server.ui.routes.models._get_missing_keys",
                return_value=["ANTHROPIC_API_KEY"],
            ):
                resp = test_client.post(
                    "/api/models/verify", json={"model": "anthropic/claude-sonnet-4-5"}
                )
        data = resp.json()
        assert data["ok"] is False
        assert data["status"] == "missing_keys"
        assert "ANTHROPIC_API_KEY" in data["missing_keys"]

    def test_verify_success(self, client):
        test_client, registry, config_mgr, _ = client
        registry.add("anthropic/claude-sonnet-4-5")
        config_mgr.set("ANTHROPIC_API_KEY", "sk-test-123")
        with (
            patch(
                "supyagent.server.ui.routes.models._get_missing_keys",
                return_value=[],
            ),
            patch(
                "litellm.completion",
                return_value={"choices": [{"message": {"content": "h"}}]},
            ),
        ):
            resp = test_client.post(
                "/api/models/verify", json={"model": "anthropic/claude-sonnet-4-5"}
            )
        data = resp.json()
        assert data["ok"] is True
        assert data["status"] == "verified"
        assert data["missing_keys"] == []
        assert data["error"] is None

    def test_verify_auth_error(self, client):
        test_client, registry, config_mgr, _ = client
        registry.add("anthropic/claude-sonnet-4-5")
        config_mgr.set("ANTHROPIC_API_KEY", "sk-bad-key")
        with (
            patch(
                "supyagent.server.ui.routes.models._get_missing_keys",
                return_value=[],
            ),
            patch(
                "litellm.completion",
                side_effect=Exception("AuthenticationError: Invalid API key"),
            ),
        ):
            resp = test_client.post(
                "/api/models/verify", json={"model": "anthropic/claude-sonnet-4-5"}
            )
        data = resp.json()
        assert data["ok"] is False
        assert data["status"] == "error"
        assert "AuthenticationError" in data["error"]
        assert data["missing_keys"] == []


class TestModelsStateMissingKeys:
    def test_state_includes_missing_keys(self, client):
        test_client, registry, *_ = client
        registry.add("anthropic/claude-sonnet-4-5")
        with patch(
            "supyagent.server.ui.routes.models._get_missing_keys",
            return_value=["ANTHROPIC_API_KEY"],
        ):
            resp = test_client.get("/api/models/state")
        data = resp.json()
        assert len(data["models"]) == 1
        assert "missing_keys" in data["models"][0]
        assert "ANTHROPIC_API_KEY" in data["models"][0]["missing_keys"]


class TestKeysAPI:
    def test_keys_status(self, client):
        test_client, *_ = client
        resp = test_client.get("/api/keys/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "keys" in data

    def test_set_and_delete_key(self, client):
        test_client, _, config_mgr, _ = client
        resp = test_client.post(
            "/api/keys/set",
            json={"key_name": "TEST_API_KEY", "value": "sk-test-123"},
        )
        assert resp.json()["ok"] is True

        resp = test_client.post(
            "/api/keys/delete", json={"key_name": "TEST_API_KEY"}
        )
        assert resp.json()["ok"] is True


class TestDoneEndpoint:
    def test_done(self, client):
        test_client, _, _, done_data = client
        resp = test_client.post(
            "/api/done", json={"action": "done", "data": {"from": "test"}}
        )
        assert resp.status_code == 200
        assert done_data["action"] == "done"
