"""
Tests for supyagent.server.ui.routes.hello — Hello Wizard UI API.
"""

import os
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from supyagent.core.config import ConfigManager
from supyagent.core.model_registry import ModelRegistry
from supyagent.server.ui import create_ui_app


@pytest.fixture
def temp_config(tmp_path):
    return ConfigManager(base_dir=tmp_path / "config")


@pytest.fixture
def temp_registry(tmp_path):
    return ModelRegistry(base_dir=tmp_path / "registry")


@pytest.fixture
def client(tmp_path, temp_config, temp_registry):
    """Create a test client with mocked dependencies."""
    done_data = {}

    def on_done(data):
        done_data.update(data)

    app = create_ui_app(mode="hello", done_callback=on_done)

    # Change to a temp directory for workspace operations
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    # Patch at source modules since routes use deferred imports
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
        yield TestClient(app), temp_registry, temp_config, done_data, tmp_path

    os.chdir(original_cwd)


class TestHelloPage:
    def test_get_hello_page(self, client):
        test_client, *_ = client
        resp = test_client.get("/hello")
        assert resp.status_code == 200
        assert "Welcome to Supyagent" in resp.text


class TestHelloState:
    def test_initial_state(self, client):
        test_client, *_ = client
        resp = test_client.get("/api/hello/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "service_connected" in data
        assert "has_models" in data
        assert "env_keys" in data
        assert "llm_keys" in data
        assert data["has_models"] is False


class TestServiceConnection:
    @patch("supyagent.core.service.request_device_code")
    def test_service_start_success(self, mock_request, client):
        test_client, *_ = client
        mock_request.return_value = {
            "device_code": "DC-123",
            "user_code": "ABCD-1234",
            "verification_uri": "https://example.com/device",
            "expires_in": 900,
            "interval": 5,
        }
        resp = test_client.post("/api/hello/service/start", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["user_code"] == "ABCD-1234"
        assert data["device_code"] == "DC-123"

    @patch("supyagent.core.service.request_device_code")
    def test_service_start_failure(self, mock_request, client):
        test_client, *_ = client
        mock_request.side_effect = ConnectionError("Cannot reach service")
        resp = test_client.post("/api/hello/service/start", json={})
        data = resp.json()
        assert data["ok"] is False
        assert "error" in data


class TestIntegrations:
    def test_integrations_list_no_service(self, client):
        test_client, *_ = client
        resp = test_client.get("/api/hello/integrations")
        assert resp.status_code == 200
        data = resp.json()
        assert "integrations" in data
        assert data["service_connected"] is False
        assert len(data["integrations"]) > 0

    def test_integration_connect_url(self, client):
        test_client, *_ = client
        resp = test_client.post(
            "/api/hello/integrations/connect", json={"provider": "google"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "google" in data["url"]


class TestProviders:
    def test_list_providers(self, client):
        test_client, *_ = client
        resp = test_client.get("/api/hello/providers")
        assert resp.status_code == 200
        data = resp.json()
        assert "providers" in data
        providers = data["providers"]
        assert len(providers) > 0
        names = [p["name"] for p in providers]
        assert "OpenAI" in names
        assert "Anthropic" in names


class TestImportEnvKeys:
    def test_import_env_keys(self, client):
        test_client, _, config_mgr, *_ = client
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-123"}):
            resp = test_client.post(
                "/api/hello/models/import-env",
                json={"key_names": ["OPENAI_API_KEY"]},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "OPENAI_API_KEY" in data["imported"]


class TestProfiles:
    def test_list_profiles(self, client):
        test_client, *_ = client
        resp = test_client.get("/api/hello/profiles")
        assert resp.status_code == 200
        data = resp.json()
        assert "profiles" in data
        assert "roles" in data
        profile_names = [p["name"] for p in data["profiles"]]
        assert "coding" in profile_names
        assert "full" in profile_names

    def test_install_profile(self, client):
        test_client, _, _, _, tmp_path = client
        resp = test_client.post(
            "/api/hello/profile/install",
            json={"profile": "coding", "model": None},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["profile"] == "coding"
        assert "agents" in data
        assert (tmp_path / "agents").exists()


class TestGoals:
    def test_save_goals(self, client):
        test_client, _, _, _, tmp_path = client
        resp = test_client.post(
            "/api/hello/goals",
            json={"goals": "Build an API\nWrite tests"},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        goals_file = tmp_path / "GOALS.md"
        assert goals_file.exists()
        content = goals_file.read_text()
        assert "Build an API" in content

    def test_save_empty_goals(self, client):
        test_client, *_ = client
        resp = test_client.post("/api/hello/goals", json={"goals": ""})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True


class TestSettings:
    def test_save_settings(self, client):
        test_client, _, _, _, tmp_path = client
        resp = test_client.post(
            "/api/hello/settings",
            json={
                "execution_mode": "yolo",
                "heartbeat_enabled": True,
                "heartbeat_interval": "10m",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        ws_config = tmp_path / ".supyagent" / "workspace.yaml"
        assert ws_config.exists()


class TestFinish:
    def test_finish_signals_done(self, client):
        test_client, _, _, done_data, _ = client
        resp = test_client.post(
            "/api/hello/finish",
            json={"start_chat": True, "agent_name": "assistant"},
        )
        assert resp.status_code == 200
        assert done_data["action"] == "hello_done"
        assert done_data["start_chat"] is True
        assert done_data["agent_name"] == "assistant"

    def test_finish_no_chat(self, client):
        test_client, _, _, done_data, _ = client
        resp = test_client.post(
            "/api/hello/finish",
            json={"start_chat": False},
        )
        assert resp.status_code == 200
        assert done_data["start_chat"] is False
