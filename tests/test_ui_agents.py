"""
Tests for supyagent.server.ui.routes.agents — Agents Dashboard UI API.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from supyagent.core.registry import AgentInstance, AgentRegistry
from supyagent.server.ui import create_ui_app

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def temp_registry(tmp_path):
    """Create a temp AgentRegistry."""
    return AgentRegistry(base_dir=tmp_path / "registry")


def _make_agent_config(**overrides):
    """Create a mock AgentConfig with sensible defaults."""
    cfg = MagicMock()
    cfg.name = overrides.get("name", "test-agent")
    cfg.description = overrides.get("description", "A test agent")
    cfg.type = overrides.get("type", "interactive")
    cfg.model = MagicMock()
    cfg.model.provider = overrides.get("model_provider", "anthropic/claude-sonnet-4-5")
    cfg.delegates = overrides.get("delegates", [])
    cfg.workspace = overrides.get("workspace", None)
    cfg.service = MagicMock()
    cfg.service.enabled = overrides.get("service_enabled", False)
    cfg.memory = MagicMock()
    cfg.memory.enabled = overrides.get("memory_enabled", True)
    cfg.schedule = MagicMock()
    cfg.schedule.interval = "5m"
    cfg.schedule.max_events_per_cycle = 10
    cfg.schedule.prompt = None
    cfg.credentials = overrides.get("credentials", [])
    return cfg


def _make_session_meta(**overrides):
    """Create a mock SessionMeta."""
    meta = MagicMock()
    meta.session_id = overrides.get("session_id", "abc12345")
    meta.title = overrides.get("title", "Test session")
    meta.model = overrides.get("model", "anthropic/claude-sonnet-4-5")
    meta.created_at = overrides.get("created_at", datetime(2025, 1, 1, tzinfo=UTC))
    meta.updated_at = overrides.get("updated_at", datetime(2025, 1, 2, tzinfo=UTC))
    return meta


def _make_message(**overrides):
    """Create a mock Message."""
    msg = MagicMock()
    msg.type = overrides.get("type", "human")
    msg.content = overrides.get("content", "Hello")
    msg.tool_calls = overrides.get("tool_calls", None)
    msg.tool_call_id = overrides.get("tool_call_id", None)
    msg.name = overrides.get("name", None)
    msg.ts = overrides.get("ts", datetime(2025, 1, 1, 12, 0, tzinfo=UTC))
    return msg


# Patch targets: routes use deferred imports, so patch at the SOURCE modules
_P_REGISTRY = "supyagent.core.registry.AgentRegistry"
_P_SESSION_MGR = "supyagent.core.session_manager.SessionManager"
_P_CRED_MGR = "supyagent.core.credentials.CredentialManager"
_P_LOAD_CONFIG = "supyagent.models.agent_config.load_agent_config"
_P_CONTENT_TO_TEXT = "supyagent.utils.media.content_to_text"


@pytest.fixture
def client(temp_registry):
    """Create a test client with mocked dependencies."""
    done_data = {}

    def on_done(data):
        done_data.update(data)

    app = create_ui_app(mode="agents", done_callback=on_done)

    with patch(_P_REGISTRY, return_value=temp_registry):
        yield TestClient(app), temp_registry, done_data


# ── Page tests ──────────────────────────────────────────────────────


class TestAgentsPage:
    def test_get_agents_page(self, client):
        test_client, *_ = client
        resp = test_client.get("/agents")
        assert resp.status_code == 200
        assert "Agents Dashboard" in resp.text


# ── State endpoint tests ────────────────────────────────────────────


class TestAgentsState:
    def test_empty_state_no_agents_dir(self, client):
        """With no agents/ directory, returns empty configs."""
        test_client, *_ = client
        with patch("supyagent.server.ui.routes.agents.Path") as mock_path_cls:
            agents_dir = MagicMock()
            agents_dir.exists.return_value = False
            mock_path_cls.return_value = agents_dir

            resp = test_client.get("/api/agents/state")

        assert resp.status_code == 200
        data = resp.json()
        assert data["agents"] == []
        assert data["instances"] == []
        assert "configs" in data["stats"]

    def test_state_with_agents_and_instances(self, client, tmp_path):
        test_client, registry, _ = client

        # Add an instance directly
        inst = AgentInstance(
            name="assistant",
            instance_id="x1y2z3",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
            status="active",
        )
        registry._instances["x1y2z3"] = inst
        registry._save()

        mock_config = _make_agent_config(name="assistant")
        mock_sessions = [_make_session_meta()]

        # Create a temporary agents dir with a YAML file
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "assistant.yaml").write_text("name: assistant")

        with (
            patch(
                "supyagent.server.ui.routes.agents.Path",
                return_value=agents_dir,
            ),
            patch(_P_LOAD_CONFIG, return_value=mock_config),
            patch(_P_SESSION_MGR) as mock_sm,
        ):
            mock_sm.return_value.list_sessions.return_value = mock_sessions
            resp = test_client.get("/api/agents/state")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["instances"]) == 1
        assert data["instances"][0]["instance_id"] == "x1y2z3"
        assert data["instances"][0]["status"] == "active"
        assert data["stats"]["active"] == 1
        assert data["stats"]["total_sessions"] == 1

    def test_state_handles_config_error(self, client, tmp_path):
        """When an agent config fails to load, it shows up with an error."""
        test_client, *_ = client

        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "broken.yaml").write_text("name: broken")

        with (
            patch(
                "supyagent.server.ui.routes.agents.Path",
                return_value=agents_dir,
            ),
            patch(
                _P_LOAD_CONFIG,
                side_effect=Exception("Invalid config"),
            ),
        ):
            resp = test_client.get("/api/agents/state")

        data = resp.json()
        assert len(data["agents"]) == 1
        assert data["agents"][0]["name"] == "broken"
        assert "error" in data["agents"][0]


# ── Detail endpoint tests ───────────────────────────────────────────


class TestAgentDetail:
    def test_detail_not_found(self, client):
        test_client, *_ = client
        from supyagent.models.agent_config import AgentNotFoundError

        with patch(
            _P_LOAD_CONFIG,
            side_effect=AgentNotFoundError("ghost"),
        ):
            resp = test_client.get("/api/agents/ghost/detail")

        data = resp.json()
        assert data["ok"] is False
        assert "not found" in data["error"]

    def test_detail_success(self, client):
        test_client, *_ = client

        mock_config = _make_agent_config(name="myagent")
        meta = _make_session_meta(session_id="sess1")
        mock_session = MagicMock()
        mock_session.messages = [_make_message(), _make_message(type="ai")]

        with (
            patch(_P_LOAD_CONFIG, return_value=mock_config),
            patch(_P_SESSION_MGR) as mock_sm,
            patch(_P_CRED_MGR) as mock_cm,
            patch(
                "supyagent.server.ui.routes.agents._aggregate_telemetry",
                return_value={"available": False},
            ),
            patch(
                "supyagent.server.ui.routes.agents._get_memory_stats",
                return_value={"available": False},
            ),
        ):
            mock_sm.return_value.list_sessions.return_value = [meta]
            mock_sm.return_value.load_session.return_value = mock_session
            mock_cm.return_value.list_credentials.return_value = ["API_KEY"]

            resp = test_client.get("/api/agents/myagent/detail")

        data = resp.json()
        assert data["ok"] is True
        assert data["name"] == "myagent"
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["session_id"] == "sess1"
        assert data["sessions"][0]["message_count"] == 2
        assert data["credentials"] == ["API_KEY"]


# ── Session messages endpoint tests ─────────────────────────────────


class TestSessionMessages:
    def test_session_not_found(self, client):
        test_client, *_ = client

        with patch(_P_SESSION_MGR) as mock_sm:
            mock_sm.return_value.load_session.return_value = None
            resp = test_client.get("/api/agents/myagent/sessions/nonexistent")

        data = resp.json()
        assert data["ok"] is False

    def test_session_messages_returned(self, client):
        test_client, *_ = client

        msgs = [
            _make_message(type="human", content="Hi there"),
            _make_message(
                type="ai",
                content="Hello!",
                tool_calls=[{"function": {"name": "search"}}],
            ),
        ]
        mock_session = MagicMock()
        mock_session.messages = msgs
        mock_session.meta.title = "Chat"

        with (
            patch(_P_SESSION_MGR) as mock_sm,
            patch(_P_CONTENT_TO_TEXT, side_effect=lambda c: str(c)),
        ):
            mock_sm.return_value.load_session.return_value = mock_session
            resp = test_client.get("/api/agents/myagent/sessions/sess1")

        data = resp.json()
        assert data["ok"] is True
        assert data["title"] == "Chat"
        assert len(data["messages"]) == 2
        assert data["messages"][0]["type"] == "human"
        assert data["messages"][1]["tool_names"] == ["search"]


# ── Action endpoint tests ───────────────────────────────────────────


class TestCleanupAction:
    def test_cleanup_stale(self, client):
        test_client, registry, _ = client

        with patch.object(registry, "prune_stale", return_value=2):
            resp = test_client.post(
                "/api/agents/cleanup", json={"mode": "stale"}
            )

        data = resp.json()
        assert data["ok"] is True
        assert data["removed"] == 2

    def test_cleanup_completed(self, client):
        test_client, registry, _ = client

        with patch.object(registry, "cleanup_completed", return_value=3):
            resp = test_client.post(
                "/api/agents/cleanup", json={"mode": "completed"}
            )

        data = resp.json()
        assert data["ok"] is True
        assert data["removed"] == 3

    def test_cleanup_all(self, client):
        test_client, registry, _ = client

        inst = AgentInstance(
            name="a",
            instance_id="abc",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        registry._instances["abc"] = inst

        resp = test_client.post(
            "/api/agents/cleanup", json={"mode": "all"}
        )

        data = resp.json()
        assert data["ok"] is True
        assert data["removed"] == 1
        assert len(registry._instances) == 0


class TestRemoveInstance:
    def test_remove_existing(self, client):
        test_client, registry, _ = client

        inst = AgentInstance(
            name="a",
            instance_id="rem123",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        registry._instances["rem123"] = inst

        resp = test_client.post(
            "/api/agents/remove-instance",
            json={"instance_id": "rem123"},
        )

        data = resp.json()
        assert data["ok"] is True
        assert "rem123" not in registry._instances

    def test_remove_nonexistent(self, client):
        test_client, *_ = client

        resp = test_client.post(
            "/api/agents/remove-instance",
            json={"instance_id": "ghost"},
        )

        data = resp.json()
        assert data["ok"] is False
        assert "not found" in data["error"].lower()


class TestDeleteSession:
    def test_delete_session_success(self, client):
        test_client, *_ = client

        with patch(_P_SESSION_MGR) as mock_sm:
            mock_sm.return_value.delete_session.return_value = True
            resp = test_client.post(
                "/api/agents/myagent/delete-session",
                json={"session_id": "sess1"},
            )

        data = resp.json()
        assert data["ok"] is True

    def test_delete_session_not_found(self, client):
        test_client, *_ = client

        with patch(_P_SESSION_MGR) as mock_sm:
            mock_sm.return_value.delete_session.return_value = False
            resp = test_client.post(
                "/api/agents/myagent/delete-session",
                json={"session_id": "nonexistent"},
            )

        data = resp.json()
        assert data["ok"] is False


# ── Helper function tests ───────────────────────────────────────────


class TestTelemetryHelper:
    def test_no_telemetry_dir(self, tmp_path):
        from supyagent.server.ui.routes.agents import _aggregate_telemetry

        with patch(
            "supyagent.server.ui.routes.agents.Path",
            return_value=tmp_path / "missing",
        ):
            result = _aggregate_telemetry("test-agent")

        assert result["available"] is False

    def test_telemetry_aggregation(self, tmp_path):
        from supyagent.server.ui.routes.agents import _aggregate_telemetry

        tel_dir = tmp_path / "telemetry" / "test-agent"
        tel_dir.mkdir(parents=True)
        (tel_dir / "s1.jsonl").write_text(
            '{"type":"turn","duration_ms":100}\n'
            '{"type":"tool_call"}\n'
            '{"type":"llm_call","input_tokens":50,"output_tokens":20}\n'
        )
        (tel_dir / "s2.jsonl").write_text(
            '{"type":"error"}\n'
        )

        with patch(
            "supyagent.server.ui.routes.agents.Path",
        ) as mock_p:
            mock_path = MagicMock()
            mock_path.__truediv__ = lambda self, x: tel_dir
            mock_p.return_value = mock_path

            result = _aggregate_telemetry("test-agent")

        assert result["available"] is True
        assert result["sessions"] == 2
        assert result["turns"] == 1
        assert result["tool_calls"] == 1
        assert result["llm_calls"] == 1
        assert result["errors"] == 1
        assert result["input_tokens"] == 50
        assert result["output_tokens"] == 20
        assert result["total_duration_ms"] == 100


class TestMemoryHelper:
    def test_no_memory_db(self):
        from supyagent.server.ui.routes.agents import _get_memory_stats

        with patch(
            "supyagent.server.ui.routes.agents.Path",
        ) as mock_p:
            mock_home = MagicMock()
            mock_path = MagicMock()
            mock_path.exists.return_value = False
            mock_home.__truediv__ = lambda self, x: mock_path
            mock_path.__truediv__ = lambda self, x: mock_path
            mock_p.home.return_value = mock_home

            result = _get_memory_stats("test-agent")

        assert result["available"] is False
