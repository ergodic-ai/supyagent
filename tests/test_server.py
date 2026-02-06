"""Integration tests for the FastAPI server endpoints."""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from supyagent.server.app import create_app
from supyagent.server.dependencies import reset_agent_pool


@pytest.fixture(autouse=True)
def _reset_pool():
    """Reset the agent pool between tests."""
    reset_agent_pool()
    yield
    reset_agent_pool()


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "supyagent"


class TestListAgents:
    def test_no_agents_dir(self, client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        resp = client.get("/api/agents")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_with_agents(self, client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        # Create agents dir with a YAML
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "myagent.yaml").write_text(
            "name: myagent\n"
            "description: Test agent\n"
            "type: interactive\n"
            "model:\n"
            "  provider: openai\n"
            "  name: gpt-4o\n"
            "system_prompt: You are a test agent.\n"
            "tools:\n"
            "  allow:\n"
            "    - shell_exec\n"
        )

        # Also need supypowers dir for tool discovery
        (tmp_path / "supypowers").mkdir()

        resp = client.get("/api/agents")
        assert resp.status_code == 200
        agents = resp.json()
        assert len(agents) == 1
        assert agents[0]["name"] == "myagent"
        assert agents[0]["description"] == "Test agent"
        assert agents[0]["tools_count"] == 1


class TestGetAgent:
    def test_not_found(self, client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agents").mkdir()
        resp = client.get("/api/agents/nonexistent")
        assert resp.status_code == 404

    def test_found(self, client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "bot.yaml").write_text(
            "name: bot\n"
            "description: A bot\n"
            "type: interactive\n"
            "model:\n"
            "  provider: anthropic\n"
            "  name: claude-sonnet-4-5-20250929\n"
            "system_prompt: Hello\n"
            "tools:\n"
            "  allow: []\n"
        )
        (tmp_path / "supypowers").mkdir()

        resp = client.get("/api/agents/bot")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "bot"
        assert data["model"] == "anthropic"


class TestTools:
    def test_list_tools(self, client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Create a minimal supypowers dir with a tool
        sp_dir = tmp_path / "supypowers"
        sp_dir.mkdir()
        (sp_dir / "test_tool.py").write_text(
            'def test_func(x: str) -> str:\n'
            '    """A test tool."""\n'
            '    return x\n'
        )

        resp = client.get("/api/tools")
        assert resp.status_code == 200
        tools = resp.json()
        assert isinstance(tools, list)


class TestSessions:
    def test_list_sessions_empty(self, client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        # Mock the agent pool to return an empty session list
        mock_pool = MagicMock()
        mock_pool.session_manager.list_sessions.return_value = []

        with patch("supyagent.server.routes.sessions.get_agent_pool", return_value=mock_pool):
            resp = client.get("/api/agents/myagent/sessions")

        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_session_not_found(self, client):
        mock_pool = MagicMock()
        mock_pool.session_manager.load_session.return_value = None

        with patch("supyagent.server.routes.sessions.get_agent_pool", return_value=mock_pool):
            resp = client.get("/api/agents/myagent/sessions/nonexistent")

        assert resp.status_code == 404

    def test_delete_session_not_found(self, client):
        mock_pool = MagicMock()
        mock_pool.session_manager.delete_session.return_value = False

        with patch("supyagent.server.routes.sessions.get_agent_pool", return_value=mock_pool):
            resp = client.delete("/api/agents/myagent/sessions/nonexistent")

        assert resp.status_code == 404

    def test_delete_session_success(self, client):
        mock_pool = MagicMock()
        mock_pool.session_manager.delete_session.return_value = True

        with patch("supyagent.server.routes.sessions.get_agent_pool", return_value=mock_pool):
            resp = client.delete("/api/agents/myagent/sessions/sess123")

        assert resp.status_code == 200
        assert resp.json() == {"ok": True}


class TestChat:
    def test_no_user_message(self, client):
        mock_pool = MagicMock()

        with patch("supyagent.server.routes.chat.get_agent_pool", return_value=mock_pool):
            resp = client.post(
                "/api/chat",
                json={
                    "messages": [{"role": "system", "content": "You are helpful."}],
                    "agent": "myagent",
                },
            )

        assert resp.status_code == 200
        body = resp.text
        # Should contain an error about no user message
        assert "3:" in body
        assert "No user message" in body

    def test_agent_not_found_error(self, client):
        mock_pool = MagicMock()
        mock_pool.get_or_create.side_effect = Exception("Agent 'bad' not found")

        with patch("supyagent.server.routes.chat.get_agent_pool", return_value=mock_pool):
            resp = client.post(
                "/api/chat",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "agent": "bad",
                },
            )

        assert resp.status_code == 200
        body = resp.text
        assert "3:" in body
        assert "not found" in body

    def test_streaming_chat(self, client):
        """Test that a successful chat stream produces valid AI SDK format."""
        mock_agent = MagicMock()
        mock_session_meta = MagicMock()
        mock_session_meta.session_id = "sess_test"
        mock_agent.session.meta = mock_session_meta

        # Simulate agent streaming events
        def fake_stream(msg):
            yield ("text", "Hello ")
            yield ("text", "world!")
            yield ("done", "")

        mock_agent.send_message_stream = fake_stream

        import threading
        mock_pool = MagicMock()
        mock_pool.get_or_create.return_value = mock_agent
        mock_pool.get_lock.return_value = threading.Lock()

        with patch("supyagent.server.routes.chat.get_agent_pool", return_value=mock_pool):
            resp = client.post(
                "/api/chat",
                json={
                    "messages": [{"role": "user", "content": "hello"}],
                    "agent": "test",
                },
            )

        assert resp.status_code == 200
        assert "x-vercel-ai-data-stream" in resp.headers
        assert resp.headers["x-vercel-ai-data-stream"] == "v1"

        body = resp.text
        lines = [l for l in body.strip().split("\n") if l]

        # First line: message start
        assert lines[0].startswith("f:")
        msg_start = json.loads(lines[0][2:])
        assert "messageId" in msg_start

        # Text deltas
        text_lines = [l for l in lines if l.startswith("0:")]
        assert len(text_lines) == 2
        assert json.loads(text_lines[0][2:]) == "Hello "
        assert json.loads(text_lines[1][2:]) == "world!"

        # Step finish and message finish
        step_lines = [l for l in lines if l.startswith("e:")]
        assert len(step_lines) >= 1

        finish_lines = [l for l in lines if l.startswith("d:")]
        assert len(finish_lines) == 1
        finish = json.loads(finish_lines[0][2:])
        assert finish["finishReason"] == "stop"

    def test_streaming_with_tools(self, client):
        """Test tool call and result events in the stream."""
        mock_agent = MagicMock()
        mock_session_meta = MagicMock()
        mock_session_meta.session_id = "sess_tools"
        mock_agent.session.meta = mock_session_meta

        def fake_stream(msg):
            yield ("text", "Let me check that.")
            yield ("tool_start", {
                "id": "call_abc",
                "name": "shell_exec",
                "arguments": '{"cmd": "ls"}',
            })
            yield ("tool_end", {
                "id": "call_abc",
                "name": "shell_exec",
                "result": "file1.txt\nfile2.txt",
            })
            yield ("text", "Here are your files.")
            yield ("done", "")

        mock_agent.send_message_stream = fake_stream

        import threading
        mock_pool = MagicMock()
        mock_pool.get_or_create.return_value = mock_agent
        mock_pool.get_lock.return_value = threading.Lock()

        with patch("supyagent.server.routes.chat.get_agent_pool", return_value=mock_pool):
            resp = client.post(
                "/api/chat",
                json={
                    "messages": [{"role": "user", "content": "list files"}],
                    "agent": "test",
                },
            )

        assert resp.status_code == 200
        body = resp.text
        lines = [l for l in body.strip().split("\n") if l]

        # Tool call
        tool_call_lines = [l for l in lines if l.startswith("9:")]
        assert len(tool_call_lines) == 1
        tc = json.loads(tool_call_lines[0][2:])
        assert tc["toolCallId"] == "call_abc"
        assert tc["toolName"] == "shell_exec"
        assert tc["args"] == {"cmd": "ls"}

        # Tool result
        tool_result_lines = [l for l in lines if l.startswith("a:")]
        assert len(tool_result_lines) == 1
        tr = json.loads(tool_result_lines[0][2:])
        assert tr["toolCallId"] == "call_abc"
        assert tr["result"] == "file1.txt\nfile2.txt"

        # Step finish after tool (isContinued)
        step_lines = [l for l in lines if l.startswith("e:")]
        # Should have at least 2: one for tool-calls, one for stop
        assert len(step_lines) >= 2
        tool_step = json.loads(step_lines[0][2:])
        assert tool_step["finishReason"] == "tool-calls"
        assert tool_step["isContinued"] is True


class TestCORS:
    def test_cors_headers(self):
        app = create_app(cors_origins=["http://localhost:3000"])
        client = TestClient(app)

        resp = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"
