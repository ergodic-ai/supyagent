"""
Pytest fixtures for supyagent tests.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from supyagent.models.agent_config import AgentConfig, ModelConfig, ToolPermissions


@pytest.fixture(autouse=True)
def _clean_env():
    """Prevent environment variable pollution between tests.

    Some CLI commands (chat, run, batch, plan) call load_config() which loads
    real credentials from ~/.supyagent/config/ into os.environ. Without this
    fixture, those variables persist across tests, causing failures in tests
    that expect a clean environment (e.g. test_status_not_connected) and hangs
    in tests whose input flow depends on whether a service key is present
    (e.g. test_hello wizard tests).
    """
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def agents_dir(temp_dir):
    """Create a temporary agents directory."""
    agents = temp_dir / "agents"
    agents.mkdir()
    return agents


@pytest.fixture
def sessions_dir(temp_dir):
    """Create a temporary sessions directory."""
    sessions = temp_dir / ".supyagent" / "sessions"
    sessions.mkdir(parents=True)
    return sessions


@pytest.fixture
def sample_agent_yaml():
    """Sample agent YAML content."""
    return '''name: test-agent
description: A test agent
version: "1.0"
type: interactive

model:
  provider: anthropic/claude-3-5-sonnet-20241022
  temperature: 0.7
  max_tokens: 4096

system_prompt: |
  You are a helpful test assistant.

tools:
  allow:
    - "*"

limits:
  max_tool_calls_per_turn: 10
'''


@pytest.fixture
def sample_agent_config():
    """Create a sample AgentConfig."""
    return AgentConfig(
        name="test-agent",
        description="A test agent",
        version="1.0",
        type="interactive",
        model=ModelConfig(
            provider="anthropic/claude-3-5-sonnet-20241022",
            temperature=0.7,
            max_tokens=4096,
        ),
        system_prompt="You are a helpful test assistant.",
        tools=ToolPermissions(allow=["*"]),
        limits={"max_tool_calls_per_turn": 10},
    )


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "Hello! I'm a test response."
    response.choices[0].message.tool_calls = None
    return response


@pytest.fixture
def mock_llm_response_with_tool_call():
    """Create a mock LLM response with a tool call."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = None

    tool_call = MagicMock()
    tool_call.id = "call_123"
    tool_call.function.name = "test__hello"
    tool_call.function.arguments = '{"name": "World"}'

    response.choices[0].message.tool_calls = [tool_call]
    return response
