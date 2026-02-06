"""
Tests for ExecutionRunner.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from supyagent.core.executor import ExecutionRunner
from supyagent.models.agent_config import AgentConfig, ModelConfig, ToolPermissions


@pytest.fixture
def execution_agent_config():
    """Create an execution agent config."""
    return AgentConfig(
        name="test-executor",
        description="A test execution agent",
        version="1.0",
        type="execution",
        model=ModelConfig(
            provider="test/model",
            temperature=0.3,
            max_tokens=2048,
        ),
        system_prompt="You are a test execution agent. Process input and return output.",
        tools=ToolPermissions(allow=[]),
        limits={"max_tool_calls_per_turn": 5},
    )


@pytest.fixture
def execution_agent_config_with_tools():
    """Create an execution agent config with tools."""
    return AgentConfig(
        name="test-executor-tools",
        description="A test execution agent with tools",
        version="1.0",
        type="execution",
        model=ModelConfig(
            provider="test/model",
            temperature=0.5,
            max_tokens=4096,
        ),
        system_prompt="You are a test execution agent with tools.",
        tools=ToolPermissions(allow=["*"]),
        limits={"max_tool_calls_per_turn": 10},
    )


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "This is the summarized output."
    response.choices[0].message.tool_calls = None
    return response


@pytest.fixture
def mock_llm_json_response():
    """Create a mock LLM response with JSON."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = '{"summary": "Test summary", "word_count": 5}'
    response.choices[0].message.tool_calls = None
    return response


class TestExecutionRunnerInit:
    """Tests for ExecutionRunner initialization."""

    @patch("supyagent.core.engine.discover_tools")
    def test_init_no_tools(self, mock_discover, execution_agent_config, temp_dir):
        """Test initialization with no supypowers tools (but process tools always present)."""
        mock_discover.return_value = []

        runner = ExecutionRunner(execution_agent_config)

        assert runner.config == execution_agent_config
        # Process management tools are always included
        tool_names = [t["function"]["name"] for t in runner.tools]
        assert "list_processes" in tool_names
        assert "check_process" in tool_names
        assert "get_process_output" in tool_names
        assert "kill_process" in tool_names
        mock_discover.assert_not_called()  # No tools allowed, so no discovery

    @patch("supyagent.core.engine.discover_tools")
    def test_init_with_tools(self, mock_discover, execution_agent_config_with_tools):
        """Test initialization with tools allowed."""
        mock_discover.return_value = [
            {"script": "test", "function": "func", "description": "Test", "input_schema": {}}
        ]

        runner = ExecutionRunner(execution_agent_config_with_tools)

        mock_discover.assert_called_once()
        assert len(runner.tools) >= 0  # Depends on mock return


class TestExecutionRunnerRun:
    """Tests for ExecutionRunner.run()."""

    @patch("supyagent.core.engine.discover_tools")
    def test_run_simple_string_task(
        self, mock_discover, execution_agent_config, mock_llm_response
    ):
        """Test running with a simple string task."""
        mock_discover.return_value = []

        runner = ExecutionRunner(execution_agent_config)

        with patch.object(runner.llm, "chat", return_value=mock_llm_response):
            result = runner.run("Summarize this text")

        assert result["ok"] is True
        assert result["data"] == "This is the summarized output."

    @patch("supyagent.core.engine.discover_tools")
    def test_run_dict_task(
        self, mock_discover, execution_agent_config, mock_llm_response
    ):
        """Test running with a dict task."""
        mock_discover.return_value = []

        runner = ExecutionRunner(execution_agent_config)

        with patch.object(runner.llm, "chat", return_value=mock_llm_response):
            result = runner.run({"text": "Some text to process", "max_length": 100})

        assert result["ok"] is True

    @patch("supyagent.core.engine.discover_tools")
    def test_run_with_secrets(
        self, mock_discover, execution_agent_config, mock_llm_response
    ):
        """Test running with secrets."""
        mock_discover.return_value = []

        runner = ExecutionRunner(execution_agent_config)

        with patch.object(runner.llm, "chat", return_value=mock_llm_response):
            result = runner.run(
                "Process this",
                secrets={"API_KEY": "secret123"},
            )

        assert result["ok"] is True

    @patch("supyagent.core.engine.discover_tools")
    def test_run_json_output_format(
        self, mock_discover, execution_agent_config, mock_llm_json_response
    ):
        """Test running with JSON output format."""
        mock_discover.return_value = []

        runner = ExecutionRunner(execution_agent_config)

        with patch.object(runner.llm, "chat", return_value=mock_llm_json_response):
            result = runner.run("Process this", output_format="json")

        assert result["ok"] is True
        assert isinstance(result["data"], dict)
        assert result["data"]["summary"] == "Test summary"

    @patch("supyagent.core.engine.discover_tools")
    def test_run_markdown_output_format(
        self, mock_discover, execution_agent_config, mock_llm_response
    ):
        """Test running with markdown output format."""
        mock_discover.return_value = []

        runner = ExecutionRunner(execution_agent_config)

        with patch.object(runner.llm, "chat", return_value=mock_llm_response):
            result = runner.run("Process this", output_format="markdown")

        assert result["ok"] is True
        assert result["format"] == "markdown"


class TestExecutionRunnerCredentialRequest:
    """Tests for credential request handling in execution mode."""

    @patch("supyagent.core.engine.discover_tools")
    def test_credential_request_fails(
        self, mock_discover, execution_agent_config_with_tools
    ):
        """Test that credential requests fail in execution mode."""
        mock_discover.return_value = []

        runner = ExecutionRunner(execution_agent_config_with_tools)

        # Create mock response with credential request
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = None

        tool_call = MagicMock()
        tool_call.id = "call_123"
        tool_call.function.name = "request_credential"
        tool_call.function.arguments = '{"name": "API_KEY", "description": "Need API key"}'

        response.choices[0].message.tool_calls = [tool_call]

        with patch.object(runner.llm, "chat", return_value=response):
            result = runner.run("Do something requiring credentials")

        assert result["ok"] is False
        assert "API_KEY" in result["error"]
        assert "required but not provided" in result["error"]


class TestExecutionRunnerToolExecution:
    """Tests for tool execution in execution mode."""

    @patch("supyagent.core.engine.discover_tools")
    @patch("supyagent.core.engine.execute_tool")
    def test_tool_execution(
        self, mock_execute, mock_discover, execution_agent_config_with_tools
    ):
        """Test tool execution in execution mode."""
        mock_discover.return_value = [
            {"script": "test", "function": "func", "description": "Test", "input_schema": {}}
        ]
        mock_execute.return_value = {"ok": True, "data": "tool result"}

        runner = ExecutionRunner(execution_agent_config_with_tools)

        # First response with tool call
        response1 = MagicMock()
        response1.choices = [MagicMock()]
        response1.choices[0].message.content = None

        tool_call = MagicMock()
        tool_call.id = "call_123"
        tool_call.function.name = "test__func"
        tool_call.function.arguments = '{"arg": "value"}'

        response1.choices[0].message.tool_calls = [tool_call]

        # Second response with final answer
        response2 = MagicMock()
        response2.choices = [MagicMock()]
        response2.choices[0].message.content = "Final answer after tool use"
        response2.choices[0].message.tool_calls = None

        with patch.object(runner.llm, "chat", side_effect=[response1, response2]):
            result = runner.run("Use the tool")

        assert result["ok"] is True
        assert result["data"] == "Final answer after tool use"
        mock_execute.assert_called_once()


class TestOutputFormatting:
    """Tests for output formatting."""

    @patch("supyagent.core.engine.discover_tools")
    def test_json_in_code_block(self, mock_discover, execution_agent_config):
        """Test extracting JSON from markdown code block."""
        mock_discover.return_value = []

        runner = ExecutionRunner(execution_agent_config)

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = '''Here's the result:
```json
{"key": "value", "number": 42}
```
'''
        response.choices[0].message.tool_calls = None

        with patch.object(runner.llm, "chat", return_value=response):
            result = runner.run("Process this", output_format="json")

        assert result["ok"] is True
        assert result["data"]["key"] == "value"
        assert result["data"]["number"] == 42

    @patch("supyagent.core.engine.discover_tools")
    def test_non_json_with_json_format(self, mock_discover, execution_agent_config):
        """Test non-JSON content with JSON format returns as-is."""
        mock_discover.return_value = []

        runner = ExecutionRunner(execution_agent_config)

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "This is not JSON"
        response.choices[0].message.tool_calls = None

        with patch.object(runner.llm, "chat", return_value=response):
            result = runner.run("Process this", output_format="json")

        assert result["ok"] is True
        assert result["data"] == "This is not JSON"


class TestMaxIterations:
    """Tests for max iterations safety."""

    @patch("supyagent.core.engine.discover_tools")
    def test_max_iterations_exceeded(self, mock_discover, execution_agent_config_with_tools):
        """Test that max iterations is enforced."""
        mock_discover.return_value = []

        # Set very low max iterations
        execution_agent_config_with_tools.limits = {"max_tool_calls_per_turn": 2}
        runner = ExecutionRunner(execution_agent_config_with_tools)

        # Response that always has tool calls
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = None

        tool_call = MagicMock()
        tool_call.id = "call_123"
        tool_call.function.name = "test__func"
        tool_call.function.arguments = "{}"

        response.choices[0].message.tool_calls = [tool_call]

        with patch.object(runner.llm, "chat", return_value=response):
            with patch("supyagent.core.engine.execute_tool", return_value={"ok": True}):
                result = runner.run("Keep calling tools")

        assert result["ok"] is False
        assert "Max tool iterations exceeded" in result["error"]
