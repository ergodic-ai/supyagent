"""
Comprehensive tests for nested agent delegation.

Covers:
1. Agent → Agent subprocess delegation mechanics
2. Agent → Agent → Agent (3-level) nested chains
3. Agent → Agent → Tool execution through delegation
4. Context passing and summarization skip optimization
5. Depth tracking and max-depth enforcement
6. CLI exec command with delegation support
7. Real subprocess chain integration tests
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from supyagent.core.context import DelegationContext, summarize_conversation
from supyagent.core.delegation import DelegationManager
from supyagent.core.registry import AgentRegistry
from supyagent.core.supervisor import (
    ProcessSupervisor,
    ProcessStatus,
    SupervisorConfig,
    TimeoutAction,
    reset_supervisor,
    run_supervisor_coroutine,
)
from supyagent.models.agent_config import (
    AgentConfig,
    AgentNotFoundError,
    ModelConfig,
    SupervisorSettings,
    ToolPermissions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_dir():
    """Temp directory for registry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def supervisor(tmp_path):
    """Fresh supervisor with short timeouts for tests."""
    reset_supervisor()
    config = SupervisorConfig(
        default_timeout=2.0,
        max_execution_time=10.0,
        on_timeout=TimeoutAction.BACKGROUND,
        log_dir=tmp_path / "logs",
        max_background_processes=5,
    )
    return ProcessSupervisor(config)


@pytest.fixture(autouse=True)
def _cleanup():
    """Ensure supervisor is reset after every test."""
    yield
    reset_supervisor()


def _make_config(
    name: str,
    agent_type: str = "interactive",
    delegates: list[str] | None = None,
    delegation_mode: str = "subprocess",
) -> AgentConfig:
    """Helper: create an AgentConfig."""
    return AgentConfig(
        name=name,
        description=f"Test {name} agent",
        version="1.0",
        type=agent_type,
        model=ModelConfig(provider="test/model", temperature=0.7),
        system_prompt=f"You are the {name} agent.",
        tools=ToolPermissions(allow=["*"]),
        delegates=delegates or [],
        supervisor=SupervisorSettings(
            default_delegation_mode=delegation_mode,
            delegation_timeout=30,
        ),
    )


def _mock_agent(config: AgentConfig, messages: list | None = None):
    """Helper: create a mock agent object."""
    agent = MagicMock()
    agent.config = config
    agent.messages = messages or []
    agent.llm = MagicMock()
    agent.instance_id = None
    return agent


# ===========================================================================
# 1. Single-level delegation: Agent → Agent
# ===========================================================================


class TestSingleLevelDelegation:
    """Tests for direct parent → child delegation."""

    def test_delegate_to_child_subprocess_calls_supervisor(self, temp_dir):
        """Subprocess delegation should invoke the ProcessSupervisor."""
        config = _make_config("parent", delegates=["child"])
        agent = _mock_agent(config)

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            child_config = _make_config("child", agent_type="execution")
            mock_load.return_value = child_config

            registry = AgentRegistry(base_dir=temp_dir)
            manager = DelegationManager(registry, agent)

            # Mock the supervisor (lazy import inside _delegate_subprocess)
            with patch("supyagent.core.supervisor.run_supervisor_coroutine") as mock_run, \
                 patch("supyagent.core.supervisor.get_supervisor") as mock_get_sup:
                mock_supervisor = MagicMock()
                mock_get_sup.return_value = mock_supervisor
                mock_run.return_value = {"ok": True, "data": "child result"}

                mock_tool_call = MagicMock()
                mock_tool_call.function.name = "delegate_to_child"
                mock_tool_call.function.arguments = json.dumps({
                    "task": "Do something",
                    "mode": "subprocess",
                })

                result = manager.execute_delegation(mock_tool_call)

        assert result["ok"] is True
        assert result["data"] == "child result"
        mock_run.assert_called_once()

    def test_delegate_to_child_in_process(self, temp_dir):
        """In-process delegation should run execution runner directly."""
        config = _make_config("parent", delegates=["child"])
        agent = _mock_agent(config)

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            child_config = _make_config("child", agent_type="execution")
            mock_load.return_value = child_config

            registry = AgentRegistry(base_dir=temp_dir)
            manager = DelegationManager(registry, agent)

            with patch("supyagent.core.executor.ExecutionRunner") as mock_runner_cls:
                mock_runner = MagicMock()
                mock_runner.run.return_value = {"ok": True, "data": "in-process result"}
                mock_runner_cls.return_value = mock_runner

                mock_tool_call = MagicMock()
                mock_tool_call.function.name = "delegate_to_child"
                mock_tool_call.function.arguments = json.dumps({
                    "task": "Do something in-process",
                    "mode": "in_process",
                })

                result = manager.execute_delegation(mock_tool_call)

        assert result["ok"] is True
        assert result["data"] == "in-process result"

    def test_subprocess_cmd_includes_task_and_context(self, temp_dir):
        """The subprocess command should include --task and --context."""
        config = _make_config("parent", delegates=["child"])
        agent = _mock_agent(config)

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            child_config = _make_config("child", agent_type="execution")
            mock_load.return_value = child_config

            registry = AgentRegistry(base_dir=temp_dir)
            manager = DelegationManager(registry, agent)

            with patch("supyagent.core.supervisor.run_supervisor_coroutine") as mock_run, \
                 patch("supyagent.core.supervisor.get_supervisor") as mock_get_sup:

                mock_supervisor = MagicMock()
                mock_get_sup.return_value = mock_supervisor

                # Capture what gets passed to supervisor.execute
                captured_coro = None
                def capture_coro(coro):
                    # The coro is the supervisor.execute(...) call
                    # We can't easily inspect it, but we verify mock_supervisor.execute was called
                    return {"ok": True, "data": "result"}

                mock_run.side_effect = capture_coro

                mock_tool_call = MagicMock()
                mock_tool_call.function.name = "delegate_to_child"
                mock_tool_call.function.arguments = json.dumps({
                    "task": "Write a haiku",
                })

                result = manager.execute_delegation(mock_tool_call)

        assert result["ok"] is True
        # Verify supervisor.execute was called with correct args
        mock_supervisor.execute.assert_called_once()
        call_kwargs = mock_supervisor.execute.call_args
        cmd = call_kwargs[0][0]  # First positional arg
        assert "supyagent" in cmd
        assert "exec" in cmd
        assert "child" in cmd
        assert "--task" in cmd
        assert "--output" in cmd

    def test_delegate_to_unknown_agent_fails(self, temp_dir):
        """Delegation to non-existent agent returns error."""
        config = _make_config("parent", delegates=["child"])
        agent = _mock_agent(config)

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            mock_load.return_value = _make_config("child")

            registry = AgentRegistry(base_dir=temp_dir)
            manager = DelegationManager(registry, agent)

        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "delegate_to_nonexistent"
        mock_tool_call.function.arguments = json.dumps({"task": "anything"})

        result = manager.execute_delegation(mock_tool_call)
        assert result["ok"] is False
        assert "not in the delegates list" in result["error"]

    def test_delegate_with_background_flag(self, temp_dir):
        """Background delegation should pass force_background=True."""
        config = _make_config("parent", delegates=["child"])
        agent = _mock_agent(config)

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            child_config = _make_config("child", agent_type="execution")
            mock_load.return_value = child_config

            registry = AgentRegistry(base_dir=temp_dir)
            manager = DelegationManager(registry, agent)

            with patch("supyagent.core.supervisor.run_supervisor_coroutine") as mock_run, \
                 patch("supyagent.core.supervisor.get_supervisor") as mock_get_sup:
                mock_supervisor = MagicMock()
                mock_get_sup.return_value = mock_supervisor
                mock_run.return_value = {
                    "ok": True,
                    "data": {"status": "backgrounded", "process_id": "agent_1"},
                }

                mock_tool_call = MagicMock()
                mock_tool_call.function.name = "delegate_to_child"
                mock_tool_call.function.arguments = json.dumps({
                    "task": "Long running task",
                    "background": True,
                })

                result = manager.execute_delegation(mock_tool_call)

        assert result["ok"] is True
        # Verify force_background was passed
        call_kwargs = mock_supervisor.execute.call_args
        assert call_kwargs.kwargs.get("force_background") is True

    def test_spawn_agent_delegates_correctly(self, temp_dir):
        """spawn_agent tool should delegate to the correct agent."""
        config = _make_config("parent", delegates=["worker", "writer"])
        agent = _mock_agent(config)

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            mock_load.return_value = _make_config("worker", agent_type="execution")

            registry = AgentRegistry(base_dir=temp_dir)
            manager = DelegationManager(registry, agent)

            with patch("supyagent.core.supervisor.run_supervisor_coroutine") as mock_run, \
                 patch("supyagent.core.supervisor.get_supervisor") as mock_get_sup:
                mock_supervisor = MagicMock()
                mock_get_sup.return_value = mock_supervisor
                mock_run.return_value = {"ok": True, "data": "spawned result"}

                mock_tool_call = MagicMock()
                mock_tool_call.function.name = "spawn_agent"
                mock_tool_call.function.arguments = json.dumps({
                    "agent_type": "worker",
                    "task": "Compute 2+2",
                })

                result = manager.execute_delegation(mock_tool_call)

        assert result["ok"] is True


# ===========================================================================
# 2. Context passing and summarization
# ===========================================================================


class TestContextPassing:
    """Tests for context building and summarization optimization."""

    def test_short_conversation_skips_summarization(self, temp_dir):
        """Conversations with ≤4 non-system messages should skip LLM summarization."""
        config = _make_config("parent", delegates=["child"])
        agent = _mock_agent(config, messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ])

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            mock_load.return_value = _make_config("child")

            registry = AgentRegistry(base_dir=temp_dir)
            manager = DelegationManager(registry, agent)

        context = manager._build_context("Do something")

        # Should NOT have called the LLM for summarization
        agent.llm.chat.assert_not_called()
        assert context.conversation_summary is None

    def test_long_conversation_triggers_summarization(self, temp_dir):
        """Conversations with >4 non-system messages should trigger LLM summarization."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "msg 1"},
            {"role": "assistant", "content": "reply 1"},
            {"role": "user", "content": "msg 2"},
            {"role": "assistant", "content": "reply 2"},
            {"role": "user", "content": "msg 3"},  # 5th non-system message
        ]
        config = _make_config("parent", delegates=["child"])
        agent = _mock_agent(config, messages=messages)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "User asked three questions."
        agent.llm.chat.return_value = mock_response

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            mock_load.return_value = _make_config("child")

            registry = AgentRegistry(base_dir=temp_dir)
            manager = DelegationManager(registry, agent)

        context = manager._build_context("Do something")

        # SHOULD have called the LLM for summarization
        agent.llm.chat.assert_called_once()
        assert context.conversation_summary == "User asked three questions."

    def test_context_includes_parent_info(self, temp_dir):
        """Context prompt should include parent agent name and task."""
        config = _make_config("planner", delegates=["coder"])
        agent = _mock_agent(config)

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            mock_load.return_value = _make_config("coder")

            registry = AgentRegistry(base_dir=temp_dir)
            manager = DelegationManager(registry, agent)

        context = manager._build_context("Write a Python function")
        prompt = context.to_prompt()

        assert "planner" in prompt
        assert "Write a Python function" in prompt

    def test_context_with_extra_context(self, temp_dir):
        """Extra context should be included as relevant facts."""
        config = _make_config("planner", delegates=["coder"])
        agent = _mock_agent(config)

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            mock_load.return_value = _make_config("coder")

            registry = AgentRegistry(base_dir=temp_dir)
            manager = DelegationManager(registry, agent)

        context = manager._build_context(
            "Write code", extra_context="Use TypeScript"
        )
        prompt = context.to_prompt()

        assert "TypeScript" in prompt

    def test_context_prompt_empty_messages(self, temp_dir):
        """Agent with no messages should produce context without summary."""
        config = _make_config("parent", delegates=["child"])
        agent = _mock_agent(config, messages=[])

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            mock_load.return_value = _make_config("child")

            registry = AgentRegistry(base_dir=temp_dir)
            manager = DelegationManager(registry, agent)

        context = manager._build_context("Do task")
        assert context.conversation_summary is None
        assert context.parent_agent == "parent"

    def test_context_to_prompt_full(self):
        """DelegationContext.to_prompt() should include all fields when present."""
        context = DelegationContext(
            parent_agent="orchestrator",
            parent_task="Build a web app",
            conversation_summary="User wants a React dashboard.",
            relevant_facts=["Use React 18", "Must support dark mode"],
            shared_data={"framework": "React", "version": "18"},
        )
        prompt = context.to_prompt()

        assert "orchestrator" in prompt
        assert "Build a web app" in prompt
        assert "React dashboard" in prompt
        assert "React 18" in prompt
        assert "dark mode" in prompt
        assert "framework" in prompt


# ===========================================================================
# 3. Depth tracking and max-depth enforcement
# ===========================================================================


class TestDepthTracking:
    """Tests for delegation depth tracking and limits."""

    def test_initial_depth_is_zero(self, temp_dir):
        """A top-level agent should have depth 0."""
        config = _make_config("top", delegates=["sub"])
        agent = _mock_agent(config)

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            mock_load.return_value = _make_config("sub")

            registry = AgentRegistry(base_dir=temp_dir)
            manager = DelegationManager(registry, agent)

        depth = registry.get_depth(manager.parent_id)
        assert depth == 0

    def test_child_depth_increments(self, temp_dir):
        """A child agent should have depth = parent depth + 1."""
        registry = AgentRegistry(base_dir=temp_dir)

        # Register parent at depth 0
        parent = MagicMock()
        parent.config.name = "parent"
        parent_id = registry.register(parent)
        assert registry.get_depth(parent_id) == 0

        # Register child at depth 1
        child = MagicMock()
        child.config.name = "child"
        child_id = registry.register(child, parent_id=parent_id)
        assert registry.get_depth(child_id) == 1

    def test_grandchild_depth(self, temp_dir):
        """Grandchild should have depth 2."""
        registry = AgentRegistry(base_dir=temp_dir)

        grandparent = MagicMock()
        grandparent.config.name = "grandparent"
        gp_id = registry.register(grandparent)

        parent = MagicMock()
        parent.config.name = "parent"
        p_id = registry.register(parent, parent_id=gp_id)

        child = MagicMock()
        child.config.name = "child"
        c_id = registry.register(child, parent_id=p_id)

        assert registry.get_depth(gp_id) == 0
        assert registry.get_depth(p_id) == 1
        assert registry.get_depth(c_id) == 2

    def test_max_depth_prevents_registration(self, temp_dir):
        """Registration should fail when MAX_DEPTH is exceeded."""
        registry = AgentRegistry(base_dir=temp_dir)

        # Build chain to MAX_DEPTH
        current_id = None
        for i in range(AgentRegistry.MAX_DEPTH):
            agent = MagicMock()
            agent.config.name = f"agent-{i}"
            current_id = registry.register(agent, parent_id=current_id)

        # Should be at MAX_DEPTH - 1 (0-indexed)
        assert registry.get_depth(current_id) == AgentRegistry.MAX_DEPTH - 1

        # One more should succeed (at exactly MAX_DEPTH)
        agent = MagicMock()
        agent.config.name = "agent-at-max"
        max_id = registry.register(agent, parent_id=current_id)
        assert registry.get_depth(max_id) == AgentRegistry.MAX_DEPTH

        # Beyond MAX_DEPTH should fail
        agent = MagicMock()
        agent.config.name = "agent-over-max"
        with pytest.raises(ValueError, match="Maximum delegation depth"):
            registry.register(agent, parent_id=max_id)

    def test_delegation_refuses_at_max_depth(self, temp_dir):
        """DelegationManager should refuse to delegate at max depth."""
        registry = AgentRegistry(base_dir=temp_dir)

        # Build chain to MAX_DEPTH
        current_id = None
        for i in range(AgentRegistry.MAX_DEPTH):
            agent = MagicMock()
            agent.config.name = f"agent-{i}"
            current_id = registry.register(agent, parent_id=current_id)

        # Now create a DelegationManager at MAX_DEPTH
        config = _make_config("deep-agent", delegates=["child"])
        agent = _mock_agent(config)

        # Keep the patch active through execute_delegation so load_agent_config works
        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            mock_load.return_value = _make_config("child")

            manager = DelegationManager(registry, agent, grandparent_instance_id=current_id)

            # The manager's parent is now at depth MAX_DEPTH
            assert registry.get_depth(manager.parent_id) == AgentRegistry.MAX_DEPTH

            # Attempting to delegate should fail
            mock_tool_call = MagicMock()
            mock_tool_call.function.name = "delegate_to_child"
            mock_tool_call.function.arguments = json.dumps({"task": "Do something"})

            result = manager.execute_delegation(mock_tool_call)
            assert result["ok"] is False
            assert "Maximum delegation depth" in result["error"]

    def test_parallel_children_same_depth(self, temp_dir):
        """Multiple children of the same parent should have the same depth."""
        registry = AgentRegistry(base_dir=temp_dir)

        parent = MagicMock()
        parent.config.name = "parent"
        p_id = registry.register(parent)

        for name in ["child_a", "child_b", "child_c"]:
            child = MagicMock()
            child.config.name = name
            c_id = registry.register(child, parent_id=p_id)
            assert registry.get_depth(c_id) == 1


# ===========================================================================
# 4. Nested delegation chain: Agent → Agent → Agent
# ===========================================================================


class TestNestedDelegation:
    """Tests for multi-level delegation chains."""

    def test_two_level_chain_in_process(self, temp_dir):
        """Parent → Middle → Worker in-process chain works."""
        # Set up configs
        parent_config = _make_config(
            "parent", delegates=["middle"], delegation_mode="in_process"
        )
        middle_config = _make_config(
            "middle", delegates=["worker"], delegation_mode="in_process"
        )
        worker_config = _make_config("worker", agent_type="execution")

        parent_agent = _mock_agent(parent_config)

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            # First call: loading "middle" config for parent
            mock_load.return_value = middle_config

            registry = AgentRegistry(base_dir=temp_dir)
            parent_mgr = DelegationManager(registry, parent_agent)

        # Verify parent is at depth 0
        assert registry.get_depth(parent_mgr.parent_id) == 0

        # Simulate middle agent creation at depth 1
        middle_agent = MagicMock()
        middle_agent.config = middle_config
        middle_id = registry.register(middle_agent, parent_id=parent_mgr.parent_id)
        assert registry.get_depth(middle_id) == 1

        # Simulate worker creation at depth 2
        worker_agent = MagicMock()
        worker_agent.config = worker_config
        worker_id = registry.register(worker_agent, parent_id=middle_id)
        assert registry.get_depth(worker_id) == 2

    def test_three_level_subprocess_commands(self, temp_dir):
        """Verify the correct subprocess commands for a 3-level chain."""
        parent_config = _make_config("tester", delegates=["middle"])
        parent_agent = _mock_agent(parent_config)

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            mock_load.return_value = _make_config("middle")

            registry = AgentRegistry(base_dir=temp_dir)
            manager = DelegationManager(registry, parent_agent)

            with patch("supyagent.core.supervisor.run_supervisor_coroutine") as mock_run, \
                 patch("supyagent.core.supervisor.get_supervisor") as mock_get_sup:
                mock_supervisor = MagicMock()
                mock_get_sup.return_value = mock_supervisor
                mock_run.return_value = {"ok": True, "data": "final result"}

                mock_tool_call = MagicMock()
                mock_tool_call.function.name = "delegate_to_middle"
                mock_tool_call.function.arguments = json.dumps({
                    "task": "Delegate to worker to compute 1+1",
                })

                result = manager.execute_delegation(mock_tool_call)

        assert result["ok"] is True
        # Verify the command targets the correct agent
        call_args = mock_supervisor.execute.call_args
        cmd = call_args[0][0]
        assert cmd[0] == "supyagent"
        assert cmd[1] == "exec"
        assert cmd[2] == "middle"

    def test_delegation_chain_metadata(self, temp_dir):
        """Metadata should include agent name and task excerpt."""
        config = _make_config("parent", delegates=["child"])
        agent = _mock_agent(config)

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            mock_load.return_value = _make_config("child")

            registry = AgentRegistry(base_dir=temp_dir)
            manager = DelegationManager(registry, agent)

            with patch("supyagent.core.supervisor.run_supervisor_coroutine") as mock_run, \
                 patch("supyagent.core.supervisor.get_supervisor") as mock_get_sup:
                mock_supervisor = MagicMock()
                mock_get_sup.return_value = mock_supervisor
                mock_run.return_value = {"ok": True, "data": "result"}

                mock_tool_call = MagicMock()
                mock_tool_call.function.name = "delegate_to_child"
                mock_tool_call.function.arguments = json.dumps({
                    "task": "Do important work",
                })

                manager.execute_delegation(mock_tool_call)

        call_kwargs = mock_supervisor.execute.call_args.kwargs
        assert call_kwargs["process_type"] == "agent"
        assert call_kwargs["tool_name"] == "agent__child"
        assert "agent_name" in call_kwargs["metadata"]
        assert call_kwargs["metadata"]["agent_name"] == "child"


# ===========================================================================
# 5. Agent → Agent → Tool execution
# ===========================================================================


class TestDelegationToTool:
    """Tests for delegation chains that end with tool execution."""

    def test_subprocess_delegation_passes_context_json(self, temp_dir):
        """Context JSON should be well-formed for the child's --context flag."""
        config = _make_config("parent", delegates=["worker"])
        agent = _mock_agent(config, messages=[
            {"role": "user", "content": "Run a shell command"},
            {"role": "assistant", "content": "I'll delegate to worker"},
        ])

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            mock_load.return_value = _make_config("worker", agent_type="execution")

            registry = AgentRegistry(base_dir=temp_dir)
            manager = DelegationManager(registry, agent)

            with patch("supyagent.core.supervisor.run_supervisor_coroutine") as mock_run, \
                 patch("supyagent.core.supervisor.get_supervisor") as mock_get_sup:
                mock_supervisor = MagicMock()
                mock_get_sup.return_value = mock_supervisor
                mock_run.return_value = {"ok": True, "data": "echo result"}

                mock_tool_call = MagicMock()
                mock_tool_call.function.name = "delegate_to_worker"
                mock_tool_call.function.arguments = json.dumps({
                    "task": "Run: echo hello",
                })

                manager.execute_delegation(mock_tool_call)

        # Extract the --context argument from the command
        call_args = mock_supervisor.execute.call_args
        cmd = call_args[0][0]
        context_idx = cmd.index("--context")
        context_json = cmd[context_idx + 1]

        # Verify it's valid JSON with expected fields
        parsed = json.loads(context_json)
        assert "parent_agent" in parsed
        assert parsed["parent_agent"] == "parent"
        assert "parent_task" in parsed
        assert "relevant_facts" in parsed

    def test_exec_agent_builds_task_with_context(self):
        """CLI _build_task_with_context should produce correct prompt."""
        from supyagent.cli.main import _build_task_with_context

        context = {
            "parent_agent": "tester",
            "parent_task": "Run echo hello",
            "conversation_summary": "User wants to test shell commands.",
            "relevant_facts": ["Using bash shell"],
        }

        result = _build_task_with_context("Run echo hello world", context)

        assert "tester" in result
        assert "Run echo hello" in result
        assert "test shell commands" in result
        assert "bash shell" in result
        assert "Run echo hello world" in result

    def test_exec_agent_empty_context(self):
        """CLI _build_task_with_context with empty context returns task as-is."""
        from supyagent.cli.main import _build_task_with_context

        result = _build_task_with_context("Just do this", {})
        assert result == "Just do this"


# ===========================================================================
# 6. Delegation tool generation
# ===========================================================================


class TestDelegationTools:
    """Tests for delegation tool schema generation."""

    def test_generates_delegate_to_tools(self, temp_dir):
        """Should generate delegate_to_X for each delegate."""
        config = _make_config("parent", delegates=["coder", "writer", "tester"])
        agent = _mock_agent(config)

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            mock_load.return_value = _make_config("generic")

            registry = AgentRegistry(base_dir=temp_dir)
            manager = DelegationManager(registry, agent)
            tools = manager.get_delegation_tools()

        tool_names = {t["function"]["name"] for t in tools}
        assert "delegate_to_coder" in tool_names
        assert "delegate_to_writer" in tool_names
        assert "delegate_to_tester" in tool_names
        assert "spawn_agent" in tool_names

    def test_delegate_tool_has_required_params(self, temp_dir):
        """Each delegate tool should have task as required parameter."""
        config = _make_config("parent", delegates=["coder"])
        agent = _mock_agent(config)

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            mock_load.return_value = _make_config("coder")

            registry = AgentRegistry(base_dir=temp_dir)
            manager = DelegationManager(registry, agent)
            tools = manager.get_delegation_tools()

        delegate_tool = next(
            t for t in tools if t["function"]["name"] == "delegate_to_coder"
        )
        params = delegate_tool["function"]["parameters"]
        assert "task" in params["required"]
        assert "task" in params["properties"]
        assert "mode" in params["properties"]
        assert "background" in params["properties"]
        assert "timeout" in params["properties"]

    def test_spawn_agent_has_enum(self, temp_dir):
        """spawn_agent tool should have enum of available agents."""
        config = _make_config("parent", delegates=["alpha", "beta"])
        agent = _mock_agent(config)

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            mock_load.return_value = _make_config("generic")

            registry = AgentRegistry(base_dir=temp_dir)
            manager = DelegationManager(registry, agent)
            tools = manager.get_delegation_tools()

        spawn_tool = next(
            t for t in tools if t["function"]["name"] == "spawn_agent"
        )
        agent_type_prop = spawn_tool["function"]["parameters"]["properties"]["agent_type"]
        assert set(agent_type_prop["enum"]) == {"alpha", "beta"}

    def test_no_tools_without_delegates(self, temp_dir):
        """Agent with no delegates should generate no delegation tools."""
        config = _make_config("loner", delegates=[])
        agent = _mock_agent(config)

        registry = AgentRegistry(base_dir=temp_dir)
        # DelegationManager shouldn't be created for agents without delegates
        # but test the tool generation if it was
        with patch("supyagent.core.delegation.load_agent_config"):
            manager = DelegationManager(registry, agent)
            tools = manager.get_delegation_tools()

        assert tools == []

    def test_missing_delegate_config_skipped(self, temp_dir):
        """If a delegate config can't be loaded, it's silently skipped."""
        config = _make_config("parent", delegates=["valid", "missing"])
        agent = _mock_agent(config)

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            def side_effect(name):
                if name == "valid":
                    return _make_config("valid")
                raise AgentNotFoundError(name)

            mock_load.side_effect = side_effect

            registry = AgentRegistry(base_dir=temp_dir)
            manager = DelegationManager(registry, agent)
            tools = manager.get_delegation_tools()

        tool_names = {t["function"]["name"] for t in tools}
        assert "delegate_to_valid" in tool_names
        assert "delegate_to_missing" not in tool_names
        # spawn_agent should still be there
        assert "spawn_agent" in tool_names


# ===========================================================================
# 7. Integration: Real subprocess delegation via supervisor
# ===========================================================================


class TestSubprocessDelegationIntegration:
    """Integration tests using real subprocesses through the supervisor."""

    async def test_supervisor_runs_subprocess(self, supervisor):
        """Supervisor should execute a simple subprocess."""
        result = await supervisor.execute(
            ["echo", "hello from child"],
            process_type="agent",
            tool_name="agent__echo_child",
        )
        assert result["ok"] is True
        assert "hello from child" in result["data"]

    async def test_supervisor_captures_json_output(self, supervisor):
        """Supervisor should parse JSON output from subprocess."""
        result = await supervisor.execute(
            ["sh", "-c", 'echo \'{"ok": true, "data": "json result"}\''],
            process_type="agent",
            tool_name="agent__json_child",
        )
        assert result["ok"] is True
        assert result["data"] == "json result"

    async def test_supervisor_handles_failure(self, supervisor):
        """Supervisor should report failure for non-zero exit code."""
        result = await supervisor.execute(
            ["sh", "-c", "echo 'error msg' >&2; exit 1"],
            process_type="agent",
            tool_name="agent__fail_child",
        )
        assert result["ok"] is False
        assert "error msg" in result["error"]

    async def test_background_delegation(self, supervisor):
        """Background delegation should return immediately."""
        result = await supervisor.execute(
            ["sleep", "60"],
            process_type="agent",
            tool_name="agent__slow_child",
            force_background=True,
        )
        assert result["ok"] is True
        assert result["data"]["status"] == "backgrounded"
        process_id = result["data"]["process_id"]

        # Verify it's tracked
        procs = supervisor.list_processes()
        assert len(procs) == 1
        assert procs[0]["process_id"] == process_id
        assert procs[0]["status"] == "backgrounded"

        # Kill it
        kill_result = await supervisor.kill(process_id)
        assert kill_result["ok"] is True

    async def test_auto_background_on_timeout(self, supervisor):
        """Process should auto-background when it exceeds timeout."""
        result = await supervisor.execute(
            ["sleep", "60"],
            process_type="agent",
            tool_name="agent__slow_child",
            timeout=0.5,
            on_timeout=TimeoutAction.BACKGROUND,
        )
        assert result["ok"] is True
        assert result["data"]["status"] == "backgrounded"

        # Clean up
        process_id = result["data"]["process_id"]
        await supervisor.kill(process_id)

    async def test_kill_on_timeout(self, supervisor):
        """Process should be killed when timeout action is KILL."""
        result = await supervisor.execute(
            ["sleep", "60"],
            process_type="agent",
            tool_name="agent__slow_child",
            timeout=0.5,
            on_timeout=TimeoutAction.KILL,
        )
        assert result["ok"] is False
        assert "killed" in result["error"].lower() or "timeout" in result["error"].lower()

    async def test_chained_subprocess_echo(self, supervisor):
        """Simulate a 2-level chain: parent → child using nested sh commands."""
        # This simulates what happens when tester delegates to worker:
        # The outer subprocess (parent) spawns an inner subprocess (child)
        inner_cmd = "echo 'worker says: 42'"
        result = await supervisor.execute(
            ["sh", "-c", f"echo 'middle running...'; {inner_cmd}"],
            process_type="agent",
            tool_name="agent__middle",
        )
        assert result["ok"] is True
        assert "worker says: 42" in result["data"]

    async def test_chained_subprocess_json_pipeline(self, supervisor):
        """Simulate a 3-level JSON pipeline through subprocesses."""
        # Worker produces JSON → middle wraps it → parent receives it
        worker_output = '{"ok": true, "data": "579"}'
        result = await supervisor.execute(
            ["sh", "-c", f"echo '{worker_output}'"],
            process_type="agent",
            tool_name="agent__worker",
        )
        assert result["ok"] is True
        assert result["data"] == "579"

    async def test_concurrent_delegations(self, supervisor):
        """Multiple delegations running concurrently."""
        tasks = [
            supervisor.execute(
                ["sh", "-c", f"echo 'agent_{i} result'"],
                process_type="agent",
                tool_name=f"agent__child_{i}",
            )
            for i in range(3)
        ]
        results = await asyncio.gather(*tasks)

        for i, result in enumerate(results):
            assert result["ok"] is True
            assert f"agent_{i} result" in result["data"]


# ===========================================================================
# 8. Registry persistence and parent-child tracking
# ===========================================================================


class TestRegistryTracking:
    """Tests for registry parent-child tracking."""

    def test_list_children(self, temp_dir):
        """Should list all children of a parent."""
        registry = AgentRegistry(base_dir=temp_dir)

        parent = MagicMock()
        parent.config.name = "parent"
        p_id = registry.register(parent)

        child_ids = []
        for name in ["child_a", "child_b"]:
            child = MagicMock()
            child.config.name = name
            c_id = registry.register(child, parent_id=p_id)
            child_ids.append(c_id)

        children = registry.list_children(p_id)
        assert len(children) == 2
        assert {c.instance_id for c in children} == set(child_ids)

    def test_mark_completed(self, temp_dir):
        """Marking an agent as completed should update its status."""
        registry = AgentRegistry(base_dir=temp_dir)

        agent = MagicMock()
        agent.config.name = "agent"
        agent_id = registry.register(agent)

        assert registry.get_instance(agent_id).status == "active"

        registry.mark_completed(agent_id)
        assert registry.get_instance(agent_id).status == "completed"

    def test_cleanup_completed(self, temp_dir):
        """cleanup_completed should remove finished agents."""
        registry = AgentRegistry(base_dir=temp_dir)

        agent1 = MagicMock()
        agent1.config.name = "agent1"
        id1 = registry.register(agent1)
        registry.mark_completed(id1)

        agent2 = MagicMock()
        agent2.config.name = "agent2"
        id2 = registry.register(agent2)

        count = registry.cleanup_completed()
        assert count == 1

        assert registry.get_instance(id1) is None
        assert registry.get_instance(id2) is not None
        assert registry.get_instance(id2).status == "active"

    def test_registry_serialization(self, temp_dir):
        """Registry should survive save/load cycle."""
        registry = AgentRegistry(base_dir=temp_dir)

        parent = MagicMock()
        parent.config.name = "parent"
        p_id = registry.register(parent)

        child = MagicMock()
        child.config.name = "child"
        c_id = registry.register(child, parent_id=p_id)

        # Load a new registry from the same directory
        registry2 = AgentRegistry(base_dir=temp_dir)

        # Instances should be preserved
        assert registry2.get_instance(p_id) is not None
        assert registry2.get_instance(c_id) is not None
        assert registry2.get_instance(c_id).parent_id == p_id
        assert registry2.get_depth(c_id) == 1


# ===========================================================================
# 9. CLI exec command
# ===========================================================================


class TestCliExecCommand:
    """Tests for the supyagent exec CLI command."""

    def test_build_task_with_full_context(self):
        """Full context should produce structured task prompt."""
        from supyagent.cli.main import _build_task_with_context

        context = {
            "parent_agent": "orchestrator",
            "parent_task": "Build project",
            "conversation_summary": "We're building a web app.",
            "relevant_facts": ["React 18", "TypeScript"],
        }

        result = _build_task_with_context("Write code", context)
        assert "orchestrator" in result
        assert "Build project" in result
        assert "web app" in result
        assert "React 18" in result
        assert "TypeScript" in result
        assert "Write code" in result

    def test_build_task_with_partial_context(self):
        """Partial context should only include present fields."""
        from supyagent.cli.main import _build_task_with_context

        context = {"parent_agent": "tester"}
        result = _build_task_with_context("Do something", context)

        assert "tester" in result
        assert "Do something" in result

    def test_build_task_no_context(self):
        """Empty context should return raw task."""
        from supyagent.cli.main import _build_task_with_context

        result = _build_task_with_context("Raw task", {})
        assert result == "Raw task"

    def test_build_task_with_none_values(self):
        """None values in context should be handled gracefully."""
        from supyagent.cli.main import _build_task_with_context

        context = {
            "parent_agent": "tester",
            "parent_task": None,
            "conversation_summary": None,
            "relevant_facts": [],
        }
        result = _build_task_with_context("Task", context)
        assert "tester" in result
        assert "Task" in result


# ===========================================================================
# 10. Edge cases and error handling
# ===========================================================================


class TestEdgeCases:
    """Tests for error handling and edge cases in delegation."""

    def test_invalid_json_arguments(self, temp_dir):
        """Invalid JSON in tool arguments should return error."""
        config = _make_config("parent", delegates=["child"])
        agent = _mock_agent(config)

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            mock_load.return_value = _make_config("child")

            registry = AgentRegistry(base_dir=temp_dir)
            manager = DelegationManager(registry, agent)

        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "delegate_to_child"
        mock_tool_call.function.arguments = "not valid json"

        result = manager.execute_delegation(mock_tool_call)
        assert result["ok"] is False
        assert "Invalid JSON" in result["error"]

    def test_agent_not_found_during_delegation(self, temp_dir):
        """Agent config not found during delegation should return error."""
        config = _make_config("parent", delegates=["missing"])
        agent = _mock_agent(config)

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            # Config found during DelegationManager init, but not during delegation
            mock_load.side_effect = [
                _make_config("missing"),  # For init
                AgentNotFoundError("missing"),  # For delegation
            ]

            registry = AgentRegistry(base_dir=temp_dir)
            manager = DelegationManager(registry, agent)

            mock_tool_call = MagicMock()
            mock_tool_call.function.name = "delegate_to_missing"
            mock_tool_call.function.arguments = json.dumps({"task": "anything"})

            result = manager.execute_delegation(mock_tool_call)

        assert result["ok"] is False
        assert "not found" in result["error"]

    def test_subprocess_delegation_exception(self, temp_dir):
        """Exception during subprocess delegation should be caught."""
        config = _make_config("parent", delegates=["child"])
        agent = _mock_agent(config)

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            mock_load.return_value = _make_config("child")

            registry = AgentRegistry(base_dir=temp_dir)
            manager = DelegationManager(registry, agent)

            with patch("supyagent.core.supervisor.run_supervisor_coroutine") as mock_run, \
                 patch("supyagent.core.supervisor.get_supervisor"):
                mock_run.side_effect = RuntimeError("Supervisor crashed")

                mock_tool_call = MagicMock()
                mock_tool_call.function.name = "delegate_to_child"
                mock_tool_call.function.arguments = json.dumps({"task": "anything"})

                result = manager.execute_delegation(mock_tool_call)

        assert result["ok"] is False
        assert "Subprocess delegation failed" in result["error"]

    def test_empty_task_delegation(self, temp_dir):
        """Delegating with an empty task should still work (LLM decides)."""
        config = _make_config("parent", delegates=["child"])
        agent = _mock_agent(config)

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            mock_load.return_value = _make_config("child")

            registry = AgentRegistry(base_dir=temp_dir)
            manager = DelegationManager(registry, agent)

            with patch("supyagent.core.supervisor.run_supervisor_coroutine") as mock_run, \
                 patch("supyagent.core.supervisor.get_supervisor") as mock_get_sup:
                mock_supervisor = MagicMock()
                mock_get_sup.return_value = mock_supervisor
                mock_run.return_value = {"ok": True, "data": "empty task handled"}

                mock_tool_call = MagicMock()
                mock_tool_call.function.name = "delegate_to_child"
                mock_tool_call.function.arguments = json.dumps({"task": ""})

                result = manager.execute_delegation(mock_tool_call)

        assert result["ok"] is True

    def test_is_delegation_tool(self, temp_dir):
        """is_delegation_tool should correctly identify delegation tools."""
        config = _make_config("parent", delegates=["coder"])
        agent = _mock_agent(config)

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            mock_load.return_value = _make_config("coder")

            registry = AgentRegistry(base_dir=temp_dir)
            manager = DelegationManager(registry, agent)

        assert manager.is_delegation_tool("delegate_to_coder") is True
        assert manager.is_delegation_tool("delegate_to_unknown") is True  # prefix check
        assert manager.is_delegation_tool("spawn_agent") is True
        assert manager.is_delegation_tool("shell__run_command") is False
        assert manager.is_delegation_tool("list_processes") is False

    def test_unknown_delegation_tool(self, temp_dir):
        """Unknown tool names should return error."""
        config = _make_config("parent", delegates=["child"])
        agent = _mock_agent(config)

        with patch("supyagent.core.delegation.load_agent_config") as mock_load:
            mock_load.return_value = _make_config("child")

            registry = AgentRegistry(base_dir=temp_dir)
            manager = DelegationManager(registry, agent)

        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "totally_unknown_tool"
        mock_tool_call.function.arguments = json.dumps({"task": "anything"})

        result = manager.execute_delegation(mock_tool_call)
        assert result["ok"] is False
        assert "Unknown delegation tool" in result["error"]


# ===========================================================================
# 11. ExecutionRunner delegation setup
# ===========================================================================


class TestExecutionRunnerDelegation:
    """Tests for ExecutionRunner's delegation support."""

    def test_execution_runner_has_delegation_tools_when_delegates_configured(self):
        """ExecutionRunner should include delegation tools if agent has delegates."""
        config = _make_config("runner", agent_type="execution", delegates=["worker"])

        with patch("supyagent.core.engine.discover_tools", return_value=[]), \
             patch("supyagent.core.delegation.load_agent_config") as mock_load:
            mock_load.return_value = _make_config("worker", agent_type="execution")

            from supyagent.core.executor import ExecutionRunner
            runner = ExecutionRunner(config)

        tool_names = [t["function"]["name"] for t in runner.tools]
        assert "delegate_to_worker" in tool_names
        assert "spawn_agent" in tool_names

    def test_execution_runner_has_process_tools(self):
        """ExecutionRunner should always include process management tools."""
        config = _make_config("runner", agent_type="execution", delegates=[])

        with patch("supyagent.core.engine.discover_tools", return_value=[]):
            from supyagent.core.executor import ExecutionRunner
            runner = ExecutionRunner(config)

        tool_names = [t["function"]["name"] for t in runner.tools]
        assert "list_processes" in tool_names
        assert "check_process" in tool_names
        assert "kill_process" in tool_names

    def test_execution_runner_no_delegation_without_delegates(self):
        """ExecutionRunner without delegates should have no delegation tools."""
        config = _make_config("runner", agent_type="execution", delegates=[])

        with patch("supyagent.core.engine.discover_tools", return_value=[]):
            from supyagent.core.executor import ExecutionRunner
            runner = ExecutionRunner(config)

        tool_names = [t["function"]["name"] for t in runner.tools]
        assert not any(name.startswith("delegate_to_") for name in tool_names)
        assert "spawn_agent" not in tool_names
