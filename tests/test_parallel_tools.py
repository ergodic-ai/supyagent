"""
Tests for parallel tool execution in BaseAgentEngine.

Verifies that multiple tool calls execute concurrently and
results are returned in the correct order.
"""

import json
import time
from unittest.mock import MagicMock, patch

from supyagent.core.engine import BaseAgentEngine
from supyagent.models.agent_config import AgentConfig


def _make_config(**overrides) -> AgentConfig:
    """Create a minimal AgentConfig for testing."""
    defaults = {
        "name": "test",
        "description": "Test agent",
        "model": {"provider": "test/model", "temperature": 0.5, "max_tokens": 100},
        "system_prompt": "You are a test agent.",
        "tools": {"allow": ["*"]},
        "delegates": [],
        "limits": {},
    }
    defaults.update(overrides)
    return AgentConfig(**defaults)


class ConcreteEngine(BaseAgentEngine):
    """Concrete subclass for testing — implements the abstract _get_secrets."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)

    def _get_secrets(self):
        return {}


def _make_tool_call_response(tool_calls_data):
    """Create a mock LLM response with tool calls."""
    msg = MagicMock()
    msg.content = ""
    tc_mocks = []
    for tc_id, name, args in tool_calls_data:
        tc = MagicMock()
        tc.id = tc_id
        tc.index = len(tc_mocks)
        tc.function = MagicMock()
        tc.function.name = name
        tc.function.arguments = json.dumps(args)
        tc_mocks.append(tc)
    msg.tool_calls = tc_mocks if tc_mocks else None
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_text_response(content):
    """Create a mock LLM response with text only."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


class TestParallelExecution:
    """Test that multiple tool calls execute in parallel."""

    def test_parallel_faster_than_sequential(self):
        """N tool calls with sleep should complete in ~1x sleep time, not Nx."""
        config = _make_config()
        engine = ConcreteEngine(config)
        engine.tools = [{"function": {"name": "shell__run", "parameters": {}}}]

        sleep_duration = 0.15  # seconds per tool call
        num_tools = 3

        # Make dispatch simulate a slow tool
        def slow_dispatch(tc):
            time.sleep(sleep_duration)
            return {"ok": True, "data": f"result_{tc.function.name}"}

        engine._dispatch_tool_call = slow_dispatch

        # First call returns 3 tool calls, second returns text
        tool_calls = [
            ("tc1", "shell__run", {"command": "echo 1"}),
            ("tc2", "shell__run", {"command": "echo 2"}),
            ("tc3", "shell__run", {"command": "echo 3"}),
        ]
        call_count = [0]

        def mock_chat(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_tool_call_response(tool_calls)
            return _make_text_response("done")

        engine.llm = MagicMock()
        engine.llm.chat = mock_chat
        engine.messages = [{"role": "user", "content": "test"}]

        start = time.monotonic()
        result = engine._run_loop(max_iterations=5)
        elapsed = time.monotonic() - start

        assert result == "done"
        # Parallel: should take ~1x sleep, not 3x
        # Allow generous margin (2x) but reject truly sequential (3x)
        assert elapsed < sleep_duration * (num_tools - 0.5), (
            f"Took {elapsed:.2f}s — expected parallel execution under "
            f"{sleep_duration * (num_tools - 0.5):.2f}s"
        )

    def test_results_in_original_order(self):
        """Results should be appended to messages in the same order as tool calls."""
        config = _make_config()
        engine = ConcreteEngine(config)
        engine.tools = [{"function": {"name": "shell__run", "parameters": {}}}]

        # Make tool calls finish in reverse order
        def variable_dispatch(tc):
            args = json.loads(tc.function.arguments)
            idx = args.get("idx", 0)
            # Higher index sleeps less → finishes first
            time.sleep(0.05 * (3 - idx))
            return {"ok": True, "data": f"result_{idx}"}

        engine._dispatch_tool_call = variable_dispatch

        tool_calls = [
            ("tc0", "shell__run", {"idx": 0}),
            ("tc1", "shell__run", {"idx": 1}),
            ("tc2", "shell__run", {"idx": 2}),
        ]
        call_count = [0]

        def mock_chat(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_tool_call_response(tool_calls)
            return _make_text_response("done")

        engine.llm = MagicMock()
        engine.llm.chat = mock_chat
        engine.messages = [{"role": "user", "content": "test"}]

        engine._run_loop(max_iterations=5)

        # Extract tool result messages (skip assistant and user messages)
        tool_msgs = [m for m in engine.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 3
        assert tool_msgs[0]["tool_call_id"] == "tc0"
        assert tool_msgs[1]["tool_call_id"] == "tc1"
        assert tool_msgs[2]["tool_call_id"] == "tc2"

        # Verify data ordering
        for i, msg in enumerate(tool_msgs):
            data = json.loads(msg["content"])
            assert data["data"] == f"result_{i}"

    def test_single_tool_no_thread_overhead(self):
        """Single tool call should NOT use ThreadPoolExecutor."""
        config = _make_config()
        engine = ConcreteEngine(config)
        engine.tools = [{"function": {"name": "shell__run", "parameters": {}}}]

        engine._dispatch_tool_call = lambda tc: {"ok": True, "data": "single"}

        call_count = [0]

        def mock_chat(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_tool_call_response([("tc1", "shell__run", {})])
            return _make_text_response("done")

        engine.llm = MagicMock()
        engine.llm.chat = mock_chat
        engine.messages = [{"role": "user", "content": "test"}]

        with patch(
            "supyagent.core.engine.ThreadPoolExecutor"
        ) as mock_pool:
            engine._run_loop(max_iterations=5)
            # Should NOT have created a thread pool for a single call
            mock_pool.assert_not_called()

    def test_callbacks_fire_in_order(self):
        """on_tool_start fires for all before execution, on_tool_result fires in order after."""
        config = _make_config()
        engine = ConcreteEngine(config)
        engine.tools = [{"function": {"name": "shell__run", "parameters": {}}}]

        engine._dispatch_tool_call = lambda tc: {"ok": True, "data": "ok"}

        events = []

        def on_start(tc_id, name, args):
            events.append(("start", tc_id))

        def on_result(tc_id, name, result):
            events.append(("result", tc_id))

        tool_calls = [
            ("tc0", "shell__run", {"cmd": "a"}),
            ("tc1", "shell__run", {"cmd": "b"}),
        ]
        call_count = [0]

        def mock_chat(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_tool_call_response(tool_calls)
            return _make_text_response("done")

        engine.llm = MagicMock()
        engine.llm.chat = mock_chat
        engine.messages = [{"role": "user", "content": "test"}]

        engine._run_loop(
            max_iterations=5, on_tool_start=on_start, on_tool_result=on_result
        )

        # All starts should come before all results
        assert events == [
            ("start", "tc0"),
            ("start", "tc1"),
            ("result", "tc0"),
            ("result", "tc1"),
        ]

    def test_stream_parallel_execution(self):
        """_run_loop_stream should also execute tool calls in parallel."""
        config = _make_config()
        engine = ConcreteEngine(config)
        engine.tools = [{"function": {"name": "shell__run", "parameters": {}}}]

        sleep_duration = 0.15
        num_tools = 3

        def slow_dispatch(tc):
            time.sleep(sleep_duration)
            return {"ok": True, "data": f"result_{tc.function.name}"}

        engine._dispatch_tool_call = slow_dispatch

        # Build a streaming response mock
        def make_stream_chunks(tool_calls_data):
            """Generate stream chunks that build up tool calls."""
            chunks = []
            for i, (tc_id, name, args) in enumerate(tool_calls_data):
                delta = MagicMock()
                delta.content = None
                delta.reasoning_content = None
                tc_delta = MagicMock()
                tc_delta.index = i
                tc_delta.id = tc_id
                tc_delta.function = MagicMock()
                tc_delta.function.name = name
                tc_delta.function.arguments = json.dumps(args)
                delta.tool_calls = [tc_delta]
                if not hasattr(delta, "reasoning_content"):
                    delta.reasoning_content = None
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta = delta
                chunks.append(chunk)
            return chunks

        tool_calls = [
            ("tc1", "shell__run", {"command": "echo 1"}),
            ("tc2", "shell__run", {"command": "echo 2"}),
            ("tc3", "shell__run", {"command": "echo 3"}),
        ]

        call_count = [0]

        def mock_chat(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return iter(make_stream_chunks(tool_calls))
            # Second call: text-only response
            delta = MagicMock()
            delta.content = "done"
            delta.tool_calls = None
            delta.reasoning_content = None
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = delta
            return iter([chunk])

        engine.llm = MagicMock()
        engine.llm.chat = mock_chat
        engine.messages = [{"role": "user", "content": "test"}]

        start = time.monotonic()
        events = list(engine._run_loop_stream(max_iterations=5))
        elapsed = time.monotonic() - start

        # Verify parallel timing
        assert elapsed < sleep_duration * (num_tools - 0.5), (
            f"Stream took {elapsed:.2f}s — expected parallel"
        )

        # Verify events contain tool_start and tool_end in order
        tool_starts = [e for e in events if e[0] == "tool_start"]
        tool_ends = [e for e in events if e[0] == "tool_end"]
        assert len(tool_starts) == 3
        assert len(tool_ends) == 3


class TestDispatchLockSafety:
    """Test that the dispatch lock correctly protects shared state."""

    def test_circuit_breaker_thread_safe(self):
        """Circuit breaker tracking should be consistent under parallel access."""
        config = _make_config(limits={"circuit_breaker_threshold": 2})
        engine = ConcreteEngine(config)
        engine.tools = [{"function": {"name": "shell__run", "parameters": {}}}]

        # All tool calls fail
        def failing_dispatch(tc):
            # Simulate work
            time.sleep(0.01)
            return {"ok": False, "error": "test failure"}

        # Directly call _dispatch_tool_call from multiple threads
        from concurrent.futures import ThreadPoolExecutor

        from supyagent.core.models import ToolCallObj

        results = []
        tcs = [
            ToolCallObj("tc1", "shell__run", json.dumps({"cmd": "a"})),
            ToolCallObj("tc2", "shell__run", json.dumps({"cmd": "b"})),
            ToolCallObj("tc3", "shell__run", json.dumps({"cmd": "c"})),
        ]

        # Override execution methods to return failures
        engine._execute_supypowers_tool = lambda tc: {"ok": False, "error": "fail"}

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(engine._dispatch_tool_call, tc) for tc in tcs]
            results = [f.result() for f in futures]

        # All should have returned (no crashes from concurrent dict access)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, dict)
