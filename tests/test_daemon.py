"""Tests for daemon runner (poll-based event processing)."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from supyagent.core.daemon import (
    DaemonConfigError,
    DaemonCycleResult,
    DaemonRunner,
    parse_interval,
)
from supyagent.core.engine import MaxIterationsError
from supyagent.models.agent_config import (
    AgentConfig,
    ModelConfig,
    ScheduleConfig,
    ServiceConfig,
    ToolPermissions,
)

# ── Helpers ──────────────────────────────────────────────────────────

_PATCH_DISCOVER = "supyagent.core.engine.discover_tools"
_PATCH_SERVICE = "supyagent.core.service.get_service_client"


def _make_config(
    name="test-daemon",
    schedule_kwargs=None,
    memory_enabled=False,
    service_enabled=True,
    **overrides,
):
    schedule = ScheduleConfig(**(schedule_kwargs or {}))
    return AgentConfig(
        name=name,
        type="daemon",
        model=ModelConfig(provider="test/model"),
        system_prompt="You are a test daemon.",
        tools=ToolPermissions(allow=["*"]),
        limits={"max_tool_calls_per_turn": 10},
        schedule=schedule,
        service=ServiceConfig(enabled=service_enabled),
        memory={"enabled": memory_enabled},
        **overrides,
    )


def _mock_service_client(inbox_events=None):
    client = MagicMock()
    client.discover_tools.return_value = []
    if inbox_events is not None:
        client.inbox_list.return_value = {
            "events": inbox_events,
            "total": len(inbox_events),
        }
    else:
        client.inbox_list.return_value = {"events": [], "total": 0}
    client.inbox_archive.return_value = True
    client.inbox_archive_all.return_value = 0
    client.health_check.return_value = True
    return client


def _sample_events(n=2):
    events = []
    for i in range(n):
        events.append({
            "id": f"evt_{i}",
            "provider": "slack",
            "event_type": "message",
            "created_at": f"2025-01-15T10:{i:02d}:00Z",
            "summary": f"Test event {i}",
            "data": {"text": f"Hello {i}", "channel": "#general"},
        })
    return events


# ── parse_interval ───────────────────────────────────────────────────


class TestParseInterval:
    def test_seconds(self):
        assert parse_interval("30s") == 30.0

    def test_minutes(self):
        assert parse_interval("5m") == 300.0

    def test_hours(self):
        assert parse_interval("1h") == 3600.0

    def test_fractional(self):
        assert parse_interval("0.5m") == 30.0

    def test_with_spaces(self):
        assert parse_interval(" 10s ") == 10.0

    def test_case_insensitive(self):
        assert parse_interval("5M") == 300.0

    def test_invalid_no_unit(self):
        with pytest.raises(ValueError, match="Invalid interval"):
            parse_interval("30")

    def test_invalid_unit(self):
        with pytest.raises(ValueError, match="Invalid interval"):
            parse_interval("5d")

    def test_invalid_empty(self):
        with pytest.raises(ValueError, match="Invalid interval"):
            parse_interval("")

    def test_invalid_text(self):
        with pytest.raises(ValueError, match="Invalid interval"):
            parse_interval("five minutes")


# ── ScheduleConfig / AgentConfig ─────────────────────────────────────


class TestDaemonConfig:
    def test_schedule_defaults(self):
        config = _make_config()
        assert config.schedule.interval == "5m"
        assert config.schedule.max_events_per_cycle == 10
        assert config.schedule.prompt is None

    def test_schedule_custom(self):
        config = _make_config(
            schedule_kwargs={"interval": "30s", "max_events_per_cycle": 5, "prompt": "Check tasks"}
        )
        assert config.schedule.interval == "30s"
        assert config.schedule.max_events_per_cycle == 5
        assert config.schedule.prompt == "Check tasks"

    def test_daemon_type_accepted(self):
        config = _make_config()
        assert config.type == "daemon"

    def test_backward_compatible(self):
        """Existing interactive/execution configs still work."""
        config = AgentConfig(
            name="test",
            type="interactive",
            model=ModelConfig(provider="test/model"),
            system_prompt="Test.",
        )
        assert config.schedule.interval == "5m"  # default, unused


# ── DaemonRunner init ────────────────────────────────────────────────


@patch(_PATCH_DISCOVER, return_value=[])
class TestDaemonRunnerInit:
    def test_init_with_service(self, mock_discover):
        config = _make_config()
        with patch(_PATCH_SERVICE, return_value=_mock_service_client()):
            runner = DaemonRunner(config)

        assert runner._interval_seconds == 300.0
        assert runner._max_events == 10
        assert runner._cycle_count == 0
        assert not runner._shutdown_event.is_set()

    def test_init_without_service_raises(self, mock_discover):
        config = _make_config(service_enabled=False)
        with pytest.raises(DaemonConfigError, match="service connection"):
            DaemonRunner(config)

    def test_init_with_memory(self, mock_discover):
        config = _make_config(memory_enabled=True)
        with (
            patch(_PATCH_SERVICE, return_value=_mock_service_client()),
            patch("supyagent.core.memory.MemoryManager") as mock_memory,
        ):
            mock_memory.return_value = MagicMock()
            runner = DaemonRunner(config)

        assert runner.memory_mgr is not None
        mock_memory.assert_called_once()

    def test_tools_exclude_credential_tool(self, mock_discover):
        config = _make_config()
        with patch(_PATCH_SERVICE, return_value=_mock_service_client()):
            runner = DaemonRunner(config)

        tool_names = [t.get("function", {}).get("name", "") for t in runner.tools]
        assert "request_credential" not in tool_names

    def test_tools_include_memory_tools_when_enabled(self, mock_discover):
        config = _make_config(memory_enabled=True)
        with (
            patch(_PATCH_SERVICE, return_value=_mock_service_client()),
            patch("supyagent.core.memory.MemoryManager") as mock_memory,
        ):
            mock_memory.return_value = MagicMock()
            runner = DaemonRunner(config)

        tool_names = [t.get("function", {}).get("name", "") for t in runner.tools]
        assert "memory_search" in tool_names
        assert "memory_write" in tool_names

    def test_custom_interval(self, mock_discover):
        config = _make_config(schedule_kwargs={"interval": "30s"})
        with patch(_PATCH_SERVICE, return_value=_mock_service_client()):
            runner = DaemonRunner(config)

        assert runner._interval_seconds == 30.0

    def test_credential_request_rejected(self, mock_discover):
        config = _make_config()
        with patch(_PATCH_SERVICE, return_value=_mock_service_client()):
            runner = DaemonRunner(config)

        mock_tc = MagicMock()
        mock_tc.function.name = "request_credential"
        mock_tc.function.arguments = '{"name": "API_KEY"}'

        result = runner._dispatch_tool_call(mock_tc)
        assert result["ok"] is False
        assert "daemon mode" in result["error"]


# ── Cycle prompt construction ────────────────────────────────────────


@patch(_PATCH_DISCOVER, return_value=[])
class TestDaemonCyclePrompt:
    def _make_runner(self, **kwargs):
        config = _make_config(**kwargs)
        with patch(_PATCH_SERVICE, return_value=_mock_service_client()):
            return DaemonRunner(config)

    def test_prompt_with_events(self, mock_discover):
        runner = self._make_runner()
        events = _sample_events(2)
        prompt = runner._build_cycle_prompt(events)

        assert "2 new inbox event(s)" in prompt
        assert "evt_0" in prompt
        assert "evt_1" in prompt
        assert "slack" in prompt
        assert "Hello 0" in prompt
        assert "Archive" in prompt or "archive" in prompt

    def test_prompt_with_no_events_no_schedule(self, mock_discover):
        runner = self._make_runner()
        prompt = runner._build_cycle_prompt([])
        assert prompt == ""

    def test_prompt_with_scheduled_prompt_only(self, mock_discover):
        runner = self._make_runner(schedule_kwargs={"prompt": "Check pending tasks"})
        prompt = runner._build_cycle_prompt([])
        assert "Check pending tasks" in prompt
        assert "inbox event" not in prompt

    def test_prompt_with_events_and_scheduled_prompt(self, mock_discover):
        runner = self._make_runner(schedule_kwargs={"prompt": "Also check deployments"})
        events = _sample_events(1)
        prompt = runner._build_cycle_prompt(events)

        assert "1 new inbox event(s)" in prompt
        assert "Also check deployments" in prompt
        assert "Additionally" in prompt

    def test_prompt_includes_event_data(self, mock_discover):
        runner = self._make_runner()
        events = [{"id": "e1", "provider": "github", "event_type": "pr_review",
                    "created_at": "2025-01-01", "data": {"pr": 42, "action": "approved"}}]
        prompt = runner._build_cycle_prompt(events)

        assert "github" in prompt
        assert "pr_review" in prompt
        assert '"pr": 42' in prompt

    def test_prompt_includes_summary(self, mock_discover):
        runner = self._make_runner()
        events = [{"id": "e1", "provider": "slack", "event_type": "message",
                    "created_at": "2025-01-01", "summary": "Alice asked about deployment"}]
        prompt = runner._build_cycle_prompt(events)
        assert "Alice asked about deployment" in prompt


# ── run_once ─────────────────────────────────────────────────────────


@patch(_PATCH_DISCOVER, return_value=[])
class TestDaemonRunOnce:
    def _make_runner(self, inbox_events=None, **kwargs):
        config = _make_config(**kwargs)
        client = _mock_service_client(inbox_events=inbox_events)
        with patch(_PATCH_SERVICE, return_value=client):
            runner = DaemonRunner(config)
        return runner

    def test_skip_when_no_events(self, mock_discover):
        runner = self._make_runner(inbox_events=[])
        result = runner.run_once()

        assert result.skipped is True
        assert result.events_found == 0
        assert result.cycle == 0  # cycle count not incremented on skip
        assert result.elapsed_seconds >= 0

    def test_cycle_with_events(self, mock_discover):
        runner = self._make_runner(inbox_events=_sample_events(2))

        with patch.object(runner, "_run_loop", return_value="Processed 2 events"):
            result = runner.run_once()

        assert result.skipped is False
        assert result.events_found == 2
        assert result.events_processed == 2
        assert result.cycle == 1
        assert result.response == "Processed 2 events"
        assert result.error is None

    def test_cycle_increments(self, mock_discover):
        runner = self._make_runner(inbox_events=_sample_events(1))

        with patch.object(runner, "_run_loop", return_value="done"):
            r1 = runner.run_once()
            r2 = runner.run_once()

        assert r1.cycle == 1
        assert r2.cycle == 2

    def test_cycle_with_scheduled_prompt_no_events(self, mock_discover):
        runner = self._make_runner(
            inbox_events=[],
            schedule_kwargs={"prompt": "Check deployments"},
        )

        with patch.object(runner, "_run_loop", return_value="Checked deployments"):
            result = runner.run_once()

        assert result.skipped is False
        assert result.events_found == 0
        assert result.response == "Checked deployments"

    def test_cycle_error_handling(self, mock_discover):
        runner = self._make_runner(inbox_events=_sample_events(1))

        with patch.object(runner, "_run_loop", side_effect=RuntimeError("LLM failed")):
            result = runner.run_once()

        assert result.error == "LLM failed"
        assert result.response == ""

    def test_max_iterations_error(self, mock_discover):
        runner = self._make_runner(inbox_events=_sample_events(1))

        with patch.object(runner, "_run_loop", side_effect=MaxIterationsError(10)):
            result = runner.run_once()

        assert result.error == "Max tool iterations exceeded"

    def test_secrets_injected(self, mock_discover):
        runner = self._make_runner(inbox_events=_sample_events(1))

        with patch.object(runner, "_run_loop", return_value="done"):
            runner.run_once(secrets={"MY_KEY": "my_val"})

        assert runner._run_secrets.get("MY_KEY") == "my_val"

    def test_system_prompt_includes_daemon_instructions(self, mock_discover):
        runner = self._make_runner(inbox_events=_sample_events(1))

        with patch.object(runner, "_run_loop", return_value="done"):
            runner.run_once()

        system_msg = runner.messages[0]
        assert system_msg["role"] == "system"
        assert "Daemon Mode" in system_msg["content"]


# ── run loop ─────────────────────────────────────────────────────────


@patch(_PATCH_DISCOVER, return_value=[])
class TestDaemonRunLoop:
    def _make_runner(self, interval="1s", inbox_events=None, **kwargs):
        config = _make_config(schedule_kwargs={"interval": interval}, **kwargs)
        client = _mock_service_client(inbox_events=inbox_events)
        with patch(_PATCH_SERVICE, return_value=client):
            runner = DaemonRunner(config)
        return runner

    def test_shutdown_stops_loop(self, mock_discover):
        runner = self._make_runner(interval="1s")
        cycles = []

        def on_cycle(result):
            cycles.append(result)
            runner.shutdown()  # Stop after first cycle

        runner.run(on_cycle=on_cycle)

        assert len(cycles) == 1
        assert cycles[0].skipped is True

    def test_multiple_cycles(self, mock_discover):
        runner = self._make_runner(interval="0.1s")
        cycles = []

        def on_cycle(result):
            cycles.append(result)
            if len(cycles) >= 3:
                runner.shutdown()

        runner.run(on_cycle=on_cycle)

        assert len(cycles) == 3

    def test_error_in_cycle_doesnt_stop_loop(self, mock_discover):
        runner = self._make_runner(interval="0.1s", inbox_events=_sample_events(1))
        cycles = []
        call_count = 0

        def mock_run_loop(max_iter, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Transient error")
            return "OK"

        with patch.object(runner, "_run_loop", side_effect=mock_run_loop):
            def on_cycle(result):
                cycles.append(result)
                if len(cycles) >= 2:
                    runner.shutdown()

            runner.run(on_cycle=on_cycle)

        assert len(cycles) == 2
        assert cycles[0].error == "Transient error"
        assert cycles[1].error is None

    def test_wall_clock_timing(self, mock_discover):
        runner = self._make_runner(interval="0.5s")
        timestamps = []

        def on_cycle(result):
            timestamps.append(time.monotonic())
            if len(timestamps) >= 2:
                runner.shutdown()

        runner.run(on_cycle=on_cycle)

        # Second cycle should start ~0.5s after first
        if len(timestamps) >= 2:
            gap = timestamps[1] - timestamps[0]
            assert 0.3 < gap < 1.0  # Allow some tolerance

    def test_shutdown_via_external_thread(self, mock_discover):
        runner = self._make_runner(interval="10s")
        started = threading.Event()

        def on_cycle(result):
            started.set()

        def stop_later():
            started.wait(timeout=5)
            runner.shutdown()

        stopper = threading.Thread(target=stop_later, daemon=True)
        stopper.start()

        runner.run(on_cycle=on_cycle)
        # Should return quickly (not wait 10s)
        stopper.join(timeout=2)


# ── DaemonCycleResult ────────────────────────────────────────────────


class TestDaemonCycleResult:
    def test_defaults(self):
        result = DaemonCycleResult(
            cycle=1, events_found=3, events_processed=3,
            elapsed_seconds=1.5, response="Done",
        )
        assert result.skipped is False
        assert result.total_unread == 0
        assert result.error is None

    def test_skipped(self):
        result = DaemonCycleResult(
            cycle=0, events_found=0, events_processed=0,
            elapsed_seconds=0.01, response="", skipped=True,
        )
        assert result.skipped is True
