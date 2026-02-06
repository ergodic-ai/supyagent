"""
Tests for the ToolResult model and timed_execution.
"""

import json
import time

from supyagent.models.tool_result import ToolResult, timed_execution


class TestToolResult:
    """Tests for the ToolResult model."""

    def test_success_result(self):
        result = ToolResult.success(data="hello")
        assert result.ok is True
        assert result.data == "hello"
        assert result.error is None

    def test_failure_result(self):
        result = ToolResult.fail("something broke", error_type="execution_error")
        assert result.ok is False
        assert result.error == "something broke"
        assert result.error_type == "execution_error"

    def test_to_dict_excludes_none(self):
        result = ToolResult.success(data=42)
        d = result.to_dict()
        assert "error" not in d
        assert "error_type" not in d
        assert "process_id" not in d
        assert d["ok"] is True
        assert d["data"] == 42

    def test_to_dict_includes_metadata(self):
        result = ToolResult(
            ok=True,
            data="result",
            tool_name="script__func",
            duration_ms=150,
            process_id="proc-123",
        )
        d = result.to_dict()
        assert d["tool_name"] == "script__func"
        assert d["duration_ms"] == 150
        assert d["process_id"] == "proc-123"

    def test_from_dict(self):
        d = {"ok": True, "data": "hello", "tool_name": "test__fn"}
        result = ToolResult.from_dict(d)
        assert result.ok is True
        assert result.data == "hello"
        assert result.tool_name == "test__fn"

    def test_from_dict_legacy_format(self):
        """Backward compatible with raw dict format."""
        d = {"ok": False, "error": "bad input"}
        result = ToolResult.from_dict(d)
        assert result.ok is False
        assert result.error == "bad input"

    def test_to_llm_content_success_string(self):
        result = ToolResult.success(data="hello world")
        assert result.to_llm_content() == "hello world"

    def test_to_llm_content_success_dict(self):
        result = ToolResult.success(data={"key": "value"})
        content = result.to_llm_content()
        assert json.loads(content) == {"key": "value"}

    def test_to_llm_content_success_none(self):
        result = ToolResult.success()
        assert result.to_llm_content() == ""

    def test_to_llm_content_failure(self):
        result = ToolResult.fail("oops", error_type="invalid_args")
        content = json.loads(result.to_llm_content())
        assert content["error"] == "oops"
        assert content["error_type"] == "invalid_args"


class TestTimedExecution:
    """Tests for the timed_execution context manager."""

    def test_records_duration(self):
        with timed_execution() as timing:
            time.sleep(0.01)  # 10ms minimum

        assert "duration_ms" in timing
        assert timing["duration_ms"] >= 10

    def test_records_duration_on_exception(self):
        timing_ref = None
        try:
            with timed_execution() as timing:
                timing_ref = timing
                raise ValueError("test error")
        except ValueError:
            pass

        assert timing_ref is not None
        assert "duration_ms" in timing_ref
