"""Unit tests for the AI SDK Data Stream Protocol encoder."""

import json

import pytest

from supyagent.server.protocol import DataStreamEncoder, get_encoder


@pytest.fixture
def encoder():
    return DataStreamEncoder()


class TestGetEncoder:
    def test_default_returns_data_stream_encoder(self):
        enc = get_encoder()
        assert isinstance(enc, DataStreamEncoder)

    def test_explicit_data_stream(self):
        enc = get_encoder("data-stream")
        assert isinstance(enc, DataStreamEncoder)


class TestMessageStart:
    def test_format(self, encoder):
        result = encoder.message_start("abc123")
        text = result.decode()
        assert text.startswith("f:")
        assert text.endswith("\n")
        payload = json.loads(text[2:])
        assert payload == {"messageId": "abc123"}

    def test_returns_bytes(self, encoder):
        assert isinstance(encoder.message_start("x"), bytes)


class TestTextDelta:
    def test_simple_text(self, encoder):
        result = encoder.text_delta("hello")
        text = result.decode()
        assert text == '0:"hello"\n'

    def test_text_with_quotes(self, encoder):
        result = encoder.text_delta('say "hi"')
        text = result.decode()
        assert text.startswith("0:")
        parsed = json.loads(text[2:])
        assert parsed == 'say "hi"'

    def test_text_with_newlines(self, encoder):
        result = encoder.text_delta("line1\nline2")
        text = result.decode()
        parsed = json.loads(text[2:])
        assert parsed == "line1\nline2"

    def test_empty_text(self, encoder):
        result = encoder.text_delta("")
        text = result.decode()
        assert text == '0:""\n'


class TestToolCall:
    def test_format(self, encoder):
        result = encoder.tool_call("call_123", "shell_exec", {"cmd": "ls"})
        text = result.decode()
        assert text.startswith("9:")
        assert text.endswith("\n")
        payload = json.loads(text[2:])
        assert payload["toolCallId"] == "call_123"
        assert payload["toolName"] == "shell_exec"
        assert payload["args"] == {"cmd": "ls"}

    def test_empty_args(self, encoder):
        result = encoder.tool_call("call_1", "noop", {})
        payload = json.loads(result.decode()[2:])
        assert payload["args"] == {}


class TestToolResult:
    def test_string_result(self, encoder):
        result = encoder.tool_result("call_123", "output text")
        text = result.decode()
        assert text.startswith("a:")
        payload = json.loads(text[2:])
        assert payload["toolCallId"] == "call_123"
        assert payload["result"] == "output text"

    def test_dict_result(self, encoder):
        result = encoder.tool_result("call_1", {"status": "ok", "data": [1, 2]})
        payload = json.loads(result.decode()[2:])
        assert payload["result"] == {"status": "ok", "data": [1, 2]}


class TestStepFinish:
    def test_stop_reason(self, encoder):
        result = encoder.step_finish(finish_reason="stop")
        text = result.decode()
        assert text.startswith("e:")
        payload = json.loads(text[2:])
        assert payload["finishReason"] == "stop"
        assert payload["isContinued"] is False

    def test_tool_calls_reason(self, encoder):
        result = encoder.step_finish(finish_reason="tool-calls")
        payload = json.loads(result.decode()[2:])
        assert payload["finishReason"] == "tool-calls"
        assert payload["isContinued"] is True

    def test_default_usage(self, encoder):
        result = encoder.step_finish()
        payload = json.loads(result.decode()[2:])
        assert payload["usage"] == {"promptTokens": 0, "completionTokens": 0}

    def test_custom_usage(self, encoder):
        result = encoder.step_finish(
            usage={"promptTokens": 100, "completionTokens": 50}
        )
        payload = json.loads(result.decode()[2:])
        assert payload["usage"]["promptTokens"] == 100
        assert payload["usage"]["completionTokens"] == 50


class TestMessageFinish:
    def test_format(self, encoder):
        result = encoder.message_finish(finish_reason="stop")
        text = result.decode()
        assert text.startswith("d:")
        payload = json.loads(text[2:])
        assert payload["finishReason"] == "stop"

    def test_error_finish(self, encoder):
        result = encoder.message_finish(finish_reason="error")
        payload = json.loads(result.decode()[2:])
        assert payload["finishReason"] == "error"


class TestError:
    def test_format(self, encoder):
        result = encoder.error("something went wrong")
        text = result.decode()
        assert text.startswith("3:")
        assert text.endswith("\n")
        parsed = json.loads(text[2:])
        assert parsed == "something went wrong"

    def test_error_with_special_chars(self, encoder):
        result = encoder.error('err: "bad input" & <tag>')
        parsed = json.loads(result.decode()[2:])
        assert parsed == 'err: "bad input" & <tag>'


class TestContentType:
    def test_content_type(self, encoder):
        assert encoder.content_type() == "text/plain; charset=utf-8"


class TestExtraHeaders:
    def test_data_stream_header(self, encoder):
        headers = encoder.extra_headers()
        assert headers["x-vercel-ai-data-stream"] == "v1"
