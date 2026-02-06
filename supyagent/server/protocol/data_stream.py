"""AI SDK Data Stream Protocol encoder.

Wire format: {type_code}:{json_data}\n

Type codes:
    f  message start
    0  text delta
    9  tool call
    a  tool result
    e  step finish
    d  message finish
    3  error
"""

import json
from typing import Any

from supyagent.server.protocol.base import StreamEncoder


class DataStreamEncoder(StreamEncoder):
    """Encodes events using the AI SDK Data Stream Protocol prefix-code format."""

    def message_start(self, message_id: str) -> bytes:
        return f'f:{json.dumps({"messageId": message_id})}\n'.encode()

    def text_delta(self, text: str) -> bytes:
        return f"0:{json.dumps(text)}\n".encode()

    def tool_call(
        self, tool_call_id: str, tool_name: str, args: dict[str, Any]
    ) -> bytes:
        payload = {
            "toolCallId": tool_call_id,
            "toolName": tool_name,
            "args": args,
        }
        return f"9:{json.dumps(payload)}\n".encode()

    def tool_result(self, tool_call_id: str, result: Any) -> bytes:
        payload = {"toolCallId": tool_call_id, "result": result}
        return f"a:{json.dumps(payload)}\n".encode()

    def step_finish(
        self, finish_reason: str = "stop", usage: dict[str, int] | None = None
    ) -> bytes:
        payload: dict[str, Any] = {
            "finishReason": finish_reason,
            "usage": usage or {"promptTokens": 0, "completionTokens": 0},
            "isContinued": finish_reason == "tool-calls",
        }
        return f"e:{json.dumps(payload)}\n".encode()

    def message_finish(
        self, finish_reason: str = "stop", usage: dict[str, int] | None = None
    ) -> bytes:
        payload = {
            "finishReason": finish_reason,
            "usage": usage or {"promptTokens": 0, "completionTokens": 0},
        }
        return f"d:{json.dumps(payload)}\n".encode()

    def error(self, message: str) -> bytes:
        return f"3:{json.dumps(message)}\n".encode()

    def content_type(self) -> str:
        return "text/plain; charset=utf-8"

    def extra_headers(self) -> dict[str, str]:
        return {"x-vercel-ai-data-stream": "v1"}
