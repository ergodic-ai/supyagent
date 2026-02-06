"""
Shared models for the supyagent core.
"""


class _FunctionObj:
    """Lightweight function call object matching LiteLLM interface."""

    __slots__ = ("name", "arguments")

    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class ToolCallObj:
    """
    Lightweight wrapper matching the LiteLLM tool_call interface.

    Used when reconstructing tool calls from streaming chunks,
    where we collect raw dicts instead of LiteLLM objects.
    """

    __slots__ = ("id", "function")

    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.function = _FunctionObj(name, arguments)
