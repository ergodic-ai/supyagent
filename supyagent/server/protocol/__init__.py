"""AI SDK stream protocol encoders."""

from supyagent.server.protocol.base import StreamEncoder
from supyagent.server.protocol.data_stream import DataStreamEncoder

__all__ = ["StreamEncoder", "DataStreamEncoder", "get_encoder"]


def get_encoder(protocol: str = "data-stream") -> StreamEncoder:
    """
    Get the appropriate stream encoder.

    Args:
        protocol: "data-stream" for the AI SDK Data Stream Protocol
    """
    return DataStreamEncoder()
