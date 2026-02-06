"""Bridge between Agent.send_message_stream() and AI SDK wire format."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import uuid
from typing import Any, AsyncGenerator

from supyagent.core.agent import Agent
from supyagent.server.protocol.base import StreamEncoder

_SENTINEL = object()


async def agent_to_aisdk_stream(
    agent: Agent,
    user_message: str | list,
    encoder: StreamEncoder,
    agent_lock: threading.Lock,
) -> AsyncGenerator[bytes, None]:
    """
    Consume agent.send_message_stream() in a background thread,
    encode each event with the given protocol encoder,
    and yield bytes asynchronously.
    """
    message_id = uuid.uuid4().hex[:12]
    yield encoder.message_start(message_id)

    # Use a thread-safe queue to bridge sync generator -> async generator
    q: queue.Queue[tuple[str, Any] | object] = queue.Queue(maxsize=256)
    error_holder: list[Exception] = []

    def _consume():
        """Run in background thread: iterate the sync generator, push to queue."""
        try:
            with agent_lock:
                for event in agent.send_message_stream(user_message):
                    q.put(event)
        except Exception as exc:
            error_holder.append(exc)
        finally:
            q.put(_SENTINEL)

    thread = threading.Thread(target=_consume, daemon=True)
    thread.start()

    try:
        while True:
            try:
                item = q.get(timeout=0.05)
            except queue.Empty:
                await asyncio.sleep(0)
                continue

            if item is _SENTINEL:
                break

            event_type, data = item

            if event_type == "text":
                yield encoder.text_delta(data)

            elif event_type == "reasoning":
                # Reasoning is not part of the standard AI SDK protocol.
                # Skip it to keep the stream clean for useChat.
                pass

            elif event_type == "tool_start":
                tool_call_id = data.get("id", f"call_{uuid.uuid4().hex[:8]}")
                tool_name = data["name"]
                try:
                    args = json.loads(data["arguments"])
                except (json.JSONDecodeError, TypeError):
                    args = {}
                yield encoder.tool_call(tool_call_id, tool_name, args)

            elif event_type == "tool_end":
                tool_call_id = data.get("id", "unknown")
                result = data.get("result", {})
                yield encoder.tool_result(tool_call_id, result)
                yield encoder.step_finish(finish_reason="tool-calls")

            elif event_type == "done":
                pass  # handled below

    except Exception as exc:
        yield encoder.error(str(exc))
        return
    finally:
        thread.join(timeout=10)

    # Check for errors from the consumer thread
    if error_holder:
        yield encoder.error(str(error_holder[0]))
        return

    # Emit final step finish and message finish
    yield encoder.step_finish(finish_reason="stop")
    yield encoder.message_finish(finish_reason="stop")
