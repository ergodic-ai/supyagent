"""POST /api/chat - Vercel AI SDK compatible streaming chat endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from supyagent.server.dependencies import get_agent_pool
from supyagent.server.models import ChatRequest
from supyagent.server.protocol import get_encoder
from supyagent.server.stream import agent_to_aisdk_stream

router = APIRouter()


@router.post("/chat")
async def chat(body: ChatRequest):
    """
    AI SDK compatible chat endpoint.

    Accepts { messages: [...], agent: "name", sessionId: "..." }
    Returns a streaming response in AI SDK Data Stream Protocol format.
    """
    encoder = get_encoder()
    pool = get_agent_pool()

    try:
        agent = pool.get_or_create(body.agent, body.session_id)
    except Exception as e:
        return StreamingResponse(
            iter([encoder.error(str(e)), encoder.message_finish("error")]),
            media_type=encoder.content_type(),
            headers=encoder.extra_headers(),
        )

    # Extract the last user message (AI SDK sends full history)
    user_message: str | list = ""
    for msg in reversed(body.messages):
        if msg.role == "user":
            user_message = msg.content
            break

    if not user_message:
        return StreamingResponse(
            iter([encoder.error("No user message found"), encoder.message_finish("error")]),
            media_type=encoder.content_type(),
            headers=encoder.extra_headers(),
        )

    # Get per-session lock to serialize requests
    agent_lock = pool.get_lock(body.agent, agent.session.meta.session_id)

    return StreamingResponse(
        agent_to_aisdk_stream(agent, user_message, encoder, agent_lock),
        media_type=encoder.content_type(),
        headers=encoder.extra_headers(),
    )
