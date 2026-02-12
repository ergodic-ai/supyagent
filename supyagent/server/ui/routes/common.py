"""Shared UI routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class DoneBody(BaseModel):
    action: str = "done"
    data: dict = {}


@router.post("/api/done")
async def done(request: Request, body: DoneBody | None = None):
    """Signal that the user is finished with the UI."""
    payload = body.model_dump() if body else {"action": "done", "data": {}}
    callback = request.app.state.done_callback
    if callback:
        callback(payload)
    return {"ok": True}
