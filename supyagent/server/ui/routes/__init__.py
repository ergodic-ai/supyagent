"""Route registration for the browser UI."""

from __future__ import annotations

from fastapi import FastAPI


def register_ui_routes(app: FastAPI, mode: str) -> None:
    """Register routes based on the UI mode."""
    from supyagent.server.ui.routes.common import router as common_router

    app.include_router(common_router)

    if mode in ("models", "hello"):
        from supyagent.server.ui.routes.models import router as models_router

        app.include_router(models_router)

    if mode == "hello":
        from supyagent.server.ui.routes.hello import router as hello_router

        app.include_router(hello_router)

    if mode == "agents":
        from supyagent.server.ui.routes.agents import router as agents_router

        app.include_router(agents_router)
