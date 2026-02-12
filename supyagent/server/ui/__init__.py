"""
Ephemeral browser UI for supyagent CLI commands.

Provides a local FastAPI app that serves HTML templates for interactive
management of models, keys, and the onboarding wizard.
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def create_ui_app(
    mode: str,
    done_callback: Callable[[dict[str, Any]], None] | None = None,
) -> FastAPI:
    """
    Create a short-lived FastAPI app for the browser UI.

    Args:
        mode: Which UI to serve — "models" or "hello".
        done_callback: Called when user clicks Done / finishes the wizard.
    """
    app = FastAPI(title=f"supyagent {mode}", docs_url=None, redoc_url=None)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.mode = mode
    app.state.done_callback = done_callback or (lambda _: None)

    from supyagent.server.ui.routes import register_ui_routes

    register_ui_routes(app, mode)

    return app
