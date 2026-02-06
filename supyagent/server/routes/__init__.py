"""Route registration."""

from fastapi import FastAPI


def register_routes(app: FastAPI) -> None:
    """Register all API routes."""
    from supyagent.server.routes.agents import router as agents_router
    from supyagent.server.routes.chat import router as chat_router
    from supyagent.server.routes.health import router as health_router
    from supyagent.server.routes.sessions import router as sessions_router
    from supyagent.server.routes.tools import router as tools_router

    app.include_router(health_router)
    app.include_router(chat_router, prefix="/api")
    app.include_router(agents_router, prefix="/api")
    app.include_router(sessions_router, prefix="/api")
    app.include_router(tools_router, prefix="/api")
