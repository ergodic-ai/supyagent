"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from supyagent.server.routes import register_routes


def create_app(cors_origins: list[str] | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        # Cleanup: reset agent pool on shutdown
        from supyagent.server.dependencies import reset_agent_pool

        reset_agent_pool()

    app = FastAPI(
        title="supyagent",
        description="AI agent server compatible with Vercel AI SDK useChat",
        lifespan=lifespan,
    )

    # CORS
    origins = cors_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_routes(app)
    return app
