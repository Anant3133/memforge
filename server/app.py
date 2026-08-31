"""
FastAPI Server Entrypoint for MemForge LLM Inference Engine.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from server.api.routes import router as rest_router
from server.api.websocket import router as ws_router
from server.config import SERVER_CONFIG


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize engine structures
    print("🚀 [MemForge Server] Starting up LLM Inference Engine...")
    yield
    # Shutdown: cleanup
    print("🛑 [MemForge Server] Shutting down...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="MemForge LLM Inference Engine",
        description="High-performance custom LLM inference engine with PagedAttention, Continuous Batching, and Speculative Decoding.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(rest_router)
    app.include_router(ws_router)

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "server.app:app",
        host=SERVER_CONFIG.host,
        port=SERVER_CONFIG.port,
        reload=SERVER_CONFIG.reload,
    )
