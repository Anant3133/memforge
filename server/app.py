"""FastAPI entrypoint for the MemForge inference engine."""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.api.routes import router as rest_router
from server.api.websocket import router as ws_router
from server.config import SERVER_CONFIG
from server.engine_manager import ENGINE_MANAGER


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[MemForge Server] starting up...")
    await ENGINE_MANAGER.start()
    app.state.engine_manager = ENGINE_MANAGER
    if ENGINE_MANAGER.ready:
        print("[MemForge Server] engine ready; accepting requests")
    else:
        print(f"[MemForge Server] engine not ready: {ENGINE_MANAGER.load_error}")
    yield
    print("[MemForge Server] shutting down...")
    await ENGINE_MANAGER.stop()


def create_app() -> FastAPI:
    app = FastAPI(
        title="MemForge LLM Inference Engine",
        description=(
            "High-performance custom LLM inference engine with PagedAttention, "
            "continuous batching and speculative decoding."
        ),
        version="0.2.0",
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
