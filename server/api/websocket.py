"""
WebSocket endpoints: live token streaming and an engine telemetry feed.

`/ws/stream` streams real generated tokens from the engine. Each token carries
its provenance (`source`: target / draft / resample / bonus) so the UI can colour
draft-accepted tokens differently from target-corrected ones, plus a snapshot of
engine metrics for the live gauges.

`/ws/telemetry` is a prompt-independent feed of engine state for the dashboard,
so the memory heatmap keeps updating even when this client is idle.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from memforge_ml.runtime.sampling import SamplingParams
from server.config import SERVER_CONFIG
from server.engine_manager import ENGINE_MANAGER

router = APIRouter()


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_json(self, message: Dict[str, Any]) -> None:
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)


manager = ConnectionManager()


def _params_from(payload: Dict[str, Any]) -> SamplingParams:
    return SamplingParams(
        max_new_tokens=int(payload.get("max_new_tokens", SERVER_CONFIG.default_max_tokens)),
        temperature=float(payload.get("temperature", 0.7)),
        top_p=float(payload.get("top_p", 0.9)),
        seed=payload.get("seed"),
    )


@router.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            payload = json.loads(await websocket.receive_text())
            prompt = (payload.get("prompt") or "").strip()
            if not prompt:
                await websocket.send_json({"type": "error", "message": "prompt is required"})
                continue

            use_speculative = bool(payload.get("use_speculative", False))
            params = _params_from(payload)

            await websocket.send_json(
                {
                    "type": "start",
                    "mode": "speculative" if use_speculative else "continuous_batching",
                    "prompt": prompt,
                    "max_new_tokens": params.max_new_tokens,
                }
            )

            try:
                if use_speculative:
                    source = ENGINE_MANAGER.speculative_stream(prompt, params)
                else:
                    source = ENGINE_MANAGER.stream(prompt, params)
                async for event in source:
                    await websocket.send_json(event)
            except Exception as exc:
                await websocket.send_json(
                    {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


@router.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    """Push engine telemetry and the block occupancy map a few times a second."""
    await websocket.accept()
    try:
        while True:
            if ENGINE_MANAGER.engine is not None:
                occupancy = ENGINE_MANAGER.block_occupancy()
                await websocket.send_json(
                    {
                        "type": "telemetry",
                        "telemetry": ENGINE_MANAGER.engine.telemetry(),
                        "last_step": ENGINE_MANAGER._last_report,
                        "occupancy": occupancy,
                    }
                )
            else:
                await websocket.send_json(
                    {
                        "type": "telemetry",
                        "telemetry": None,
                        "ready": ENGINE_MANAGER.ready,
                        "load_error": ENGINE_MANAGER.load_error,
                    }
                )
            await asyncio.sleep(0.4)
    except WebSocketDisconnect:
        return
    except Exception:
        return
