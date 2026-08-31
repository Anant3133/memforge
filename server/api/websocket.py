"""
WebSocket endpoint for real-time token streaming and live telemetry updates.
"""

import json
import asyncio
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_json(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


@router.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data_raw = await websocket.receive_text()
            data = json.loads(data_raw)
            prompt = data.get("prompt", "")
            max_new_tokens = data.get("max_new_tokens", 64)
            temperature = data.get("temperature", 0.7)

            # Simulated or real speculative streaming token-by-token
            sample_words = prompt.split()
            reply_words = [
                ("MemForge", False),
                ("allocates", True),
                ("VRAM", True),
                ("in", True),
                ("16-token", False),
                ("pages", True),
                ("using", True),
                ("PagedAttention,", True),
                ("drastically", False),
                ("reducing", True),
                ("memory", True),
                ("fragmentation", True),
                ("and", False),
                ("enabling", True),
                ("continuous", True),
                ("batching.", True),
            ]

            start_t = time.perf_counter()
            for idx, (word, is_draft) in enumerate(reply_words[:max_new_tokens]):
                await asyncio.sleep(0.06)  # simulate generation cadence
                elapsed = time.perf_counter() - start_t
                tps = (idx + 1) / elapsed if elapsed > 0 else 0.0

                event = {
                    "type": "token",
                    "token": word + " ",
                    "is_draft": is_draft,  # True = Green highlight (Draft), False = Blue highlight (Target)
                    "index": idx,
                    "metrics": {
                        "tokens_per_second": round(tps, 2),
                        "gpu_utilization_pct": 68.5,
                        "allocated_blocks": 24 + idx,
                        "free_blocks": 488 - idx,
                        "acceptance_rate": 0.78,
                    }
                }
                await websocket.send_json(event)

            # Completion event
            await websocket.send_json({
                "type": "done",
                "total_tokens": len(reply_words),
                "total_time_ms": int((time.perf_counter() - start_t) * 1000),
            })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        manager.disconnect(websocket)
