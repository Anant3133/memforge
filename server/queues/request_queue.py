"""
Async Request Queue Manager: Queues incoming user requests and feeds them
to the continuous batching scheduler.
"""

import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import uuid


@dataclass
class GenerationRequest:
    request_id: str
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    created_at: float = field(default_factory=asyncio.get_event_loop().time if asyncio.get_event_loop().is_running() else lambda: 0.0)
    future: asyncio.Future = field(default_factory=asyncio.Future)


class AsyncRequestQueue:
    def __init__(self, max_queue_size: int = 1024):
        self.queue: asyncio.Queue[GenerationRequest] = asyncio.Queue(maxsize=max_queue_size)
        self.active_requests: Dict[str, GenerationRequest] = {}

    async def submit_request(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> GenerationRequest:
        req_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        req = GenerationRequest(
            request_id=req_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            future=future,
        )

        self.active_requests[req_id] = req
        await self.queue.put(req)
        return req

    async def get_next_batch(self, max_batch_size: int = 8) -> list[GenerationRequest]:
        batch = []
        if self.queue.empty():
            return batch

        while not self.queue.empty() and len(batch) < max_batch_size:
            req = await self.queue.get()
            batch.append(req)
        return batch

    def complete_request(self, request_id: str, result: Any) -> None:
        if request_id in self.active_requests:
            req = self.active_requests.pop(request_id)
            if not req.future.done():
                req.future.set_result(result)

    def fail_request(self, request_id: str, error: Exception) -> None:
        if request_id in self.active_requests:
            req = self.active_requests.pop(request_id)
            if not req.future.done():
                req.future.set_exception(error)
