"""
Client-disconnect propagation: from an HTTP hangup down to a scheduler abort.

A generation that nobody is listening to still occupies physical KV blocks and
still costs a slice of every forward pass. `PROGRESS_AND_NEXT_PHASES.md` §8
Phase F records the gap: the WebSocket path aborts, HTTP disconnects are not
detected. Two distinct cases:

  * **Streaming (SSE).** Starlette closes the response generator when the client
    goes away, so `EngineManager.stream()`'s `finally` already aborts. That path
    is covered; what it lacked was a test proving the blocks come back.
  * **Non-streaming.** `await ENGINE_MANAGER.generate(...)` waits on a queue
    until the sequence finishes. A client hanging up is not noticed at all: the
    request generates every one of its tokens into a response that will never be
    read, holding blocks the whole time. Under a saturated pool that is a
    self-inflicted memory-pressure event.

`run_cancellable()` closes the second case by racing the work against the
client's connection state.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

from fastapi import Request

# How often to ask whether the client is still there. 250 ms is fast enough that
# the blocks come back within an engine step or two, and slow enough to cost
# nothing measurable per in-flight request.
DISCONNECT_POLL_INTERVAL_S = 0.25


class ClientDisconnected(Exception):
    """Raised when the caller went away before its result was ready."""


async def _client_is_gone(request: Request) -> bool:
    """
    Ask whether the client disconnected, without ever blocking.

    Must be awaited from the request handler's **own** task. Starlette's
    `Request.is_disconnected()` peeks at the receive channel inside an
    already-cancelled `anyio.CancelScope`, and an anyio cancel scope entered
    from a task anyio does not manage does not behave: driving this from a bare
    `asyncio` task hangs the request instead of returning. That is why the poll
    below is inline rather than a watcher task -- the obvious design, and the
    wrong one.

    A transport that cannot answer at all is treated as "still connected":
    failing to detect a disconnect degrades to the old behaviour, whereas
    guessing "gone" would abort live requests.
    """
    try:
        return await request.is_disconnected()
    except Exception:
        return False


async def run_cancellable(
    request: Request,
    work: Awaitable[Any],
    on_cancel: Optional[Callable[[], Any]] = None,
    *,
    poll_interval: float = DISCONNECT_POLL_INTERVAL_S,
) -> Any:
    """
    Await `work`, abandoning it if the client disconnects first.

    On disconnect, `on_cancel` runs -- this is where the engine is told to abort
    the sequence and free its blocks -- and `ClientDisconnected` is raised. The
    work task is cancelled on every exit path, so nothing keeps generating
    behind a connection that no longer exists.

    An exception from `on_cancel` is deliberately swallowed: a cleanup failure
    must not mask the disconnect, and there is no client left to report it to.
    """
    work_task = asyncio.ensure_future(work)
    try:
        while True:
            # A timeout on asyncio.wait does *not* cancel the task it is waiting
            # on, so this polls without disturbing the work in progress.
            done, _ = await asyncio.wait({work_task}, timeout=poll_interval)
            if work_task in done:
                return work_task.result()

            if await _client_is_gone(request):
                if on_cancel is not None:
                    try:
                        result = on_cancel()
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:
                        pass
                raise ClientDisconnected(
                    "client disconnected before the response was ready"
                )
    finally:
        if not work_task.done():
            work_task.cancel()
        # Let the cancellation settle so no "task was destroyed but it is
        # pending" warning leaks out of the event loop.
        await asyncio.gather(work_task, return_exceptions=True)
