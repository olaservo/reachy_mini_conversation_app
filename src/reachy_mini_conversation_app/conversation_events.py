"""Pub/sub bus for conversation activity events, surfaced to UI clients via SSE.

Why this module exists
----------------------
The realtime handler (``base_realtime.BaseRealtimeHandler``) calls
``_mark_activity(reason)`` on every meaningful turn transition (user speech
started/stopped, response created, assistant audio chunk, tool call, ...). It
exposes ``set_activity_observer(callback)`` to opt into this stream without
coupling the handler to any specific transport.

This module wires the observer to a fan-out bus that any number of HTTP/SSE
clients can subscribe to. The new ``static_v2`` web UI uses it to drive the
state of its conversation orb (idle / listening / thinking / speaking) without
needing audio access in the browser - the truth lives on the Python side.

Threading model
---------------
The realtime handler runs on its own asyncio event loop (the LocalStream
runner thread). FastAPI/uvicorn serves SSE clients on a different loop.
Each subscriber owns an ``asyncio.Queue`` created on the loop that called
``subscribe()`` (the FastAPI request handler), and ``publish()`` schedules
inserts into those queues via ``call_soon_threadsafe`` so the realtime
loop never blocks on UI clients and never crashes when one disconnects.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Subscriber:
    """A single SSE consumer.

    ``loop`` is captured at ``subscribe()`` time so the publisher can schedule
    ``queue.put_nowait`` thread-safely back onto the consumer's loop.
    """

    loop: asyncio.AbstractEventLoop
    queue: "asyncio.Queue[str]"


class ConversationEventBus:
    """Thread-safe fan-out of conversation activity events.

    The bus is created once per app instance and bound as the activity observer
    of the realtime handler via ``BaseRealtimeHandler.set_activity_observer``.
    Subscribers are SSE request handlers; each one gets its own bounded queue
    so a slow consumer cannot starve the others.
    """

    _MAX_QUEUE_SIZE = 64

    def __init__(self) -> None:
        self._subscribers: list[_Subscriber] = []
        self._lock = threading.Lock()

    def subscribe(self) -> tuple["asyncio.Queue[str]", Callable[[], None]]:
        """Register a new subscriber and return its queue plus an unsubscribe.

        Must be called from inside an asyncio coroutine so the running loop can
        be captured. The returned queue receives every future event published
        on the bus until ``unsubscribe()`` is invoked.
        """
        loop = asyncio.get_running_loop()
        queue: "asyncio.Queue[str]" = asyncio.Queue(maxsize=self._MAX_QUEUE_SIZE)
        sub = _Subscriber(loop=loop, queue=queue)
        with self._lock:
            self._subscribers.append(sub)

        def unsubscribe() -> None:
            with self._lock:
                try:
                    self._subscribers.remove(sub)
                except ValueError:
                    pass

        return queue, unsubscribe

    def publish(self, event: str) -> None:
        """Broadcast ``event`` to every subscriber. Safe from any thread.

        Each delivery is scheduled on the subscriber's own loop with
        ``call_soon_threadsafe``. If a subscriber's queue is full we drop the
        event for that subscriber (the UI is best-effort: the next transition
        will catch it back up). If the loop has been closed already we silently
        skip the subscriber - it will be cleaned up when its consumer detects
        the disconnect.
        """
        with self._lock:
            snapshot = list(self._subscribers)
        for sub in snapshot:
            try:
                sub.loop.call_soon_threadsafe(self._enqueue_safely, sub.queue, event)
            except RuntimeError:
                # Loop closed; the subscriber will be removed by its own
                # cleanup path. Nothing actionable here.
                pass

    @staticmethod
    def _enqueue_safely(queue: "asyncio.Queue[str]", event: str) -> None:
        """Best-effort, non-blocking insert. Drops the event when the queue is full."""
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.debug("conversation event dropped (subscriber queue full): %s", event)
