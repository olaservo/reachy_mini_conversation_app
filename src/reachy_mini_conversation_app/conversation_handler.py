from __future__ import annotations
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TypeAlias
from collections.abc import Callable

import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler
from numpy.typing import NDArray

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


logger = logging.getLogger(__name__)


AudioFrame: TypeAlias = tuple[int, NDArray[np.int16]]
HandlerOutput: TypeAlias = AudioFrame | AdditionalOutputs | None
QueueItem: TypeAlias = AudioFrame | AdditionalOutputs


class ConversationHandler(AsyncStreamHandler, ABC):
    """Shared app handler contract for realtime conversation backends."""

    deps: ToolDependencies
    output_queue: asyncio.Queue[QueueItem]
    _clear_queue: Callable[[], None] | None = None

    # Event-injection plumbing (morning-briefing demo). Shared across backends so any
    # backend can be woken by an ha-events-bridge push via inject_user_turn. Initialized
    # by _init_event_injection(), which each backend calls early in its own __init__.
    _connected_event: asyncio.Event
    _events_task: asyncio.Task[None] | None = None

    def _init_event_injection(self) -> None:
        """Initialize the shared event-injection fields. Call early in subclass __init__."""
        self._connected_event = asyncio.Event()
        self._events_task = None

    def _maybe_start_events_loop(self) -> "asyncio.Task[None] | None":
        """Start the ha-events-bridge injection loop if enabled (morning-briefing demo).

        Opt-in via config.HA_EVENTS_ENABLED. Idempotent: if a loop is already running it
        is reused (a backend whose _restart_session re-enters start_up must not leak a
        second loop). Stores the task on self._events_task and returns it, or None when
        disabled / on failure. Never raises into the startup path.
        """
        if not getattr(config, "HA_EVENTS_ENABLED", False):
            return None
        existing = self._events_task
        if existing is not None and not existing.done():
            return existing
        try:
            from reachy_mini_conversation_app.events_loop import run_events_injection_loop

            self._events_task = asyncio.create_task(
                run_events_injection_loop(self), name="events-injection"
            )
            logger.info("Started HA events injection loop (bridge=%s)", config.HA_EVENTS_BRIDGE_URL)
            return self._events_task
        except Exception as exc:
            logger.warning("Failed to start HA events injection loop: %s", exc)
            return None

    @staticmethod
    async def _stop_events_loop(task: "asyncio.Task[None] | None") -> None:
        """Cancel and await the events loop task. Safe to call with None."""
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug("Events loop task ended with: %s", exc)

    @abstractmethod
    async def inject_user_turn(self, text: str) -> None:
        """Inject an out-of-band user turn into the live session and request a response.

        Sanctioned extension point for event-driven wake-ups (see events_loop). Concrete
        backends create a user message in their session and trigger a model response.
        """
        ...

    @abstractmethod
    def copy(self) -> ConversationHandler:
        """Create a copy of the handler."""
        ...

    @abstractmethod
    async def start_up(self) -> None:
        """Start the realtime handler."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Shut down the realtime handler."""
        ...

    @abstractmethod
    async def receive(self, frame: AudioFrame) -> None:
        """Receive an input audio frame."""
        ...

    @abstractmethod
    async def emit(self) -> HandlerOutput:
        """Emit the next output item."""
        ...

    @abstractmethod
    async def apply_personality(self, profile: str | None) -> str:
        """Apply a personality profile."""
        ...

    @abstractmethod
    async def get_available_voices(self) -> list[str]:
        """Return voices available for the active backend."""
        ...

    @abstractmethod
    def get_current_voice(self) -> str:
        """Return the current voice."""
        ...

    @abstractmethod
    async def change_voice(self, voice: str) -> str:
        """Change the current voice."""
        ...
