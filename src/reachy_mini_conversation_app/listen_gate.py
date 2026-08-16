"""Single source of truth for whether Reachy is currently listening (push-to-talk).

``always_on`` (the default) keeps the gate permanently open, so behaviour is
identical to the original always-listening app. In ``push_to_talk`` mode the gate
is driven by control sources (a momentary/latching keyboard or USB-HID key, and
the companion UI toggle relayed via the Node server). They all funnel into one
boolean:

    enabled = (mode == always_on) OR latched OR held

- ``held``    — a momentary push-to-talk key is physically down (hold-to-talk).
- ``latched`` — a toggle/switch (companion UI or a latch key) is on.

``console.LocalStream.record_loop`` reads ``enabled`` to drop mic frames while the
gate is closed (one chokepoint covering every backend). Each false<->true
transition of ``enabled`` schedules the active handler's ``on_listen_open`` /
``on_listen_close`` hook so the backend can finalise the turn correctly (cascade
force-ends the utterance; realtime commits the input buffer and asks for a
response).
"""

from __future__ import annotations
import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Callable, Optional


if TYPE_CHECKING:
    from reachy_mini_conversation_app.conversation_handler import ConversationHandler

logger = logging.getLogger(__name__)

ALWAYS_ON = "always_on"
PUSH_TO_TALK = "push_to_talk"


def normalize_mode(value: Optional[str]) -> str:
    """Coerce an env/UI string to a known listen mode (defaulting to always_on)."""
    return PUSH_TO_TALK if (value or "").strip().lower() == PUSH_TO_TALK else ALWAYS_ON


class ListenGate:
    """Thread-safe gate that decides when mic audio reaches the active handler."""

    def __init__(self, mode: str = ALWAYS_ON) -> None:
        """Create a gate in the given mode (``always_on`` or ``push_to_talk``)."""
        self._mode = normalize_mode(mode)
        self._held = False
        self._latched = False
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._get_handler: Optional[Callable[[], Optional["ConversationHandler"]]] = None
        self._enabled = self._compute_enabled()

    def bind(
        self,
        loop: asyncio.AbstractEventLoop,
        get_handler: Callable[[], Optional["ConversationHandler"]],
    ) -> None:
        """Attach the running asyncio loop + active-handler accessor.

        Called from the stream runner once the loop and handler exist. Until
        bound, the gate still tracks state but fires no handler hooks.
        """
        self._loop = loop
        self._get_handler = get_handler

    # ── state ────────────────────────────────────────────────────────────────
    def _compute_enabled(self) -> bool:
        return self._mode == ALWAYS_ON or self._latched or self._held

    @property
    def enabled(self) -> bool:
        """Whether mic audio should currently flow to the handler."""
        return self._enabled

    @property
    def mode(self) -> str:
        """Current listen mode (``always_on`` or ``push_to_talk``)."""
        return self._mode

    @property
    def latched(self) -> bool:
        """Whether the latching toggle/switch is on."""
        return self._latched

    # ── mutations (safe to call from any thread) ───────────────────────────────
    def set_mode(self, mode: str) -> None:
        """Switch listen mode; leaving push_to_talk clears the momentary key."""
        with self._lock:
            new_mode = normalize_mode(mode)
            if new_mode == self._mode:
                return
            logger.info("Listen mode -> %s", new_mode)
            self._mode = new_mode
            if new_mode == ALWAYS_ON:
                self._held = False
            self._apply_locked(continuous=False)

    def set_held(self, held: bool) -> None:
        """Momentary push-to-talk key down (True) / up (False)."""
        with self._lock:
            if bool(held) == self._held:
                return
            self._held = bool(held)
            # A momentary press with no latch => capture continuously until release.
            self._apply_locked(continuous=self._held and not self._latched)

    def set_latched(self, latched: bool) -> None:
        """Set the latching toggle/switch explicitly (companion UI / SSE relay)."""
        with self._lock:
            if bool(latched) == self._latched:
                return
            self._latched = bool(latched)
            self._apply_locked(continuous=False)

    def toggle(self) -> None:
        """Flip the latching toggle (latch key / companion button)."""
        with self._lock:
            self._latched = not self._latched
            logger.info("Listen latch -> %s", self._latched)
            self._apply_locked(continuous=False)

    # ── internals ──────────────────────────────────────────────────────────────
    def _apply_locked(self, *, continuous: bool) -> None:
        new_enabled = self._compute_enabled()
        if new_enabled == self._enabled:
            return
        self._enabled = new_enabled
        self._fire(opening=new_enabled, continuous=continuous)

    def _fire(self, *, opening: bool, continuous: bool) -> None:
        loop = self._loop
        get_handler = self._get_handler
        if loop is None or get_handler is None:
            return
        handler = get_handler()
        if handler is None:
            return
        if opening:
            coro = handler.on_listen_open(continuous=continuous)
        else:
            coro = handler.on_listen_close(commit=True)
        try:
            asyncio.run_coroutine_threadsafe(coro, loop)
        except Exception as e:  # never let a control source crash on a dead loop
            logger.debug("ListenGate hook scheduling failed: %s", e)
