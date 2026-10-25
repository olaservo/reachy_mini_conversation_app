"""Push-to-talk control source: a keyboard / USB-HID key listener for a ListenGate.

Active only in ``push_to_talk`` mode. Uses ``pynput`` (an optional dependency) to
bind a momentary hold-to-talk key (``config.PTT_KEY``) and a latching toggle key
(``config.PTT_TOGGLE_KEY``). A missing ``pynput`` is logged and otherwise ignored,
leaving the gate driven by the other control source (the ``conversation.listen``
JSON-RPC toggle the settings UI can call).
"""

from __future__ import annotations
import logging
from typing import Any, Callable

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.listen_gate import ListenGate


logger = logging.getLogger(__name__)


def _make_matcher(keyboard: Any, spec: str) -> Callable[[Any], bool]:
    """Return a predicate matching a pynput key against a config string.

    ``spec`` is either a special key name (``space``, ``ctrl``, ``f1`` …) or a
    single character (``m``).
    """
    name = (spec or "").strip().lower()
    special = getattr(keyboard.Key, name, None) if name else None
    if special is not None:
        return lambda key: key == special
    char = name[:1]

    def match(key: Any) -> bool:
        k = getattr(key, "char", None)
        return isinstance(k, str) and k.lower() == char

    return match


def start_keyboard_listener(gate: ListenGate) -> Any:
    """Start a background pynput listener wiring keys to the gate.

    The momentary key (``config.PTT_KEY``) drives hold-to-talk; the toggle key
    (``config.PTT_TOGGLE_KEY``) latches listening on/off. Returns the listener
    (which has a ``.stop()`` method) so the caller can shut it down. Raises if
    ``pynput`` is unavailable so the caller can log and continue.
    """
    from pynput import keyboard  # lazy: optional dependency

    ptt_match = _make_matcher(keyboard, config.PTT_KEY)
    toggle_match = _make_matcher(keyboard, config.PTT_TOGGLE_KEY)
    # pynput repeats on_press while a key is held; debounce the latching key so one
    # physical press flips the latch exactly once.
    state = {"toggle_down": False}

    def on_press(key: Any) -> None:
        try:
            if ptt_match(key):
                gate.set_held(True)
            elif toggle_match(key) and not state["toggle_down"]:
                state["toggle_down"] = True
                gate.toggle()
        except Exception as e:
            logger.debug("PTT key press error: %s", e)

    def on_release(key: Any) -> None:
        try:
            if ptt_match(key):
                gate.set_held(False)
            elif toggle_match(key):
                state["toggle_down"] = False
        except Exception as e:
            logger.debug("PTT key release error: %s", e)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()
    return listener
