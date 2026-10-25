"""Tests for the push-to-talk listen gate."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from reachy_mini_conversation_app.listen_gate import ALWAYS_ON, PUSH_TO_TALK, ListenGate, normalize_mode


def test_normalize_mode_defaults_to_always_on() -> None:
    """Anything but push_to_talk means always_on."""
    assert normalize_mode(None) == ALWAYS_ON
    assert normalize_mode("nonsense") == ALWAYS_ON
    assert normalize_mode(" Push_To_Talk ") == PUSH_TO_TALK


def test_always_on_gate_stays_open_whatever_the_controls_do() -> None:
    """The default mode never gates the mic."""
    gate = ListenGate(ALWAYS_ON)
    assert gate.enabled is True
    gate.set_held(True)
    gate.set_held(False)
    gate.set_latched(False)
    assert gate.enabled is True


def test_push_to_talk_gate_opens_on_hold_or_latch() -> None:
    """A hold or a latch opens the gate; releasing both closes it."""
    gate = ListenGate(PUSH_TO_TALK)
    assert gate.enabled is False
    gate.set_held(True)
    assert gate.enabled is True
    gate.set_held(False)
    assert gate.enabled is False
    gate.set_latched(True)
    assert gate.enabled is True
    gate.toggle()
    assert gate.enabled is False


def test_leaving_push_to_talk_releases_the_hold() -> None:
    """Switching modes drops a hold in progress."""
    gate = ListenGate(PUSH_TO_TALK)
    gate.set_held(True)
    gate.set_mode(ALWAYS_ON)
    assert gate.enabled is True
    gate.set_mode(PUSH_TO_TALK)
    assert gate.enabled is False


def test_hold_and_release_fire_the_handler_hooks() -> None:
    """The bound handler sees one open and one close per hold."""
    handler = MagicMock()
    handler.on_listen_open = AsyncMock()
    handler.on_listen_close = AsyncMock()

    async def run() -> None:
        gate = ListenGate(PUSH_TO_TALK)
        gate.bind(asyncio.get_running_loop(), lambda: handler)
        gate.set_held(True)
        gate.set_held(False)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(run())
    handler.on_listen_open.assert_awaited_once_with(continuous=True)
    handler.on_listen_close.assert_awaited_once_with(commit=True)


def test_unbound_gate_tracks_state_without_hooks() -> None:
    """Before bind() the gate still computes enabled."""
    gate = ListenGate(PUSH_TO_TALK)
    gate.set_latched(True)
    assert gate.enabled is True
