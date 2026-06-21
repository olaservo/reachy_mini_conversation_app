"""Tests for the lifted event-injection hook (morning-briefing demo).

Event injection is a sanctioned ``ConversationHandler`` extension point: the launcher
(``_maybe_start_events_loop``) and teardown (``_stop_events_loop``) live on the base class,
and every backend implements ``inject_user_turn``. These tests lock:
  - the opt-in default (no loop unless ``HA_EVENTS_ENABLED``), for both realtime and Gemini;
  - the idempotent launcher (a re-entered start_up must not leak a second loop);
  - the events loop driving ``inject_user_turn`` on a qualifying push;
  - each backend's ``inject_user_turn`` (OpenAI realtime + Gemini).
"""

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastrtc import AdditionalOutputs

import reachy_mini_conversation_app.events_loop as events_loop_mod
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.events_loop import INJECTION_PROMPT, run_events_injection_loop
from reachy_mini_conversation_app.gemini_live import GeminiLiveHandler
from reachy_mini_conversation_app.openai_realtime import OpenaiRealtimeHandler
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


def _openai_handler() -> OpenaiRealtimeHandler:
    return OpenaiRealtimeHandler(_deps())


def _gemini_handler() -> GeminiLiveHandler:
    return GeminiLiveHandler(_deps())


# --------------------------------------------------------------------------- #
# Opt-in default: disabled => no loop. Locked for BOTH backends.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("make_handler", [_openai_handler, _gemini_handler])
async def test_maybe_start_events_loop_disabled_returns_none(
    monkeypatch: Any, make_handler: Any
) -> None:
    monkeypatch.setattr(config, "HA_EVENTS_ENABLED", False)
    # If the disabled guard ever regressed, this sentinel would blow up the import path.
    monkeypatch.setattr(
        events_loop_mod,
        "run_events_injection_loop",
        lambda *_a, **_k: pytest.fail("events loop must not start when disabled"),
    )
    handler = make_handler()
    assert handler._maybe_start_events_loop() is None
    assert handler._events_task is None


# --------------------------------------------------------------------------- #
# Enabled: starts exactly one task, idempotent on re-entry, stops cleanly.
# Covers the Gemini-restart double-start guard (start_up re-enters start_up).
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("make_handler", [_openai_handler, _gemini_handler])
async def test_maybe_start_events_loop_idempotent_and_stoppable(
    monkeypatch: Any, make_handler: Any
) -> None:
    monkeypatch.setattr(config, "HA_EVENTS_ENABLED", True)

    started: list[Any] = []

    async def fake_loop(handler: Any) -> None:
        started.append(handler)
        await asyncio.Event().wait()  # long-lived; never completes on its own

    monkeypatch.setattr(events_loop_mod, "run_events_injection_loop", fake_loop)

    handler = make_handler()
    task1 = handler._maybe_start_events_loop()
    assert task1 is not None
    assert task1.get_name() == "events-injection"

    # Re-entry (e.g. Gemini _restart_session re-running start_up) must reuse the live task.
    task2 = handler._maybe_start_events_loop()
    assert task2 is task1

    await asyncio.sleep(0)  # let the coroutine body run once
    assert len(started) == 1  # exactly one loop, no leak

    await handler._stop_events_loop(task1)
    assert task1.cancelled()


@pytest.mark.asyncio
async def test_stop_events_loop_handles_none() -> None:
    # No task (disabled path) -> no-op, no error.
    await OpenaiRealtimeHandler._stop_events_loop(None)


# --------------------------------------------------------------------------- #
# The events loop drives inject_user_turn on a qualifying push.
# Depends only on the UserTurnInjector protocol, not on any concrete handler.
# --------------------------------------------------------------------------- #


class _FakeInjector:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fired = asyncio.Event()

    async def inject_user_turn(self, text: str) -> None:
        self.calls.append(text)
        self.fired.set()


class _FakeEventsClient:
    """Yields one matching occurrence then blocks (so the loop doesn't reconnect-spin)."""

    def __init__(self, _url: str) -> None:
        pass

    async def __aenter__(self) -> "_FakeEventsClient":
        return self

    async def __aexit__(self, *_a: Any) -> bool:
        return False

    async def initialize(self, client_name: str = "") -> None:
        return None

    async def list_events(self) -> list[dict[str, str]]:
        return [{"name": "ha.state_changed"}]

    async def stream(self, _name: str, params: Any = None):
        yield {
            "method": "notifications/events/event",
            "params": {"data": {"entity_id": "light.test", "to": "on"}},
        }
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_run_events_injection_loop_injects_on_qualifying_fire(monkeypatch: Any) -> None:
    # Make the snapshot deterministic: any hour passes, no debounce, repeatable, matching entity.
    monkeypatch.setattr(config, "HA_EVENTS_BRIDGE_URL", "http://bridge.test/")
    monkeypatch.setattr(config, "HA_EVENTS_ENTITY_ID", "light.test")
    monkeypatch.setattr(config, "HA_EVENTS_TO_STATE", "on")
    monkeypatch.setattr(config, "HA_EVENTS_MORNING_START", 0)
    monkeypatch.setattr(config, "HA_EVENTS_MORNING_END", 24)
    monkeypatch.setattr(config, "HA_EVENTS_DEBOUNCE_S", 0.0)
    monkeypatch.setattr(config, "HA_EVENTS_ONCE_PER_MORNING", False)
    monkeypatch.setattr(events_loop_mod, "EventsClient", _FakeEventsClient)

    injector = _FakeInjector()
    task = asyncio.create_task(run_events_injection_loop(injector))
    try:
        await asyncio.wait_for(injector.fired.wait(), timeout=2.0)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert injector.calls == [INJECTION_PROMPT]


@pytest.mark.asyncio
async def test_run_events_injection_loop_skips_non_matching(monkeypatch: Any) -> None:
    monkeypatch.setattr(config, "HA_EVENTS_BRIDGE_URL", "http://bridge.test/")
    monkeypatch.setattr(config, "HA_EVENTS_ENTITY_ID", "light.other")  # mismatch
    monkeypatch.setattr(config, "HA_EVENTS_TO_STATE", "on")
    monkeypatch.setattr(config, "HA_EVENTS_MORNING_START", 0)
    monkeypatch.setattr(config, "HA_EVENTS_MORNING_END", 24)
    monkeypatch.setattr(config, "HA_EVENTS_ONCE_PER_MORNING", False)
    monkeypatch.setattr(events_loop_mod, "EventsClient", _FakeEventsClient)

    injector = _FakeInjector()
    task = asyncio.create_task(run_events_injection_loop(injector))
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(injector.fired.wait(), timeout=0.3)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert injector.calls == []


# --------------------------------------------------------------------------- #
# Per-backend inject_user_turn.
# --------------------------------------------------------------------------- #


class _FakeRealtimeItem:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    async def create(self, *, item: dict[str, Any]) -> None:
        self.created.append(item)


class _FakeRealtimeConn:
    def __init__(self) -> None:
        self.conversation = MagicMock()
        self.conversation.item = _FakeRealtimeItem()


@pytest.mark.asyncio
async def test_openai_inject_user_turn_creates_item_and_requests_response() -> None:
    handler = _openai_handler()
    handler.connection = _FakeRealtimeConn()
    handler._connected_event.set()

    await handler.inject_user_turn("good morning")

    created = handler.connection.conversation.item.created
    assert len(created) == 1
    assert created[0]["role"] == "user"
    assert created[0]["content"][0]["text"] == "good morning"
    # Response was enqueued through the one-active-response sender queue.
    assert handler._pending_responses.qsize() == 1
    # UI echo of the user turn.
    echo = handler.output_queue.get_nowait()
    assert isinstance(echo, AdditionalOutputs)


@pytest.mark.asyncio
async def test_openai_inject_user_turn_drops_when_not_connected() -> None:
    handler = _openai_handler()
    # _connected_event unset -> wait times out fast and the turn is dropped.
    await handler.inject_user_turn("dropped", connect_timeout=0.01)
    assert handler._pending_responses.qsize() == 0


class _FakeGeminiSession:
    def __init__(self) -> None:
        self.client_content: list[dict[str, Any]] = []

    async def send_client_content(self, **kwargs: Any) -> None:
        self.client_content.append(kwargs)


@pytest.mark.asyncio
async def test_gemini_inject_user_turn_sends_client_content() -> None:
    handler = _gemini_handler()
    handler.session = _FakeGeminiSession()
    handler._connected_event.set()

    await handler.inject_user_turn("good morning")

    assert len(handler.session.client_content) == 1
    assert handler.session.client_content[0]["turn_complete"] is True
    echo = handler.output_queue.get_nowait()
    assert isinstance(echo, AdditionalOutputs)


@pytest.mark.asyncio
async def test_gemini_inject_user_turn_drops_when_not_connected() -> None:
    handler = _gemini_handler()
    handler.session = _FakeGeminiSession()
    # _connected_event unset -> dropped before any send.
    await handler.inject_user_turn("dropped", connect_timeout=0.01)
    assert handler.session.client_content == []
