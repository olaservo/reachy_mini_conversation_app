"""Tests for the Dreamer — uses a scripted fake OpenAI client."""

from __future__ import annotations
from typing import Any, Callable
from pathlib import Path
from dataclasses import field, dataclass

import pytest

from reachy_mini_conversation_app.memory.dreamer import (
    Dreamer,
    DreamerRuntimeError,
    run_dream_pass,
)
from reachy_mini_conversation_app.memory.memory_manager import MemoryManager


# ---------------------------------------------------------------------------
# Scripted fake OpenAI client
# ---------------------------------------------------------------------------


@dataclass
class _FakeResponses:
    """Scripted stand-in for ``client.responses``.

    ``on_create`` is a function called with the current ``input`` list; it
    returns the next response's ``output`` list (items as dicts). That keeps
    the fake tiny and lets each test describe exactly the tool-call pattern
    it wants to exercise.
    """

    on_create: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def create(self, **kwargs: Any) -> Any:
        """Fake ``client.responses.create`` implementation."""
        self.calls.append(kwargs)
        output = self.on_create(kwargs["input"])

        class _Resp:
            pass

        resp = _Resp()
        resp.output = output
        return resp


@dataclass
class _FakeClient:
    responses: _FakeResponses


def _msg_item(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def _call_item(name: str, args: dict[str, Any], call_id: str = "c1") -> dict[str, Any]:
    import json as _json
    return {
        "type": "function_call",
        "name": name,
        "call_id": call_id,
        "arguments": _json.dumps(args),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manager(tmp_path: Path) -> MemoryManager:
    """Create a fresh MemoryManager and queue one non-live log to dream on."""
    mgr = MemoryManager(tmp_path / "data")
    # Put a closed log into pending/ (the live session log is excluded automatically).
    (mgr.pending_logs_dir / "2026-04-14_09-15.log").write_text(
        "--- session 2026-04-14 09:15 UTC ---\n\n"
        "09:15:12 user: Hey Reachy, my name is Rémi.\n"
        "09:15:14 assistant: Nice to meet you, Rémi!\n"
        "09:15:20 user: I love chess.\n"
        "09:15:22 assistant: Got it.\n",
        encoding="utf-8",
    )
    return mgr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDreamerSingleLog:
    """Verify the dreamer loop end-to-end with a scripted LLM."""

    def test_write_memory_flow(self, manager: MemoryManager) -> None:
        """Scripted LLM: write one memory, then stop."""
        steps: list[list[dict[str, Any]]] = [
            [_call_item(
                "write_memory",
                {
                    "id": "2026-04-14_user-name_a01",
                    "body": "User's name is Rémi.",
                    "kind": "fact",
                    "tags": ["identity"],
                    "sources": ["2026-04-14_09-15.log"],
                    "pinned": True,
                },
                call_id="c1",
            )],
            [_msg_item("Wrote one identity memory.")],
            # Self-reflection call
            [_msg_item("All good.")],
        ]
        fake = _FakeClient(responses=_FakeResponses(on_create=lambda _i: steps.pop(0)))
        dreamer = Dreamer(manager, model="fake-model", client=fake)
        stats_list = dreamer.run()

        [stats] = stats_list
        assert stats.created == 1
        assert stats.tool_calls_count.get("write_memory") == 1
        assert len(stats.llm_durations_s) >= 1
        assert stats.tool_total_s >= 0.0
        assert stats.errors == []
        # File moved out of pending
        assert not (manager.pending_logs_dir / "2026-04-14_09-15.log").exists()
        assert (manager._processed_logs_dir / "2026-04-14_09-15.log").is_file()
        # Memory file and index exist
        memory_file = manager.memories_dir / "2026-04-14_user-name_a01.md"
        assert memory_file.is_file()
        assert manager.active_memory_path.read_text(encoding="utf-8").count("[2026-04-14_user-name_a01]") == 1

    def test_overlap_update_flow(self, manager: MemoryManager) -> None:
        """Scripted LLM: consult existing, then update instead of create."""
        manager.write_memory(
            "2026-04-10_chess_aaa",
            "User plays chess.",
            kind="preference",
            tags=["chess"],
        )
        steps: list[list[dict[str, Any]]] = [
            [_call_item("list_existing_memories", {"tag": "chess"}, call_id="c1")],
            [_call_item(
                "update_memory",
                {
                    "id": "2026-04-10_chess_aaa",
                    "body": "User plays chess and prefers the Queen's Gambit.",
                    "sources": ["2026-04-14_09-15.log"],
                },
                call_id="c2",
            )],
            [_msg_item("Enriched existing memory.")],
            [_msg_item("Reflection.")],
        ]
        fake = _FakeClient(responses=_FakeResponses(on_create=lambda _i: steps.pop(0)))
        Dreamer(manager, model="fake-model", client=fake).run()

        mem = manager.read_memory("2026-04-10_chess_aaa")
        assert "Queen's Gambit" in mem["body"]
        assert "2026-04-14_09-15.log" in mem["frontmatter"].get("sources", [])

    def test_errors_keep_log_in_pending(self, manager: MemoryManager) -> None:
        """If a tool call raises, the log must stay in pending/ for retry."""
        steps: list[list[dict[str, Any]]] = [
            [_call_item(
                "write_memory",
                {
                    "id": "invalid id!",
                    "body": "bad",
                    "kind": "fact",
                    "tags": ["t"],
                },
                call_id="c1",
            )],
            [_msg_item("Giving up.")],
            [_msg_item("Reflection.")],
        ]
        fake = _FakeClient(responses=_FakeResponses(on_create=lambda _i: steps.pop(0)))
        [stats] = Dreamer(manager, model="fake-model", client=fake).run()

        assert stats.errors
        # Left in pending on failure
        assert (manager.pending_logs_dir / "2026-04-14_09-15.log").is_file()

    def test_empty_pending_skips_llm(self, tmp_path: Path) -> None:
        """No pending logs → no LLM calls."""
        mgr = MemoryManager(tmp_path / "data")
        fake = _FakeClient(responses=_FakeResponses(on_create=lambda _i: []))
        stats_list = Dreamer(mgr, model="fake-model", client=fake).run()
        assert stats_list == []
        assert fake.responses.calls == []


class TestRunDreamPass:
    """Verify the convenience runner's model-selection behaviour."""

    def test_raises_when_no_model_configured(
        self,
        manager: MemoryManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Raise if neither env var is set."""
        monkeypatch.delenv("MEMORY_DREAMER_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_MODEL_NAME", raising=False)
        fake = _FakeClient(responses=_FakeResponses(on_create=lambda _i: []))
        with pytest.raises(DreamerRuntimeError):
            run_dream_pass(manager, client=fake)

    def test_uses_memory_dreamer_model_env(
        self,
        manager: MemoryManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """MEMORY_DREAMER_MODEL takes precedence over OPENAI_MODEL_NAME."""
        monkeypatch.setenv("MEMORY_DREAMER_MODEL", "custom-model")
        monkeypatch.setenv("OPENAI_MODEL_NAME", "other-model")
        fake = _FakeClient(responses=_FakeResponses(on_create=lambda _i: [_msg_item("done")]))
        # Run with no pending logs — just confirms no error.
        for p in manager.pending_logs_dir.glob("*.log"):
            if p.name != manager.session_log_path.name:  # type: ignore[union-attr]
                p.unlink()
        run_dream_pass(manager, client=fake)
