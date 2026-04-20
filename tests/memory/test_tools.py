"""Tests for the live-conversation memory tools."""

from pathlib import Path
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.memory.memory_manager import MemoryManager
from reachy_mini_conversation_app.memory.index_renderer import rebuild_index
from reachy_mini_conversation_app.tools.recall_memory import RecallMemory
from reachy_mini_conversation_app.tools.recall_topic import RecallTopic
from reachy_mini_conversation_app.tools.short_term_memory import ShortTermMemory


@dataclass
class _FakeDeps:
    memory_manager: MemoryManager | None


@pytest.fixture
def deps(tmp_path: Path) -> _FakeDeps:
    """Build a ToolDependencies-like object backed by a real MemoryManager."""
    return _FakeDeps(memory_manager=MemoryManager(tmp_path / "data"))


def _mid(slug: str, hex3: str = "abc", date: str = "2026-04-17") -> str:
    return f"{date}_{slug}_{hex3}"


# ------------------------------------------------------------------
# recall_memory
# ------------------------------------------------------------------


class TestRecallMemory:
    """Verify recall_memory returns bundled target + related memories."""

    @pytest.mark.asyncio
    async def test_returns_memory(self, deps: _FakeDeps) -> None:
        """Happy path: existing memory with no related_to."""
        assert deps.memory_manager is not None
        mid = _mid("chess")
        deps.memory_manager.write_memory(
            mid,
            "Loves Queen's Gambit.",
            kind="preference",
            tags=["chess"],
        )
        result = await RecallMemory()(deps, id=mid)
        assert result["memory"]["id"] == mid
        assert "Queen" in result["memory"]["body"]
        assert result["related"] == []

    @pytest.mark.asyncio
    async def test_bundles_related(self, deps: _FakeDeps) -> None:
        """Every memory referenced in related_to must be returned."""
        assert deps.memory_manager is not None
        main = _mid("chess-openings", "111")
        neighbour = _mid("chess-match", "222")
        deps.memory_manager.write_memory(
            neighbour, "Lost to Jean.", kind="event", tags=["chess"]
        )
        deps.memory_manager.write_memory(
            main,
            "Prefers Queen's Gambit.",
            kind="preference",
            tags=["chess", "openings"],
            related_to=[neighbour],
        )
        result = await RecallMemory()(deps, id=main)
        assert [m["id"] for m in result["related"]] == [neighbour]

    @pytest.mark.asyncio
    async def test_missing_returns_error(self, deps: _FakeDeps) -> None:
        """Unknown ID returns an error plus a sample of known IDs."""
        assert deps.memory_manager is not None
        deps.memory_manager.write_memory(
            _mid("known"), "body", kind="fact", tags=["foo"]
        )
        result = await RecallMemory()(deps, id=_mid("unknown", "fff"))
        assert "error" in result
        assert _mid("known") in result["known_ids_sample"]

    @pytest.mark.asyncio
    async def test_disabled(self) -> None:
        """Missing memory_manager is reported as memory_disabled."""
        deps = _FakeDeps(memory_manager=None)
        result = await RecallMemory()(deps, id="anything")
        assert result == {"status": "memory_disabled"}


# ------------------------------------------------------------------
# recall_topic
# ------------------------------------------------------------------


class TestRecallTopic:
    """Verify recall_topic's tag filtering and limit behaviour."""

    @pytest.mark.asyncio
    async def test_filters_and_limits(self, deps: _FakeDeps) -> None:
        """Returns only memories matching `tag`, bounded by `limit`."""
        assert deps.memory_manager is not None
        for idx in range(3):
            deps.memory_manager.write_memory(
                _mid(f"chess{idx}", hex3=f"a{idx:02d}"),
                f"chess memory {idx}",
                kind="preference",
                tags=["chess"],
                created=None,
            )
        deps.memory_manager.write_memory(
            _mid("cooking"), "cooking", kind="preference", tags=["cooking"]
        )
        result = await RecallTopic()(deps, tag="chess", limit=2)
        assert result["returned"] == 2
        assert result["total_matches"] == 3
        for entry in result["memories"]:
            assert "chess" in entry["frontmatter"]["tags"]

    @pytest.mark.asyncio
    async def test_empty_tag_returns_nothing_matching(self, deps: _FakeDeps) -> None:
        """Tag with no matches returns an empty bundle (not an error)."""
        result = await RecallTopic()(deps, tag="missing")
        assert result["returned"] == 0
        assert result["memories"] == []

    @pytest.mark.asyncio
    async def test_missing_tag_is_error(self, deps: _FakeDeps) -> None:
        """Missing or blank tag returns an error."""
        result = await RecallTopic()(deps, tag="")
        assert "error" in result


# ------------------------------------------------------------------
# short_term_memory
# ------------------------------------------------------------------


class TestShortTermMemory:
    """Verify short_term_memory reads the current log verbatim."""

    @pytest.mark.asyncio
    async def test_returns_session_content(self, deps: _FakeDeps) -> None:
        """Turns logged during the session show up in the returned content."""
        assert deps.memory_manager is not None
        deps.memory_manager.log_turn("user", "Hello!")
        deps.memory_manager.log_turn("assistant", "Hi Rémi.")
        result = await ShortTermMemory()(deps)
        assert "Hello!" in result["content"]
        assert "Hi Rémi." in result["content"]
        assert result["length_chars"] == len(result["content"])

    @pytest.mark.asyncio
    async def test_handles_empty_session(self, deps: _FakeDeps) -> None:
        """Returns empty content when no turns have been logged.

        The session log is created lazily on the first write, so a session
        with no conversation produces no file and the read returns "".
        """
        assert deps.memory_manager is not None
        result = await ShortTermMemory()(deps)
        assert result["content"] == ""
