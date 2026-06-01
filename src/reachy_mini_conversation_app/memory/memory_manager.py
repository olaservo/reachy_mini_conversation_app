"""Memory manager for Reachy Mini conversation app.

Storage layout::

    $DATA_DIRECTORY/memory/
    ├── active_memory.md              # Rendered index, injected into system prompt
    ├── memories/                     # One atomic memory per file
    │   └── YYYY-MM-DD_<slug>_<hex3>.md
    └── logs/
        ├── pending/                  # Live log + logs waiting to be dreamed
        └── processed/                # Logs already dreamed

See ``docs/memory-rework-dreaming-spec.md``.
"""

from __future__ import annotations
import os
import re
import json
import shutil
import logging
import tempfile
import threading
from typing import Any
from pathlib import Path
from datetime import datetime, timezone

from reachy_mini_conversation_app.memory.frontmatter import (
    dump_frontmatter,
    parse_frontmatter,
)


logger = logging.getLogger(__name__)

ALLOWED_KINDS = {"fact", "preference", "event", "skill", "relationship", "goal", "other"}
# date_<slug>_<3-hex>. Slug may contain a–z, 0–9, hyphens, underscores; must
# start with an alphanumeric character. The 3-hex suffix makes the split
# unambiguous since ``[0-9a-f]{3}`` is always the last token.
_MEMORY_ID_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_[a-z0-9][a-z0-9_-]*_[0-9a-f]{3}$")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically (temp file + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def _one_line_summary(body: str) -> str:
    """Extract a short, single-line summary from a memory body for the index."""
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("---"):
            continue
        return stripped
    return ""


class MemoryManager:
    """Owns on-disk memory storage for the conversation app.

    Responsibilities:
      - Append conversation transcripts to the live session log (``logs/pending/``).
      - Provide CRUD-style primitives the dreamer uses on ``memories/*.md``.
      - Rebuild ``active_memory.md`` from frontmatters.
      - Expose a session-scoped invariant: the dreamer must never read or move
        the file at ``_session_log_path`` while it is being written to.
    """

    def __init__(self, data_dir: Path) -> None:
        """Initialize memory under ``data_dir`` and run fresh-start migration."""
        self._lock = threading.Lock()
        self._data_dir = data_dir
        self._memory_dir = data_dir / "memory"
        self._active_path = self._memory_dir / "active_memory.md"
        self._memories_dir = self._memory_dir / "memories"
        self._logs_dir = self._memory_dir / "logs"
        self._pending_logs_dir = self._logs_dir / "pending"
        self._processed_logs_dir = self._logs_dir / "processed"
        self._session_log_path: Path | None = None
        self._session_log_header: str = ""
        self._ensure_dirs()
        self._migrate_legacy_layout()
        self._ensure_index()
        self._start_session_log()
        logger.info("MemoryManager initialized: data_dir=%s", data_dir)

    # ------------------------------------------------------------------
    # Paths and filesystem bring-up
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        for d in (
            self._memory_dir,
            self._memories_dir,
            self._logs_dir,
            self._pending_logs_dir,
            self._processed_logs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def _migrate_legacy_layout(self) -> None:
        """Idempotent fresh-start migration from the old two-tier layout.

        - Move any ``logs/*.log`` (old top-level logs) into ``logs/pending/``.
        - Remove any legacy ``archive/`` directory.

        Note: this runs on every construction, so it must stay idempotent. It
        deliberately does NOT touch ``active_memory.md`` — that file is the index
        injected into the system prompt and must persist across restarts. (An
        earlier version wiped it here on every init, which meant the summary was
        never present in the prompt; see ``_ensure_index`` for the rebuild path.)
        """
        moved = 0
        for entry in self._logs_dir.iterdir():
            if entry.is_file() and entry.suffix == ".log":
                target = self._pending_logs_dir / entry.name
                if target.exists():
                    logger.warning("Legacy log already in pending/: %s", entry.name)
                    continue
                shutil.move(str(entry), str(target))
                moved += 1
        if moved:
            logger.info("Migrated %d legacy log(s) to logs/pending/", moved)

        archive_dir = self._memory_dir / "archive"
        if archive_dir.exists():
            try:
                shutil.rmtree(archive_dir)
                logger.info("Removed legacy archive/ directory")
            except OSError as e:
                logger.warning("Failed to remove legacy archive/: %s", e)

    def _ensure_index(self) -> None:
        """Guarantee ``active_memory.md`` exists when there are memories to index.

        The index is normally rebuilt at the end of each dream pass, but dreaming
        only runs when there are pending logs. If the index is missing (fresh
        install with copied memories, manual deletion, or a legacy upgrade) yet
        memory files exist, rebuild it now from frontmatter — cheap, no LLM — so
        the very next session still gets its summary injected into the prompt.
        """
        if self._active_path.exists():
            return
        if not any(self._memories_dir.glob("*.md")):
            return
        try:
            # Local import avoids a module-load cycle (index_renderer is leaf-level).
            from reachy_mini_conversation_app.memory.index_renderer import rebuild_index

            rebuild_index(self)
            logger.info("Rebuilt missing active_memory.md from existing memories.")
        except Exception as e:
            logger.warning("Failed to rebuild missing active_memory.md: %s", e)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def new_session(self) -> None:
        """Rotate the live session log file."""
        self._start_session_log()
        logger.info(
            "MemoryManager new session: %s",
            self._session_log_path.name if self._session_log_path else "?",
        )

    def _start_session_log(self) -> None:
        """Reserve a session log path under logs/pending/ without writing it.

        The file is created lazily on the first append (see ``_append_log``).
        Boots that never produce conversation leave nothing behind, so the
        dreamer doesn't waste an LLM call processing an empty stub.
        """
        now = _now_utc()
        base = now.strftime("%Y-%m-%d_%H-%M")
        path = self._pending_logs_dir / f"{base}.log"
        suffix = 2
        while path.exists():
            path = self._pending_logs_dir / f"{base}_{suffix}.log"
            suffix += 1
        self._session_log_path = path
        self._session_log_header = f"--- session {now.strftime('%Y-%m-%d %H:%M')} UTC ---\n\n"

    # ------------------------------------------------------------------
    # Live log append (same as before, just targeted at pending/)
    # ------------------------------------------------------------------

    def _append_log(self, line: str) -> None:
        """Append a plain-text line to the current session log."""
        if self._session_log_path is None:
            return
        try:
            with open(self._session_log_path, "a", encoding="utf-8") as f:
                if f.tell() == 0:
                    f.write(self._session_log_header)
                f.write(line + "\n")
        except OSError as e:
            logger.warning("Failed to write conversation log: %s", e)

    def log_turn(self, role: str, content: str) -> None:
        """Log a user or assistant transcript turn."""
        if not content or not content.strip():
            return
        ts = _now_utc().strftime("%H:%M:%S")
        self._append_log(f"{ts} {role}: {content.strip()}")

    def log_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Log a completed tool call."""
        ts = _now_utc().strftime("%H:%M:%S")
        args_str = json.dumps(args or {}, ensure_ascii=False)
        result_str = json.dumps(result or {}, ensure_ascii=False)
        self._append_log(f"{ts} tool: {tool_name}({args_str}) -> {result_str}")

    # ------------------------------------------------------------------
    # Log inspection (used by dreamer and short_term_memory tool)
    # ------------------------------------------------------------------

    def read_current_session_log(self) -> str:
        """Return the whole current session log, or empty string if none."""
        if self._session_log_path is None or not self._session_log_path.exists():
            return ""
        try:
            return self._session_log_path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("Failed to read current session log: %s", e)
            return ""

    def list_pending_logs(self, exclude_session: bool = True) -> list[str]:
        """Return pending log filenames in chronological order (oldest first).

        If ``exclude_session`` is True (the default), the currently-open
        live log file is skipped — this is the invariant the dreamer relies
        on.
        """
        try:
            names = sorted(p.name for p in self._pending_logs_dir.glob("*.log"))
        except OSError:
            return []
        if exclude_session and self._session_log_path is not None:
            active = self._session_log_path.name
            names = [n for n in names if n != active]
        return names

    def read_pending_log(self, filename: str) -> str:
        """Read a pending log file. Raises FileNotFoundError if missing."""
        path = self._pending_logs_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"pending log not found: {filename}")
        return path.read_text(encoding="utf-8")

    def mark_log_processed(self, filename: str) -> None:
        """Move ``logs/pending/<filename>`` to ``logs/processed/<filename>``.

        Refuses to move the currently-open live session log.
        """
        if self._session_log_path is not None and filename == self._session_log_path.name:
            raise RuntimeError(f"cannot mark active session log as processed: {filename}")
        src = self._pending_logs_dir / filename
        dst = self._processed_logs_dir / filename
        if not src.is_file():
            raise FileNotFoundError(f"pending log not found: {filename}")
        shutil.move(str(src), str(dst))

    # ------------------------------------------------------------------
    # Atomic memory files (CRUD)
    # ------------------------------------------------------------------

    def _memory_path(self, memory_id: str) -> Path:
        if not _MEMORY_ID_PATTERN.match(memory_id):
            raise ValueError(
                f"invalid memory id: {memory_id!r}. Expected format: YYYY-MM-DD_<slug>_<3-hex>, ASCII lowercase."
            )
        return self._memories_dir / f"{memory_id}.md"

    def _load_memory(self, memory_id: str) -> tuple[dict[str, Any], str]:
        path = self._memory_path(memory_id)
        if not path.is_file():
            raise FileNotFoundError(f"memory not found: {memory_id}")
        text = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        if not meta or meta.get("id") != memory_id:
            logger.warning("Memory %s has missing/mismatched id in frontmatter", memory_id)
        return meta, body

    def read_memory(self, memory_id: str) -> dict[str, Any]:
        """Return ``{id, frontmatter, body}`` for an existing memory file."""
        meta, body = self._load_memory(memory_id)
        return {"id": memory_id, "frontmatter": meta, "body": body}

    def memory_exists(self, memory_id: str) -> bool:
        """Return True if a memory file exists on disk."""
        try:
            return self._memory_path(memory_id).is_file()
        except ValueError:
            return False

    def write_memory(
        self,
        memory_id: str,
        body: str,
        *,
        kind: str,
        tags: list[str],
        sources: list[str] | None = None,
        related_to: list[str] | None = None,
        pinned: bool = False,
        supersedes: str | None = None,
        superseded_by: str | None = None,
        created: datetime | None = None,
    ) -> Path:
        """Create a new memory file. Raises ``FileExistsError`` if it exists."""
        path = self._memory_path(memory_id)
        if path.exists():
            raise FileExistsError(f"memory already exists: {memory_id}")
        if kind not in ALLOWED_KINDS:
            raise ValueError(f"invalid kind: {kind!r}. One of {sorted(ALLOWED_KINDS)}")
        meta: dict[str, Any] = {
            "id": memory_id,
            "created": _iso(created or _now_utc()),
            "sources": list(sources or []),
            "kind": kind,
            "tags": list(tags),
            "related_to": list(related_to or []),
            "pinned": bool(pinned),
            "supersedes": supersedes,
            "superseded_by": superseded_by,
        }
        _atomic_write(path, dump_frontmatter(meta, body.strip() + "\n"))
        return path

    def update_memory(
        self,
        memory_id: str,
        *,
        body: str | None = None,
        frontmatter_updates: dict[str, Any] | None = None,
    ) -> Path:
        """Overwrite an existing memory. Merges ``frontmatter_updates`` over the existing frontmatter."""
        meta, existing_body = self._load_memory(memory_id)
        if frontmatter_updates:
            new_kind = frontmatter_updates.get("kind", meta.get("kind"))
            if new_kind is not None and new_kind not in ALLOWED_KINDS:
                raise ValueError(f"invalid kind: {new_kind!r}")
            meta.update(frontmatter_updates)
        meta["id"] = memory_id  # never let callers rename via update
        new_body = (body if body is not None else existing_body).strip() + "\n"
        path = self._memory_path(memory_id)
        _atomic_write(path, dump_frontmatter(meta, new_body))
        return path

    def list_memories(
        self,
        *,
        tag: str | None = None,
        kind: str | None = None,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        """List memory summaries, optionally filtered by tag and/or kind.

        Each entry is ``{id, summary, tags, kind, pinned, created, sources, superseded_by}``.
        Dropped from default output: any memory with a non-null ``superseded_by`` —
        the replacement is what the dreamer (and live LLM) care about.
        """
        out: list[dict[str, Any]] = []
        for path in sorted(self._memories_dir.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
                meta, body = parse_frontmatter(text)
            except (OSError, ValueError) as e:
                logger.warning("Skipping unreadable memory %s: %s", path.name, e)
                continue
            mem_id = meta.get("id") or path.stem
            if not include_superseded and meta.get("superseded_by"):
                continue
            tags_val = meta.get("tags") or []
            if tag is not None and tag not in tags_val:
                continue
            if kind is not None and meta.get("kind") != kind:
                continue
            out.append(
                {
                    "id": mem_id,
                    "summary": _one_line_summary(body),
                    "tags": list(tags_val),
                    "kind": meta.get("kind"),
                    "pinned": bool(meta.get("pinned", False)),
                    "created": meta.get("created"),
                    "sources": list(meta.get("sources") or []),
                    "superseded_by": meta.get("superseded_by"),
                }
            )
        return out

    def find_related_memories(
        self,
        *,
        query: str = "",
        tags: list[str] | None = None,
        limit: int = 10,
        body_preview_chars: int = 0,
    ) -> list[dict[str, Any]]:
        """Rank memories by how many substring matches they get for ``query``/``tags``.

        Scoring is plain case-insensitive substring containment over a
        per-memory haystack built from ``id``, ``tags``, ``kind``, the
        one-line summary, and the full body. No fuzzy matching, no
        embeddings — just ``needle in haystack``.

        When ``body_preview_chars`` > 0, each result carries a
        ``body_preview`` field — the first N characters of the body —
        so the dreamer can decide whether it needs a follow-up
        ``read_memory`` call at all. 0 omits the field, matching the
        original behaviour.

        Returns up to ``limit`` entries, ranked by descending score.
        Same shape as :py:meth:`list_memories` plus a ``score`` field.
        """
        needles: list[str] = []
        if query:
            needles.extend(tok for tok in query.lower().split() if tok)
        if tags:
            needles.extend(t.lower() for t in tags if t)
        if not needles:
            return []

        scored: list[tuple[int, dict[str, Any]]] = []
        for path in sorted(self._memories_dir.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
                meta, body = parse_frontmatter(text)
            except (OSError, ValueError) as e:
                logger.warning("Skipping unreadable memory %s: %s", path.name, e)
                continue
            if meta.get("superseded_by"):
                continue
            mem_id = meta.get("id") or path.stem
            tags_val = meta.get("tags") or []
            summary = _one_line_summary(body)
            haystack = " ".join(
                [
                    mem_id,
                    " ".join(tags_val),
                    str(meta.get("kind") or ""),
                    summary,
                    body,
                ]
            ).lower()
            score = sum(1 for needle in needles if needle in haystack)
            if score == 0:
                continue
            entry: dict[str, Any] = {
                "id": mem_id,
                "summary": summary,
                "tags": list(tags_val),
                "kind": meta.get("kind"),
                "pinned": bool(meta.get("pinned", False)),
                "created": meta.get("created"),
                "score": score,
            }
            if body_preview_chars > 0:
                stripped = body.strip()
                entry["body_preview"] = (
                    stripped if len(stripped) <= body_preview_chars else stripped[:body_preview_chars].rstrip() + "…"
                )
            scored.append((score, entry))
        scored.sort(key=lambda sc: (-sc[0], sc[1]["id"]))
        return [entry for _, entry in scored[: max(1, limit)]]

    # ------------------------------------------------------------------
    # Prompt injection
    # ------------------------------------------------------------------

    def get_memory_block(self) -> str:
        """Return the formatted memory block for system prompt injection."""
        if not self._active_path.exists():
            return ""
        try:
            content = self._active_path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
        if not content:
            return ""
        return (
            "\n\n## MEMORY\n"
            "The index below is automatically curated between sessions. "
            "Use `recall_memory(id)` to read a specific memory (and its related neighbours), "
            "`recall_memories(tag=..., date_from=..., date_to=...)` to filter by topic and/or "
            "conversation date, or `short_term_memory()` to re-read the current session. "
            "Dates always refer to when something was discussed.\n\n" + content
        )

    # ------------------------------------------------------------------
    # Path accessors (used by dreamer and tests)
    # ------------------------------------------------------------------

    @property
    def memories_dir(self) -> Path:
        """Return the on-disk directory containing atomic memory files."""
        return self._memories_dir

    @property
    def active_memory_path(self) -> Path:
        """Return the on-disk path to the rendered index file."""
        return self._active_path

    @property
    def pending_logs_dir(self) -> Path:
        """Return the on-disk directory containing pending (not-yet-dreamed) logs."""
        return self._pending_logs_dir

    @property
    def session_log_path(self) -> Path | None:
        """Return the path of the currently-open live log, or None."""
        return self._session_log_path

    def _atomic_write_active(self, content: str) -> None:
        """Write ``content`` atomically to active_memory.md. Used by index rendering."""
        _atomic_write(self._active_path, content)
