"""Render the Core / Recent / Older index from memory frontmatters.

Input: a list of memory summaries (as produced by ``MemoryManager.list_memories``).
Output: a markdown string written to ``active_memory.md``.

See §4 of ``docs/memory-rework-dreaming-spec.md``.
"""

from __future__ import annotations
from typing import Any
from datetime import datetime, timezone, timedelta


RECENT_WINDOW = timedelta(days=30)
OLDER_TAG_LIMIT = 15


def _parse_created(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        # Canonical form from dump: YYYY-MM-DDTHH:MM:SSZ
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            return datetime.fromisoformat(value.rstrip("Z"))
        except ValueError:
            return None


def _fmt_entry(mem: dict[str, Any]) -> str:
    summary = mem.get("summary") or "(empty memory)"
    return f"- [{mem['id']}] {summary}"


def _group_by_primary_tag(memories: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group memories by first tag; memories without tags go under 'untagged'."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for mem in memories:
        tags = mem.get("tags") or []
        primary = tags[0] if tags else "untagged"
        groups.setdefault(primary, []).append(mem)
    return groups


def _tag_counts(memories: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """Return ranked (tag, count) pairs across the given memories."""
    counts: dict[str, int] = {}
    for mem in memories:
        for tag in mem.get("tags") or []:
            counts[tag] = counts.get(tag, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def render_index(memories: list[dict[str, Any]], now: datetime | None = None) -> str:
    """Render the memory index as markdown.

    Memories are split into three tiers:
      - Core: ``pinned: true``, regardless of age.
      - Recent: non-pinned, ``created`` within the last 30 days, grouped by primary tag.
      - Older: everything else, summarised as ranked tag counts.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - RECENT_WINDOW

    visible = [m for m in memories if not m.get("superseded_by")]

    core = [m for m in visible if m.get("pinned")]
    non_core = [m for m in visible if not m.get("pinned")]

    recent: list[dict[str, Any]] = []
    older: list[dict[str, Any]] = []
    for mem in non_core:
        created = _parse_created(mem.get("created"))
        if created is None or created >= cutoff:
            recent.append(mem)
        else:
            older.append(mem)

    lines: list[str] = ["# Memory index", ""]

    lines.append("## Core (pinned)")
    if core:
        for mem in sorted(core, key=lambda m: m["id"]):
            lines.append(_fmt_entry(mem))
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("## Recent (last 30 days)")
    if recent:
        groups = _group_by_primary_tag(recent)
        for tag in sorted(groups):
            lines.append(f"### {tag.capitalize() if tag != 'untagged' else 'Untagged'}")
            for mem in sorted(groups[tag], key=lambda m: m.get("created") or m["id"]):
                lines.append(_fmt_entry(mem))
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("## Older")
    if older:
        counts = _tag_counts(older)
        lines.append("Tags (count), ranked by frequency:")
        truncated: list[tuple[str, int]] = []
        remaining = 0
        if len(counts) > OLDER_TAG_LIMIT:
            truncated = counts[:OLDER_TAG_LIMIT]
            remaining = len(counts) - OLDER_TAG_LIMIT
        else:
            truncated = counts
        for tag, count in truncated:
            lines.append(f"- {tag} ({count})")
        if remaining:
            lines.append(f"- … +{remaining} more tags")
        lines.append("")
        lines.append("Use `recall_topic(tag)` to load.")
    else:
        lines.append("(none)")

    return "\n".join(lines).rstrip() + "\n"


def rebuild_index(manager: Any) -> str:
    """Rebuild ``active_memory.md`` from all on-disk memories.

    ``manager`` is a :class:`MemoryManager`. This function is the public
    entry point used by both the dreamer and test code.
    """
    memories = manager.list_memories(include_superseded=False)
    rendered = render_index(memories)
    manager._atomic_write_active(rendered)
    return rendered
