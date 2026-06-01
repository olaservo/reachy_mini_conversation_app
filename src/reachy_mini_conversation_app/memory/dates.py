"""Event-date helpers for memories.

A memory's date is the date of the *conversation(s)* it was drawn from — parsed
from its ``sources`` log filenames (``YYYY-MM-DD_HH-MM.log``) — never the date the
dreamer happened to write the file (``created``). A single memory can span several
conversations on different days, so a memory has potentially many event dates; the
``created`` timestamp is only a last-resort fallback when no source date is parseable.

This is the one place that defines "when a memory happened", used by the index
renderer (grouping/ordering) and by ``recall_memories`` (filtering/sorting).
"""

from __future__ import annotations
import re
from typing import Any
from datetime import datetime, timezone


# Log filenames start with the conversation date, e.g. "2026-04-17_14-37.log".
_LOG_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")


def parse_created(value: Any) -> datetime | None:
    """Parse a frontmatter ``created`` timestamp (fallback only)."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            return datetime.fromisoformat(value.rstrip("Z"))
        except ValueError:
            return None


def parse_log_date(filename: Any) -> datetime | None:
    """Extract the conversation date from a log filename, or None."""
    if not isinstance(filename, str):
        return None
    match = _LOG_DATE_RE.match(filename)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def source_dates(memory: dict[str, Any]) -> list[datetime]:
    """All parseable conversation dates for a memory, from its ``sources``."""
    out: list[datetime] = []
    for src in memory.get("sources") or []:
        parsed = parse_log_date(src)
        if parsed is not None:
            out.append(parsed)
    return out


def event_date(memory: dict[str, Any]) -> datetime | None:
    """Return the representative event date (the most recent conversation date).

    Falls back to ``created`` only when no source date is parseable. Used for
    ordering and for the Recent/Older split in the index.
    """
    dates = source_dates(memory)
    if dates:
        return max(dates)
    return parse_created(memory.get("created"))


def event_dates(memory: dict[str, Any]) -> list[str]:
    """Sorted unique conversation dates (``YYYY-MM-DD``) shown to the model."""
    return sorted({d.strftime("%Y-%m-%d") for d in source_dates(memory)})


def present_memory(mem: dict[str, Any]) -> dict[str, Any]:
    """Model-facing view of a ``read_memory`` result (``{id, frontmatter, body}``).

    Drops the dreamer's ``created`` timestamp — which the model could mistake for
    when the conversation happened — and adds ``dates_discussed`` (the actual
    conversation dates), so the only dates the model ever sees are the real ones.
    """
    frontmatter = dict(mem.get("frontmatter") or {})
    discussed = event_dates({"sources": frontmatter.get("sources")})
    frontmatter.pop("created", None)
    out = dict(mem)
    out["frontmatter"] = frontmatter
    out["dates_discussed"] = discussed
    return out
