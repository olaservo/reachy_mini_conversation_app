"""Dreaming agent: offline memory consolidation.

The dreamer runs as a blocking startup phase. It walks through every log in
``logs/pending/``, calls an LLM with dedicated tools, and lets the LLM
create/update/merge atomic memory files. After every log it rebuilds the
index. At the end of the run it asks the LLM to reflect on its own work.

The conversation LLM never sees any of this — it simply inherits the
curated memory state once the conversation app boots.

Every tool call, every LLM input/output, and every per-log statistic is
printed to the terminal logger. See §7 of the spec for the expected
behaviour.
"""

from __future__ import annotations
import os
import json
import time
import logging
from typing import Any, Callable
from dataclasses import field, dataclass

from openai import OpenAI

from reachy_mini_conversation_app.memory.memory_manager import (
    ALLOWED_KINDS,
    MemoryManager,
)
from reachy_mini_conversation_app.memory.index_renderer import rebuild_index


logger = logging.getLogger(__name__)


DREAMER_SYSTEM_PROMPT = """\
You are the Dreamer — a background memory-consolidation agent running between
live conversations with a human.

Your job is to turn raw conversation logs into atomic, well-tagged memory
files that a live conversational robot (separate from you) can retrieve
next time it talks with this user. You never talk to the user directly.

## Memory file model

Each memory file has the following frontmatter:
- id: YYYY-MM-DD_<slug>_<3-hex>  (date is the creation date)
  The slug is lowercase ASCII using letters, digits, hyphens or underscores
  and MUST start with a letter or digit. The 3-hex suffix is exactly three
  characters from [0-9a-f]. Good: 2026-04-20_chess-openings_a3f,
  2026-04-20_user_name_01d. Bad: 2026-04-20_chess_xyz (xyz not hex),
  2026-04-20_-chess_a3f (slug starts with '-').
- created: ISO8601 UTC
- sources: [log filenames the memory is drawn from]
- kind: one of fact | preference | event | skill | relationship | goal | other
- tags: lowercase ASCII tokens; first tag is the primary topic (drives index grouping)
- related_to: memory IDs that MUST be read alongside this one; keep sparse
- pinned: true only for identity/core facts (name, language, key relationships)
- supersedes / superseded_by: explicit replacement links

## Rules (follow all five)

1. **Atomicity** — One memory = one `kind` + one primary topic. If your draft
   covers two, split it into two memories.
2. **Overlap-first** — Before creating a new memory, call
   `list_existing_memories(tag=X)` for each tag you're considering. Prefer
   `update_memory` over `write_memory` when an existing memory covers the
   same topic.
3. **Evidence** — You choose how to represent each fact. Direct quotes from
   the log are self-justifying. Paraphrase and compression are fine *provided
   you state (in your reasoning messages) which log lines back them*. Never
   synthesise across memories — use `related_to` instead.
4. **Conflict** — If new info contradicts an existing memory, create a new
   one and set `supersedes=<old_id>`, then update the old with
   `superseded_by=<new_id>`. Never silently overwrite.
5. **Pin** — Set `pinned: true` ONLY for identity/core facts. When in doubt,
   don't pin.

## Workflow for each log

1. Read the log with `read_log(filename)`.
2. Check for overlap with `list_existing_memories(tag=...)` for any tag you
   plan to use.
3. For each distinct (kind, primary topic) you extract, either
   `write_memory(...)` or `update_memory(...)`.
4. When you're finished with this log, respond with a plain-text summary of
   what you did. Do NOT call `mark_log_processed` — the runner marks the log
   processed automatically once you stop making tool calls. Do NOT call
   `rebuild_index` — the runner rebuilds it at the end of the run.

Be terse in your chat messages. The audit trail is in the tool calls and
the log lines, not your prose.
"""


DREAMER_SELF_REFLECTION_PROMPT = """\
The dream pass just ended. You processed {n_logs} log(s) in {total_seconds:.1f}s.
Per-log statistics:

{stats_block}

Please reflect honestly on this run (short, specific, concrete):

1. Were the available tools sufficient? Any task you wanted to do but couldn't?
2. Did the five rules (atomicity, overlap-first, evidence, conflict, pin) fit
   the material? Any rule that was ambiguous or missing?
3. Any tool call you repeated unnecessarily — a sign a helper tool is missing?
4. One concrete improvement you'd suggest for the next run.

Your reply is printed to the terminal logger for Rémi to read between
sessions. It is NOT stored in memory and NOT acted on automatically.
"""


# ---------------------------------------------------------------------------
# Tool spec + dispatcher
# ---------------------------------------------------------------------------


DREAMER_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "read_log",
        "description": "Read the full text of a pending conversation log.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename from the pending list, e.g. 2026-04-15_21-04.log",
                },
            },
            "required": ["filename"],
        },
    },
    {
        "type": "function",
        "name": "list_existing_memories",
        "description": (
            "List existing memory summaries on disk, optionally filtered by tag and/or kind. "
            "Returns {id, summary, tags, kind, pinned, created}. Always call this BEFORE "
            "creating a new memory so you don't duplicate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tag": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": sorted(ALLOWED_KINDS),
                },
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "read_memory",
        "description": "Read a memory's full body + frontmatter by ID.",
        "parameters": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    },
    {
        "type": "function",
        "name": "write_memory",
        "description": (
            "Create a new memory file. Fails if the ID already exists — call update_memory "
            "in that case. The ID must be lowercase ASCII and shaped as "
            "YYYY-MM-DD_<slug>_<3-hex>."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "body": {"type": "string"},
                "kind": {"type": "string", "enum": sorted(ALLOWED_KINDS)},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Lowercase ASCII tags; first entry is the primary topic.",
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Log filenames this memory draws from.",
                },
                "related_to": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Memory IDs that MUST be read alongside this one. Keep sparse.",
                },
                "pinned": {"type": "boolean"},
                "supersedes": {
                    "type": ["string", "null"],
                    "description": "ID of the memory this one replaces, if any.",
                },
            },
            "required": ["id", "body", "kind", "tags"],
        },
    },
    {
        "type": "function",
        "name": "update_memory",
        "description": (
            "Update an existing memory. Only include fields you want to change. "
            "To mark a memory as replaced, set superseded_by."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "body": {"type": "string"},
                "kind": {"type": "string", "enum": sorted(ALLOWED_KINDS)},
                "tags": {"type": "array", "items": {"type": "string"}},
                "sources": {"type": "array", "items": {"type": "string"}},
                "related_to": {"type": "array", "items": {"type": "string"}},
                "pinned": {"type": "boolean"},
                "supersedes": {"type": ["string", "null"]},
                "superseded_by": {"type": ["string", "null"]},
            },
            "required": ["id"],
        },
    },
]


@dataclass
class DreamLogStats:
    """Per-log runtime statistics printed after every log."""

    filename: str
    duration_s: float = 0.0
    tool_calls: dict[str, int] = field(default_factory=dict)
    created: int = 0
    updated: int = 0
    errors: list[str] = field(default_factory=list)

    def inc_tool(self, name: str) -> None:
        """Increment the call count for a given dreamer tool."""
        self.tool_calls[name] = self.tool_calls.get(name, 0) + 1

    def one_line(self) -> str:
        """Render a single-line summary for the terminal logger."""
        total = sum(self.tool_calls.values())
        parts = [f"{k}×{v}" for k, v in sorted(self.tool_calls.items())]
        tools_str = ", ".join(parts) if parts else "(no tool calls)"
        outcome = f"created {self.created}, updated {self.updated}"
        if self.errors:
            outcome += f", errors {len(self.errors)}"
        return (
            f"[DREAM] {self.filename} — {self.duration_s:.1f}s, "
            f"{total} tool calls ({tools_str}), {outcome}"
        )


class DreamerRuntimeError(RuntimeError):
    """Raised when the dreamer encounters an unrecoverable condition."""


class Dreamer:
    """LLM-driven memory consolidation runner.

    Usage::

        dreamer = Dreamer(manager, model="gpt-5.4", api_key=OPENAI_API_KEY)
        dreamer.run()

    The runner is sync because it is a blocking boot phase that must finish
    before the conversation app accepts connections. Internally it uses
    OpenAI's sync ``responses`` API, matching the s2s pipeline pattern.
    """

    def __init__(
        self,
        manager: MemoryManager,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        client: OpenAI | None = None,
        max_tool_calls_per_log: int = 40,
    ) -> None:
        """Initialize the dreamer. Pass ``client`` in tests to bypass OpenAI."""
        self.manager = manager
        self.model = model
        self.max_tool_calls_per_log = max_tool_calls_per_log
        self.client = client if client is not None else OpenAI(api_key=api_key, base_url=base_url)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> list[DreamLogStats]:
        """Run a full dream pass and return the per-log stats list."""
        pending = self.manager.list_pending_logs(exclude_session=True)
        if not pending:
            logger.info("[DREAM] No pending logs; skipping dream pass.")
            return []

        logger.info("[DREAM] Starting dream pass on %d log(s): %s", len(pending), pending)
        t_run0 = time.monotonic()
        stats_list: list[DreamLogStats] = []
        for filename in pending:
            stats = self._process_one_log(filename)
            stats_list.append(stats)
            logger.info(stats.one_line())
            if not stats.errors:
                try:
                    self.manager.mark_log_processed(filename)
                    logger.info("[DREAM] Marked %s as processed.", filename)
                except Exception as e:
                    logger.exception("[DREAM] Failed to mark %s processed: %s", filename, e)
                    stats.errors.append(f"mark_log_processed: {e}")
            else:
                logger.warning(
                    "[DREAM] %s left in pending/ due to errors: %s",
                    filename,
                    stats.errors,
                )

        rendered = rebuild_index(self.manager)
        logger.info("[DREAM] Rebuilt active_memory.md (%d chars).", len(rendered))

        total = time.monotonic() - t_run0
        self._self_reflection(stats_list, total_seconds=total)
        logger.info("[DREAM] Dream pass finished in %.1fs.", total)
        return stats_list

    # ------------------------------------------------------------------
    # Per-log loop
    # ------------------------------------------------------------------

    def _process_one_log(self, filename: str) -> DreamLogStats:
        stats = DreamLogStats(filename=filename)
        t0 = time.monotonic()

        try:
            log_content = self.manager.read_pending_log(filename)
        except OSError as e:
            stats.errors.append(f"read_pending_log: {e}")
            stats.duration_s = time.monotonic() - t0
            return stats

        existing = self.manager.list_memories(include_superseded=False)
        summaries = "\n".join(
            f"- [{m['id']}] ({m['kind']}, tags={m['tags']}) {m['summary']}"
            for m in existing
        ) or "(none yet)"

        try:
            index_text = self.manager.active_memory_path.read_text(encoding="utf-8")
        except OSError:
            index_text = "(index not yet built)"

        user_message = (
            f"## Pending log: {filename}\n\n"
            f"--- current index ---\n{index_text}\n\n"
            f"--- existing memory summaries ({len(existing)}) ---\n{summaries}\n\n"
            f"--- log contents ---\n{log_content}\n\n"
            f"Process this log. Call tools as needed. When you are done, respond with a one-paragraph plain-text summary."
        )

        logger.info("[DREAM] === Processing %s ===", filename)
        logger.debug("[DREAM] Prompt payload for %s (%d chars)", filename, len(user_message))

        input_items: list[dict[str, Any]] = [
            {
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": DREAMER_SYSTEM_PROMPT}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": user_message}],
            },
        ]

        for iteration in range(self.max_tool_calls_per_log + 1):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=input_items,
                    tools=DREAMER_TOOL_SPECS,  # type: ignore[arg-type]
                )
            except Exception as e:
                logger.exception("[DREAM] LLM call failed on %s: %s", filename, e)
                stats.errors.append(f"llm_call: {e}")
                break

            did_call_tool = False
            for item in response.output:
                item_dict = self._item_as_dict(item)
                item_type = item_dict.get("type")
                if item_type == "function_call":
                    did_call_tool = True
                    input_items.append(item_dict)
                    tool_result, ok = self._dispatch_tool(item_dict, stats)
                    input_items.append({
                        "type": "function_call_output",
                        "call_id": item_dict["call_id"],
                        "output": json.dumps(tool_result, ensure_ascii=False),
                    })
                    if not ok:
                        stats.errors.append(
                            f"tool {item_dict.get('name')}: {tool_result.get('error')}"
                        )
                elif item_type == "message":
                    text_chunks = []
                    for chunk in item_dict.get("content", []) or []:
                        if chunk.get("type") == "output_text":
                            text_chunks.append(chunk.get("text", ""))
                    final_text = "".join(text_chunks).strip()
                    if final_text:
                        logger.info("[DREAM] Dreamer on %s said: %s", filename, final_text)
                    input_items.append(item_dict)
                else:
                    logger.debug("[DREAM] Ignoring output item type=%s", item_type)

            if not did_call_tool:
                break
        else:
            stats.errors.append(
                f"max_tool_calls_per_log ({self.max_tool_calls_per_log}) exceeded"
            )
            logger.error("[DREAM] %s: tool-call budget exceeded", filename)

        stats.duration_s = time.monotonic() - t0
        return stats

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch_tool(
        self,
        call: dict[str, Any],
        stats: DreamLogStats,
    ) -> tuple[dict[str, Any], bool]:
        name = call.get("name", "")
        raw_args = call.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except Exception as e:
            return {"error": f"invalid JSON arguments: {e}"}, False

        stats.inc_tool(name)
        logger.info("[DREAM] tool call: %s(%s)", name, json.dumps(args, ensure_ascii=False))
        handler: Callable[[dict[str, Any], DreamLogStats], dict[str, Any]] | None = getattr(
            self, f"_tool_{name}", None
        )
        if handler is None:
            return {"error": f"unknown tool: {name}"}, False
        try:
            result = handler(args, stats)
        except Exception as e:
            logger.exception("[DREAM] tool %s raised: %s", name, e)
            return {"error": f"{type(e).__name__}: {e}"}, False
        logger.debug("[DREAM] tool result: %s", result)
        return result, "error" not in result

    # Individual handlers ------------------------------------------------

    def _tool_read_log(self, args: dict[str, Any], _: DreamLogStats) -> dict[str, Any]:
        filename = args.get("filename") or ""
        content = self.manager.read_pending_log(filename)
        return {"filename": filename, "content": content}

    def _tool_list_existing_memories(
        self, args: dict[str, Any], _: DreamLogStats
    ) -> dict[str, Any]:
        tag = args.get("tag") or None
        kind = args.get("kind") or None
        items = self.manager.list_memories(tag=tag, kind=kind)
        return {"count": len(items), "memories": items}

    def _tool_read_memory(self, args: dict[str, Any], _: DreamLogStats) -> dict[str, Any]:
        return self.manager.read_memory(args.get("id", ""))

    def _tool_write_memory(
        self, args: dict[str, Any], stats: DreamLogStats
    ) -> dict[str, Any]:
        memory_id = args.get("id") or ""
        body = args.get("body") or ""
        kind = args.get("kind") or ""
        tags = args.get("tags") or []
        sources = args.get("sources") or []
        related_to = args.get("related_to") or []
        pinned = bool(args.get("pinned", False))
        supersedes = args.get("supersedes")
        self.manager.write_memory(
            memory_id,
            body,
            kind=kind,
            tags=tags,
            sources=sources,
            related_to=related_to,
            pinned=pinned,
            supersedes=supersedes,
        )
        if supersedes:
            try:
                self.manager.update_memory(
                    supersedes,
                    frontmatter_updates={"superseded_by": memory_id},
                )
            except FileNotFoundError:
                logger.warning("[DREAM] supersedes target %s not found", supersedes)
        stats.created += 1
        return {"status": "created", "id": memory_id}

    def _tool_update_memory(
        self, args: dict[str, Any], stats: DreamLogStats
    ) -> dict[str, Any]:
        memory_id = args.get("id") or ""
        body = args.get("body")
        frontmatter_updates: dict[str, Any] = {}
        for key in ("kind", "tags", "sources", "related_to", "pinned", "supersedes", "superseded_by"):
            if key in args:
                frontmatter_updates[key] = args[key]
        self.manager.update_memory(
            memory_id,
            body=body,
            frontmatter_updates=frontmatter_updates or None,
        )
        stats.updated += 1
        return {"status": "updated", "id": memory_id}

    # ------------------------------------------------------------------
    # Self-reflection
    # ------------------------------------------------------------------

    def _self_reflection(self, stats_list: list[DreamLogStats], total_seconds: float) -> None:
        if not stats_list:
            return
        stats_block = "\n".join(s.one_line() for s in stats_list)
        prompt = DREAMER_SELF_REFLECTION_PROMPT.format(
            n_logs=len(stats_list),
            total_seconds=total_seconds,
            stats_block=stats_block,
        )
        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "type": "message",
                        "role": "system",
                        "content": [{"type": "input_text", "text": DREAMER_SYSTEM_PROMPT}],
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": prompt}],
                    },
                ],
            )
        except Exception as e:
            logger.warning("[DREAM] Self-reflection LLM call failed: %s", e)
            return
        reflection = self._extract_message_text(response)
        logger.info("[DREAM] --- Self-reflection ---\n%s", reflection or "(empty)")
        logger.info("[DREAM] --- End self-reflection ---")

    # ------------------------------------------------------------------
    # Response parsing helpers (tolerant of dicts and pydantic objects)
    # ------------------------------------------------------------------

    @staticmethod
    def _item_as_dict(item: Any) -> dict[str, Any]:
        if isinstance(item, dict):
            return dict(item)
        if hasattr(item, "model_dump"):
            return item.model_dump()
        if hasattr(item, "to_dict"):
            return item.to_dict()  # type: ignore[no-any-return]
        return dict(item)

    @classmethod
    def _extract_message_text(cls, response: Any) -> str:
        items = getattr(response, "output", None) or []
        chunks: list[str] = []
        for item in items:
            item_dict = cls._item_as_dict(item)
            if item_dict.get("type") != "message":
                continue
            for part in item_dict.get("content", []) or []:
                if part.get("type") == "output_text":
                    chunks.append(part.get("text", ""))
        return "".join(chunks).strip()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


def run_dream_pass(
    manager: MemoryManager,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    client: OpenAI | None = None,
) -> list[DreamLogStats]:
    """Run one dream pass and return the per-log stats list.

    If ``model`` is None, the environment variable ``MEMORY_DREAMER_MODEL``
    wins; otherwise it falls back to ``OPENAI_MODEL_NAME``. That matches the
    §9 spec: "default = live LLM model".
    """
    resolved_model = model or os.getenv("MEMORY_DREAMER_MODEL") or os.getenv("OPENAI_MODEL_NAME") or ""
    if not resolved_model:
        raise DreamerRuntimeError(
            "No dreamer model configured. Set MEMORY_DREAMER_MODEL or OPENAI_MODEL_NAME."
        )
    dreamer = Dreamer(
        manager,
        model=resolved_model,
        api_key=api_key,
        base_url=base_url,
        client=client,
    )
    return dreamer.run()
