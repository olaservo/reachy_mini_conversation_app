"""System tool ``remember``: synchronously persist a memory the DM wants to keep.

Unlike the offline dreamer (which consolidates *past* sessions), this writes ONE
atomic memory file immediately so the fact is recallable later this same long
session — essential for a TTRPG where story beats accrue mid-session.
"""

import asyncio
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

_ALLOWED_KINDS = ["fact", "preference", "event", "skill", "relationship", "goal"]


class Remember(Tool):
    """Persist a durable memory immediately (atomic write + index rebuild)."""

    name = "remember"
    description = (
        "Save a durable memory right now so you can recall it in future turns and "
        "future sessions. Use this for anything worth keeping: story beats and events, "
        "facts about the world or characters, the party's goals/quests, relationships "
        "between people or factions, and player preferences. The memory is written "
        "immediately and becomes recallable via recall_memory/recall_memories. Keep "
        "`content` to a concise, self-contained sentence or two."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The fact or event to remember, as a concise self-contained statement. Required.",
            },
            "kind": {
                "type": "string",
                "enum": _ALLOWED_KINDS,
                "description": (
                    "Category of memory. event=things that happened, fact=world/lore, "
                    "goal=quests/objectives, relationship=people/factions, "
                    "preference=player likes/dislikes, skill=abilities. Default 'event'."
                ),
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional topic tags to make this easier to recall later.",
            },
        },
        "required": ["content"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Write one memory synchronously (off-thread) and return its id."""
        if deps.memory_manager is None:
            return {"status": "memory_disabled", "message": "Memory is not enabled; nothing was saved."}

        content = (kwargs.get("content") or "").strip()
        if not content:
            return {"error": "content is required"}
        kind = (kwargs.get("kind") or "event").strip() or "event"
        tags = kwargs.get("tags") or []
        if not isinstance(tags, list):
            tags = [str(tags)]

        logger.info("Tool call: remember kind=%r tags=%r", kind, tags)
        manager = deps.memory_manager
        try:
            memory_id = await asyncio.to_thread(
                manager.save_memory_sync, content, kind=kind, tags=list(tags)
            )
        except ValueError as e:
            return {"error": str(e)}

        return {"status": "remembered", "id": memory_id, "kind": kind}
