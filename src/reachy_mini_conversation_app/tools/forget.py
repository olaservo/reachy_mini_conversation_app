"""System tool ``forget``: archive a stored memory so it stops surfacing.

Never hard-deletes — it marks the matched memory superseded (preserving the file),
mirroring the dreamer's supersede-don't-delete invariant.
"""

import asyncio
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class Forget(Tool):
    """Archive a memory by exact id or best-match query (no hard delete)."""

    name = "forget"
    description = (
        "Forget (archive) a stored memory so it no longer appears in your memory index "
        "or recall results. Provide either the exact memory `id` from the MEMORY index, "
        "or a free-text `query` describing the memory to forget (the closest match is "
        "archived). The underlying file is preserved, not deleted. Provide at least one "
        "of id or query."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Exact memory ID to forget, e.g. '2026-04-17_freed-prisoner_a3f'. Optional.",
            },
            "query": {
                "type": "string",
                "description": "Free-text description of the memory to forget; closest match is archived. Optional.",
            },
        },
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Archive the matched memory (off-thread) and return a status dict."""
        if deps.memory_manager is None:
            return {"status": "memory_disabled", "message": "Memory is not enabled; nothing was forgotten."}

        target = (kwargs.get("id") or kwargs.get("query") or "").strip()
        if not target:
            return {"error": "provide an id or a query"}

        logger.info("Tool call: forget target=%r", target)
        manager = deps.memory_manager
        return await asyncio.to_thread(manager.forget_memory, target)
