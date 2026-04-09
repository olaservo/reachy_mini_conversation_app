import asyncio
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)


class SleepWake(Tool):
    """Put the robot to sleep or wake it up."""

    name = "sleep_wake"
    description = (
        "Put the robot to sleep or wake it up. "
        "ONLY use this tool when the user EXPLICITLY and SPECIFICALLY asks the robot "
        "to go to sleep or to wake up. Do NOT use this tool for any other purpose."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["sleep", "wake_up"],
                "description": (
                    "The action to perform: 'sleep' to put the robot to sleep, "
                    "'wake_up' to wake it up."
                ),
            }
        },
        "required": ["action"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        action = kwargs.get("action")
        if action not in ("sleep", "wake_up"):
            return {"error": f"Invalid action: {action}. Must be 'sleep' or 'wake_up'."}

        loop = asyncio.get_running_loop()

        if action == "sleep":
            logger.info("sleep_wake: going to sleep")
            # Suppress audio output
            deps.is_sleeping = True
            # Gate set_target in the movement loop
            deps.movement_manager.set_sleeping(True)
            # Sleep animation (blocks ~4s) — no set_target interference
            await loop.run_in_executor(None, deps.reachy_mini.goto_sleep)
            return {"status": "sleeping"}

        else:  # wake_up
            logger.info("sleep_wake: waking up")
            # Wake-up animation (blocks ~2.5s) — set_target still gated
            await loop.run_in_executor(None, deps.reachy_mini.wake_up)
            # Resume set_target — triggers BreathingMove from current pose
            deps.movement_manager.set_sleeping(False)
            # Resume audio output
            deps.is_sleeping = False
            return {"status": "awake"}
