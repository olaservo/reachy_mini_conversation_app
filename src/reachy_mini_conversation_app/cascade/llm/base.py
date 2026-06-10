"""LLM abstraction for cascade pipeline."""

from __future__ import annotations
import abc
import json
import logging
from typing import Any, Dict, List, Literal, Optional, AsyncIterator
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class LLMChunk:
    """Represents a chunk from LLM streaming response."""

    type: Literal["text_delta", "tool_call", "done"]
    content: Optional[str] = None  # Text content for text_delta
    tool_call: Optional[Dict[str, Any]] = None  # Tool call data for tool_call type


class LLMProvider(abc.ABC):
    """Abstract base class for LLM providers."""

    @abc.abstractmethod
    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[LLMChunk]:
        """Generate streaming response from LLM.

        Args:
            messages: Conversation history in OpenAI format
            tools: Optional list of available tools
            temperature: Sampling temperature

        Yields:
            LLMChunk objects with text deltas, tool calls, or completion signal

        """
        raise NotImplementedError

    async def warmup(self) -> None:
        """Warm up the LLM provider. Override if needed."""

    def parse_tool_call(self, tool_call: Dict[str, Any]) -> tuple[str, str, Dict[str, Any]]:
        """Parse a tool call into its components.

        Default implementation handles OpenAI-style tool call format.

        Args:
            tool_call: Tool call dictionary

        Returns:
            Tuple of (call_id, tool_name, arguments_dict)

        """
        call_id = tool_call.get("id", "")
        function_data = tool_call.get("function", {})
        tool_name = function_data.get("name", "")
        args_json = function_data.get("arguments", "{}")

        try:
            arguments = json.loads(args_json) if isinstance(args_json, str) else args_json
        except json.JSONDecodeError:
            logger.error(f"Failed to parse tool arguments: {args_json}")
            arguments = {}

        return call_id, tool_name, arguments
