"""Google Gemini LLM provider for cascade pipeline."""

from __future__ import annotations
import json
import logging
from typing import Any, Dict, List, Optional, AsyncIterator

from google import genai
from google.genai import types

from .base import LLMChunk, LLMProvider


logger = logging.getLogger(__name__)


class GeminiLLM(LLMProvider):
    """Google Gemini implementation for LLM."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        system_instructions: Optional[str] = None,
        input_cost_per_1m: float = 0.0,
        output_cost_per_1m: float = 0.0,
    ):
        """Initialize Gemini LLM.

        Args:
            api_key: Google Gemini API key
            model: Model name (default: gemini-2.5-flash)
            system_instructions: System prompt
            input_cost_per_1m: Cost per 1M input tokens (from cascade.yaml)
            output_cost_per_1m: Cost per 1M output tokens (from cascade.yaml)

        """
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.input_cost_per_1m = input_cost_per_1m
        self.output_cost_per_1m = output_cost_per_1m
        self.last_cost: float = 0.0
        logger.info(f"Initialized Gemini LLM with model: {model}")
        self.system_instructions = system_instructions

    async def warmup(self) -> None:
        """Pre-warm HTTP connection by making a minimal request.

        This establishes the connection and reduces first-request latency.
        """
        logger.info("Warming up Gemini REST API (pre-establishing HTTP connection)...")
        try:
            # Make minimal request to establish connection
            warmup_messages = [{"role": "user", "content": "ping"}]

            # Convert to Gemini format
            gemini_contents = self._convert_messages_to_gemini(warmup_messages)

            # Make request with minimal tokens
            config = types.GenerateContentConfig(
                system_instruction=self.system_instructions if self.system_instructions else None,
                max_output_tokens=1,  # Minimal response
                temperature=0.0,
            )

            # Send request (this establishes the connection)
            _ = await self.client.aio.models.generate_content(
                model=self.model,
                contents=gemini_contents,  # type: ignore[arg-type,unused-ignore]
                config=config,
            )

            logger.info("Gemini REST API connection ready!")

        except Exception as e:
            # Warmup failure is non-critical
            logger.warning(f"Gemini warmup failed (non-critical): {e}")

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[LLMChunk]:
        """Generate streaming response from Gemini.

        Args:
            messages: Conversation history in OpenAI format
            tools: Available tools in OpenAI format
            temperature: Sampling temperature

        Yields:
            LLMChunk with text deltas or tool calls

        """
        from reachy_mini_conversation_app.cascade.timing import tracker

        logger.debug(f"Generating with {len(messages)} messages, {len(tools) if tools else 0} tools")

        tracker.mark("llm_start", {"messages": len(messages), "tools": len(tools) if tools else 0})

        try:
            # Convert OpenAI format to Gemini format
            tracker.mark("llm_request_prep_start")
            gemini_contents = self._convert_messages_to_gemini(messages)
            gemini_tools = self._convert_tools_to_gemini(tools) if tools else None

            # Build config
            config_params: Dict[str, Any] = {
                "temperature": temperature,
            }

            if self.system_instructions:
                config_params["system_instruction"] = self.system_instructions

            if gemini_tools:
                config_params["tools"] = gemini_tools
                # Disable automatic function calling - we handle it manually
                config_params["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)

            config = types.GenerateContentConfig(**config_params)
            tracker.mark("llm_request_prep_end")

            # Stream response
            tracker.mark("llm_request_sending")

            import time

            request_start = time.perf_counter()

            stream = await self.client.aio.models.generate_content_stream(
                model=self.model,
                contents=gemini_contents,  # type: ignore[arg-type,unused-ignore]
                config=config,
            )

            stream_open_time = (time.perf_counter() - request_start) * 1000
            tracker.mark("llm_stream_opened", {"stream_open_ms": round(stream_open_time, 1)})
            logger.debug(f"Stream opened in {stream_open_time:.1f}ms")

            accumulated_text = ""
            # Track tool calls by (candidate_idx, part_idx) to prevent duplicates
            # when a function call is streamed across multiple chunks
            tool_calls_by_position: Dict[tuple[int, int], Dict[str, Any]] = {}
            first_token = True
            chunk_count = 0

            first_chunk_time = None
            usage_metadata = None
            async for chunk in stream:
                chunk_count += 1
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()
                # Track first token (critical metric!)
                if first_token and (chunk.text or (hasattr(chunk, "candidates") and chunk.candidates)):
                    tracker.mark("llm_first_token")
                    first_token = False

                # Capture usage metadata (available on final chunk)
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    usage_metadata = chunk.usage_metadata

                # Handle text content
                if chunk.text:
                    accumulated_text += chunk.text
                    yield LLMChunk(type="text_delta", content=chunk.text)

                # Handle function calls — deduplicate by position
                if hasattr(chunk, "candidates") and chunk.candidates:
                    for cand_idx, candidate in enumerate(chunk.candidates):
                        if hasattr(candidate, "content") and candidate.content:
                            if hasattr(candidate.content, "parts") and candidate.content.parts:
                                for part_idx, part in enumerate(candidate.content.parts):
                                    if hasattr(part, "function_call") and part.function_call:
                                        key = (cand_idx, part_idx)
                                        tool_call = self._convert_function_call_to_openai(part.function_call)
                                        if key not in tool_calls_by_position:
                                            logger.info(f"Function call: {part.function_call.name}")
                                        tool_calls_by_position[key] = tool_call

            # Yield deduplicated tool calls at the end
            for tool_call in tool_calls_by_position.values():
                yield LLMChunk(type="tool_call", tool_call=tool_call)

            total_time = (time.perf_counter() - request_start) * 1000
            first_chunk_latency = (first_chunk_time - request_start) * 1000 if first_chunk_time else None

            logger.debug(
                f"Streaming complete: {chunk_count} chunks, {total_time:.1f}ms total"
                + (f", first chunk at {first_chunk_latency:.1f}ms" if first_chunk_latency else "")
            )

            tracker.mark(
                "llm_complete",
                {
                    "text_len": len(accumulated_text),
                    "tool_calls": len(tool_calls_by_position),
                    "chunks": chunk_count,
                    "total_ms": round(total_time, 1),
                    "first_chunk_ms": round(first_chunk_latency, 1) if first_chunk_latency else None,
                },
            )

            # Calculate cost from usage metadata
            if usage_metadata and (self.input_cost_per_1m > 0 or self.output_cost_per_1m > 0):
                prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
                completion_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
                self.last_cost = (
                    prompt_tokens * self.input_cost_per_1m / 1e6
                    + completion_tokens * self.output_cost_per_1m / 1e6
                )
                logger.info(f"LLM Cost: ${self.last_cost:.6f} (in={prompt_tokens}, out={completion_tokens})")

            yield LLMChunk(type="done")

        except Exception as e:
            logger.error(f"Gemini LLM generation failed: {e}")
            raise

    def _convert_messages_to_gemini(self, messages: List[Dict[str, Any]]) -> List[types.Content]:
        """Convert OpenAI message format to Gemini content format.

        OpenAI format:
            [
                {"role": "system", "content": "..."},  # Handled separately
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "...", "tool_calls": [...]},
                {"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}
            ]

        Gemini format:
            [
                Content(role="user", parts=[Part(text="...")]),
                Content(role="model", parts=[Part(text="...")]),
            ]
        """
        gemini_contents = []

        for msg in messages:
            role = msg.get("role")

            # Skip system messages (handled separately in config)
            if role == "system":
                continue

            # Convert role names
            if role == "assistant":
                gemini_role = "model"
            elif role == "user":
                gemini_role = "user"
            elif role == "tool":
                # Tool results go back as user role with function response
                gemini_role = "user"
            else:
                logger.warning(f"Unknown role: {role}, treating as user")
                gemini_role = "user"

            # Build parts
            parts = []

            # Handle regular content
            if "content" in msg and msg["content"]:
                # Handle tool results
                if role == "tool":
                    # Tool result format for Gemini
                    tool_name = msg.get("name", "unknown")
                    tool_response = msg["content"]

                    # Parse JSON if it's a string
                    if isinstance(tool_response, str):
                        try:
                            tool_response = json.loads(tool_response)
                        except json.JSONDecodeError:
                            pass

                    # Create function response part
                    function_response = types.FunctionResponse(
                        name=tool_name,
                        response=tool_response if isinstance(tool_response, dict) else {"result": tool_response},
                    )
                    parts.append(types.Part(function_response=function_response))
                elif isinstance(msg["content"], list):
                    # Content is a list (e.g., multimodal with text and images)
                    for content_part in msg["content"]:
                        if isinstance(content_part, dict):
                            content_type = content_part.get("type")
                            if content_type == "text":
                                parts.append(types.Part(text=content_part.get("text", "")))
                            elif content_type == "image":
                                # Image content (raw bytes)
                                image_bytes = content_part.get("image")
                                if image_bytes:
                                    parts.append(
                                        types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_bytes))
                                    )
                else:
                    # Regular text content
                    parts.append(types.Part(text=msg["content"]))

            # Handle tool calls (from assistant)
            if "tool_calls" in msg and msg["tool_calls"]:
                for tool_call in msg["tool_calls"]:
                    function_data = tool_call.get("function", {})
                    function_name = function_data.get("name", "")
                    args_json = function_data.get("arguments", "{}")

                    # Parse arguments
                    try:
                        arguments = json.loads(args_json) if isinstance(args_json, str) else args_json
                    except json.JSONDecodeError:
                        arguments = {}

                    # Create function call part
                    function_call = types.FunctionCall(name=function_name, args=arguments)
                    parts.append(types.Part(function_call=function_call))

            if parts:
                gemini_contents.append(types.Content(role=gemini_role, parts=parts))

        return gemini_contents

    def _convert_tools_to_gemini(self, tools: List[Dict[str, Any]]) -> List[types.Tool]:
        """Convert OpenAI tool format to Gemini function declarations.

        OpenAI format:
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather info",
                        "parameters": {...}
                    }
                }
            ]

        Gemini format:
            [
                Tool(
                    function_declarations=[
                        FunctionDeclaration(
                            name="get_weather",
                            description="Get weather info",
                            parameters={"type": "object", ...}
                        )
                    ]
                )
            ]
        """
        function_declarations = []

        for tool in tools:
            if tool.get("type") == "function":
                function_data = tool.get("function", {})

                # Create function declaration
                func_decl = types.FunctionDeclaration(
                    name=function_data.get("name", ""),
                    description=function_data.get("description", ""),
                    parameters=function_data.get("parameters", {}),
                )
                function_declarations.append(func_decl)

        # Wrap in Tool object
        if function_declarations:
            return [types.Tool(function_declarations=function_declarations)]
        return []

    def _convert_function_call_to_openai(self, function_call: types.FunctionCall) -> Dict[str, Any]:
        """Convert Gemini function call to OpenAI tool call format.

        Gemini format:
            FunctionCall(name="get_weather", args={"location": "Paris"})

        OpenAI format:
            {
                "id": "call_xxx",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Paris"}'
                }
            }
        """
        import uuid

        return {
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": function_call.name,
                "arguments": json.dumps(function_call.args or {}),
            },
        }
