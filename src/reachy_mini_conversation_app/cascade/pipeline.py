"""LLM response processing and tool execution pipeline."""

from __future__ import annotations
import json
import base64
import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, List
from dataclasses import dataclass

from reachy_mini_conversation_app.cascade.llm import LLMProvider
from reachy_mini_conversation_app.cascade.tts import TTSProvider
from reachy_mini_conversation_app.cascade.config import get_config
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies, dispatch_tool_call
from reachy_mini_conversation_app.cascade.turn_result import TurnItem, PipelineResult


def _log_prompt(messages: list[dict[str, Any]], tools: list[dict[str, Any]], system: str | None, depth: int) -> None:
    """Log a human-readable snapshot of the LLM request at debug level."""
    import datetime

    lines: list[str] = []
    lines.append(f"\n{'=' * 80}")
    lines.append(f"LLM REQUEST  depth={depth}  {datetime.datetime.now().isoformat(timespec='milliseconds')}")
    lines.append(f"{'=' * 80}")

    # System instructions
    if system:
        lines.append(f"\n--- SYSTEM ({len(system)} chars) ---")
        lines.append(system)

    # Tools
    lines.append(f"\n--- TOOLS ({len(tools)}) ---")
    for t in tools:
        fn = t.get("function", t)
        lines.append(f"  - {fn.get('name', '?')}: {fn.get('description', '')[:120]}")

    # Messages
    lines.append(f"\n--- MESSAGES ({len(messages)}) ---")
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        # Tool result message
        if role == "tool":
            content = msg.get("content", "")
            if len(content) > 300:
                content = content[:300] + "..."
            lines.append(f"[{i}] {role} ({msg.get('name', '?')}): {content}")
        # Assistant with tool calls
        elif "tool_calls" in msg:
            text = msg.get("content", "") or ""
            tc_summary = ", ".join(tc.get("function", {}).get("name", "?") for tc in msg["tool_calls"])
            lines.append(f"[{i}] {role}: {text[:200]}  [tool_calls: {tc_summary}]")
        # User with image
        elif isinstance(msg.get("content"), list):
            parts = []
            for p in msg["content"]:
                if isinstance(p, dict) and p.get("type") == "image":
                    parts.append("<image>")
                elif isinstance(p, dict) and p.get("type") == "text":
                    parts.append(p.get("text", "")[:200])
                else:
                    parts.append(str(p)[:200])
            lines.append(f"[{i}] {role}: {' | '.join(parts)}")
        else:
            content = str(msg.get("content", ""))
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"[{i}] {role}: {content}")

    lines.append("")

    logger.debug("\n".join(lines))


if TYPE_CHECKING:
    from reachy_mini_conversation_app.cascade.speech_output import SpeechOutput


logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Bundle of references passed through the LLM/tool pipeline."""

    llm: LLMProvider
    tts: TTSProvider
    speech_output: SpeechOutput | None
    conversation_history: list[dict[str, Any]]
    tool_specs: list[dict[str, Any]]
    deps: ToolDependencies
    result: PipelineResult


def _track_cost(ctx: PipelineContext, provider: Any) -> None:
    """Accumulate provider cost into the pipeline result."""
    if hasattr(provider, "last_cost") and provider.last_cost > 0:
        ctx.result.cost += provider.last_cost
        provider.last_cost = 0.0


async def process_llm_response(ctx: PipelineContext) -> PipelineResult:
    """Process LLM response with retry on failure."""
    max_retries = 2
    for attempt in range(1 + max_retries):
        try:
            await _process_llm_response_once(ctx)
            return ctx.result
        except Exception as e:
            if attempt < max_retries:
                logger.warning("LLM failed (attempt %d), retrying: %s", attempt + 1, e)
                if ctx.speech_output:
                    await ctx.speech_output.speak("Give me a moment.")
                await asyncio.sleep(2)
            else:
                logger.error("LLM failed after %d attempts: %s", max_retries + 1, e)
                if ctx.speech_output:
                    await ctx.speech_output.speak("Sorry, I'm having trouble responding right now.")
    return ctx.result


async def _process_llm_response_once(ctx: PipelineContext, _depth: int = 0) -> None:
    """Single attempt at processing LLM response with streaming, tool calls, and TTS."""
    # Log the full prompt for debugging
    system = getattr(ctx.llm, "system_instructions", None)
    _log_prompt(ctx.conversation_history, ctx.tool_specs, system, _depth)

    # Generate streaming response
    text_chunks: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    async for chunk in ctx.llm.generate(
        messages=ctx.conversation_history,
        tools=ctx.tool_specs,
        temperature=get_config().llm_temperature,
    ):
        if chunk.type == "text_delta" and chunk.content:
            text_chunks.append(chunk.content)
            logger.debug(f"LLM text delta: {chunk.content}")

        elif chunk.type == "tool_call" and chunk.tool_call:
            tool_calls.append(chunk.tool_call)
            logger.info(f"LLM tool call: {chunk.tool_call}")

        elif chunk.type == "done":
            logger.debug("LLM generation complete")
            break

    # Aggregate LLM cost after generator completes
    _track_cost(ctx, ctx.llm)

    # Create assistant message with text, tool calls...
    assistant_message: Dict[str, Any] = {"role": "assistant"}
    full_text = ""
    if text_chunks:
        full_text = "".join(text_chunks)
        assistant_message["content"] = full_text
    if tool_calls:
        assistant_message["tool_calls"] = tool_calls

    logger.debug(
        f"process_llm_response: text_chunks={len(text_chunks)}, tool_calls={len(tool_calls)}, full_text_len={len(full_text)}"
    )

    if text_chunks or tool_calls:
        ctx.conversation_history.append(assistant_message)
        logger.debug(f"Added assistant message to history, history_len={len(ctx.conversation_history)}")

    # Handle text-only responses: auto-inject speak tool call
    # This handles cases where LLM returns text without using the speak tool
    # In principle it should not happen thanks to the extra instructions.
    # If it happens, we create a synthetic tool call for speaking
    if full_text and not tool_calls:
        logger.info("❓LLM returned text without speak tool - auto-injecting speak call")

        synthetic_tool_call = {
            "id": "auto_speak",
            "type": "function",
            "function": {"name": "speak", "arguments": json.dumps({"message": full_text})},
        }
        await execute_tool_calls([synthetic_tool_call], ctx)
    elif tool_calls and not any(
        tc.get("function", {}).get("name") in ("speak", "speak_as") for tc in tool_calls
    ):
        # Tool calls but no speak — record assistant text if present
        if full_text:
            ctx.result.turn_items.append(TurnItem(kind="assistant", text=full_text))
    if tool_calls:
        # Process normal tool calls
        await execute_tool_calls(tool_calls, ctx)

        # If no speak tool was called, re-invoke LLM so it can react to tool results.
        # speak_as is also a verbal response, so it counts as speaking.
        has_speak = any(tc.get("function", {}).get("name") in ("speak", "speak_as") for tc in tool_calls)
        if not has_speak and _depth < 5:
            logger.info("No speak in tool calls — re-invoking LLM to react to tool results")
            await _process_llm_response_once(ctx, _depth=_depth + 1)


async def execute_tool_calls(
    tool_calls: list[dict[str, Any]],
    ctx: PipelineContext,
) -> None:
    """Execute tool calls and handle see_image_through_camera and speak specially."""
    camera_image_bytes: bytes | None = None

    # First pass: execute all tools and add ALL tool results to conversation
    # This must be done before adding any other messages (OpenAI requires all tool
    # responses immediately after the assistant message with tool_calls)
    for tool_call in tool_calls:
        call_id = ""
        tool_name = "unknown"
        try:
            call_id, tool_name, arguments = ctx.llm.parse_tool_call(tool_call)

            logger.info(f"Executing tool: {tool_name}({arguments})")

            # Execute tool. "speak" is cascade-specific (not in the shared tool registry):
            # the handler injects its spec and turns the returned message into TTS below.
            if tool_name == "speak":
                result = {"message": arguments.get("message", "")}
            elif tool_name == "speak_as":
                result = {
                    "voice_id": arguments.get("voice_id", ""),
                    "message": arguments.get("message", ""),
                }
            else:
                result = await dispatch_tool_call(
                    tool_name,
                    json.dumps(arguments),
                    ctx.deps,
                )

            # Do not log full result if the tool returned base64 (huge)
            if "b64_im" in result:
                logger.info("Tool result: [image in base64, not shown]")
            else:
                logger.info(f"Tool result: {result}")

            # Add tool result to conversation
            ctx.conversation_history.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": json.dumps(result),
                }
            )
            logger.debug(
                f"Added tool result to history: name={tool_name}, history_len={len(ctx.conversation_history)}"
            )

            # Special handling for see_image_through_camera - store frame, replace heavy b64
            if tool_name == "see_image_through_camera":
                if "b64_im" in result:
                    b64_im = result["b64_im"]
                    camera_image_bytes = base64.b64decode(b64_im)
                    frame_index = len(ctx.result.captured_frames)
                    ctx.result.captured_frames.append(camera_image_bytes)
                    # Replace the heavy b64 blob in conversation history with a lightweight marker
                    ctx.conversation_history[-1]["content"] = json.dumps(
                        {"status": "image_captured", "frame_index": frame_index}
                    )
                    ctx.result.turn_items.append(TurnItem(kind="image", image_jpeg=camera_image_bytes))
                    logger.info(
                        "see_image_through_camera: stored frame %d, will add image to conversation", frame_index
                    )
                else:
                    logger.warning(f"see_image_through_camera returned error: {result}")

            # Special handling for speak tool
            elif tool_name == "speak" and "message" in result:
                message = result["message"]
                logger.info(f"Speaking: {message}")
                ctx.result.turn_items.append(TurnItem(kind="speak", text=message))

                if ctx.speech_output:
                    await ctx.speech_output.speak(message)
                _track_cost(ctx, ctx.tts)

            # Special handling for speak_as tool — like speak, but in a named voice
            elif tool_name == "speak_as" and "message" in result:
                voice_id = result["voice_id"]
                message = result["message"]
                logger.info(f"Speaking as {voice_id}: {message}")
                ctx.result.turn_items.append(TurnItem(kind="speak", text=message))

                if ctx.speech_output:
                    await ctx.speech_output.speak(message, voice=voice_id)
                _track_cost(ctx, ctx.tts)

            # Other tools
            elif tool_name not in ("speak", "speak_as", "see_image_through_camera"):
                ctx.result.turn_items.append(
                    TurnItem(kind="tool", tool_name=tool_name, tool_content=json.dumps(result))
                )

        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}: {e}")

            # Add error to conversation
            ctx.conversation_history.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": json.dumps({"error": str(e)}),
                }
            )

    # After all tool results are added, add camera image as user message and call LLM
    if camera_image_bytes is not None:
        ctx.conversation_history.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": camera_image_bytes,  # Will be converted to provider format in LLM
                    }
                ],
            }
        )
        logger.info("Camera image added to conversation - calling LLM to analyze it")
        await process_llm_response(ctx)
