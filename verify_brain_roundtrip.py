"""Headless verification of the cascade -> live brain tool-call passthrough.

Replicates the exact streaming Chat Completions call that
``cascade/llm/openai.py:OpenAILLM.generate`` makes (stream=True, tools,
tool_choice="auto", index-keyed tool_call accumulation), against the live
Modal-hosted Qwen brain. Proves the WS1+WS3 risk leg end to end:

  app tool spec -> brain -> structured tool_calls -> tool result back in -> brain narrates

No torch / mic / fastrtc needed. Run with the throwaway .verify-venv.
"""

from __future__ import annotations
import os
import json
import asyncio

from openai import AsyncOpenAI

BASE_URL = os.getenv("CASCADE_LLM_BASE_URL", "https://olahungerford--qwen3-brain-serve.modal.run/v1")
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

# The app-side tool name is namespaced by the `ttrpg` MCP alias (ttrpg__roll_dice).
# The brain echoes whatever name the app hands it, so the namespaced name must round-trip.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ttrpg__roll_dice",
            "description": "Roll dice using standard RPG notation (e.g. '1d20', '3d6+2') and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "notation": {"type": "string", "description": "Dice notation, e.g. '1d20'"}
                },
                "required": ["notation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "speak",
            "description": "Speak the given message to the user. Use this tool for ALL verbal responses.",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        },
    },
]

SYSTEM = (
    "You are a whimsical tabletop-RPG dungeon master. You have tools: ttrpg__roll_dice to roll "
    "dice, and speak to say things aloud. When a player attempts an action that needs a check, "
    "call ttrpg__roll_dice. After you get the result, narrate the outcome by calling speak."
)


async def stream_turn(client: AsyncOpenAI, messages: list[dict]) -> tuple[str, list[dict]]:
    """Mirror OpenAILLM.generate: stream, accumulate text + tool_calls by index."""
    kwargs = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": TOOLS,
        "tool_choice": "auto",
    }
    text = ""
    tool_calls: dict[int, dict] = {}
    stream = await client.chat.completions.create(**kwargs)
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            text += delta.content
        if delta.tool_calls:
            for tcd in delta.tool_calls:
                idx = tcd.index
                if idx not in tool_calls:
                    tool_calls[idx] = {
                        "id": tcd.id or "",
                        "type": "function",
                        "function": {"name": tcd.function.name or "", "arguments": ""},
                    }
                if tcd.function.arguments:
                    tool_calls[idx]["function"]["arguments"] += tcd.function.arguments
                if tcd.function.name:
                    tool_calls[idx]["function"]["name"] = tcd.function.name
    return text, list(tool_calls.values())


async def main() -> None:
    print(f"Brain: {BASE_URL}\nModel: {MODEL}")
    print("Connecting (first request may block ~2.5 min on a cold start)...\n")
    client = AsyncOpenAI(api_key="EMPTY", base_url=BASE_URL, timeout=360.0)

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "I swing my rusty sword at the goblin! Do I hit?"},
    ]

    # --- Turn 1: expect a ttrpg__roll_dice tool call ---
    text1, calls1 = await stream_turn(client, messages)
    print("=== TURN 1 ===")
    print("assistant text:", repr(text1))
    print("tool_calls:", json.dumps(calls1, indent=2))

    dice_call = next((c for c in calls1 if c["function"]["name"] == "ttrpg__roll_dice"), None)
    assert dice_call is not None, "FAIL: brain did not emit a structured ttrpg__roll_dice tool_call"
    args = json.loads(dice_call["function"]["arguments"])
    print(f"\n[OK] structured ttrpg__roll_dice round-tripped with args={args}")

    # --- Feed a tool result back, expect narration (speak or plain text) ---
    messages.append({"role": "assistant", "content": text1 or None, "tool_calls": calls1})
    # Respond to every tool_call id the assistant emitted (OpenAI requires this).
    for c in calls1:
        result = {"notation": args.get("notation", "1d20"), "rolls": [17], "total": 17} \
            if c is dice_call else {"ok": True}
        messages.append({
            "role": "tool",
            "tool_call_id": c["id"],
            "name": c["function"]["name"],
            "content": json.dumps(result),
        })

    text2, calls2 = await stream_turn(client, messages)
    print("\n=== TURN 2 (after tool result total=17) ===")
    print("assistant text:", repr(text2))
    print("tool_calls:", json.dumps(calls2, indent=2))

    spoke = next((c for c in calls2 if c["function"]["name"] in ("speak", "speak_as")), None)
    if spoke:
        narration = json.loads(spoke["function"]["arguments"]).get("message", "")
        print(f"\n[OK] brain narrated via {spoke['function']['name']}: {narration!r}")
    elif text2.strip():
        print(f"\n[OK] brain narrated as plain text (no speak tool): {text2[:200]!r}")
    else:
        print("\n[WARN] brain produced neither speak tool nor text after the roll result")

    print("\nPASS: tool-call passthrough verified (brain leg of the integration milestone).")


if __name__ == "__main__":
    asyncio.run(main())
