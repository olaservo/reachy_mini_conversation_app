"""Headless DM-behavior check against the warm brain with the REAL dm instructions.

Verifies the brain *chooses* the right tools during play (the plumbing is already
proven): does it call speak_as with valid roster voice_ids, hass__HassLightSet on the
Ola Office LIFX bulb, ttrpg__roll_dice on checks, remember on facts, and the camera tool
when asked to look. Scripts a few player turns and prints every tool call.
"""
from __future__ import annotations
import os, json, asyncio
from openai import AsyncOpenAI
from reachy_mini_conversation_app.cascade.provider_factory import cascade_system_instructions

BASE_URL = "https://olahungerford--qwen3-brain-serve.modal.run/v1"
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
ROSTER = {"gm_narrator","augusta_byron","tommy_doyle","bailey_bigsmile","old_tallman",
          "hazel_johnson","marvin","npc_raider","npc_settler","npc_merchant","npc_overseer"}

def tool(name, desc, props, req):
    return {"type":"function","function":{"name":name,"description":desc,
            "parameters":{"type":"object","properties":props,"required":req}}}

TOOLS = [
    tool("speak","Speak narration to the user.",{"message":{"type":"string"}},["message"]),
    tool("speak_as","Speak a line in a specific character's designed voice.",
         {"voice_id":{"type":"string"},"message":{"type":"string"}},["voice_id","message"]),
    tool("ttrpg__roll_dice","Roll dice in RPG notation.",{"notation":{"type":"string"}},["notation"]),
    tool("ttrpg__present_player_choice","Offer the player a set of choices.",
         {"prompt":{"type":"string"},"options":{"type":"array","items":{"type":"string"}}},["prompt","options"]),
    tool("hass__HassLightSet","Set a light's color/brightness.",
         {"name":{"type":"string"},"area":{"type":"string"},"color":{"type":"string"},
          "brightness":{"type":"integer"}},[]),
    tool("remember","Store a durable memory (party, scene, plot thread).",
         {"content":{"type":"string"}},["content"]),
    tool("see_image_through_camera","Look at what the player is showing the camera.",{},[]),
]

# Player turns crafted to provoke each leg.
SCRIPT = [
    "Let's begin. I'm Tommy, a survivor gambler. I walk into the Junktown saloon.",      # scene -> light + narration
    "I greet the grizzled bartender and ask what jobs are around.",                       # NPC -> speak_as
    "I try to pick the lock on the back room.",                                            # check -> roll_dice
    "Remember that the bartender owes me 50 caps.",                                        # fact -> remember
    "Look at my character sheet.",                                                          # AMBIGUOUS -> should ASK, not camera
    "I'm holding up my dice to the camera — what did I just roll?",                         # EXPLICIT show -> camera
]

async def turn(client, messages):
    calls={}
    stream= await client.chat.completions.create(model=MODEL,messages=messages,temperature=0.7,
        stream=True,tools=TOOLS,tool_choice="auto")
    text=""
    async for ch in stream:
        if not ch.choices: continue
        d=ch.choices[0].delta
        if d.content: text+=d.content
        for tc in (d.tool_calls or []):
            i=tc.index
            calls.setdefault(i,{"id":tc.id or "","function":{"name":tc.function.name or "","arguments":""}})
            if tc.function.name: calls[i]["function"]["name"]=tc.function.name
            if tc.function.arguments: calls[i]["function"]["arguments"]+=tc.function.arguments
    return text, list(calls.values())

async def main():
    sys_prompt = cascade_system_instructions()
    print(f"dm system prompt: {len(sys_prompt)} chars\n")
    client = AsyncOpenAI(api_key="EMPTY", base_url=BASE_URL, timeout=120.0)
    messages=[{"role":"system","content":sys_prompt}]
    seen=set()
    for i,player in enumerate(SCRIPT,1):
        messages.append({"role":"user","content":player})
        text,calls=await turn(client,messages)
        print(f"=== TURN {i}: {player[:60]}")
        if text.strip(): print("  (text)", text[:120])
        for c in calls:
            n=c["function"]["name"]; a=c["function"]["arguments"]; seen.add(n)
            flag=""
            try: args=json.loads(a)
            except: args={}
            if n=="speak_as":
                vid=args.get("voice_id"); flag=" <-- VALID roster voice" if vid in ROSTER else f" <-- !! UNKNOWN voice '{vid}'"
            if n=="hass__HassLightSet":
                nm=args.get("name"); flag=" <-- office LIFX" if nm=="LIFX Color 65631B" else f" <-- !! target={nm!r} area={args.get('area')!r}"
            print(f"  TOOL {n}({a}){flag}")
        # feed tool results so the convo can continue
        messages.append({"role":"assistant","content":text or None,"tool_calls":[{"id":c["id"],"type":"function","function":c["function"]} for c in calls] or None})
        for c in calls:
            res={"ok":True}
            if c["function"]["name"]=="ttrpg__roll_dice": res={"total":14,"rolls":[14]}
            messages.append({"role":"tool","tool_call_id":c["id"],"name":c["function"]["name"],"content":json.dumps(res)})
        print()
    print("TOOLS THE BRAIN USED:", sorted(seen))
    for want in ["speak_as","ttrpg__roll_dice","hass__HassLightSet","remember","see_image_through_camera"]:
        print(f"  {'YES' if want in seen else 'no ':3} {want}")

if __name__=="__main__":
    asyncio.run(main())
