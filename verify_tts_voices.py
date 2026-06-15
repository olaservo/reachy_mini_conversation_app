"""Headless check: does the Qwen3-TTS endpoint resolve the 11 roster voice names?

Mirrors cascade/tts/qwen3_tts.py:Qwen3TTS.synthesize — an OpenAI-compatible
POST /v1/audio/speech with response_format=pcm and voice=<roster id>. Confirms
the server accepts each roster speaker name (the speak_as leg) and returns audio.
"""

from __future__ import annotations
import os
import sys
import asyncio

from openai import AsyncOpenAI

BASE_URL = os.getenv("CASCADE_TTS_BASE_URL", "https://olahungerford--qwen3-tts-voices-serve.modal.run/v1")
MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B"

ROSTER = [
    "gm_narrator", "augusta_byron", "tommy_doyle", "bailey_bigsmile", "old_tallman",
    "hazel_johnson", "marvin", "npc_raider", "npc_settler", "npc_merchant", "npc_overseer",
]


async def synth(client: AsyncOpenAI, voice: str) -> int:
    total = 0
    async with client.audio.speech.with_streaming_response.create(
        model=MODEL, voice=voice, input="Greetings, traveler.", response_format="pcm",
    ) as resp:
        async for chunk in resp.iter_bytes(chunk_size=4096):
            total += len(chunk)
    return total


async def main() -> None:
    print(f"TTS: {BASE_URL}\nModel: {MODEL}")
    print("Probing roster voices (first request may cold-start)...\n")
    client = AsyncOpenAI(api_key="EMPTY", base_url=BASE_URL, timeout=300.0)
    ok, bad = [], []
    for v in ROSTER:
        try:
            nbytes = await synth(client, v)
            status = "OK" if nbytes > 0 else "EMPTY"
            (ok if nbytes > 0 else bad).append(v)
            print(f"  [{status:5}] {v:15} {nbytes} bytes")
        except Exception as e:
            bad.append(v)
            print(f"  [FAIL ] {v:15} {type(e).__name__}: {str(e)[:120]}")
    print(f"\n{len(ok)}/{len(ROSTER)} roster voices resolved.")
    if bad:
        print("Unresolved:", ", ".join(bad))
        sys.exit(1)
    print("PASS: all roster voices resolve on the TTS endpoint (speak_as leg).")


if __name__ == "__main__":
    asyncio.run(main())
