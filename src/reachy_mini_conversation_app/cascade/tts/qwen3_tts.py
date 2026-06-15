"""Qwen3-TTS provider for the cascade pipeline (per-character voices via `speak_as`).

Qwen/Qwen3-TTS-12Hz-1.7B (1.7B params, under the hackathon 32B cap) is the voice
engine for the whimsical TTRPG dungeon master. The DM narrates in a default voice
and switches to distinct designed voices for each NPC. The 11-voice roster and the
VoiceDesign / voice-clone workflow live in:
  reachy-dm/voices/character-voices.md

Call shape
----------
Qwen3-TTS is served behind an **OpenAI-compatible** ``POST /v1/audio/speech``
endpoint (this is how vLLM and the ``huggingface/speech-to-speech`` cascade expose
it), so we reuse ``AsyncOpenAI`` with a configurable ``base_url`` instead of a
bespoke HTTP client. The same code therefore targets either:

* a **local** Qwen3-TTS server running beside the robot
  (default ``base_url=http://localhost:8001/v1``, $0, preferred for home use), or
* a **Modal** deployment (set ``base_url`` to the Modal web endpoint + ``/v1``;
  pass the bearer token via the ``api_key`` provider setting, which maps to an env
  var in ``cascade.yaml`` — never hardcode it).

We request ``response_format="pcm"`` (raw 16-bit little-endian PCM) and stream the
bytes through the same leading-silence trim + 1 KiB sub-chunking used by the OpenAI
and Gradium providers, so the cascade's ``QueueSpeechOutput`` can play it directly.

``voice`` is the per-utterance speaker name (a ``voice_id`` from the roster, e.g.
``gm_narrator`` or ``npc_raider``). When omitted it falls back to ``default_voice``.
The voice names must be registered as speakers on the Qwen3-TTS server (each is a
stored voice-clone prompt produced by the VoiceDesign workflow). The deployed Modal
endpoint resolves all 11 roster names **by name** (verified 2026-06-15 — each returns
distinct audio), so ``VOICE_PROMPTS`` stays empty. Only populate it if you stand up a
server that instead wants the clone prompt passed inline via ``extra_body``.

speak_as integration (DONE — wired end to end in commit 3a75c07)
----------------------------------------------------------------
The DM voices an NPC by calling a ``speak_as(voice_id, message)`` tool. The wiring is
live across three files; the seams (for reference / future edits) are:

1. **Tool spec** — ``SPEAK_AS_TOOL_SPEC`` sits next to ``SPEAK_TOOL_SPEC`` in
   ``cascade/handler.py`` and is included in ``_build_tool_specs``. Params:
   ``voice_id`` (one of the roster ids) and ``message``.

2. **Interception** — ``cascade/pipeline.py:execute_tool_calls`` special-cases
   ``speak_as`` alongside ``speak``: it pulls ``voice_id``/``message`` from the tool
   arguments and calls ``ctx.speech_output.speak(message, voice=voice_id)``, then
   ``_track_cost(ctx, ctx.tts)``.

3. **Per-utterance voice override** — ``QueueSpeechOutput.speak`` takes an optional
   ``voice`` and threads ``voice=voice or handler._voice`` into
   ``handler.tts.synthesize(...)``; the ``SpeechOutput`` Protocol carries the same
   optional kwarg, and ``synthesize`` (below) forwards it to the endpoint's ``voice``
   field. Backward compatible: bare ``speak(text)`` callers keep the default voice.

Narration stays on ``gm_narrator`` (the handler default) while ``speak_as`` swaps
voices per line without touching handler state.
"""

from __future__ import annotations
import time
import logging
from typing import Optional, AsyncIterator

import numpy as np

from .base import TTSProvider
from .utils import trim_leading_silence


logger = logging.getLogger(__name__)

SILENCE_THRESHOLD = 327  # ~0.01 * 32767, matches trim_leading_silence default
SUB_CHUNK_BYTES = 1024

# The 11-voice roster (see reachy-dm/voices/character-voices.md). These names must
# exist as registered speakers / stored voice-clone prompts on the Qwen3-TTS server.
ROSTER_VOICE_IDS = (
    "gm_narrator",     # default narrator
    "augusta_byron",   # pregen: Vault Dweller scientist
    "tommy_doyle",     # pregen: Survivor gambler
    "bailey_bigsmile",  # pregen: Ghoul wanderer
    "old_tallman",     # pregen: Super Mutant philosopher
    "hazel_johnson",   # pregen: Brotherhood Field Scribe
    "marvin",          # pregen: Mister Handy robot
    "npc_raider",      # archetype: hostile raiders
    "npc_settler",     # archetype: nervous townsfolk
    "npc_merchant",    # archetype: traders
    "npc_overseer",    # archetype: authority/comms
)

# Optional name -> inline voice-clone-prompt map. The deployed Modal endpoint
# resolves all 11 roster voices by name (verified), so this stays empty. Populate
# it only for a server that wants the clone prompt passed inline via extra_body.
VOICE_PROMPTS: dict[str, str] = {}


class Qwen3TTS(TTSProvider):
    """Qwen3-TTS via an OpenAI-compatible /v1/audio/speech endpoint.

    Supports a default voice plus a per-utterance ``voice`` override, which is what
    the DM's ``speak_as(voice_id, text)`` tool needs to voice different NPCs.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001/v1",
        api_key: Optional[str] = None,
        model: str = "Qwen/Qwen3-TTS-12Hz-1.7B",
        voice: str = "gm_narrator",
        sample_rate: int = 24000,
        response_format: str = "pcm",
        cost_per_1m_chars: float = 0.0,
    ):
        """Initialize Qwen3-TTS.

        Args:
            base_url: OpenAI-compatible base URL of the Qwen3-TTS server. Default is
                a local serve; point at a Modal web endpoint (+ ``/v1``) for cloud.
            api_key: Bearer token for the endpoint (maps to an env var in
                cascade.yaml; local serve usually needs none — a placeholder is used).
            model: Served model name.
            voice: Default voice_id (narrator). Overridden per utterance by ``speak_as``.
            sample_rate: PCM sample rate the server emits (Qwen3-TTS-12Hz -> 24kHz).
            response_format: Audio format requested; "pcm" = raw int16 LE (streamable).
            cost_per_1m_chars: Cost accounting (0 for local).

        """
        # Heavy/optional dep guarded the same way as other providers; openai is a
        # core dep here, but keep the import local to match the lazy pattern and so
        # a missing/old client raises at init with a clear message, not at import.
        from openai import AsyncOpenAI

        # Local servers typically ignore auth; AsyncOpenAI requires a non-empty key.
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key or "EMPTY")
        self.base_url = base_url
        self.model = model
        self.default_voice = voice
        self._sample_rate = sample_rate
        self.response_format = response_format
        self.cost_per_1m_chars = cost_per_1m_chars
        self.last_cost: float = 0.0
        # Exposed so the handler/UI can list selectable voices (get_available_voices
        # reads provider.available_voices — see handler.py:193).
        self.available_voices = list(ROSTER_VOICE_IDS)
        logger.info(
            "Initialized Qwen3-TTS (model=%s, base_url=%s, default_voice=%s, sr=%d)",
            model,
            base_url,
            voice,
            sample_rate,
        )

    @property
    def sample_rate(self) -> int:
        """PCM sample rate emitted by the server (24kHz for Qwen3-TTS-12Hz)."""
        return self._sample_rate

    async def synthesize(self, text: str, voice: Optional[str] = None) -> AsyncIterator[bytes]:
        """Synthesize ``text`` with optional per-utterance ``voice`` override.

        Streams raw PCM (int16 LE) from the endpoint, trims leading silence on the
        initial chunk(s), then yields ~1 KiB sub-chunks for low time-to-first-audio.

        Yields:
            Audio bytes (PCM 16-bit mono at ``self.sample_rate``).

        """
        from reachy_mini_conversation_app.cascade.timing import tracker

        if not text.strip():
            logger.warning("Empty text provided for synthesis")
            return

        voice_to_use = voice or self.default_voice
        logger.info("Qwen3-TTS: synthesizing as '%s': '%s...'", voice_to_use, text[:50])

        tracker.mark("tts_start", {"text_len": len(text)})

        # If the server resolves voices by inline clone prompt rather than by name,
        # forward it via extra_body. Empty VOICE_PROMPTS -> name-based resolution.
        extra_body: dict = {}
        if voice_to_use in VOICE_PROMPTS:
            extra_body["voice_clone_prompt"] = VOICE_PROMPTS[voice_to_use]

        try:
            tracker.mark("tts_api_request_sending")
            request_start = time.perf_counter()

            is_leading = True
            leading_buffer = bytearray()
            first_byte = True
            chunk_count = 0

            async with self.client.audio.speech.with_streaming_response.create(
                model=self.model,
                voice=voice_to_use,
                input=text,
                response_format=self.response_format,
                extra_body=extra_body or None,
            ) as response:
                async for chunk in response.iter_bytes(chunk_size=SUB_CHUNK_BYTES):
                    if not chunk:
                        continue

                    if first_byte:
                        ttfb_ms = (time.perf_counter() - request_start) * 1000
                        tracker.mark("tts_api_first_byte", {"ttfb_ms": round(ttfb_ms, 1)})
                        first_byte = False

                    if is_leading:
                        leading_buffer.extend(chunk)
                        samples = np.frombuffer(chunk, dtype=np.int16)
                        if np.any(np.abs(samples) > SILENCE_THRESHOLD):
                            # Found audio — trim accumulated buffer and start yielding.
                            is_leading = False
                            full_buffer = np.frombuffer(bytes(leading_buffer), dtype=np.int16)
                            trimmed = trim_leading_silence(
                                full_buffer, sample_rate=self.sample_rate, provider_name="Qwen3-TTS"
                            )
                            tracker.mark("tts_first_chunk_ready")
                            trimmed_bytes = trimmed.tobytes()
                            for i in range(0, len(trimmed_bytes), SUB_CHUNK_BYTES):
                                sub = trimmed_bytes[i : i + SUB_CHUNK_BYTES]
                                if sub:
                                    if chunk_count == 0:
                                        logger.info("Qwen3-TTS: First chunk ready (can start playback now!)")
                                    chunk_count += 1
                                    yield sub
                    else:
                        chunk_count += 1
                        yield chunk

            # Edge case: entire audio was silent — yield it unchanged.
            if is_leading and leading_buffer:
                tracker.mark("tts_first_chunk_ready")
                logger.info("Qwen3-TTS: First chunk ready (can start playback now!)")
                for i in range(0, len(leading_buffer), SUB_CHUNK_BYTES):
                    sub = bytes(leading_buffer[i : i + SUB_CHUNK_BYTES])
                    if sub:
                        chunk_count += 1
                        yield sub

            tracker.mark("tts_api_complete")

            if self.cost_per_1m_chars > 0:
                call_cost = len(text) * self.cost_per_1m_chars / 1e6
                self.last_cost += call_cost
                logger.info("TTS Cost: $%.6f (%d chars, model=%s)", call_cost, len(text), self.model)

            logger.info("Qwen3-TTS: Synthesis complete - %d chunks for '%s...'", chunk_count, text[:50])

        except Exception as e:
            logger.error("Qwen3-TTS synthesis failed: %s", e)
            raise
