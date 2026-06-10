"""Kokoro TTS provider for cascade pipeline (Apple Silicon optimized)."""

from __future__ import annotations
import time
import asyncio
import logging
from typing import Optional, AsyncIterator

import numpy as np

from .base import TTSProvider
from .utils import trim_leading_silence


logger = logging.getLogger(__name__)

CHUNK_SIZE = 4096  # samples per sub-chunk (~170ms at 24kHz)


class KokoroTTS(TTSProvider):
    """Kokoro-82M TTS implementation optimized for Apple Silicon."""

    def __init__(
        self,
        voice: str = "af_bella",
        lang_code: str = "a",  # 'a' for American English
        repo_id: str = "hexgrad/Kokoro-82M",
    ):
        """Initialize Kokoro TTS.

        Args:
            voice: Voice to use (af_bella, af_sarah, am_adam, am_michael, bf_emma, bf_isabella, bm_george, bm_lewis)
            lang_code: Language code ('a' for American English, 'b' for British English)
            repo_id: HuggingFace repo ID for model

        """
        self.default_voice = voice
        self.lang_code = lang_code
        self.repo_id = repo_id

        # Load pipeline immediately (not lazy) to avoid loading delay on first synthesis
        from kokoro import KPipeline

        logger.info(f"Loading Kokoro pipeline from {repo_id}...")
        self.pipeline = KPipeline(lang_code=self.lang_code, repo_id=self.repo_id)

        # Preload voice to avoid download delay on first synthesis
        logger.info(f"Preloading voice: {voice}...")
        try:
            # Generate a tiny test sentence to trigger voice caching
            result = self.pipeline(".", voice=voice)
            for _ in result:
                break  # Only need first chunk to trigger cache
            logger.info(f"Voice {voice} preloaded successfully")
        except Exception as e:
            logger.warning(f"Failed to preload voice {voice}: {e}")

        logger.info(f"Kokoro TTS initialized with voice: {voice}, lang: {lang_code}")

    async def synthesize(self, text: str, voice: Optional[str] = None) -> AsyncIterator[bytes]:
        """Synthesize text using Kokoro TTS with true streaming.

        Launches a producer thread that iterates KPipeline (sync generator) and
        pushes sub-chunks through an asyncio.Queue. The async generator yields
        each sub-chunk as soon as it arrives, giving ~150-200ms time-to-first-audio
        instead of waiting for the full synthesis to complete.

        Yields:
            Audio bytes (PCM 16-bit, 24kHz mono, 4096-sample sub-chunks)

        """
        from reachy_mini_conversation_app.cascade.timing import tracker

        if not text.strip():
            logger.warning("Empty text provided for synthesis")
            return

        voice_to_use = voice or self.default_voice
        logger.info(f"Kokoro TTS: Starting synthesis for '{text[:50]}...'")

        tracker.mark("tts_start", {"text_len": len(text)})

        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _producer() -> None:
            """Iterate KPipeline in a thread, push sub-chunks to the async queue."""
            tracker.mark("tts_model_generation_start")
            generation_start = time.perf_counter()
            is_first_chunk = True

            try:
                for result in self.pipeline(text, voice=voice_to_use):
                    audio_data = result.audio if hasattr(result, "audio") else result
                    # KPipeline yields PyTorch tensors — convert to numpy
                    if hasattr(audio_data, "numpy"):
                        audio_data = audio_data.numpy()

                    if is_first_chunk:
                        tracker.mark("tts_model_first_chunk")
                        audio_data = trim_leading_silence(audio_data, sample_rate=self.sample_rate, provider_name="Kokoro TTS")
                        is_first_chunk = False

                    # Convert float32 to int16 PCM
                    audio_int16 = (audio_data * 32767).astype(np.int16)

                    # Split into sub-chunks and push to queue
                    for i in range(0, len(audio_int16), CHUNK_SIZE):
                        sub_chunk = audio_int16[i : i + CHUNK_SIZE].tobytes()
                        loop.call_soon_threadsafe(queue.put_nowait, sub_chunk)

                generation_time_ms = (time.perf_counter() - generation_start) * 1000
                tracker.mark("tts_model_generation_complete", {"generation_ms": round(generation_time_ms, 1)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        # Launch producer thread
        future = asyncio.ensure_future(asyncio.to_thread(_producer))

        try:
            first_yielded = True
            chunk_count = 0
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                chunk_count += 1
                if first_yielded:
                    tracker.mark("tts_first_chunk_ready")
                    logger.info("Kokoro TTS: First chunk ready (can start playback now!)")
                    first_yielded = False
                yield chunk

            logger.info(f"Kokoro TTS: Generated {chunk_count} audio chunks for '{text[:50]}...'")
        finally:
            # Propagate any exception from the producer thread
            await future
