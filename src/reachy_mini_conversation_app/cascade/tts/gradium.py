"""Gradium TTS provider for cascade pipeline."""

from __future__ import annotations
import time
import logging
from typing import Optional, AsyncIterator

import numpy as np

from .base import TTSProvider
from .utils import trim_leading_silence


logger = logging.getLogger(__name__)

SILENCE_THRESHOLD = 327  # ~0.01 * 32767, matches trim_leading_silence default


class GradiumTTS(TTSProvider):
    """Gradium TTS implementation with WebSocket streaming."""

    def __init__(
        self,
        api_key: str,
        voice_id: str = "YTpq7expH9539ERJ",
        model: str = "default",
        cost_per_1m_chars: float = 0.0,
    ):
        """Initialize Gradium TTS.

        Args:
            api_key: Gradium API key
            voice_id: Voice ID (default: Emma US/Feminine)
            model: TTS model name
            cost_per_1m_chars: Cost per 1M characters

        """
        from gradium import GradiumClient

        self.client = GradiumClient(api_key=api_key)
        self.voice_id = voice_id
        self.model = model
        self.cost_per_1m_chars = cost_per_1m_chars
        self.last_cost: float = 0.0
        logger.info(f"Initialized Gradium TTS with model: {model}, voice: {voice_id}")

    async def synthesize(self, text: str, voice: Optional[str] = None) -> AsyncIterator[bytes]:
        """Synthesize text using Gradium streaming TTS.

        Streams PCM audio at 24kHz via WebSocket, trims leading silence
        on initial chunks, then yields directly.

        Yields:
            Audio bytes (PCM 16-bit, 24kHz mono)

        """
        from reachy_mini_conversation_app.cascade.timing import tracker

        if not text.strip():
            logger.warning("Empty text provided for synthesis")
            return

        voice_to_use = voice or self.voice_id
        logger.info(f"Gradium TTS: Starting synthesis for '{text[:50]}...'")

        tracker.mark("tts_start", {"text_len": len(text)})

        try:
            tracker.mark("tts_api_request_sending")
            request_start = time.perf_counter()

            stream = await self.client.tts_stream(
                setup={
                    "model_name": self.model,
                    "voice_id": voice_to_use,
                    "output_format": "pcm_24000",
                },
                text=text,
            )

            is_leading = True
            leading_buffer = bytearray()
            first_byte = True
            chunk_count = 0

            async for chunk in stream.iter_bytes():
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
                        # Found audio - trim accumulated buffer and start yielding
                        is_leading = False
                        full_buffer = np.frombuffer(bytes(leading_buffer), dtype=np.int16)
                        trimmed = trim_leading_silence(
                            full_buffer, sample_rate=self.sample_rate, provider_name="Gradium TTS"
                        )
                        tracker.mark("tts_first_chunk_ready")
                        trimmed_bytes = trimmed.tobytes()
                        for i in range(0, len(trimmed_bytes), 1024):
                            sub = trimmed_bytes[i : i + 1024]
                            if sub:
                                if chunk_count == 0:
                                    logger.info("Gradium TTS: First chunk ready (can start playback now!)")
                                chunk_count += 1
                                yield sub
                else:
                    chunk_count += 1
                    yield chunk

            # Edge case: entire audio was silent
            if is_leading and leading_buffer:
                tracker.mark("tts_first_chunk_ready")
                logger.info("Gradium TTS: First chunk ready (can start playback now!)")
                for i in range(0, len(leading_buffer), 1024):
                    sub = bytes(leading_buffer[i : i + 1024])
                    if sub:
                        chunk_count += 1
                        yield sub

            tracker.mark("tts_api_complete")

            if self.cost_per_1m_chars > 0:
                call_cost = len(text) * self.cost_per_1m_chars / 1e6
                self.last_cost += call_cost
                logger.info(f"TTS Cost: ${call_cost:.6f} ({len(text)} chars, model={self.model})")

            logger.info(f"Gradium TTS: Synthesis complete - {chunk_count} chunks for '{text[:50]}...'")

        except Exception as e:
            logger.error(f"Gradium TTS synthesis failed: {e}")
            raise
