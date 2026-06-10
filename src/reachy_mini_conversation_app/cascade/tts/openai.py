"""OpenAI TTS implementation."""

from __future__ import annotations
import logging
from typing import Literal, Optional, AsyncIterator

from openai import AsyncOpenAI

from .base import TTSProvider
from .utils import trim_leading_silence


AudioFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


logger = logging.getLogger(__name__)


class OpenAITTS(TTSProvider):
    """OpenAI TTS API implementation."""

    def __init__(
        self,
        api_key: str,
        model: str = "tts-1",
        voice: str = "alloy",
        response_format: AudioFormat = "pcm",
        cost_per_1m_chars: float = 0.0,
    ):
        """Initialize OpenAI TTS.

        Args:
            api_key: OpenAI API key
            model: TTS model (tts-1 or tts-1-hd)
                   Note: tts-1 has lower latency than tts-1-hd
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            response_format: Audio format (pcm, mp3, opus, aac, flac, wav)
            cost_per_1m_chars: Cost per 1M characters (from cascade.yaml)

        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.default_voice = voice
        self.response_format = response_format
        self.cost_per_1m_chars = cost_per_1m_chars
        self.last_cost: float = 0.0
        logger.info(f"Initialized OpenAI TTS with model: {model}, voice: {voice} (format: {response_format})")

    async def synthesize(self, text: str, voice: Optional[str] = None) -> AsyncIterator[bytes]:
        """Synthesize text using OpenAI TTS API with true streaming.

        Trims leading silence on the first chunk only, then yields subsequent
        chunks immediately as they arrive from the API.

        Yields:
            Audio bytes (PCM 16-bit)

        """
        from reachy_mini_conversation_app.cascade.timing import tracker

        if not text.strip():
            logger.warning("Empty text provided for synthesis")
            return

        voice_to_use = voice or self.default_voice
        logger.info(f"TTS: Starting synthesis for '{text[:50]}...'")

        tracker.mark("tts_start", {"text_len": len(text)})

        try:
            import time

            import numpy as np

            SILENCE_THRESHOLD = 327  # ~0.01 * 32767, matches trim_leading_silence default

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
            ) as response:
                async for chunk in response.iter_bytes(chunk_size=1024):
                    if not chunk:
                        continue

                    if first_byte:
                        ttfb_ms = (time.perf_counter() - request_start) * 1000
                        tracker.mark("tts_api_first_byte", {"ttfb_ms": round(ttfb_ms, 1)})
                        first_byte = False

                    if is_leading:
                        leading_buffer.extend(chunk)
                        # Check if this chunk contains non-silent audio
                        samples = np.frombuffer(chunk, dtype=np.int16)
                        if np.any(np.abs(samples) > SILENCE_THRESHOLD):
                            # Found audio — trim accumulated buffer and start yielding
                            is_leading = False
                            full_buffer = np.frombuffer(bytes(leading_buffer), dtype=np.int16)
                            trimmed = trim_leading_silence(
                                full_buffer, sample_rate=self.sample_rate, provider_name="OpenAI TTS"
                            )
                            tracker.mark("tts_first_chunk_ready")
                            trimmed_bytes = trimmed.tobytes()
                            for i in range(0, len(trimmed_bytes), 1024):
                                sub = trimmed_bytes[i : i + 1024]
                                if sub:
                                    if chunk_count == 0:
                                        logger.info("TTS: First chunk ready (can start playback now!)")
                                    chunk_count += 1
                                    yield sub
                    else:
                        # Past leading silence — stream directly
                        chunk_count += 1
                        yield chunk

            # Edge case: entire audio was silent (yield it unchanged)
            if is_leading and leading_buffer:
                tracker.mark("tts_first_chunk_ready")
                logger.info("TTS: First chunk ready (can start playback now!)")
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

            logger.info(f"TTS: Synthesis complete - {chunk_count} chunks for '{text[:50]}...'")

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise
