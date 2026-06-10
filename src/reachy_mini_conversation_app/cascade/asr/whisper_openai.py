"""Whisper ASR via OpenAI API."""

from __future__ import annotations
import io
import wave
import asyncio
import logging
from typing import Optional

from openai import OpenAI

from .base import ASRProvider


logger = logging.getLogger(__name__)


class WhisperOpenAIASR(ASRProvider):
    """Whisper ASR via OpenAI API (cloud, batch)."""

    def __init__(
        self,
        api_key: str,
        model: str = "whisper-1",
        cost_per_second: float = 0.0,
    ):
        """Initialize Whisper OpenAI ASR.

        Args:
            api_key: OpenAI API key
            model: Whisper model name (default: whisper-1)
            cost_per_second: Cost per second of audio (from cascade.yaml)

        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.cost_per_second = cost_per_second
        self.last_cost: float = 0.0
        logger.info(f"Initialized Whisper OpenAI ASR with model: {model}")

    def _estimate_audio_duration(self, audio_bytes: bytes) -> float:
        """Estimate duration from WAV bytes."""
        try:
            audio_file = io.BytesIO(audio_bytes)
            with wave.open(audio_file, "rb") as wav:
                return wav.getnframes() / wav.getframerate()
        except Exception:
            # Fallback: estimate from bytes (assume 16kHz mono 16-bit PCM)
            return len(audio_bytes) / (16000 * 2)

    async def transcribe(self, audio_bytes: bytes, language: Optional[str] = None) -> str:
        """Transcribe audio using OpenAI Whisper API.

        Args:
            audio_bytes: Audio data in WAV or other supported format
            language: Optional language code for transcription

        Returns:
            Transcribed text

        """
        logger.debug(f"Transcribing {len(audio_bytes)} bytes of audio")

        try:
            # Estimate audio duration for cost tracking
            duration = self._estimate_audio_duration(audio_bytes)

            # Create a file-like object from bytes
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.wav"  # Whisper API needs a filename

            # Run synchronous OpenAI call in thread pool
            def _transcribe() -> str:
                if language:
                    resp = self.client.audio.transcriptions.create(
                        model=self.model, file=audio_file, language=language
                    )
                else:
                    resp = self.client.audio.transcriptions.create(model=self.model, file=audio_file)
                return resp.text.strip()

            transcript = await asyncio.to_thread(_transcribe)

            # Calculate cost after successful transcription
            if self.cost_per_second > 0:
                self.last_cost = duration * self.cost_per_second
                logger.info(f"ASR Cost: ${self.last_cost:.6f} ({duration:.1f}s audio)")

            logger.info(f"Transcription successful: {transcript[:100]}...")
            return transcript

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
