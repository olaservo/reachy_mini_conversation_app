"""Voxtral-MLX ASR provider using Mistral's Voxtral Mini Realtime model.

Uses mlx_audio's Voxtral support for local Apple Silicon ASR.
Audio is accumulated and periodically re-transcribed (like Parakeet progressive),
since Voxtral processes audio in a single encoder pass without chunked decoding.
"""

from __future__ import annotations
import time
import logging
from typing import Any, Optional

import numpy as np
import mlx.core as mx
import numpy.typing as npt

from .audio_utils import wav_to_float32
from .base_streaming import StreamingASRProvider


logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class VoxtralMLXASR(StreamingASRProvider):
    """Streaming ASR using Voxtral Mini Realtime via mlx_audio.

    Accumulates audio and re-runs model.generate() periodically for partials.
    Final transcript uses the full concatenated audio for best accuracy.
    """

    def __init__(
        self,
        model: str = "mlx-community/Voxtral-Mini-4B-Realtime-6bit",
        language: str = "en",
        max_tokens: int = 256,
        transcription_delay_ms: int = 480,
    ):
        """Initialize Voxtral ASR with mlx_audio model."""
        self.model_name = model
        self.language = language
        self.max_tokens = max_tokens
        self.target_sample_rate = SAMPLE_RATE
        self._min_new_samples = int(SAMPLE_RATE * transcription_delay_ms / 1000)

        # Load model
        logger.info(f"Loading Voxtral model via mlx_audio: {model}")
        from mlx_audio.stt.utils import load_model

        self._model: Any = load_model(model)
        logger.info("Voxtral model loaded")

        # Warmup with 1s silence
        logger.info("Warming up Voxtral model...")
        t0 = time.perf_counter()
        silence = mx.zeros(SAMPLE_RATE, dtype=mx.float32)
        self._model.generate(silence, language=self.language, max_tokens=16)
        logger.info(f"Warmup done in {(time.perf_counter() - t0) * 1000:.0f}ms")

        # Streaming state
        self._audio_buffer: list[npt.NDArray[np.float32]] = []
        self._total_samples = 0
        self._last_transcribed_length = 0
        self._last_partial: str | None = None

    # -- StreamingASRProvider interface --

    async def start_stream(self) -> None:
        """Reset state for new streaming session."""
        self._audio_buffer = []
        self._total_samples = 0
        self._last_transcribed_length = 0
        self._last_partial = None
        logger.info("Voxtral ASR stream started")

    async def send_audio_chunk(self, audio_chunk: bytes) -> None:
        """Accumulate audio and run transcription if enough new audio."""
        audio_np = wav_to_float32(audio_chunk, self.target_sample_rate)
        if len(audio_np) == 0:
            return

        self._audio_buffer.append(audio_np)
        self._total_samples += len(audio_np)

        new_samples = self._total_samples - self._last_transcribed_length
        if new_samples < self._min_new_samples:
            return

        # Need at least 500ms of audio
        if self._total_samples < SAMPLE_RATE // 2:
            return

        self._last_transcribed_length = self._total_samples
        full_audio = np.concatenate(self._audio_buffer)
        audio_mx = mx.array(full_audio, dtype=mx.float32)
        result = self._model.generate(
            audio_mx, language=self.language, max_tokens=self.max_tokens,
        )
        text = result.text.strip()
        self._last_partial = text or None

    async def get_partial_transcript(self) -> Optional[str]:
        """Return latest partial transcript."""
        return self._last_partial

    async def end_stream(self) -> str:
        """Finalize: re-transcribe full audio for maximum accuracy."""
        if self._total_samples == 0:
            return ""

        last_partial = self._last_partial or ""

        full_audio = np.concatenate(self._audio_buffer)
        audio_mx = mx.array(full_audio, dtype=mx.float32)
        result = self._model.generate(
            audio_mx, language=self.language, max_tokens=self.max_tokens,
        )
        transcript = result.text.strip()

        if not transcript and last_partial:
            logger.warning(
                f"Full transcription returned empty ({self._total_samples} samples), "
                f"falling back to last partial: '{last_partial}'"
            )
            transcript = last_partial

        # Reset
        self._audio_buffer = []
        self._total_samples = 0
        self._last_transcribed_length = 0
        self._last_partial = None

        logger.info(f"Voxtral ASR final: '{transcript}'")
        return transcript  # type: ignore[no-any-return]

    async def transcribe(self, audio_bytes: bytes, language: Optional[str] = None) -> str:
        """Batch fallback: single generate() call on full audio."""
        audio_np = wav_to_float32(audio_bytes, self.target_sample_rate)
        if len(audio_np) == 0:
            return ""
        audio_mx = mx.array(audio_np, dtype=mx.float32)
        lang = language or self.language
        result = self._model.generate(audio_mx, language=lang, max_tokens=self.max_tokens)
        return result.text.strip()  # type: ignore[no-any-return]
