"""Shared sliding window logic for progressive ASR providers.

Extracted from ParakeetMLXProgressiveASR so that multiple backends (MLX, NeMo)
can reuse the sentence-aware sliding window without duplicating ~150 lines.
"""

from __future__ import annotations
import abc
import time
import logging
from typing import Optional
from dataclasses import field, dataclass

import numpy as np
import numpy.typing as npt

from .audio_utils import wav_to_float32
from .base_streaming import StreamingASRProvider


logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


@dataclass
class SentenceSegment:
    """A sentence with its end timestamp (seconds from window start)."""

    text: str
    end: float


@dataclass
class DecodeResult:
    """Result of a single decode call — full text plus sentence segments."""

    text: str
    sentences: list[SentenceSegment] = field(default_factory=list)


class ProgressiveASRBase(StreamingASRProvider):
    """Progressive ASR with sentence-aware sliding window.

    Subclasses only need to implement ``_decode()`` and ``_warmup()``.

    Strategy:
    1. Accumulate audio in a growing buffer
    2. Every ~250ms of new audio, re-transcribe from the last fixed sentence boundary
    3. When the window exceeds max_window_size, fix completed sentences and slide forward
    4. Fixed sentences never change; only the active tail is re-transcribed
    """

    def __init__(
        self,
        max_window_size: float = 15.0,
        sentence_buffer: float = 2.0,
    ) -> None:
        """Initialize sliding window state."""
        self.max_window_size = max_window_size
        self.sentence_buffer = sentence_buffer
        self.target_sample_rate = SAMPLE_RATE

        # Streaming state (initialized in start_stream)
        self._audio_buffer: list[npt.NDArray[np.float32]] = []
        self._total_samples = 0
        self._fixed_sentences: list[str] = []
        self._fixed_end_time = 0.0
        self._last_transcribed_length = 0
        self._last_partial: str | None = None
        self._best_partial: str = ""
        self._min_new_samples = int(SAMPLE_RATE * 0.25)  # 250ms between transcriptions

    # -- Abstract methods for backends --

    @abc.abstractmethod
    def _decode(self, audio_np: npt.NDArray[np.float32]) -> DecodeResult:
        """Run inference on an audio array and return text + sentence segments."""

    @abc.abstractmethod
    def _warmup(self) -> None:
        """Warm up the model (e.g. transcribe silence)."""

    @abc.abstractmethod
    def _decode_full(self, audio_np: npt.NDArray[np.float32]) -> str:
        """Decode the full audio for the final end-of-stream transcription.

        Backends may use a different strategy for the final pass (e.g. full-context
        decode without sliding window). Default implementation delegates to _decode().
        """

    # -- StreamingASRProvider interface --

    async def start_stream(self) -> None:
        """Reset state for new streaming session."""
        self._audio_buffer = []
        self._total_samples = 0
        self._fixed_sentences = []
        self._fixed_end_time = 0.0
        self._last_transcribed_length = 0
        self._last_partial = None
        self._best_partial = ""
        logger.info("Progressive ASR stream started")

    async def send_audio_chunk(self, audio_chunk: bytes) -> None:
        """Append audio and run incremental transcription if enough new audio."""
        audio_np = wav_to_float32(audio_chunk, self.target_sample_rate)
        if len(audio_np) == 0:
            return

        self._audio_buffer.append(audio_np)
        self._total_samples += len(audio_np)

        new_samples = self._total_samples - self._last_transcribed_length
        if new_samples < self._min_new_samples:
            return

        full_audio = np.concatenate(self._audio_buffer)
        fixed_text, active_text = self._transcribe_incremental(full_audio)

        parts = [p for p in (fixed_text, active_text) if p]
        if parts:
            self._last_partial = " ".join(parts)
            if len(self._last_partial) > len(self._best_partial):
                self._best_partial = self._last_partial

    async def get_partial_transcript(self) -> Optional[str]:
        """Return latest partial transcript."""
        return self._last_partial

    async def end_stream(self) -> str:
        """Finalize stream: re-transcribe full audio for maximum accuracy."""
        if self._total_samples == 0:
            return ""

        best_partial = self._best_partial

        full_audio = np.concatenate(self._audio_buffer)
        transcript = self._decode_full(full_audio)

        # Fall back to best partial if final is empty or much shorter
        if best_partial and len(transcript) < len(best_partial) * 0.5:
            logger.warning(
                f"Final transcript '{transcript}' is worse than best partial '{best_partial}', "
                f"using best partial"
            )
            transcript = best_partial

        # Reset
        self._audio_buffer = []
        self._total_samples = 0
        self._fixed_sentences = []
        self._fixed_end_time = 0.0
        self._last_transcribed_length = 0
        self._last_partial = None
        self._best_partial = ""

        logger.info(f"Progressive ASR final: '{transcript}'")
        return transcript

    async def transcribe(self, audio_bytes: bytes, language: Optional[str] = None) -> str:
        """Batch fallback: decode entire audio."""
        audio_np = wav_to_float32(audio_bytes, self.target_sample_rate)
        if len(audio_np) == 0:
            return ""
        return self._decode_full(audio_np)

    # -- Core sliding window logic --

    def _transcribe_incremental(self, audio: npt.NDArray[np.float32]) -> tuple[str, str]:
        """Transcribe incrementally with sentence-aware sliding window.

        Returns (fixed_text, active_text).
        """
        current_length = len(audio)

        if current_length < SAMPLE_RATE // 2:
            return " ".join(self._fixed_sentences), ""

        self._last_transcribed_length = current_length

        window_start = int(self._fixed_end_time * SAMPLE_RATE)
        audio_window = audio[window_start:]

        result = self._decode(audio_window)

        window_duration = len(audio_window) / SAMPLE_RATE
        if window_duration >= self.max_window_size and len(result.sentences) > 1:
            cutoff_time = window_duration - self.sentence_buffer
            new_fixed: list[str] = []
            new_fixed_end = self._fixed_end_time

            for sentence in result.sentences:
                if sentence.end < cutoff_time:
                    new_fixed.append(sentence.text.strip())
                    new_fixed_end = self._fixed_end_time + sentence.end
                else:
                    break

            if new_fixed:
                self._fixed_sentences.extend(new_fixed)
                self._fixed_end_time = new_fixed_end
                logger.info(f"Fixed {len(new_fixed)} sentences, window slides to {self._fixed_end_time:.1f}s")

                window_start = int(self._fixed_end_time * SAMPLE_RATE)
                audio_window = audio[window_start:]
                result = self._decode(audio_window)

        return " ".join(self._fixed_sentences), result.text.strip()

    def _do_warmup(self) -> None:
        """Run warmup with timing."""
        logger.info("Warming up model...")
        t0 = time.perf_counter()
        self._warmup()
        logger.info(f"Warmup done in {(time.perf_counter() - t0) * 1000:.0f}ms")
