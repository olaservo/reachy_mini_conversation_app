"""Parakeet-MLX progressive ASR provider (Apple Silicon).

Thin subclass of ProgressiveASRBase that uses mlx_audio's decode_chunk()
for inference. All sliding window logic lives in the base class.

Adapted from Andi Marafioti's work on https://github.com/huggingface/speech-to-speech#
"""

from __future__ import annotations
import logging
from typing import Any

import numpy as np
import mlx.core as mx
import numpy.typing as npt

from .progressive_base import DecodeResult, SentenceSegment, ProgressiveASRBase


logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class ParakeetMLXProgressiveASR(ProgressiveASRBase):
    """Progressive ASR using mlx_audio's Parakeet model on Apple Silicon."""

    def __init__(
        self,
        model: str = "mlx-community/parakeet-tdt-0.6b-v3",
        precision: str = "float16",
        max_window_size: float = 15.0,
        sentence_buffer: float = 2.0,
    ) -> None:
        """Initialize progressive ASR with mlx_audio model."""
        self.model_name = model
        self.precision = precision

        logger.info(f"Loading Parakeet model via mlx_audio: {model}")
        from mlx_audio.stt.generate import load_model

        self._model: Any = load_model(model)
        logger.info("Parakeet model loaded")

        super().__init__(max_window_size=max_window_size, sentence_buffer=sentence_buffer)
        self._do_warmup()

    def _decode(self, audio_np: npt.NDArray[np.float32]) -> DecodeResult:
        """Run mlx_audio decode_chunk and wrap into DecodeResult."""
        audio_mx = mx.array(audio_np, dtype=mx.float32)
        result = self._model.decode_chunk(audio_mx, verbose=False)
        return DecodeResult(
            text=result.text.strip(),
            sentences=[SentenceSegment(s.text, s.end) for s in result.sentences],
        )

    def _decode_full(self, audio_np: npt.NDArray[np.float32]) -> str:
        """Full-context decode for final transcription."""
        audio_mx = mx.array(audio_np, dtype=mx.float32)
        result = self._model.decode_chunk(audio_mx, verbose=False)
        return result.text.strip()  # type: ignore[no-any-return]

    def _warmup(self) -> None:
        """Transcribe 1 second of silence."""
        silence = mx.zeros(SAMPLE_RATE, dtype=mx.float32)
        self._model.decode_chunk(silence, verbose=False)
