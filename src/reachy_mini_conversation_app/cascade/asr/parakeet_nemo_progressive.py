"""Parakeet-NeMo progressive ASR provider (CUDA).

Cross-platform alternative to ParakeetMLXProgressiveASR. Runs the original
NeMo Parakeet TDT model on NVIDIA GPUs via nemo_toolkit[asr].
Reuses the shared sliding window logic from ProgressiveASRBase.
"""

from __future__ import annotations
import re
import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from .progressive_base import DecodeResult, SentenceSegment, ProgressiveASRBase


logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Regex for sentence-ending punctuation
_SENTENCE_END_RE = re.compile(r"[.!?]$")


class ParakeetNeMoProgressiveASR(ProgressiveASRBase):
    """Progressive ASR using NeMo Parakeet TDT on CUDA.

    Loads nvidia/parakeet-tdt-0.6b-v2 (or compatible) and builds
    SentenceSegment boundaries from word-level timestamps.
    """

    def __init__(
        self,
        model: str = "nvidia/parakeet-tdt-0.6b-v2",
        max_window_size: float = 15.0,
        sentence_buffer: float = 2.0,
    ) -> None:
        """Initialize Parakeet NeMo progressive ASR."""
        self.model_name = model
        self._model: Any = None

        logger.info(f"Loading Parakeet NeMo model: {model}")

        import torch
        import nemo.collections.asr as nemo_asr

        self._model = nemo_asr.models.ASRModel.from_pretrained(model)

        if not torch.cuda.is_available():
            raise RuntimeError("ParakeetNeMoProgressiveASR requires CUDA. No CUDA device found.")

        self._model = self._model.cuda()
        self._model.eval()
        logger.info("Parakeet NeMo model loaded on CUDA")

        super().__init__(max_window_size=max_window_size, sentence_buffer=sentence_buffer)
        self._do_warmup()

    def _decode(self, audio_np: npt.NDArray[np.float32]) -> DecodeResult:
        """Transcribe audio and return text with sentence segments."""
        import torch

        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).cuda()
        audio_len = torch.tensor([len(audio_np)], dtype=torch.int32).cuda()

        with torch.no_grad():
            hypotheses = self._model.transcribe(
                audio_tensor,
                lengths=audio_len,
                return_hypotheses=True,
                batch_size=1,
            )

        # NeMo returns list of hypotheses (one per batch item)
        if not hypotheses or len(hypotheses) == 0:
            return DecodeResult(text="")

        hyp = hypotheses[0]

        # Handle NeMo's nested list return format
        if isinstance(hyp, list):
            hyp = hyp[0] if hyp else None
        if hyp is None:
            return DecodeResult(text="")

        text = hyp.text if hasattr(hyp, "text") else str(hyp)
        text = text.strip()

        if not text:
            return DecodeResult(text="")

        # Try to extract word-level timestamps for sentence segmentation
        sentences = _build_sentence_segments(hyp, len(audio_np) / SAMPLE_RATE)
        return DecodeResult(text=text, sentences=sentences)

    def _decode_full(self, audio_np: npt.NDArray[np.float32]) -> str:
        """Full-context decode for final transcription."""
        result = self._decode(audio_np)
        return result.text

    def _warmup(self) -> None:
        """Transcribe 1 second of silence to warm up."""
        import torch

        silence = torch.zeros(SAMPLE_RATE, dtype=torch.float32).unsqueeze(0).cuda()
        silence_len = torch.tensor([SAMPLE_RATE], dtype=torch.int32).cuda()
        with torch.no_grad():
            self._model.transcribe(silence, lengths=silence_len, batch_size=1)


def _build_sentence_segments(hyp: Any, audio_duration: float) -> list[SentenceSegment]:
    """Build sentence segments from a NeMo hypothesis.

    Tries word-level timestamps first (TDT models produce these natively).
    Falls back to proportional timing if timestamps aren't available.
    """
    text = hyp.text.strip() if hasattr(hyp, "text") else str(hyp).strip()
    if not text:
        return []

    # Try to get word timestamps from the hypothesis
    words_with_times = _extract_word_timestamps(hyp, audio_duration)

    if words_with_times:
        return _segment_from_word_timestamps(words_with_times)

    # Fallback: proportional timing based on character position
    return _segment_proportional(text, audio_duration)


def _extract_word_timestamps(hyp: Any, audio_duration: float) -> list[tuple[str, float]]:
    """Extract (word, end_time) pairs from a NeMo hypothesis.

    NeMo TDT models store word-level timing in various places depending on version.
    """
    # Try timestep attribute (NeMo 2.x with return_hypotheses=True)
    if hasattr(hyp, "timestep") and hyp.timestep is not None:
        timestep = hyp.timestep
        # timestep may have 'word' level info
        if isinstance(timestep, dict) and "word" in timestep:
            words_info = timestep["word"]
            result = []
            for w in words_info:
                word = w.get("word", w.get("char", ""))
                end = w.get("end_offset", w.get("end", 0.0))
                if word:
                    result.append((word, float(end)))
            if result:
                return result

    # Try word_timestamps attribute
    if hasattr(hyp, "word_timestamps") and hyp.word_timestamps:
        result = []
        for w in hyp.word_timestamps:
            word = w.get("word", "")
            end = w.get("end", w.get("end_time", 0.0))
            if word:
                result.append((word, float(end)))
        if result:
            return result

    # Try words attribute (some NeMo versions)
    if hasattr(hyp, "words") and hyp.words:
        result = []
        for w in hyp.words:
            if hasattr(w, "word") and hasattr(w, "end"):
                result.append((w.word, float(w.end)))
            elif isinstance(w, dict):
                word = w.get("word", "")
                end = w.get("end", 0.0)
                if word:
                    result.append((word, float(end)))
        if result:
            return result

    return []


def _segment_from_word_timestamps(words: list[tuple[str, float]]) -> list[SentenceSegment]:
    """Group words into sentences by splitting on sentence-ending punctuation."""
    segments: list[SentenceSegment] = []
    current_words: list[str] = []
    current_end = 0.0

    for word, end_time in words:
        current_words.append(word)
        current_end = end_time

        if _SENTENCE_END_RE.search(word):
            segments.append(SentenceSegment(
                text=" ".join(current_words),
                end=current_end,
            ))
            current_words = []

    # Remaining words form the last (possibly incomplete) sentence
    if current_words:
        segments.append(SentenceSegment(
            text=" ".join(current_words),
            end=current_end,
        ))

    return segments


def _segment_proportional(text: str, audio_duration: float) -> list[SentenceSegment]:
    """Split text into sentences and assign proportional timestamps."""
    # Split on sentence boundaries
    raw_sentences = re.split(r"(?<=[.!?])\s+", text)
    if not raw_sentences:
        return [SentenceSegment(text=text, end=audio_duration)]

    total_chars = sum(len(s) for s in raw_sentences)
    if total_chars == 0:
        return [SentenceSegment(text=text, end=audio_duration)]

    segments: list[SentenceSegment] = []
    cumulative_chars = 0

    for sentence in raw_sentences:
        cumulative_chars += len(sentence)
        end_time = (cumulative_chars / total_chars) * audio_duration
        segments.append(SentenceSegment(text=sentence, end=end_time))

    return segments
