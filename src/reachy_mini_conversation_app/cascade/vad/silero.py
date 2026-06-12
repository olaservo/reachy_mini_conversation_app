"""Voice Activity Detection using Silero VAD.

Silero VAD is a pre-trained ML model for detecting speech in audio.
It works with 16kHz audio and returns speech probability for each chunk.
"""

import logging
from enum import Enum, auto
from typing import Any

import numpy as np
import torch
import numpy.typing as npt


logger = logging.getLogger(__name__)

# Silero VAD requires specific sample rates and chunk sizes
SILERO_SAMPLE_RATE = 16000

# Minimum chunk size for Silero VAD (512 samples = 32ms at 16kHz)
VAD_CHUNK_SIZE = 512


class SileroVAD:
    """Voice Activity Detection using Silero VAD model.

    The model processes audio chunks and returns speech probability.
    It maintains internal state for streaming audio processing.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 500,
    ) -> None:
        """Initialize Silero VAD.

        Args:
            threshold: Speech probability threshold (0.0-1.0). Higher = stricter.
            min_speech_duration_ms: Minimum speech duration to trigger detection.
            min_silence_duration_ms: Minimum silence duration to end speech segment.

        """
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms

        # Load Silero VAD model
        logger.info("Loading Silero VAD model...")
        self.model, _utils = torch.hub.load(  # type: ignore[no-untyped-call]
            "snakers4/silero-vad",
            "silero_vad",
            trust_repo=True,
        )
        self.model.eval()

        # State tracking for streaming
        self._speech_frames: float = 0
        self._silence_frames: float = 0
        self._is_speaking = False
        self._prob_log_count = 0

        logger.info(f"Silero VAD initialized (threshold={threshold})")

    def get_speech_prob(self, audio: npt.NDArray[Any], sample_rate: int = SILERO_SAMPLE_RATE) -> float:
        """Get speech probability for an audio chunk.

        Args:
            audio: Audio samples as int16 or float32 numpy array.
            sample_rate: Sample rate of the audio (must be 16000).

        Returns:
            Speech probability between 0.0 and 1.0.

        """
        if sample_rate != SILERO_SAMPLE_RATE:
            raise ValueError(f"Silero VAD requires {SILERO_SAMPLE_RATE}Hz audio, got {sample_rate}Hz")

        # Convert to float32 tensor normalized to [-1, 1]
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.float32:
            audio_float = audio
        else:
            audio_float = audio.astype(np.float32)

        tensor = torch.from_numpy(audio_float)

        # Run VAD model
        with torch.no_grad():
            speech_prob = self.model(tensor, sample_rate).item()

        return speech_prob  # type: ignore[no-any-return]

    def is_speech(self, audio: npt.NDArray[Any], sample_rate: int = SILERO_SAMPLE_RATE) -> bool:
        """Check if audio chunk contains speech above threshold.

        Args:
            audio: Audio samples as numpy array.
            sample_rate: Sample rate of the audio.

        Returns:
            True if speech probability exceeds threshold.

        """
        prob = self.get_speech_prob(audio, sample_rate)
        return prob >= self.threshold

    def process_chunk(self, audio: npt.NDArray[Any], sample_rate: int = SILERO_SAMPLE_RATE) -> tuple[bool, bool]:
        """Process audio chunk with hysteresis for robust speech detection.

        Implements state machine with minimum duration requirements:
        - Speech must persist for min_speech_duration_ms to trigger start
        - Silence must persist for min_silence_duration_ms to trigger end

        Args:
            audio: Audio samples as numpy array.
            sample_rate: Sample rate of the audio.

        Returns:
            Tuple of (speech_started, speech_ended) events.

        """
        chunk_duration_ms = len(audio) / sample_rate * 1000
        prob = self.get_speech_prob(audio, sample_rate)
        self._prob_log_count += 1
        if self._prob_log_count % 50 == 0:
            logger.debug(f"VAD prob={prob:.3f} (threshold={self.threshold}, speaking={self._is_speaking})")
        is_speech = prob >= self.threshold

        speech_started = False
        speech_ended = False

        if is_speech:
            self._silence_frames = 0
            self._speech_frames += chunk_duration_ms

            # Trigger speech start after minimum duration
            if not self._is_speaking and self._speech_frames >= self.min_speech_duration_ms:
                self._is_speaking = True
                speech_started = True
                logger.debug(f"Speech started (accumulated {self._speech_frames:.0f}ms)")
        else:
            self._speech_frames = 0
            if self._is_speaking:
                self._silence_frames += chunk_duration_ms

                # Trigger speech end after minimum silence duration
                if self._silence_frames >= self.min_silence_duration_ms:
                    self._is_speaking = False
                    speech_ended = True
                    logger.debug(f"Speech ended (silence {self._silence_frames:.0f}ms)")

        return speech_started, speech_ended

    @property
    def is_speaking(self) -> bool:
        """Returns True if currently in a speech segment."""
        return self._is_speaking

    def reset(self) -> None:
        """Reset VAD state between sessions."""
        self._speech_frames = 0.0
        self._silence_frames = 0.0
        self._is_speaking = False
        self._prob_log_count = 0
        self.model.reset_states()
        logger.debug("VAD state reset")


class VADEvent(Enum):
    """Events returned by VADStateMachine.process_chunk()."""

    NOTHING = auto()
    SPEECH_STARTED = auto()
    SPEECH_ENDED = auto()


class VADState(Enum):
    """States of the VAD state machine."""

    LISTENING = "listening"
    RECORDING = "recording"
    PROCESSING = "processing"


class VADStateMachine:
    """Shared VAD state machine for pre-roll buffering and speech detection.

    Callers feed audio chunks and react to returned events.
    Does NOT own audio sources, callbacks, or async/threading.
    """

    def __init__(
        self,
        vad: SileroVAD,
        chunk_size: int = VAD_CHUNK_SIZE,
        preroll_duration_s: float = 0.5,
    ) -> None:
        """Initialize the VAD state machine."""
        self._vad = vad
        self._state = VADState.LISTENING
        self._chunk_size = chunk_size

        # Pre-roll buffer: keep ~preroll_duration_s of audio before speech triggers
        self._max_preroll = int(preroll_duration_s * SILERO_SAMPLE_RATE / chunk_size)
        self._preroll_chunks: list[npt.NDArray[np.int16]] = []

        # All speech frames including pre-roll (readable after SPEECH_STARTED/ENDED)
        self.speech_chunks: list[npt.NDArray[np.int16]] = []

    @property
    def state(self) -> VADState:
        """Current state of the VAD state machine."""
        return self._state

    def process_chunk(self, audio_chunk: npt.NDArray[np.int16]) -> VADEvent:
        """Feed one audio chunk and get back an event."""
        if self._state == VADState.PROCESSING:
            return VADEvent.NOTHING

        if self._state == VADState.LISTENING:
            speech_started, _ = self._vad.process_chunk(audio_chunk, SILERO_SAMPLE_RATE)

            # Always buffer for pre-roll
            self._preroll_chunks.append(audio_chunk)
            if len(self._preroll_chunks) > self._max_preroll:
                self._preroll_chunks = self._preroll_chunks[-self._max_preroll :]

            if speech_started:
                self._state = VADState.RECORDING
                self.speech_chunks = list(self._preroll_chunks)
                self._preroll_chunks = []
                logger.info("Speech detected, recording...")
                return VADEvent.SPEECH_STARTED

            return VADEvent.NOTHING

        # RECORDING
        self.speech_chunks.append(audio_chunk)
        _, speech_ended = self._vad.process_chunk(audio_chunk, SILERO_SAMPLE_RATE)
        if speech_ended:
            self._state = VADState.PROCESSING
            logger.info(f"Speech ended, {len(self.speech_chunks)} chunks")
            return VADEvent.SPEECH_ENDED

        return VADEvent.NOTHING

    def finish_processing(self) -> None:
        """Reset buffers and transition back to LISTENING."""
        self.speech_chunks = []
        self._preroll_chunks = []
        self._vad.reset()
        self._state = VADState.LISTENING

