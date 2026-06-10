"""Nemotron ASR provider (local, CUDA/MPS streaming).

Uses nvidia/nemotron-speech-streaming-en-0.6b model with:
- FastConformer encoder with cache-aware streaming
- RNNT decoder for autoregressive token generation
- Native punctuation and capitalization
"""

from __future__ import annotations
import logging
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from .audio_utils import wav_to_float32
from .base_streaming import StreamingASRProvider


logger = logging.getLogger(__name__)


class NemotronASR(StreamingASRProvider):
    """Nemotron ASR via NeMo (local, CUDA/MPS streaming).

    Cache-aware streaming with configurable latency via chunk_size_ms.
    Smaller chunks = lower latency but potentially lower accuracy.
    """

    def __init__(
            self,
            model: str = "nvidia/nemotron-speech-streaming-en-0.6b",
            chunk_size_ms: int = 560,
            sample_rate: int = 16000,
    ) -> None:
        """Initialize Nemotron ASR.

        Args:
            model: HuggingFace model name or local path
            chunk_size_ms: Chunk duration in ms (80, 160, 560, or 1120)
            sample_rate: Audio sample rate (must be 16000 for this model)

        """
        self.model_name = model
        self.chunk_size_ms = chunk_size_ms
        self.sample_rate = sample_rate

        # Calculate chunk size in samples
        self.chunk_size_samples = int(sample_rate * chunk_size_ms / 1000)

        # Model and cache state (lazy loaded)
        self._model: Any = None
        self._cache: Any = None
        self._is_initialized = False

        # Streaming state
        self._audio_buffer: list[npt.NDArray[Any]] = []
        self._partial_transcript = ""
        self._final_transcript = ""

        logger.info(
            f"NemotronASR configured (model={model}, "
            f"chunk_size={chunk_size_ms}ms, samples_per_chunk={self.chunk_size_samples})"
        )

    def _ensure_model_loaded(self) -> None:
        """Lazy load the model on first use."""
        if self._is_initialized:
            return

        logger.info(f"Loading Nemotron model: {self.model_name}")

        try:
            import torch
            import nemo.collections.asr as nemo_asr

            # Load the streaming model
            self._model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)

            # Configure for streaming
            # The model should already be configured for streaming, but we can
            # set chunk parameters if needed
            if hasattr(self._model, "setup_streaming"):
                self._model.setup_streaming(chunk_size=self.chunk_size_ms)

            # Move to appropriate device
            if torch.cuda.is_available():
                self._model = self._model.cuda()
                logger.info("Using CUDA for Nemotron ASR")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._model = self._model.to("mps")
                logger.info("Using MPS for Nemotron ASR")
            else:
                logger.info("Using CPU for Nemotron ASR")

            self._model.eval()
            self._is_initialized = True
            logger.info("Nemotron model loaded successfully")

        except ImportError as e:
            raise ImportError(
                f"NeMo toolkit not installed. Install with: "
                f"pip install 'nemo_toolkit[asr]>=2.0'\nError: {e}"
            )
        except Exception as e:
            logger.exception(f"Failed to load Nemotron model: {e}")
            raise

    async def start_stream(self) -> None:
        """Initialize streaming session."""
        # Ensure model is loaded
        self._ensure_model_loaded()

        # Reset state
        self._audio_buffer = []
        self._partial_transcript = ""
        self._final_transcript = ""

        assert self._model is not None

        # Reset cache for new streaming session
        if hasattr(self._model, "reset_cache"):
            self._model.reset_cache()

        self._cache = None
        logger.info("Nemotron streaming session started")

    async def send_audio_chunk(self, audio_chunk: bytes) -> None:
        """Send audio chunk for processing.

        Args:
            audio_chunk: Audio data (WAV format or raw PCM int16)

        """
        # Convert to numpy array
        audio_array = wav_to_float32(audio_chunk, self.sample_rate)
        if len(audio_array) == 0:
            return

        # Add to buffer
        self._audio_buffer.append(audio_array)

        # Process when we have enough samples for a chunk
        total_samples = sum(len(chunk) for chunk in self._audio_buffer)
        if total_samples >= self.chunk_size_samples:
            await self._process_buffer()

    async def get_partial_transcript(self) -> Optional[str]:
        """Get current partial transcript.

        Returns:
            Partial transcript or None

        """
        return self._partial_transcript if self._partial_transcript else None

    async def end_stream(self) -> str:
        """Finalize stream and get final transcript.

        Returns:
            Final complete transcript

        """
        # Process any remaining audio in buffer
        if self._audio_buffer:
            # Concatenate all remaining audio
            remaining_audio = np.concatenate(self._audio_buffer)
            self._audio_buffer = []

            # Final processing (flush cache)
            await self._process_audio_final(remaining_audio)

        # Combine partial and final transcripts
        transcript = self._final_transcript.strip()
        if not transcript and self._partial_transcript:
            transcript = self._partial_transcript.strip()

        logger.info(f"Nemotron final transcript: '{transcript}'")

        # Reset state
        self._partial_transcript = ""
        self._final_transcript = ""

        return transcript

    async def _process_buffer(self) -> None:
        """Process accumulated audio buffer."""
        if not self._audio_buffer:
            return

        # Concatenate buffer
        audio = np.concatenate(self._audio_buffer)
        self._audio_buffer = []

        # Process in chunks
        offset = 0
        while offset + self.chunk_size_samples <= len(audio):
            chunk = audio[offset : offset + self.chunk_size_samples]
            await self._process_chunk(chunk)
            offset += self.chunk_size_samples

        # Keep remaining samples for next batch
        if offset < len(audio):
            self._audio_buffer.append(audio[offset:])

    async def _process_chunk(self, audio_chunk: npt.NDArray[Any]) -> None:
        """Process a single audio chunk through the model.

        Args:
            audio_chunk: Audio samples (numpy array, float32)

        """
        import torch

        if self._model is None:
            return

        try:
            # Convert to tensor
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32) / 32768.0

            audio_tensor = torch.from_numpy(audio_chunk).unsqueeze(0)

            # Move to same device as model
            device = next(self._model.parameters()).device
            audio_tensor = audio_tensor.to(device)

            # Process through streaming model
            with torch.no_grad():
                if hasattr(self._model, "transcribe_stream"):
                    # Use streaming API if available
                    result, self._cache = self._model.transcribe_stream(
                        audio_tensor,
                        cache=self._cache,
                    )
                    if result:
                        self._partial_transcript = result
                        logger.debug(f"Nemotron partial: '{result}'")
                else:
                    # Fallback to regular transcription for batch
                    result = self._model.transcribe([audio_tensor], batch_size=1)
                    if result and len(result) > 0:
                        self._partial_transcript = result[0]

        except Exception as e:
            logger.warning(f"Error processing chunk: {e}")

    async def _process_audio_final(self, audio: npt.NDArray[Any]) -> None:
        """Process final audio with cache flush.

        Args:
            audio: Remaining audio samples

        """
        import torch

        if self._model is None:
            return

        try:
            # Convert to tensor
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32) / 32768.0

            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            device = next(self._model.parameters()).device
            audio_tensor = audio_tensor.to(device)

            with torch.no_grad():
                if hasattr(self._model, "transcribe_stream"):
                    # Final call with flush=True if supported
                    result, _ = self._model.transcribe_stream(
                        audio_tensor,
                        cache=self._cache,
                        flush=True,
                    )
                    if result:
                        self._final_transcript = result
                else:
                    # Batch fallback
                    result = self._model.transcribe([audio_tensor], batch_size=1)
                    if result and len(result) > 0:
                        self._final_transcript = result[0]

        except Exception as e:
            logger.warning(f"Error in final processing: {e}")
            # Use partial as final if final processing fails
            self._final_transcript = self._partial_transcript
