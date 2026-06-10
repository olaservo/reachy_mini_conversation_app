"""TTS (Text-to-Speech) abstraction for cascade pipeline."""

from __future__ import annotations
import abc
from typing import Optional, AsyncIterator


class TTSProvider(abc.ABC):
    """Abstract base class for TTS providers."""

    @property
    def sample_rate(self) -> int:
        """Audio sample rate in Hz. Override for non-24kHz providers."""
        return 24000

    @abc.abstractmethod
    def synthesize(self, text: str, voice: Optional[str] = None) -> AsyncIterator[bytes]:
        """Synthesize text to audio stream.

        Args:
            text: Text to synthesize
            voice: Optional voice identifier

        Yields:
            Audio bytes (PCM or other format depending on provider)

        """
        raise NotImplementedError
