"""SpeechOutput protocol and a queue-backed implementation for the cascade backend."""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Protocol

import numpy as np
from fastrtc import AdditionalOutputs


if TYPE_CHECKING:
    from reachy_mini_conversation_app.cascade.handler import CascadeHandler


logger = logging.getLogger(__name__)


class SpeechOutput(Protocol):
    """Protocol for TTS playback backends."""

    async def speak(self, text: str) -> None:
        """Synthesize and play speech."""
        ...


class QueueSpeechOutput:
    """Synthesize TTS and push audio frames onto the handler's output queue.

    The fastrtc stream drains the queue via emit() and plays audio through the
    robot speaker, which also drives the daemon head wobbler. Each spoken
    segment is additionally surfaced as an assistant chat message.
    """

    def __init__(self, handler: "CascadeHandler") -> None:
        """Bind to the owning handler (whose output_queue/tts are used)."""
        self.handler = handler

    async def speak(self, text: str) -> None:
        """Stream TTS audio for `text` onto the handler's output queue."""
        from reachy_mini_conversation_app.cascade.timing import tracker

        if not text.strip():
            return

        handler = self.handler
        await handler.output_queue.put(AdditionalOutputs({"role": "assistant", "content": text}))

        sample_rate = handler.tts.sample_rate
        first_chunk = True
        async for chunk in handler.tts.synthesize(text, voice=handler._voice):
            samples = np.frombuffer(chunk, dtype=np.int16)
            if samples.size == 0:
                continue
            if first_chunk:
                tracker.mark("audio_playback_started")
                first_chunk = False
            await handler.output_queue.put((sample_rate, samples.reshape(1, -1)))
