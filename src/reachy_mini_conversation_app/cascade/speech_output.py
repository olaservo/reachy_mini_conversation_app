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

    The fastrtc stream drains the queue via emit() and plays audio in the browser.
    In Gradio mode the samples are additionally tapped to the daemon so the head
    wobbler moves. Each spoken segment is also surfaced as an assistant chat message.
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
            frame = samples.reshape(1, -1)
            # Browser plays the audio via emit(); in Gradio mode the daemon never sees
            # it, so tap the same samples to drive the head wobbler (robot speaker muted).
            if handler.gradio_mode:
                handler._tap_audio_for_daemon_wobbler(frame)
            await handler.output_queue.put((sample_rate, frame))
