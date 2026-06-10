"""OpenAI Realtime ASR provider (cloud streaming via WebSocket).

## How It Works

Audio chunks are sent continuously to OpenAI's Realtime API. Transcription only
starts AFTER the audio buffer is committed (either by server VAD detecting silence,
or manually via input_audio_buffer.commit). Once committed, transcription deltas
stream out rapidly (~200ms for full transcript).

This is "streaming transcription" in the sense that TEXT streams out quickly after
commit, NOT that you get partial transcripts while still speaking.

## Server VAD Behavior

When `use_server_vad=True`:
- Server detects speech boundaries and auto-commits on silence (500ms default)
- Transcription deltas stream after each commit
- Can coexist with local VAD (Silero) - server VAD streams partials, local VAD
  controls when end_stream() is called

When `use_server_vad=False`:
- Audio buffers until end_stream() calls input_audio_buffer.commit
- All transcription happens at the end

## Known Limitations

1. **Connection latency**: ~800-1000ms to establish WebSocket after speech starts.
   Audio recorded during this time may overflow/be lost.

2. **No true real-time partials**: Partial transcripts only appear after audio is
   committed, not while user is still speaking. For true real-time partials during
   speech, would need periodic mid-speech commits (causing fragmented transcripts).

3. **Potential fix**: Pre-warm WebSocket connection before speech starts (keep
   connection open in standby mode). Not yet implemented.
"""

from __future__ import annotations
import json
import base64
import asyncio
import logging
from typing import Any, Optional

from .audio_utils import wav_to_pcm_int16
from .base_streaming import StreamingASRProvider


logger = logging.getLogger(__name__)


class OpenAIRealtimeASR(StreamingASRProvider):
    """OpenAI Realtime ASR via WebSocket streaming (cloud).

    For true real-time streaming, audio should be sent continuously as captured
    (not buffered). Server VAD will detect speech boundaries and stream partial
    transcripts during speech.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-transcribe",
        language: str = "en",
        prompt: str = "",
        use_server_vad: bool = True,
    ):
        """Initialize OpenAI Realtime ASR.

        Args:
            api_key: OpenAI API key
            model: Transcription model (gpt-4o-transcribe or gpt-4o-mini-transcribe)
            language: Language code (e.g., 'en', 'fr')
            prompt: Optional prompt to guide transcription style/vocabulary
            use_server_vad: If True (default), use OpenAI's server-side VAD for real-time
                           partial transcripts during speech. Both server and local VAD
                           can coexist - server VAD streams partials, local VAD controls
                           when end_stream() is called.

        """
        self.api_key = api_key
        self.model = model
        self.language = language
        self.prompt = prompt
        self.use_server_vad = use_server_vad

        # WebSocket state
        self.ws: Any = None
        self.listener_task: Optional[asyncio.Task[None]] = None
        self.completed_segments: list[str] = []
        self.current_partial = ""
        self.error_message: Optional[str] = None
        self.session_ready = asyncio.Event()
        self.got_final = asyncio.Event()

        # Import websockets
        try:
            import websockets

            self.websockets = websockets
        except ImportError:
            raise ImportError("websockets not installed. Install with: pip install websockets")

        logger.info(f"OpenAI Realtime ASR initialized (model={model}, language={language})")

    async def start_stream(self) -> None:
        """Initialize streaming session with OpenAI Realtime API."""
        # Reset state
        self.completed_segments = []
        self.current_partial = ""
        self.error_message = None
        self.session_ready = asyncio.Event()
        self.got_final = asyncio.Event()

        # Connect to OpenAI Realtime API
        url = "wss://api.openai.com/v1/realtime?intent=transcription"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        try:
            self.ws = await self.websockets.connect(url, additional_headers=headers)
            logger.info("OpenAI Realtime WebSocket connection established")

            # Start listener task
            self.listener_task = asyncio.create_task(self._listen_for_events())

            # Configure transcription session
            # Build turn_detection config based on use_server_vad setting
            if self.use_server_vad:
                # Server VAD commits audio automatically when speech ends
                turn_detection = {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                }
            else:
                # Manual mode: audio committed via input_audio_buffer.commit in end_stream()
                # This is better when using local VAD (Silero) to avoid conflicts
                turn_detection = None

            # Build transcription config (omit prompt if empty)
            transcription_config: dict[str, Any] = {
                "model": self.model,
                "language": self.language,
            }
            if self.prompt:
                transcription_config["prompt"] = self.prompt

            session_config = {
                "type": "transcription_session.update",
                "session": {
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": transcription_config,
                    "turn_detection": turn_detection,
                },
            }
            await self.ws.send(json.dumps(session_config))
            logger.info(f"OpenAI Realtime session configured: model={self.model}, server_vad={self.use_server_vad}")

            # Wait for session to be ready
            try:
                await asyncio.wait_for(self.session_ready.wait(), timeout=2.0)
                logger.debug("Session ready, can send audio")
            except asyncio.TimeoutError:
                logger.warning("Session ready timeout, proceeding anyway")

        except Exception as e:
            logger.exception(f"Failed to start OpenAI Realtime stream: {e}")
            raise

    async def send_audio_chunk(self, audio_chunk: bytes) -> None:
        """Send audio chunk to OpenAI Realtime API.

        For real-time streaming, call this continuously as audio is captured,
        not after buffering. Server VAD will detect speech and trigger transcription.

        Args:
            audio_chunk: Audio data (WAV 16kHz format, will be resampled to 24kHz)

        """
        if not self.ws:
            logger.warning("OpenAI Realtime connection not established, skipping audio chunk")
            return

        try:
            # Convert WAV 16kHz to PCM 24kHz
            pcm_24k = wav_to_pcm_int16(audio_chunk, 24000)

            # Base64 encode
            audio_b64 = base64.b64encode(pcm_24k).decode("utf-8")

            # Send audio buffer append
            message = {
                "type": "input_audio_buffer.append",
                "audio": audio_b64,
            }
            await self.ws.send(json.dumps(message))

        except Exception as e:
            logger.warning(f"Failed to send audio chunk to OpenAI Realtime: {e}")

    async def get_partial_transcript(self) -> Optional[str]:
        """Get current partial transcript.

        Returns:
            Current partial transcript, or None if nothing available yet

        """
        parts = self.completed_segments + ([self.current_partial] if self.current_partial else [])
        if not parts:
            return None
        return " ".join(parts).strip()

    async def end_stream(self) -> str:
        """Finalize stream and get final transcript.

        Returns:
            Final complete transcript

        """
        try:
            if self.ws:
                # Clear event before committing
                self.got_final.clear()

                # Commit any remaining audio in buffer
                commit_message = {"type": "input_audio_buffer.commit"}
                await self.ws.send(json.dumps(commit_message))
                logger.debug("Sent input_audio_buffer.commit")

                # Wait for final transcript
                WAITING_TIME_SEC = 2.0
                try:
                    await asyncio.wait_for(self.got_final.wait(), timeout=WAITING_TIME_SEC)
                    logger.debug("Received final transcript after commit")
                except asyncio.TimeoutError:
                    logger.debug("Timeout waiting for final transcript")

                # Build final transcript
                parts = self.completed_segments + ([self.current_partial] if self.current_partial else [])
                transcript = " ".join(parts).strip()

                # Clean up in background
                ws_to_close = self.ws
                listener_to_cancel = self.listener_task
                self.ws = None
                self.listener_task = None

                asyncio.create_task(self._cleanup(ws_to_close, listener_to_cancel))

                logger.info(f"OpenAI Realtime transcript: '{transcript}' (segments: {len(self.completed_segments)})")
                return transcript
            else:
                parts = self.completed_segments + ([self.current_partial] if self.current_partial else [])
                return " ".join(parts).strip()

        except Exception as e:
            logger.exception(f"Error ending OpenAI Realtime stream: {e}")
            parts = self.completed_segments + ([self.current_partial] if self.current_partial else [])
            return " ".join(parts).strip()

    async def _listen_for_events(self) -> None:
        """Listen for events from the WebSocket."""
        try:
            async for message in self.ws:
                event = json.loads(message)
                event_type = event.get("type", "")

                if event_type == "conversation.item.input_audio_transcription.delta":
                    # Streaming partial transcript - accumulate
                    delta = event.get("delta", "")
                    self.current_partial += delta
                    logger.info(f"🎤 Partial: {self.current_partial}")

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    # Segment completed
                    transcript = event.get("transcript", self.current_partial)
                    if transcript:
                        self.completed_segments.append(transcript)
                    self.current_partial = ""
                    self.got_final.set()
                    logger.debug(f"OpenAI Realtime segment completed: '{transcript}'")

                elif event_type == "error":
                    error = event.get("error", {})
                    self.error_message = error.get("message", str(error))
                    self.got_final.set()
                    logger.error(f"OpenAI Realtime error: {self.error_message}")

                elif event_type in ("session.created", "transcription_session.created"):
                    logger.debug("OpenAI Realtime session created")
                    self.session_ready.set()

                elif event_type in ("session.updated", "transcription_session.updated"):
                    logger.debug("OpenAI Realtime session configured")
                    self.session_ready.set()

                elif event_type == "input_audio_buffer.committed":
                    logger.debug("OpenAI Realtime: audio buffer committed")

                elif event_type == "input_audio_buffer.speech_started":
                    logger.debug("OpenAI Realtime: speech started (server VAD)")

                elif event_type == "input_audio_buffer.speech_stopped":
                    logger.debug("OpenAI Realtime: speech stopped (server VAD)")

                else:
                    logger.debug(f"OpenAI Realtime event: {event_type}")

        except asyncio.CancelledError:
            logger.debug("OpenAI Realtime listener cancelled")
        except Exception as e:
            logger.warning(f"OpenAI Realtime listener error: {e}")

    async def _cleanup(self, ws: Any, listener_task: Optional[asyncio.Task[None]]) -> None:
        """Clean up WebSocket and listener task in background."""
        try:
            if listener_task:
                listener_task.cancel()
                try:
                    await listener_task
                except asyncio.CancelledError:
                    pass

            if ws:
                await ws.close()
                logger.debug("OpenAI Realtime WebSocket closed")
        except Exception as e:
            logger.debug(f"Cleanup error (non-critical): {e}")
