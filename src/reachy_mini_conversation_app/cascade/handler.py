"""Cascade backend: a ConversationHandler running a VAD -> ASR -> LLM -> TTS turn loop."""

from __future__ import annotations
import io
import wave
import asyncio
import logging
from typing import Any, Dict, List, Union, Optional

import numpy as np
from fastrtc import AdditionalOutputs, wait_for_item, audio_to_int16, audio_to_float32
from numpy.typing import NDArray
from scipy.signal import resample

from reachy_mini_conversation_app.cascade import pipeline
from reachy_mini_conversation_app.cascade.asr import ASRProvider
from reachy_mini_conversation_app.cascade.llm import LLMProvider
from reachy_mini_conversation_app.cascade.tts import TTSProvider
from reachy_mini_conversation_app.cascade.vad import (
    VAD_CHUNK_SIZE,
    SILERO_SAMPLE_RATE,
    VADEvent,
    SileroVAD,
    VADStateMachine,
)
from reachy_mini_conversation_app.cascade.pipeline import PipelineContext
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies, get_active_tool_specs
from reachy_mini_conversation_app.cascade.turn_result import PipelineResult
from reachy_mini_conversation_app.conversation_handler import AudioFrame, HandlerOutput, ConversationHandler
from reachy_mini_conversation_app.cascade.speech_output import QueueSpeechOutput
from reachy_mini_conversation_app.cascade.provider_factory import (
    init_asr_provider,
    init_llm_provider,
    init_tts_provider,
    init_transcript_analysis,
    cascade_system_instructions,
)


logger = logging.getLogger(__name__)


# The cascade pipeline has no native speech, so the LLM speaks via this tool.
# It is intentionally NOT in the shared tool registry (which the realtime
# backends share); the pipeline intercepts it and routes the message to TTS.
SPEAK_TOOL_SPEC: Dict[str, Any] = {
    "type": "function",
    "name": "speak",
    "description": "Speak the given message to the user. Use this tool for ALL verbal responses.",
    "parameters": {
        "type": "object",
        "properties": {"message": {"type": "string", "description": "The text to speak to the user"}},
        "required": ["message"],
    },
}


# Like `speak`, but voices the line as a specific character. The pipeline intercepts
# this and routes the message to TTS using the named per-character voice instead of
# the handler default, letting the DM voice NPCs distinctly without changing state.
SPEAK_AS_TOOL_SPEC: Dict[str, Any] = {
    "type": "function",
    "name": "speak_as",
    "description": (
        "Speak the given message in a specific character's voice. Use this to voice an NPC "
        "or narrator distinctly instead of the default voice."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "voice_id": {
                "type": "string",
                "description": (
                    "The id of the registered character voice to use, e.g. 'gm_narrator' or "
                    "'npc_raider'."
                ),
            },
            "message": {"type": "string", "description": "The text to speak in that voice"},
        },
        "required": ["voice_id", "message"],
    },
}


def _to_chat_tool_specs(realtime_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Realtime-API tool specs to Chat Completions format."""
    chat_specs: List[Dict[str, Any]] = []
    for spec in realtime_specs:
        if spec.get("type") == "function":
            chat_specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec["name"],
                        "description": spec["description"],
                        "parameters": spec["parameters"],
                    },
                }
            )
    return chat_specs


def _pcm_to_wav(pcm: NDArray[np.int16], sample_rate: int) -> bytes:
    """Wrap mono int16 PCM samples in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buf.getvalue()


class CascadeHandler(ConversationHandler):
    """Voice backend running a local VAD -> ASR -> LLM -> TTS turn loop.

    Plugs into the same stream interface as the realtime backends: receive()
    consumes mic audio and starts a turn when an utterance ends; emit() drains
    audio frames and transcripts from output_queue.
    """

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
    ) -> None:
        """Initialize providers, VAD, and stream buffers from cascade.yaml."""
        self.asr: ASRProvider = init_asr_provider()
        # Inject the persistent-memory index block (if memory is enabled) into the
        # cascade system prompt at construction time.
        self.llm: LLMProvider = init_llm_provider(deps.memory_manager)
        self.tts: TTSProvider = init_tts_provider()

        super().__init__(
            expected_layout="mono",
            output_sample_rate=self.tts.sample_rate,
            input_sample_rate=SILERO_SAMPLE_RATE,
        )

        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path
        self._voice: str | None = startup_voice or None

        self.output_queue: "asyncio.Queue[AudioFrame | AdditionalOutputs]" = asyncio.Queue()
        self.conversation_history: List[Dict[str, Any]] = []
        self.cumulative_cost: float = 0.0

        self._speech_output = QueueSpeechOutput(self)
        self.transcript_manager = init_transcript_analysis(deps)
        self._vad_sm = VADStateMachine(SileroVAD())
        self._vad_buffer: NDArray[np.int16] = np.zeros(0, dtype=np.int16)
        self._turn_task: asyncio.Task[None] | None = None
        self._processing_lock = asyncio.Lock()
        self._closed = asyncio.Event()

        self.tool_specs = self._build_tool_specs()
        logger.info("CascadeHandler ready (ASR/LLM/TTS configured via cascade.yaml)")

    def _build_tool_specs(self) -> List[Dict[str, Any]]:
        """Chat-format tool specs = shared robot tools + the cascade speak tool."""
        return _to_chat_tool_specs([*get_active_tool_specs(self.deps), SPEAK_TOOL_SPEC, SPEAK_AS_TOOL_SPEC])

    # ── ConversationHandler contract ─────────────────────────────────────────

    def copy(self) -> "CascadeHandler":
        """Create a fresh handler for a new stream connection."""
        return CascadeHandler(
            self.deps,
            gradio_mode=self.gradio_mode,
            instance_path=self.instance_path,
            startup_voice=self._voice,
        )

    async def start_up(self) -> None:
        """Warm up, then keep the session alive until shutdown.

        The stream manager treats start_up() returning as "session ended", so this
        must block for the lifetime of the session. receive()/emit() are driven
        concurrently by the stream's record/play loops.
        """
        await self.llm.warmup()
        # Rotate the memory session log so any synchronous remember/forget writes
        # this session land in a fresh pending log. Reads/index injection work
        # without this, so it's guarded and best-effort.
        if self.deps.memory_manager is not None:
            try:
                self.deps.memory_manager.new_session()
            except Exception as e:
                logger.warning("Failed to start memory session: %s", e)
        await self._closed.wait()

    async def shutdown(self) -> None:
        """End the session and cancel any in-flight turn."""
        self._closed.set()
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()

    async def receive(self, frame: AudioFrame) -> None:
        """Feed mic audio to the VAD; start a turn when an utterance ends."""
        in_rate, audio = frame
        audio = self._to_mono(audio)
        if in_rate != SILERO_SAMPLE_RATE:
            audio = resample(audio, int(len(audio) * SILERO_SAMPLE_RATE / in_rate))
        chunk16 = audio_to_int16(audio).reshape(-1)
        self._vad_buffer = np.concatenate([self._vad_buffer, chunk16])

        while len(self._vad_buffer) >= VAD_CHUNK_SIZE:
            vad_chunk = self._vad_buffer[:VAD_CHUNK_SIZE]
            self._vad_buffer = self._vad_buffer[VAD_CHUNK_SIZE:]
            event = self._vad_sm.process_chunk(vad_chunk)
            if event == VADEvent.SPEECH_STARTED:
                self.deps.movement_manager.set_listening(True)
            elif event == VADEvent.SPEECH_ENDED:
                speech = (
                    np.concatenate(self._vad_sm.speech_chunks)
                    if self._vad_sm.speech_chunks
                    else np.zeros(0, dtype=np.int16)
                )
                self._turn_task = asyncio.create_task(self._run_turn(_pcm_to_wav(speech, SILERO_SAMPLE_RATE)))

    async def emit(self) -> HandlerOutput:
        """Return the next queued audio frame or transcript."""
        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    async def get_available_voices(self) -> list[str]:
        """Voices available on the active TTS provider."""
        voices = getattr(self.tts, "available_voices", None)
        return list(voices) if voices else [self.get_current_voice()]

    def get_current_voice(self) -> str:
        """Return the active voice (TTS provider default if unset)."""
        return self._voice or str(getattr(self.tts, "default_voice", "") or "")

    async def change_voice(self, voice: str) -> str:
        """Switch the TTS voice for subsequent synthesis."""
        self._voice = voice
        return f"Voice changed to {voice}."

    async def apply_personality(self, profile: str | None) -> str:
        """Switch profile: reload instructions and the active tool set."""
        from reachy_mini_conversation_app.config import set_custom_profile

        set_custom_profile(profile)
        try:
            self.llm.system_instructions = cascade_system_instructions(  # type: ignore[attr-defined]
                self.deps.memory_manager
            )
        except Exception as e:
            logger.error("Failed to apply personality '%s': %s", profile, e)
            return f"Failed to apply personality: {e}"
        self.tool_specs = self._build_tool_specs()
        return "Applied personality."

    # ── Internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _to_mono(audio: NDArray[Any]) -> NDArray[Any]:
        """Collapse a possibly-2D frame to mono 1D."""
        if audio.ndim == 2:
            if audio.shape[1] > audio.shape[0]:
                audio = audio.T
            if audio.shape[1] > 1:
                audio = audio[:, 0]
            audio = audio.reshape(-1)
        return audio

    def _tap_audio_for_daemon_wobbler(self, decoded_pcm: NDArray[np.int16]) -> None:
        """Push spoken audio to the daemon so it drives the head wobbler.

        In Gradio mode audio plays in the browser (via emit), so the daemon — whose
        wobbler taps push_audio_sample — never sees it. Mirror the realtime handlers
        and push the same samples; the robot speaker is muted upstream to avoid
        double playback. See BaseRealtimeHandler._tap_audio_for_daemon_wobbler.
        """
        try:
            robot = self.deps.reachy_mini
            output_rate = robot.media.get_output_audio_samplerate()
            audio = audio_to_float32(decoded_pcm).reshape(-1)
            if self.output_sample_rate != output_rate:
                num_samples = int(len(audio) * output_rate / self.output_sample_rate)
                if num_samples == 0:
                    return
                audio = resample(audio, num_samples)
            robot.media.push_audio_sample(audio.astype(np.float32))
        except Exception as exc:
            logger.debug("Daemon wobbler audio tap failed: %s", exc)

    def _aggregate_cost(self, provider: Union[ASRProvider, LLMProvider, TTSProvider], label: str) -> None:
        """Fold a provider's per-call cost into the cumulative total."""
        cost = getattr(provider, "last_cost", 0.0)
        if cost and cost > 0:
            self.cumulative_cost += cost
            logger.info("Cost (%s): $%.4f | Cumulative: $%.4f", label, cost, self.cumulative_cost)
            setattr(provider, "last_cost", 0.0)

    async def _run_turn(self, wav_bytes: bytes) -> None:
        """Transcribe, run the LLM/tool/TTS pipeline, then return VAD to listening."""
        from reachy_mini_conversation_app.cascade.timing import tracker

        async with self._processing_lock:
            try:
                tracker.reset("vad_speech_end")
                tracker.mark("transcribing_start")
                transcript = await self.asr.transcribe(wav_bytes)
                tracker.mark("asr_complete", {"len": len(transcript)})
                self._aggregate_cost(self.asr, "ASR")
                self.deps.movement_manager.set_listening(False)

                if not transcript.strip():
                    logger.info("Empty transcript, ignoring turn")
                    return

                logger.info("User said: %s", transcript)
                await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))
                self.conversation_history.append({"role": "user", "content": transcript})

                # Live reactions: analyze the final transcript in parallel with the LLM.
                asyncio.create_task(self.transcript_manager.analyze_final(transcript))

                ctx = PipelineContext(
                    llm=self.llm,
                    tts=self.tts,
                    speech_output=self._speech_output,
                    conversation_history=self.conversation_history,
                    tool_specs=self.tool_specs,
                    deps=self.deps,
                    result=PipelineResult(),
                )
                result = await pipeline.process_llm_response(ctx)
                self.cumulative_cost += result.cost
                tracker.print_summary()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Cascade turn failed")
                self.deps.movement_manager.set_listening(False)
            finally:
                self.transcript_manager.reset()
                self._vad_sm.finish_processing()
