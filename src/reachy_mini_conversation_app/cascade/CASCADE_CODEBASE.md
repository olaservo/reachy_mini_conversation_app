# Cascade backend internals

The cascade backend runs an **ASR → LLM → TTS** pipeline as one of the app's swappable
conversation backends. Unlike the realtime backends (a single speech-to-speech model), each
stage is an independent, swappable provider chosen in `cascade.yaml`.

It is **not** a fork: `CascadeHandler` implements the same `ConversationHandler` contract
(`conversation_handler.py`) as `OpenaiRealtimeHandler` / `GeminiLiveHandler` /
`HuggingFaceRealtimeHandler`, and plugs into the same stream managers, tool registry, config,
profiles, vision, and UI. `main.build_handler()` returns it when `BACKEND_PROVIDER == "cascade"`
(via `--cascade`).

## Turn flow

```
mic frames ──▶ receive()                         emit() ──▶ robot speaker (+ daemon head wobble)
                 │  resample→16k, Silero VAD          ▲
                 │  on speech end: spawn turn         │  drains output_queue
                 ▼                                    │
            _run_turn() ── ASR ──▶ LLM (+tools) ── TTS ┘
                          │                    │
                          └ user transcript    └ QueueSpeechOutput puts (rate, int16) frames
                            + analyze_final       + assistant text  onto output_queue
```

- **`receive(frame)`** (`handler.py`): resamples mic audio to 16 kHz and feeds a
  `VADStateMachine` (Silero VAD, `vad/`) in 512-sample chunks. On `SPEECH_ENDED` it wraps the
  buffered speech in a WAV and spawns `_run_turn` as a background task, returning immediately so
  audio keeps flowing.
- **`_run_turn(wav)`**: ASR → emit the user transcript (`AdditionalOutputs`) → run the LLM/tool
  pipeline → reset VAD. The final transcript is analyzed for live reactions in parallel with the
  LLM.
- **`emit()`**: returns the next item from `output_queue` (audio frame or transcript), exactly
  like the realtime backends. The stream manager (`console.LocalStream` / fastrtc `Stream`) plays
  audio through `robot.media`, which drives the daemon head wobbler — no bespoke playback threads.
- **`start_up()`** blocks until `shutdown()` (an `asyncio.Event`): the stream manager treats
  `start_up()` returning as "session ended", so the handler must hold the session open while
  `receive`/`emit` run concurrently.

## Pipeline & tools (`pipeline.py`)

`process_llm_response` streams the LLM, accumulates tool calls, and runs them via
`execute_tool_calls`. Robot tools (dance, move_head, camera, emotions, …) dispatch through main's
shared `dispatch_tool_call`. **`speak` is cascade-specific**: it is not in the shared registry
(so realtime backends are unaffected); its spec (`SPEAK_TOOL_SPEC`) is injected into the cascade
tool list and the pipeline intercepts the call, turning the message into TTS via
`QueueSpeechOutput`. Profile instructions get `CASCADE_EXTRA_INSTRUCTIONS` appended so the LLM
always speaks through the `speak` tool.

## Providers (`asr/`, `llm/`, `tts/`)

Each stage has an abstract base (`asr/base.py` `ASRProvider`, `asr/base_streaming.py`
`StreamingASRProvider`, `llm/base.py` `LLMProvider`, `tts/base.py` `TTSProvider`) and concrete
providers loaded dynamically by `provider_factory.py` from the `cascade.yaml` catalog. Each
provider declares required API keys, hardware, and an `import_check`/`install_extra`, validated at
config load (`config.py`) with a clear "Install with: uv sync --extra cascade_<x>" error.

Streaming ASR providers also satisfy the batch `transcribe()` interface (start→send→end), so they
work through `_run_turn`'s batch path today; a real-time partial path (feeding `analyze_partial`
during speech) is a future enhancement.

## Live reactions (`transcript_analysis/`)

A profile may add `reactions.yaml` + callback modules. `TranscriptAnalysisManager` runs keyword
(literal + glob) and optional GLiNER entity analyzers on the final transcript and dispatches
profile callbacks (e.g. dance when the user says "let's dance"), deduplicated per turn. Profiles
without `reactions.yaml` get a `NoOpTranscriptManager`.

## Config (`config.py`, `cascade.yaml`)

`cascade.yaml` (bundled in the package; a copy in the working directory overrides it) selects the
active provider per stage and lists the catalog. `CASCADE_ASR_PROVIDER` / `CASCADE_LLM_PROVIDER` /
`CASCADE_TTS_PROVIDER` env vars override the selection.

## Known follow-ups

- Real-time streaming-ASR partial path (live partials + `analyze_partial` during speech).
- Non-blocking long tools via `BackgroundToolManager`, and the idle policy in `emit()`.
- Barge-in (needs acoustic echo cancellation so the robot's own TTS doesn't self-trigger the VAD).
- Multimodal `see_image_through_camera` (raw image re-injected to the LLM) vs. main's text `camera`.
