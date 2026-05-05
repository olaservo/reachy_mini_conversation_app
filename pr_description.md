# Drop the in-app head wobbler, use the core daemon's instead

The `reachy_mini` SDK now ships its own audio-reactive head wobbler that runs inside the daemon's media pipeline. Audio pushed by this app via `push_audio_sample` already flows through that pipeline, so the daemon can drive the head directly. The local DSP, threading, scheduling, and offset-application code we maintained in the app are no longer needed.

## What changes

- **`main.py`**: replace the local `HeadWobbler` lifecycle with `robot.enable_wobbling()` / `robot.disable_wobbling()`.
- **Backends (`gemini_live.py`, `openai_realtime.py`)**: drop the `if deps.head_wobbler is not None:` blocks that called `.reset()`, `.request_reset_after_current_audio()`, and `.feed()`. The daemon's wobbler reacts to the audio stream itself, so explicit reset hooks are unnecessary.
- **`console.py`**: drop the `head_wobbler.feed_pcm` call and the `_estimate_pending_playback_seconds` helper that only existed to compute its `start_delay_s`.
- **Delete `src/reachy_mini_conversation_app/audio/`**: `head_wobbler.py`, `speech_tapper.py`, and the package itself.
- **`MovementManager`**: drop `set_speech_offsets` and the supporting state (`_pending_speech_offsets`, `_speech_offsets_lock`, `_speech_offsets_dirty`, `state.speech_offsets`, the speech term in `_apply_pending_offsets` and `_get_secondary_pose`). Speech offsets are now composed server-side via `SetSpeechOffsetsCmd`; the local composition path is gone.
- **`ToolDependencies`**: drop the `head_wobbler` field.
- **Tests**: drop `tests/audio/test_head_wobbler.py`, the two head-wobbler-only tests in `tests/test_openai_realtime.py`, the playback-delay test in `tests/test_console.py`, and update `tests/test_gemini_live.py`'s remaining wobbler-aware test to drop the wobbler mock while keeping its transcript and listening-state coverage.

Net diff: roughly 970 deletions vs 11 insertions across 4 commits.

## Mathematical equivalence

The daemon composes `_speech_offsets` with the target head pose using the exact same primitives this app used to use locally (`reachy_mini.utils.create_head_pose` + `reachy_mini.utils.interpolation.compose_world_offset`). For speech offsets in isolation the result is byte-for-byte identical to the previous in-app composition. When face tracking is also active, the new path applies face on top of the primary pose first and then speech on top of that, instead of summing both Euler-angle vectors before composing once. Translations remain exactly equivalent; rotations differ by O(angle²), which is imperceptible at the angle magnitudes used here, and the new ordering is mathematically more rigorous than the old component-wise Euler sum.

## Behaviour change to be aware of

In `--gradio` mode, audio is rendered in the browser and never reaches the daemon's media pipeline, so the head no longer wobbles in that path. Production (non-gradio) paths use `push_audio_sample`, which routes audio through the daemon's pipeline and the wobbler tees off it, so wobbling continues to work transparently and now benefits from the SDK wobbler's stricter silence handling.

## Test plan

- [x] `pytest tests/test_console.py tests/test_openai_realtime.py` passes (24 / 24).
- [x] `tests/test_gemini_live.py` is syntactically clean and contains zero `head_wobbler` references. (Full collection currently blocked by an unrelated `huggingface-hub` / `transformers` env mismatch present on `develop` too.)
- [x] Manual smoke test on the physical robot in non-gradio mode: assistant speech drives the head as expected.
