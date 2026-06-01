# Background ("hidden") dreaming — implementation spec

> Supersedes the boot-time blocking dream phase described in
> `memory-rework-dreaming-spec.md` §8. Everything else in that spec
> (memory format, index, recall tools, dreamer prompt/algorithm) is unchanged.
> Status: ready to implement.

## 1. Why

The dreamer currently runs as a **blocking boot phase**: the app spins
`dizzy_spin` and refuses connections until `Dreamer.run()` has consolidated
every pending log. Good for testing, too intrusive for real use — the robot
visibly "goes away" before every conversation.

New behavior: dreaming runs **in parallel, during the conversation**, hidden.
A subtle chime marks start and finish, and the live model is given just enough
context to explain the chime if the user asks ("what was that sound?").

## 2. What changes

- **Remove** the blocking boot path: the `run_dream_phase(...)` call in
  `main.py` and the `_DizzySpinLoop` in `memory/boot.py`. `boot.py` is deleted
  (its only export was the blocking phase).
- **Add** `memory/dream_scheduler.py`: a small object that runs `Dreamer.run()`
  on **one daemon thread per session** and fires lifecycle callbacks.
- **Wire** it into the realtime session lifecycle, right where
  `memory_manager.new_session()` is already called (`base_realtime.py`, covering
  OpenAI + HuggingFace; and `gemini_live.py`).

`Dreamer` itself is unchanged — still synchronous, still uses the OpenAI
Responses API. The thread is what makes it "parallel"; the event loop never
blocks.

## 3. DreamScheduler

```
DreamScheduler(
    memory_manager,
    model: str,
    api_key: str | None,
    base_url: str | None,
    on_start: Callable[[], None],
    on_finish: Callable[[DreamSummary], None],
)
```

- `.start()` — if there are pending logs (excluding the active session log),
  spawn a daemon thread; otherwise do nothing (no chime, no thread).
- Thread body: call `on_start()`, run `Dreamer.run()`, then **always** call
  `on_finish(summary)` in a `finally` — even if the dream raises. Exceptions are
  logged, never propagated to the conversation.
- `on_start` / `on_finish` are thin: they marshal the audio + awareness side
  effects back onto the asyncio loop (see §5). The scheduler itself knows
  nothing about audio or the realtime connection — it only knows "starting" and
  "finished, here's a one-line summary".
- `DreamSummary`: tiny dataclass — `logs_processed: int`, `created: int`,
  `updated: int`, `errored: bool`. Derived from the existing `DreamLogStats`
  list that `Dreamer.run()` already returns. Used only to phrase the awareness
  note; not persisted.

## 4. The audio tell

Two short, soft generated chimes committed under
`src/reachy_mini_conversation_app/sounds/`:

- `dream_start.wav` — gentle rising two-note tone.
- `dream_end.wav` — softer falling two-note tone.

Played via `robot.media.play_sound(path)` (the documented Reachy Mini media
API). Low amplitude so it sits under speech, not over it.

**Open risk to verify during implementation:** that `play_sound` coexists with
the active realtime audio stream without disrupting it. If it does conflict,
fall back to pushing the cue through the existing `output_queue` audio path.
Whichever path works, it stays behind the scheduler callbacks so the rest of the
design is unaffected.

## 5. Robot awareness (no forced speech)

On start and finish we inject a **hidden context item** into the live
conversation — exactly the mechanism `send_idle_signal` already uses:

```python
await connection.conversation.item.create(item={
    "type": "message", "role": "user",
    "content": [{"type": "input_text", "text": NOTE}],
})
```

Crucially we do **not** call `_safe_response_create` afterwards, so the robot
does not speak — the note just sits in context. If the user later asks about the
sound, the model has the explanation; otherwise it stays silent.

Notes (final wording in code):

- start: `[Background event: a soft chime just played. Your memory-consolidation
  "dreaming" process started in the background — you're quietly reprocessing
  your recent conversations into long-term memory. Keep conversing normally; do
  not mention this unless the user asks about the sound or your memory.]`
- finish: `[Background event: a soft chime just played. Dreaming finished — your
  memories are now up to date. Do not mention this unless asked.]`

Marshalling: the daemon thread can't touch the connection directly, so callbacks
hop onto the loop via `asyncio.run_coroutine_threadsafe(coro, loop)`. The loop
reference is captured at session start. If the connection has closed by the time
a callback fires (session ended mid-dream), the injection is skipped — logged,
not fatal.

## 6. Concurrency (unchanged stance)

The dreamer is the only writer of memory files. It already excludes the active
session log (`exclude_session=True`). Memory + index writes are atomic
(temp-then-`os.replace`). Live conversation only *reads* memory. We accept the
rare, harmless read-during-write; no locks.

One process-level guard: a session never starts a second dream thread while one
is still running (the scheduler is created per session and tracks its thread).

## 7. Out of scope

- Re-triggering dreams mid-session or on a timer. One dream per session start,
  over the logs left by previous sessions.
- Exposing any dreaming tool to the live model. Dreaming stays system-initiated
  and hidden — not an LLM-callable tool. (Rationale: the background-tool
  framework is LLM-initiated and completion-only; it fits poorly. See PR
  discussion.)

## 8. Tests

- `DreamScheduler`: with a stub `Dreamer`, assert `on_start` fires before the
  run, `on_finish` fires after, and `on_finish` still fires when the run raises.
  No-pending-logs ⇒ neither callback fires, no thread.
- Awareness wording / summary mapping: `DreamLogStats` list ⇒ `DreamSummary`.
- Existing dreamer / memory tests unchanged.
