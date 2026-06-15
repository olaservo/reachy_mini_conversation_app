# Memory subsystem → cascade DM: merge plan

How to give the cascade-based TTRPG DM durable, cross-session memory by bringing in the
`origin/295-add-memory-in-s2s-realtime-backend-branch` memory system. Investigation done
2026-06-15 (read-only). Merge base of `cascade_backend` and `295` is `cd0470c3`.

## The core finding
The 295 memory system is ~80% backend-agnostic, BUT its live-side wiring targets
`base_realtime.py` (the realtime backends), and the **cascade does NOT inherit
`BaseRealtimeHandler`** — it has its own `cascade/handler.py`. So:

- **Free (merge → works):** the `memory/` module, the `recall_memory`/`recall_memories`
  tools (auto-register into the cascade via the `SystemTool` enum + name-based tool loading —
  verified), `ToolDependencies.memory_manager`, the `MemoryManager` construction in `main.py`,
  config knobs.
- **Needs cascade glue (merge → inert without it):** four hooks 295 only added to
  `base_realtime.py` must be hand-ported into `cascade/handler.py`:
  1. inject the `active_memory.md` index into the system prompt (cascade calls
     `get_session_instructions()` with NO manager → empty memory block today);
  2. `memory_manager.new_session()` + per-turn `log_turn()`/`log_tool_call()` (cascade keeps
     only an in-memory history → no logs → dreamer has nothing to consolidate);
  3. launch the background dreamer per session;
  4. dream chime + injected "background context" note.
  Without these: recall tools exist but the index is empty and nothing is ever written.

## Critical gameplay insight
The 295 dreamer-only write model assumes MANY SHORT sessions (consolidates *previous*
sessions' logs in the background). A TTRPG session is ONE LONG conversation, so mid-session
facts ("the party freed the prisoner") are NOT recallable until a future session's dreamer
runs — and the very first session has an empty index throughout. **A synchronous write tool
is required.** Recommended: port `1-add-memory`'s `save_memory` pattern as a DM-facing
`remember`/`forget` tool backed by the 295 `MemoryManager.save_memory_sync()` (atomic write,
kind=`event`); the dreamer still runs post-session to curate.

## DM fit
- The `kind` vocab maps well: event=story beats, relationship=party/NPCs/factions,
  goal=quests, fact=lore/locations, pinned=campaign premise + party roster. No schema change.
- The dreamer prompt (`memory/dreamer.py:51` `DREAMER_SYSTEM_PROMPT`, hardcoded, user-centric)
  needs a DM/campaign variant — add a `system_prompt` param to `Dreamer`/`DreamScheduler`.
- Dreamer hardcodes `DEFAULT_DREAMER_MODEL="gpt-5.4"` + `OPENAI_API_KEY` → must repoint to
  our Qwen brain's Responses endpoint for the all-Qwen stack.

## Tool-name reconciliation
`profiles/dm/tools.txt` lists `remember`/`forget` → resolve to nonexistent modules → inert
today. Recall tools load via the `SystemTool` enum (no `tools.txt` entry needed). Fix: drop the
dead lines; if we add the synchronous write tool, implement real `tools/remember.py`+`forget.py`.

## Plan (cherry-pick additive + hand-port hooks; do NOT full-merge — 295's base_realtime
rewrite is irrelevant to the cascade and conflicts)

### Phase 1 — land the module (low risk)
1. Copy `memory/` (7 files), `tools/recall_memory.py`, `tools/recall_memories.py`,
   `sounds/dream_*.wav`, `tests/memory/` verbatim from `origin/295-...`.
2. Hand-port: `SystemTool.RECALL_MEMORY/RECALL_MEMORIES` (tool_constants.py); `args_json_str`
   (background_tool_manager.py); the config block (config.py); the `memory_manager` field in
   `ToolDependencies` (core_tools.py — place after `vision_processor`; cascade dropped
   `head_wobbler`).
3. Construct `MemoryManager` in `main.py` `run()` + pass into `ToolDependencies(...)`.

### Phase 2 — cascade glue (the real work, medium risk; all in cascade/)
5. Index injection: `cascade_system_instructions()` (provider_factory.py:88) forwards a
   `memory_manager` to `get_session_instructions(memory_manager)`; thread from handler; re-inject
   in `apply_personality` (handler.py:238).
6. Lifecycle + logging: `new_session()` at session start in `handler.start_up()`; `log_turn()`
   for user transcript + assistant spoken text; `log_tool_call()` at pipeline tool execution.
   (OPEN: pin the assistant-text finalization point in cascade/pipeline.py + speech_output.py.)
7. Dreamer launch: port `_start_background_dreaming()` per session (consider a shared mixin).
8. Dream chime + context note: cascade has no realtime `conversation.item.create`; append a
   hidden note into `self.conversation_history`; chime via `deps.reachy_mini.media.play_sound`.

### Phase 3 — DM theming + synchronous write (medium risk)
9. `profiles/dm/tools.txt`: remove dead `remember`/`forget` (or repurpose per 11).
10. DM dreamer prompt: `system_prompt` param on Dreamer/DreamScheduler + `DM_DREAMER_SYSTEM_PROMPT`
    selected when profile == `dm`.
11. Synchronous `remember`/`forget`: `MemoryManager.save_memory_sync()` (atomic temp+os.replace)
    + `tools/remember.py`/`forget.py`. Reuse atomic write path so it coexists with the dreamer
    (dreamer supersedes, never deletes).

### Phase 4 — validate
12. Repoint the dreamer at the Qwen Responses endpoint (not gpt-5.4). Run a session: logs under
    `~/.reachy_mini/data/memory/logs/pending/`, index rebuilt, recall returns real content,
    `remember` writes a live memory recallable same-session.

## Open questions
- Assistant-text capture point in the cascade (step 6) — not yet pinned.
- Dreamer client base-URL override for Qwen (dreamer.py client construction).
- Single-writer invariant if synchronous `remember` added — atomic writes + dreamer-supersedes.
- Shared dreaming mixin vs duplicate in cascade handler (duplicate is faster for the hackathon).

## Minimal-viable subset (recommended first cut for the demo)
Phase 1 + hook (5) index injection + Phase 3 step 11 synchronous `remember`/`forget` + recall.
SKIP the dreamer (Phase 2 steps 6–8 logging/launch + Phase 3 step 10 theming + Qwen repoint).
Gives durable, cross-session, same-session-recallable memory with the least risk; add the
auto-consolidating dreamer later.
