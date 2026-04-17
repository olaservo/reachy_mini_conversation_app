# Memory System Design

> **Status**: Implemented
> **Branch**: `1-add-memory` (based on `amir/concurrent_execution_tool` from [PR #205](https://github.com/pollen-robotics/reachy_mini_conversation_app/pull/205))
> **Issue**: [#1 — Add memory](https://github.com/pollen-robotics/reachy_mini_conversation_app/issues/1)
> **Date**: 2026-02-20

## Problem

The robot has no way to recall past interactions once the app is restarted and the OpenAI Realtime session resets. This makes a great first interaction but a disappointing second one — the robot forgets everything.

## Goals

1. **Remember across sessions** — the robot should recall key facts (names, preferences, topics discussed) from previous conversations.
2. **Never lose raw data** — full conversation transcripts are logged and preserved indefinitely.
3. **Stay within context limits** — the active memory injected into the system prompt must fit within the `gpt-realtime` token budget.
4. **Graceful degradation** — memory is enabled by default but can be disabled. Memory failures never crash the app.

## Non-goals (for V1)

- Vector search / semantic retrieval (simple substring search is sufficient for now)
- Multi-user memory scoping (single robot = single memory)
- Automatic summarization of conversation logs
- Memory shared across multiple robots

---

## Research & Validation

### Context window budget

The `gpt-realtime` model has a **32,768 token** total context window ([source](https://platform.openai.com/docs/models/gpt-realtime)):

| Component | Token budget |
|---|---|
| Instructions + tools cap | 16,384 |
| Max response output | 4,096 |
| Conversation history | ~12,288 |

After the current system prompt (~500 tokens) and tool schemas (~1,000 tokens), **~4,000–6,000 tokens remain for injected memory** within the 16K instruction+tools cap. We cap active memory at **~1,500 tokens** to leave comfortable headroom.

Audio consumes context at ~10x the rate of text ([source](https://developers.openai.com/cookbook/examples/context_summarization_with_realtime_api/)): 600 tokens/min input + 1,200 tokens/min output. The `gpt-realtime` GA model drops audio tokens when transcripts are available, but conversations still fill context in ~10-20 minutes.

When context overflows, the server **silently drops the oldest conversation items** while preserving system instructions ([source](https://community.openai.com/t/what-does-auto-truncation-in-realtime-api-actually-do/1356153)). This means our injected memory block is always preserved.

### Prior art

| System | Approach | Benchmark (LoCoMo) | Notes |
|---|---|---|---|
| [mem0](https://github.com/mem0ai/mem0) | Vector DB + LLM extraction/deduplication | 66.9% (base), 68.4% (graph) | Requires vector DB infrastructure; p50 search 148ms ([paper](https://arxiv.org/abs/2504.19413)) |
| [Letta/MemGPT](https://github.com/letta-ai/letta) | OS virtual memory metaphor: core memory blocks + archival + recall | 74.0% (filesystem variant) | Agents using file-based grep/search beat vector DB approaches ([benchmark](https://www.letta.com/blog/benchmarking-ai-agent-memory)) |
| [memori](https://github.com/GibsonAI/memori) | SQL-native memory with async enrichment | N/A | Younger project; async background augmentation fits voice well |
| Full-context (no memory system) | Dump everything into context | ~73% | Works until context overflows; cost scales quadratically ([ConvoMem](https://arxiv.org/html/2511.10523v1)) |
| **Our approach** | Flat files: logs + curated active memory + archives + recall tool | N/A (expected: Letta-class) | Matches the Letta Filesystem pattern that scored 74% |

**Key finding**: the [Letta Filesystem benchmark](https://www.letta.com/blog/benchmarking-ai-agent-memory) showed that agents with simple file-based search tools (grep, read) achieve **74.0% on LoCoMo** — beating mem0g (68.5%) with a full vector+graph DB. This validates our file-based approach.

**Key finding**: the [ConvoMem benchmark](https://arxiv.org/html/2511.10523v1) found that simple long-context approaches outperform RAG for the **first ~150 conversations**. We don't need a vector DB yet.

### Latency considerations

Tool calls on the Realtime API add **1-3 seconds** of round-trip latency ([source](https://cresta.com/blog/engineering-for-real-time-voice-agent-latency)). For a voice conversation:

- **`save_memory`**: called after the LLM decides to remember something. It writes to a file (~1ms) and returns a short status. The LLM then speaks its response. Latency is negligible.
- **`recall_memory`**: involves file reads + substring search. The tool itself is fast (<50ms), but the full round-trip (LLM generates tool call → server executes → LLM processes result → generates audio) adds 1-3s. Acceptable for deliberate "let me think" moments, but we **pre-inject active memory into the prompt** to avoid this for common facts.

The [Letta sleep-time compute](https://www.letta.com/blog/sleep-time-compute) pattern (background memory processing between conversations) is a natural next step if latency becomes a concern.

---

## Foundation: BackgroundToolManager & SystemTools (PR #205)

This design builds on top of [PR #205](https://github.com/pollen-robotics/reachy_mini_conversation_app/pull/205) which introduces:

- **`BackgroundToolManager`**: All tool calls now run as non-blocking async tasks. The robot continues conversing while tools execute. Results arrive asynchronously via a callback.
- **`SystemTool` enum**: Tools listed in `SystemTool` are **auto-loaded regardless of profile**. They don't need to be in `tools.txt`. Currently: `task_status`, `task_cancel`.
- **Tool call history**: `BackgroundToolManager.get_all_tools()` returns all recent tool calls with name, args, result, status, and timing — a free source of structured conversation metadata.

**Memory tools (`save_memory`, `recall_memory`) are registered as `SystemTool`s.** This means:
- They are always available regardless of which personality profile is active (memory is global, not per-profile).
- They don't need to be listed in any profile's `tools.txt`.
- They follow the same auto-registration pattern as `task_status` and `task_cancel`.

**Conversation logging also captures tool calls.** Since the `BackgroundToolManager` tracks every tool invocation with timestamps, arguments, and results, we log completed tool calls alongside user/assistant transcripts in the conversation JSONL files. This gives a complete picture of each session.

---

## Architecture

### Two-tier memory model

```
┌──────────────────────────────────────────────────────────────────┐
│                        System Prompt                             │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Profile instructions (identity, rules, tool guidance)     │  │
│  ├────────────────────────────────────────────────────────────┤  │
│  │  Active Memory (grows unbounded, warning at ~1500 tokens)  │  │
│  │  User's name is Rémi (2026-02-20_14-32.log)               │  │
│  │  Rémi prefers French (2026-02-20_14-32.log)               │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  Conversation Logs (readable via recall_memory) │
│  logs/2026-02-20_14-32.log                      │
│  logs/2026-02-21_09-15.log                      │
│  ...                                            │
└─────────────────────────────────────────────────┘
```

| Tier | What | Format | Size | Access |
|---|---|---|---|---|
| **1. Conversation logs** | Full transcripts of every session | Plain text, one file per session | Unbounded (user manages) | Readable via `recall_memory` tool |
| **2. Active memory** | Curated facts the LLM saved | Plain text, single file | Unbounded (warning logged at ~1500 tokens) | Injected into system prompt automatically |

### Data directory layout

```
$REACHY_MINI_DATA_DIRECTORY/        # default: ~/.reachy_mini/data/
└── memory/
    ├── active_memory.md            # Active memory: prompt-injected facts
    └── logs/                       # Conversation logs: one per session
        ├── 2026-02-20_14-32.log
        └── 2026-02-21_09-15.log
```

### Active memory format (`active_memory.md`)

```
User's name is Rémi, he's one of the creators of Reachy Mini. (2026-02-20_14-32.log)
Rémi prefers French but speaks English at work. (2026-02-20_14-32.log)
We discussed adding object detection to Reachy's camera. (2026-02-21_09-15.log)
[Archived: 5 older memories moved to archive/2026-02-25T14-23-00.md — use recall_memory to access]
```

Each line is a free-form fact written by the LLM via the `save_memory` tool, with a parenthesized reference to the session log file where the conversation happened. The LLM can pass this filename to `recall_memory` to read the detailed conversation.

### Conversation log format (`logs/YYYY-MM-DD_HH-MM.log`)

```
--- session 2026-02-20 14:32 UTC ---

14:32:00 user: Hi, my name is Rémi!
14:32:02 assistant: Nice to meet you, Rémi! I'll remember that.
14:32:03 tool: save_memory({"fact": "User's name is Rémi"}) -> {"status": "saved"}
14:35:10 tool: dance({"name": "happy"}) -> {"status": "queued"}
```

One file per session, human-readable plain text. The session header is written once at file creation. Each line has a timestamp (HH:MM:SS), role, and content. Append-only, never modified by the app.

---

## Tools

### `save_memory`

The LLM calls this tool when it encounters something worth remembering across sessions.

| Field | Value |
|---|---|
| **Name** | `save_memory` |
| **Description** | Save an important fact to long-term memory. Use when the user shares something worth remembering across sessions (name, preferences, key facts). Be selective. |
| **Parameters** | `fact` (string, required): A concise statement to remember, e.g. "User's name is Alice" |
| **Returns** | `{"status": "saved", "fact": "..."}` |

**Behavior**: Appends the fact (with session log reference) to `active_memory.md`. If the file exceeds the token cap after the append, the oldest third of entries are moved to a new archive file and replaced with a breadcrumb line.

### `recall_memory`

The LLM calls this tool when it needs detailed context from a past conversation referenced in a memory entry.

| Field | Value |
|---|---|
| **Name** | `recall_memory` |
| **Description** | Read a past session log. Each memory has a filename in parentheses — pass it to read the full conversation. Empty string lists available logs. |
| **Parameters** | `log_ref` (string, required): Session log filename (e.g. `2026-03-26_16-28.log`) or empty string |
| **Returns** | `{"log_ref": "...", "content": "..."}` or `{"available_logs": [...]}` or `{"error": "..."}` |

**Behavior**: Reads the referenced session log file from the `logs/` directory and returns the entire content. If the file doesn't exist, returns an error with the list of available log files. If `log_ref` is empty, returns just the list of available files.

---

## Integration points

### Conversation logging (automatic)

Three hook points in `openai_realtime.py`:

1. **User transcripts**: `conversation.item.input_audio_transcription.completed` → `memory_manager.log_turn("user", transcript)`
2. **Assistant transcripts**: `response.audio_transcript.done` → `memory_manager.log_turn("assistant", transcript)`
3. **Tool calls**: In `_handle_tool_result()` callback → `memory_manager.log_tool_call(tool_name, args, result)` — logs every completed tool with its arguments and result.

These are fire-and-forget appends. Failures are silently logged, never crash the audio pipeline.

### Prompt injection (automatic)

`prompts.py:get_session_instructions()` is modified to accept an optional `memory_manager` parameter. When provided and active memory is non-empty, the memory block is appended after the profile instructions:

```
<profile instructions>

## MEMORY
The following facts were saved from previous conversations.
Use them to personalize responses. You can save new memories with save_memory
and search older context with recall_memory.

- [2026-02-20] User's name is Rémi...
- ...
```

This runs on every session start and personality switch, so newly saved facts appear in the next session automatically.

### Dependency injection

`MemoryManager` is added as an optional field on `ToolDependencies` (matching the existing pattern for `camera_worker`, `head_wobbler`, etc.). Constructed in `main.py`, passed through to tools and the realtime handler.

### SystemTool registration

`save_memory` and `recall_memory` are added to the `SystemTool` enum in `tool_constants.py`. This makes them auto-load with every profile — no `tools.txt` changes needed. They access `memory_manager` via `deps.memory_manager`, not via the `tool_manager` kwarg (that's for task management tools only).

### Configuration

| Env var | Default | Description |
|---|---|---|
| `REACHY_MINI_DATA_DIRECTORY` | `~/.reachy_mini/data/` | Root directory for all persistent data |
| `REACHY_MINI_MEMORY_ENABLED` | `true` | Set to `false` to disable the entire memory system |

---

## Memory growth

Active memory grows unbounded — every `save_memory` call appends a line. When the file exceeds ~1,500 tokens (~5,250 chars), a warning is logged. There is no automatic archival or pruning; the user or a future feature is responsible for managing growth. In practice, the context budget (~11,000 tokens available) provides ample headroom.

### Token estimation

We use a character-count heuristic (`len(text) / 3.5`) rather than a tokenizer dependency. This is intentionally conservative — GPT-4 family models average ~4 chars/token for English.

---

## Thread safety

- `MemoryManager` uses a `threading.Lock` for all active memory reads/writes.
- Conversation log appends (`log_turn`) do not acquire the lock — append-only file writes are atomic on Linux for small payloads.
- The lock is a `threading.Lock` (not asyncio) because the manager may be accessed from both the async event loop and potential future background workers.
- If profiling shows `log_turn` blocks the audio loop, it can be wrapped in `asyncio.to_thread()`.

---

## Security & privacy

- Active memory content is written by the LLM. It is read back as plain text and concatenated into the system prompt — never `eval`'d or executed.
- The default data directory (`~/.reachy_mini/data/`) is outside the project repo, so memory files are never accidentally committed.
- Conversation logs contain full transcripts. Users who need privacy can set `REACHY_MINI_MEMORY_ENABLED=false` or manually delete the data directory.
- Directory permissions inherit the process umask. On a robot this is typically world-readable. For sensitive deployments, add `mode=0o700` to the `mkdir` calls.

---

## Implementation summary

### Files created

| File | Purpose |
|---|---|
| `src/.../memory/__init__.py` | Package init, re-exports `MemoryManager` |
| `src/.../memory/memory_manager.py` | Core 3-tier manager (242 lines) |
| `src/.../tools/save_memory.py` | SystemTool: persist facts to active memory |
| `src/.../tools/recall_memory.py` | SystemTool: search active + archived memory |
| `tests/memory/__init__.py` | Test package init |
| `tests/memory/test_memory_manager.py` | 28 tests covering all tiers |

### Files modified

| File | Change |
|---|---|
| `config.py` | Added `MEMORY_ENABLED` (default: true) and `DATA_DIRECTORY` config vars |
| `tool_constants.py` | Added `SAVE_MEMORY` and `RECALL_MEMORY` to `SystemTool` enum |
| `core_tools.py` | Added `memory_manager` field to `ToolDependencies` |
| `prompts.py` | Memory block injection after prompt include expansion |
| `openai_realtime.py` | Transcript logging, tool call logging with args, session rotation |
| `main.py` | MemoryManager construction and injection |
| `prompts/default_prompt.txt` | Added `## MEMORY RULES` section |
| `background_tool_manager.py` | Added `args_json_str` to `ToolNotification` for conversation logging |

### Key decisions

1. **Flat files over vector DB** — Letta Filesystem benchmark shows 74% LoCoMo with flat files, beating mem0's 68.5% with vector search. Simpler, no dependencies, sufficient for <150 conversations.
2. **Token cap at 1500** — conservative budget within the 4-6K tokens available for memory in the 16K instruction+tools cap.
3. **LLM-driven save** — the LLM decides what to save (via `save_memory` tool call), not automatic extraction. More selective, avoids noise.
4. **Built on PR #205** — memory tools registered as `SystemTool` (auto-loaded regardless of profile), runs through `BackgroundToolManager`.
5. **Enabled by default** — `REACHY_MINI_MEMORY_ENABLED=true`, configurable via env var.

### Quality review fixes applied

1. Unified `recall_memory` response schema (`active_matches` list in all cases)
2. Loop archival until under cap (with degenerate-case guard for single huge entries)
3. Tool args now logged in JSONL (added `args_json_str` to `ToolNotification`)
4. `new_session()` race condition fixed (acquires lock)
5. Removed duplicated empty-fact guard from `save_memory` tool
6. Removed empty-query rejection from `recall_memory` tool (manager handles it)
7. Moved config import to top-level in `main.py`
8. Unified typing style (modern `list[str]`, `dict[str, Any]`)
9. Filtered breadcrumb metadata lines from archive recall results

### Test results

**75/75 tests pass** (28 memory + 47 existing)

---

## Future improvements

These are explicitly out of scope for V1 but the architecture supports them:

- **Vector search over archives**: Swap the substring search in `recall()` with an embedded vector DB (e.g., Qdrant local, SQLite with vector extensions). No changes to tools or prompt injection needed.
- **Sleep-time compute**: Run a background LLM pass between sessions to extract key facts from logs and update active memory automatically ([Letta pattern](https://www.letta.com/blog/sleep-time-compute)).
- **Conversation summarization**: Periodically summarize conversation logs and store as memory entries, reducing the need for raw log access.
- **Multi-user scoping**: Namespace the data directory by user ID. `MemoryManager(data_dir, user_id="global")` → `MemoryManager(data_dir, user_id="remi")`.
- **Memory editing/deletion**: A `forget_memory` tool for the LLM to remove outdated facts.
- **Importance scoring**: Replace oldest-first archival with relevance + frequency scoring ([ACM survey](https://dl.acm.org/doi/10.1145/3748302) found up to 10% performance gain).

---

## References

- [gpt-realtime model specs](https://platform.openai.com/docs/models/gpt-realtime) — 32K context, 16K instruction+tools cap
- [Context Summarization with Realtime API (OpenAI Cookbook)](https://developers.openai.com/cookbook/examples/context_summarization_with_realtime_api/) — audio = ~10x text tokens
- [Realtime API truncation behavior](https://community.openai.com/t/what-does-auto-truncation-in-realtime-api-actually-do/1356153) — preserves system instructions, drops oldest items
- [mem0 paper (arXiv)](https://arxiv.org/abs/2504.19413) — extraction/deduplication pipeline, benchmark results
- [Letta Filesystem benchmark](https://www.letta.com/blog/benchmarking-ai-agent-memory) — 74% LoCoMo with flat files, beating vector DB approaches
- [MemGPT paper (arXiv)](https://arxiv.org/abs/2310.08560) — virtual memory metaphor for LLM agents
- [Letta sleep-time compute](https://www.letta.com/blog/sleep-time-compute) — background memory processing between conversations
- [ConvoMem benchmark](https://arxiv.org/html/2511.10523v1) — simple approaches beat RAG for <150 conversations
- [ACM Survey on LLM Agent Memory](https://dl.acm.org/doi/10.1145/3748302) — indiscriminate storage degrades performance; utility-based deletion helps
- [Engineering for Real-Time Voice Agent Latency (Cresta)](https://cresta.com/blog/engineering-for-real-time-voice-agent-latency) — tool call latency budgets
- [Realtime API cost accumulation](https://community.openai.com/t/realtime-api-pricing-vad-and-token-accumulation-a-killer/979545) — token accumulation costs
