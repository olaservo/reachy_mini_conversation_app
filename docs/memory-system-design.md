# Memory system design

How the Reachy Mini conversation app remembers things across sessions. This
describes the current implementation.

## Core idea

Two strictly separated phases:

- **Live conversation [read-only].** The robot talks and *reads* its memory. It
  never writes memory files.
- **Dreaming [the only writer].** A separate LLM pass turns raw conversation logs
  into curated memory files. It runs in the background during a conversation, over
  the logs left by *previous* sessions.

The live model is handed a short, auto-curated index at the start of every session
and pulls full details on demand with recall tools.

## Storage layout

```
$DATA_DIRECTORY/memory/                 [default: ~/.reachy_mini/data/memory]
├── active_memory.md                    # The index, injected into the system prompt
├── memories/
│   └── YYYY-MM-DD_<slug>_<hex3>.md      # One atomic memory per file
└── logs/
    ├── pending/                         # Live log + logs waiting to be dreamed
    └── processed/                       # Logs already dreamed
```

The index is always derivable from the memory files, so it is safe to delete and
rebuild [`_ensure_index` rebuilds it at startup if missing].

## Memory file format

Each memory is one markdown file: YAML-ish frontmatter plus a body.

```markdown
---
id: 2026-04-17_chess-openings_a3f       # identifier; the date prefix is opaque
created: 2026-04-17T14:32:10Z           # audit only; not shown to the live model
sources: [2026-04-14_09-15.log, ...]    # the conversations this came from
kind: preference                        # fact | preference | event | skill | relationship | goal | other
tags: [chess, openings]                 # first tag is primary [drives index grouping]
related_to: []                          # sparse "must read together" links
pinned: false                           # true only for identity/core facts
supersedes: null                        # / superseded_by: explicit replacement links
---

First line is a one-sentence TL;DR [the only thing the index shows]. Then detail.
```

## Dates

A memory's date is the date of the **conversation** it came from [parsed from the
`sources` log filenames], never the date the dreamer wrote the file. A memory can
span several days, so it has several event dates. This is the one notion of "when
something happened", defined in `memory/dates.py` and used by the index and by
`recall_memories`. The live model is never shown `created`; it sees `dates_discussed`.

The session prompt also carries `The current date is YYYY-MM-DD.` [from the local
system clock, or "unknown" if that fails], so the model can resolve "yesterday" or
"a few weeks ago" into concrete dates.

## The index (`active_memory.md`)

Regenerated from frontmatter at the end of every dream pass. Three tiers:

- **Core**: pinned memories, always shown.
- **Recent**: non-pinned, discussed within the last 30 days, grouped by primary tag,
  each as `[id] one-line summary`.
- **Older**: ranked tag counts only [topic + volume signal, not the individual lines].

It is appended to the system prompt at session start by
`get_session_instructions` -> `get_memory_block`.

## Recall tools [live model]

- `recall_memory(id)`: read one memory by id, plus every memory in its `related_to`.
  Returns full bodies.
- `recall_memories(tag?, date_from?, date_to?, limit)`: filter by topic and/or
  conversation-date range [at least one filter required]. A memory matches a date
  range if *any* of its conversation dates falls in it. Returns the full text of up
  to `limit` matches [body + `dates_discussed`], newest first.

Both return the model-facing view: `created` stripped, `dates_discussed` added.

## Dreaming

Runs on a daemon thread per session [`DreamScheduler`], launched from
`base_realtime.py` right after the session opens, so it never blocks startup. The
dreamer [`memory/dreamer.py`] is a synchronous LLM agent with its own tools
[`read_log`, `find_related_memories`, `read_memory`, `write_memory`, `update_memory`,
`rebuild_index`, ...]. For each pending log it extracts atomic memories, then rebuilds
the index. Every step is logged to the terminal.

The dreamer's prompt enforces a few rules: atomicity [one memory = one kind + one
topic], overlap-first [prefer updating an existing memory], evidence [no unjustified
synthesis], explicit conflict resolution [`supersedes`/`superseded_by`, never silent
overwrite], and sparing use of `pinned`.

It is the **only** writer of memory files, which is why no locks are needed: writes
are atomic [temp file then `os.replace`], the live side only reads, and the rare
read-during-write is harmless. The dreamer skips the currently-open session log.

**The tell.** A soft chime marks the start [rising] and finish [falling] of a dream,
played via `robot.media.play_sound`. A hidden context note is injected into the live
conversation at each [via `conversation.item.create`, with no forced response, the
same mechanism as the idle signal], telling the robot it just consolidated memories.
So if asked "what was that sound?" it can explain, but it never raises it unprompted.

## Configuration

- `REACHY_MINI_MEMORY_ENABLED` [default true]: master switch.
- `REACHY_MINI_DATA_DIRECTORY` [default `~/.reachy_mini/data`]: where everything lives.
- `MEMORY_DREAMER_MODEL` [default `gpt-5.4`]: the dreamer's chat model. It must be a
  Responses-API model, not a realtime alias.
- `OPENAI_API_KEY`: used by the dreamer [the live audio backend is separate].

## Privacy

Logs contain full transcripts. Set `REACHY_MINI_MEMORY_ENABLED=false`, or delete the
data directory, to opt out.
