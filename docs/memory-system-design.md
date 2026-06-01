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

**The seven `kind` values** [a closed vocabulary the dreamer assigns; rejected at
write time if not one of these]: `fact` [stable truths], `preference` [likes, styles,
how they want things done], `event` [something that happened], `skill` [something the
user can do or is learning], `relationship` [people and how they relate], `goal`
[what the user is trying to achieve], `other` [escape hatch]. The `id` is ASCII-only
and cross-platform safe, matching `^\d{4}-\d{2}-\d{2}_<slug>_<3-hex>$` [the 3-hex
suffix keeps the slug split unambiguous].

A design choice visible in the body: the dreamer **cites its evidence inline with
source-log timestamps**, so any memory is auditable back to what was actually said.

**Example: a real memory file, copied verbatim from a running instance.** [It spans
five conversations across two days, links to a related `notation` memory, and its
event date is the latest source, 2026-05-05.]

```markdown
---
id: 2026-04-20_chess-fried-liver_4b7
created: "2026-04-20T20:20:08Z"
sources: [2026-04-17_14-37_2.log, 2026-04-17_14-45.log, 2026-04-17_14-54_3.log, 2026-04-17_17-34_2.log, 2026-05-05_09-29.log]
kind: goal
tags: [chess, fried-liver-attack, opening-prep, teaching]
related_to: [2026-04-20_chess-notation_6b1]
pinned: false
supersedes: null
superseded_by: null
---
User wants to teach a counter to the Fried Liver Attack and repeatedly returns to the exact trap line ending in smothered mate.
From 2026-04-17_14-37_2.log: final corrected Fried Liver setup was "e four, e five, knight to f three, knight to c six, bishop to c four, knight to f six and knight to g five" (14:42:12). At 14:43:19 the user wanted it written that they want to teach a counter to this attack.
From 2026-04-17_14-45.log: after the Fried Liver position, user said the good move for Black is "d5" (14:48:39). Then White takes and Black's c6 knight is attacked; the user likes White pushing to d4 (14:48:47). User then gave the trap: Black plays "queen to d6" to invite the fork (14:49:35); White plays Nf7 forking queen and rook (14:49:42); Black queen goes to c6 (14:50:13); a typical lower-level blunder is White taking the rook with Nh8 (14:50:26); then Black queen takes g2 (14:50:34). User clarified this does not win material immediately, but threatens the rook with checks, so White's only move is Rf1 (14:51:06). Then Black plays Qe4+ (14:51:12). White can block with the queen and lose it to the knight, or blunder bigger with Be2 (14:51:30, 14:51:38). The finish the user highlighted: the knight from d4 jumps to f3 for a smothered mate; the bishop cannot capture because it is pinned, and the king is suffocated by its own pieces (14:51:52, 14:52:23).
From 2026-04-17_14-54_3.log: user insisted on recalling "all of the moves with the explanations and everything" (14:56:05), reinforcing that this is a memorization/teaching line worth preserving.
From 2026-04-17_17-34_2.log: the user explicitly asked to remember the exact Fried Liver line again (17:37:04-17:37:22), corrected the assistant to search the deeper session (17:40:27), and asked for "everything" from it (17:42:20). The assistant's recap matched the stored line: Black's strong move is d5, the line is not about winning material immediately, but about luring overcommitment, getting initiative and checks, and finishing with Nf3 smothered mate because the bishop is pinned and the king is boxed in (17:43:24-17:44:39).
From 2026-05-05_09-29.log: after asking "Now do you remember anything about chess?" (09:31:20), the user asked, "Can you give me the full line please?" (09:31:43), and responded, "This is great. Thank you." (09:33:11) after hearing the spoken sequence again.
Keep this paired with the notation preference memory when presenting the line aloud.
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

**Example: a real `active_memory.md`, copied verbatim from a running instance.** [Note
how the pinned name memory honestly preserves uncertainty rather than guessing, and how
*Older* collapses to topic + counts instead of listing every line, keeping the
always-injected index small.]

```markdown
# Memory index

## Core (pinned)
- [2026-04-21_user-name-remy_4d2] The user's name may be Remy, but speaker identity was explicitly uncertain in this group demo.

## Recent (last 30 days)
### chess
- [2026-04-20_chess-fried-liver_4b7] User wants to teach a counter to the Fried Liver Attack and repeatedly returns to the exact trap line ending in smothered mate.
- [2026-05-05_chess-human-teaching_7af] User wants chess teaching framed for humans, not just engine-best play.
- [2026-06-01_chess-trust-memory-over-calculation_b9e] The user values honest source attribution in chess and was impressed that the assistant explained a line from stored memory rather than bluffing calculation.
### memory
- [2026-05-05_memory-recall-chess-check_c3a] The user did another live memory-recall check, then immediately verified chess recall by asking for the full Fried Liver line.
- [2026-05-05_memory-recall-colleagues_8b4] The user validated that Coco and Clément memories were retrievable with uncertainty preserved.
- [2026-05-05_memory-tool-transparency_c61] User is actively testing whether answers came from visible summaries or fresh memory-tool calls.
- [2026-06-01_memory-recall-benchmarking_7d4] The user benchmarked memory by asking for loaded recall first, then verified tool-backed chess recall and explanation quality.
- [2026-06-01_memory-recall-index-limits_5c1] The user probed that memory cannot be searched by date alone and asked whether there was another session today.
### work
- [2026-04-21_local-runtime-split_6c2] User's conversation app mostly runs locally on their laptop, with the LLM remote for now and STT/TTS split across PC and robot.

## Older
Tags (count), ranked by frequency:
- colleagues (8)
- memory (6)
- retrieval (3)
- robotics (3)
- television (3)
- work (3)
- clement (2)
- coco (2)
- remy (2)
- retrieval-test (2)
- uncertainty (2)
- voice (2)
- accessibility (1)
- animation (1)
- apps (1)
- … +54 more tags

Use `recall_memories(tag=...)` to load (also filters by date_from/date_to).
```

## Recall tools [live model]

- `recall_memory(id)`: read one memory by id, plus every memory in its `related_to`.
  Returns full bodies.
- `recall_memories(tag?, date_from?, date_to?, limit)`: filter by topic and/or
  conversation-date range [at least one filter required]. A memory matches a date
  range if *any* of its conversation dates falls in it. Returns the full text of up
  to `limit` matches [body + `dates_discussed`], newest first.

Both return the model-facing view: `created` stripped, `dates_discussed` added.

## (Day) Dreaming

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
