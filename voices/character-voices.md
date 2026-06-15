# Character Voices — Qwen3-TTS-VoiceDesign spec

Distinct voices for the DM narrator, the six pregens, and reusable NPC archetypes.
Each entry is a **natural-language voice description** for
`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` (English excels). Workflow per voice:

1. `generate_voice_design(text=<sample line>, instruct=<description below>)` → reference clip.
2. `create_voice_clone_prompt(<clip>)` → reusable prompt for consistent delivery.
3. `generate_voice_clone(voice_clone_prompt=…)` for all subsequent lines.

Register each finished voice as a named speaker in the realtime/TTS backend. `voice_id` =
the speaker name to register.

**Integration (decided):** the DM delivers an in-character line by calling
`speak_as(voice_id, text)` — an app-native tool the **WS1 adapter** intercepts, synthesizing
that voice's clone prompt and streaming the audio (instead of the default Omni speaker). The
`voice_id` values are the names below.

Each entry now also has a **sample line** — 1–2 short in-character sentences used as the
`text` for `generate_voice_design`, producing the reference clip the clone prompt is built from.
Keep them short (one breath), vivid, and distinctly in-character; the wasteland is Fallout-themed.

## Narrator
- **voice_id:** `gm_narrator`
- **description:** A warm, soft-spoken storyteller and Game Master — rich mid-range, relaxed
  and intimate, speaking calmly at a gentle conversational volume as if leaning in across the
  table. Unhurried, lightly wry, never loud or announcer-like; subtle understated timing, not
  booming theatrics. The default voice for all narration.
- **sample line:** "The rusted gate groans softly open, and beyond it the quiet wasteland
  stretches out under a fading sky. Take a breath, wanderers — and tell me where you'd like to go."

## Pregens
| voice_id | character | VoiceDesign description | sample line |
|---|---|---|---|
| `augusta_byron` | augusta-byron (Vault Dweller scientist, L1) | A bright, articulate woman in her late 20s; crisp diction, measured and inquisitive, a faint clinical warmth and unshaken optimism. | "Fascinating — the radiation readings spike near the old reactor. If my calculations hold, we can cross safely. Probably." |
| `tommy_doyle` | tommy-doyle (Survivor gambler, L2) | A smooth, fast-talking man in his mid-30s with a gambler's swagger; warm gravel, a grin you can hear, always working an angle. | "Relax, friend — I've talked my way out of worse than a few raiders. Tell you what, double or nothing on the caps?" |
| `bailey_bigsmile` | bailey-bigsmile (Ghoul wanderer, L3) | A dry, raspy ghoul voice — cracked and gravelly from centuries of radiation; slow, wry, world-weary but oddly kindly. | "Two hundred years I've walked this dust, kid. Trust me — the things that'll kill you out here are never the ones you're watchin' for." |
| `old_tallman` | old-tallman (Super Mutant philosopher, L2) | An immense, deep, slow-rumbling bass; ponderous and philosophical, with a surprising gentleness beneath the bulk. | "Small ones ask why the world ended. Old Tallman asks... what it wishes to become next." |
| `hazel_johnson` | hazel-johnson (Brotherhood Field Scribe, L1) | A young woman, earnest and disciplined; clear, slightly formal military cadence; eager, principled, by-the-book. | "Scribe Johnson reporting. The technology in this bunker must be catalogued and recovered for the Brotherhood. By the book, no exceptions." |
| `marvin` | marvin (Mister Handy robot, L2) | A chipper retro robotic butler; clipped upper-class British accent with a faint metallic, tinny modulation; eternally, almost smugly, polite. | "Ah, splendid! Another delightful brush with certain death. Might I suggest, sir, that we not all perish before tea?" |

## Reusable NPC archetypes (Machine Frequency + general wasteland)
| voice_id | use | VoiceDesign description | sample line |
|---|---|---|---|
| `npc_raider` | hostile raiders | A snarling, manic male voice; ragged and aggressive, unhinged glee. | "Well, well, well! Fresh meat wandered right into my yard! Hand over the caps and maybe — MAYBE — I let you keep your kneecaps!" |
| `npc_settler` | nervous townsfolk | A tired, wary working-class voice; soft, hesitant, hopeful underneath. | "You're... you're not with the raiders, are you? Please. We don't have much left, but we could use a little help." |
| `npc_merchant` | traders | A oily, ingratiating fast patter; sing-song salesmanship, never quite trustworthy. | "Step right up, friend! Stimpaks, scrap, secrets — all guaranteed genuine, mostly! For you? A very special, very fair price." |
| `npc_overseer` | authority/comms | A cold, clipped institutional voice over a crackle of radio static. | "Attention, vault residents. Remain calm and return to your quarters. The situation is fully under control. Repeat — fully under control." |

## Notes
- Keep descriptions short and concrete — VoiceDesign responds best to a few vivid traits.
- Generate once, store the voice-clone prompts as assets; do NOT re-design per session.
- `marvin` and `npc_overseer` benefit from light post-FX (metallic / radio) if the backend
  allows; otherwise bake the timbre into the description.
- **This file is the source of truth.** `voices/generate_voices.py` parses the
  `voice_id` / `description` / `sample line` for all 11 voices straight out of this markdown
  (Narrator block + the two tables), so editing a line here changes what gets generated. After
  editing, re-run the batch (see `voices/README.md`).
