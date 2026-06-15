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

## Narrator
- **voice_id:** `gm_narrator`
- **description:** A warm, theatrical storyteller and Game Master — rich mid-range, inviting
  and slightly wry, with confident dramatic timing. The default voice for all narration.

## Pregens
| voice_id | character | VoiceDesign description |
|---|---|---|
| `augusta_byron` | augusta-byron (Vault Dweller scientist, L1) | A bright, articulate woman in her late 20s; crisp diction, measured and inquisitive, a faint clinical warmth and unshaken optimism. |
| `tommy_doyle` | tommy-doyle (Survivor gambler, L2) | A smooth, fast-talking man in his mid-30s with a gambler's swagger; warm gravel, a grin you can hear, always working an angle. |
| `bailey_bigsmile` | bailey-bigsmile (Ghoul wanderer, L3) | A dry, raspy ghoul voice — cracked and gravelly from centuries of radiation; slow, wry, world-weary but oddly kindly. |
| `old_tallman` | old-tallman (Super Mutant philosopher, L2) | An immense, deep, slow-rumbling bass; ponderous and philosophical, with a surprising gentleness beneath the bulk. |
| `hazel_johnson` | hazel-johnson (Brotherhood Field Scribe, L1) | A young woman, earnest and disciplined; clear, slightly formal military cadence; eager, principled, by-the-book. |
| `marvin` | marvin (Mister Handy robot, L2) | A chipper retro robotic butler; clipped upper-class British accent with a faint metallic, tinny modulation; eternally, almost smugly, polite. |

## Reusable NPC archetypes (Machine Frequency + general wasteland)
| voice_id | use | VoiceDesign description |
|---|---|---|
| `npc_raider` | hostile raiders | A snarling, manic male voice; ragged and aggressive, unhinged glee. |
| `npc_settler` | nervous townsfolk | A tired, wary working-class voice; soft, hesitant, hopeful underneath. |
| `npc_merchant` | traders | A oily, ingratiating fast patter; sing-song salesmanship, never quite trustworthy. |
| `npc_overseer` | authority/comms | A cold, clipped institutional voice over a crackle of radio static. |

## Notes
- Keep descriptions short and concrete — VoiceDesign responds best to a few vivid traits.
- Generate once, store the voice-clone prompts as assets; do NOT re-design per session.
- `marvin` and `npc_overseer` benefit from light post-FX (metallic / radio) if the backend
  allows; otherwise bake the timbre into the description.
- TODO: pick the actual sample lines per character (1–2 in-character sentences each).
