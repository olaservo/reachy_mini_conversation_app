# Qwen3-TTS voices on Modal — deploy guide

The per-character voice synthesizer for the DM. Serves `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
through **vllm-omni** as an OpenAI-compatible speech endpoint. This is the **optional second
model** (see WS-Voices) that unlocks the 11-voice roster in `voices/character-voices.md` —
beyond Omni's 3 built-in speakers (Ethan/Chelsie/Aiden).

**Architecture seam:** the WS1 CPU adapter's `speak_as(voice_id, text)` handler calls this
server's **`/v1/audio/speech`** to render a line in a designed voice and streams it back as
output audio. Omni (the other Modal app) stays the brain + default narration.

```
app (Space, CPU) → adapter (local) ─┬─ Omni 30B   (Modal, /v1/chat/completions)  ← brain+default voice
                                    └─ TTS 1.7B   (Modal, /v1/audio/speech)       ← speak_as character voices
```

## ⚠️ Local is the preferred home-use target (this is the hackathon-cloud variant)

1.7B is tiny (~3.4 GB bf16). For home use, run the **same `vllm serve`** on the machine beside
the robot/adapter and point the adapter at `http://localhost:8091/v1` — a localhost call, **$0**,
lowest latency, and it frees the Modal **$250 grant** for the 30B Omni (which actually needs
cloud GPU). We're on Modal now only to ship within the hackathon window; see
"Local deployment (home use)" below for the swap.

## Prerequisites
- Modal account: `pip install modal` then `modal token new`.
- HF token secret (shared with the Omni app; one-time):
  ```bash
  modal secret create huggingface HF_TOKEN=hf_xxxxxxxx
  ```

## Deploy
```bash
modal deploy modal/qwen3_tts_modal.py                       # stable https URL
modal run    modal/qwen3_tts_modal.py::download_weights      # (optional) pre-bake weights
modal serve  modal/qwen3_tts_modal.py                        # foreground logs
```

## GPU sizing & cost
| Config | `gpu=` | Notes |
|---|---|---|
| bf16 (recommended) | `"L4"` (24GB) | Cheapest; ample for 1.7B + code2wav. |
| alt | `"A10G"` | Also fine; slightly pricier. |

Serverless / scale-to-zero: `min_containers=0` → **$0 idle**. L4 ≈ ~$0.80/GPU-hr (verify at
modal.com/pricing) and the 1.7B cold start is fast, so dev iterations are cents. Set
`min_containers=1` **only** during the live demo, then back to 0.

## Endpoint contract (for the WS1 adapter `speak_as`)
`modal deploy` prints a URL like `https://<workspace>--qwen3-tts-voices-serve.modal.run`.
```bash
# adapter .env (TTS endpoint, separate from OMNI_BASE_URL)
TTS_BASE_URL=https://<workspace>--qwen3-tts-voices-serve.modal.run/v1   # or http://localhost:8091/v1
TTS_API_KEY=EMPTY
TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
```
`speak_as` → `POST /v1/audio/speech` with `{model, voice: <voice_id>, input: <text>}` (and/or the
streaming speech websocket). Returns audio the adapter resamples to 16k and emits as
`response.output_audio.delta`. (Confirm the exact speech request/stream shape against
`vllm-omni/examples/online_serving/text_to_speech/qwen3_tts/openai_speech_client.py`.)

## Voice-generation workflow (one-time, offline — do this once, store as assets)
Per `voices/character-voices.md` (11 voices: `gm_narrator` + 6 pregens + 4 NPC archetypes):
1. **Design** each voice from its NL description with **VoiceDesign** (`generate_voice_design`).
2. **Create a reusable clone prompt** (`create_voice_clone_prompt`) per voice.
3. Store the clone prompts / speaker embeddings as **committed assets** — do NOT re-design per session.
4. At runtime, `speak_as` renders lines from those stored voices.

**This is now scripted.** `voices/generate_voices.py` parses the 11 voices from
`voices/character-voices.md` and runs the full design → clone-prompt → save workflow, writing
`voices/assets/{voices.json, ref_clips/*.wav, clone_prompts/*.pt}`. Run it on Modal L4 via the
`design_voices_main` entrypoint added to `qwen3_tts_modal.py`:
```bash
modal run modal/qwen3_tts_modal.py::design_voices_main          # all 11 (writes assets back locally)
modal run modal/qwen3_tts_modal.py::design_voices_main --only npc_raider
python voices/generate_voices.py --dry-run                      # laptop-safe spec check
```
Cost: a few minutes on an L4, one-time, **< $1**. Full details + API-signature flags +
consumer handoff: **`voices/README.md`**.

## Which TTS variant for *runtime* (open decision)
Qwen3-TTS ships three checkpoints — pick the runtime serving model:
- **VoiceDesign** — design a voice from text (the generation phase). Default here.
- **CustomVoice** — built-in named speakers + **register precomputed/designed speakers by name**
  → cleanest for `speak_as(voice_id)` (call by name). Likely the runtime choice.
- **Base** — clone from a reference clip per call.

→ TODO: validate whether we serve **CustomVoice** with the 11 designed voices precomputed (call
by `voice_id`), vs. passing clone prompts at request time. See
`vllm-omni/examples/online_serving/text_to_speech/qwen3_tts/precompute_custom_voice.py`.

## Open risks to validate on a real run
1. **`/v1/audio/speech` request/stream shape** — confirm against the vllm-omni speech client + `serving_speech.py`.
2. **Runtime variant** (CustomVoice vs Base vs VoiceDesign) — decide before wiring `speak_as`.
3. **Designed-voice persistence** — how precomputed speakers are registered/loaded at serve time.
4. **Image** — reuses `vllm/vllm-omni:v0.22.0`; confirm it serves Qwen3-TTS (run_server.sh uses `--deploy-config vllm_omni/deploy/qwen3_tts.yaml`; we rely on auto-resolution + a commented fallback).
5. **Adapter `speak_as` handler not built yet** — lives in WS1 (`reachy-dm-ws1-qwen`); this worktree is the model + deploy half.
