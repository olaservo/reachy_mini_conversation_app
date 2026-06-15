# Qwen3-TTS voices on Modal — deploy guide

The per-character voice synthesizer for the DM. Serves `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
behind a **custom OpenAI-compatible FastAPI server** (`voices/tts_server.py`). This is the
model that unlocks the 11-voice roster in `voices/character-voices.md` — beyond the cascade's
default narrator.

> ⚠️ **Why a custom server (not vllm-omni):** vLLM-Omni only supports **offline** inference
> for Qwen3-TTS — there is no online `/v1/audio/speech` server for the TTS stages yet, so the
> old `vllm serve --omni` path was a dead end. Instead we run our own FastAPI app that wraps
> the transformers `qwen_tts` voice-clone API around the **committed** clone prompts in
> `voices/assets/clone_prompts/`. The `voice` request param is a roster id resolved to a clone
> prompt **server-side**, so the cascade's `VOICE_PROMPTS` map stays empty.

**Architecture seam:** the cascade TTS provider (`reachy-dm-cascade .../cascade/tts/qwen3_tts.py`)
calls this server's **`/v1/audio/speech`** with `voice=<voice_id>` to render a line in a designed
voice and streams the raw PCM back. The brain (the other Modal app) stays the LLM + default narration.

```
Space (CPU) → cascade (local) ─┬─ Brain 30B  (Modal, /v1/responses)      ← LLM + default narration
                               └─ TTS 1.7B   (Modal, /v1/audio/speech)   ← per-character voices
```

## ⚠️ Local is the preferred home-use target (this is the hackathon-cloud variant)

1.7B is tiny (~3.4 GB bf16). For home use, run the **same FastAPI server** on the machine beside
the robot/cascade and point the cascade at `http://localhost:8091/v1` — a localhost call, **$0**,
lowest latency, and it frees the Modal grant for the 30B brain (which actually needs cloud GPU).
We're on Modal now only to ship within the hackathon window; see "Serving" below for the swap.

## Prerequisites
- Modal account: `pip install modal` then `modal token new`.
- HF token secret (shared with the brain app; one-time):
  ```bash
  modal secret create huggingface HF_TOKEN=hf_xxxxxxxx
  ```

## Serving

### Modal (hackathon-cloud variant)
```bash
modal deploy modal/qwen3_tts_modal.py                       # stable https URL
modal run    modal/qwen3_tts_modal.py::download_weights      # (optional) pre-bake Base + VoiceDesign weights
modal serve  modal/qwen3_tts_modal.py                        # foreground logs
```
`modal deploy` prints a URL like `https://<workspace>--qwen3-tts-voices-serve.modal.run`.
The custom FastAPI server (`voices/tts_server.py`) is launched via `subprocess.Popen` inside
the `@modal.web_server(port=8091)` function (ships the clone prompts + `tts_server.py` via
`.add_local_dir`, runs `uvicorn tts_server:app` with `cwd=/root/voices`). Point the cascade's
`qwen3_tts` provider `base_url` at the Modal endpoint + `/v1`:
```yaml
# cascade.yaml (tts provider)
base_url: https://<workspace>--qwen3-tts-voices-serve.modal.run/v1
api_key:  EMPTY        # the server ignores auth; AsyncOpenAI just needs a non-empty value
model:    Qwen/Qwen3-TTS-12Hz-1.7B-Base
```

### Local (preferred home-use; $0)
```bash
cd voices
QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base uvicorn tts_server:app --port 8091
```
Then set the cascade `base_url` to `http://localhost:8091/v1`. Env knobs: `CLONE_PROMPTS_DIR`
(default `voices/assets/clone_prompts`), `QWEN_TTS_DEVICE` (default `cuda:0`), `DEFAULT_VOICE`
(default `gm_narrator`), `QWEN_TTS_LANGUAGE` (default `English`).

## GPU sizing & cost
| Config | `gpu=` | Notes |
|---|---|---|
| bf16 (recommended) | `"L4"` (24GB) | Cheapest; ample for 1.7B Base + code2wav. |
| alt | `"A10G"` | Also fine; slightly pricier. |

Serverless / scale-to-zero: `min_containers=0` → **$0 idle**. L4 ≈ ~$0.80/GPU-hr (verify at
modal.com/pricing) and the 1.7B cold start is fast, so dev iterations are cents. Set
`min_containers=1` **only** during the live demo, then back to 0.

## Endpoint contract (`POST /v1/audio/speech`)
Request JSON (the cascade sends `voice`/`input`/`response_format`; extra fields are tolerated):
```json
{ "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "voice": "npc_raider",
  "input": "Hand over the caps!", "response_format": "pcm" }
```
Response: a stream of **raw int16 little-endian PCM, mono, 24000 Hz** (media_type `audio/pcm`,
4096-byte chunks) — exactly what the cascade's `response.iter_bytes()` expects. `voice` is a
roster id resolved server-side to a committed clone prompt; an unknown voice falls back to
`DEFAULT_VOICE` (`gm_narrator`) with a logged warning. `response_format:"wav"` returns a WAV
file as a courtesy. Helpers: `GET /v1/models`, `GET /health` (lists loaded voice ids).

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

## Which TTS variant for *runtime* (decided: Base)
Qwen3-TTS ships three checkpoints:
- **VoiceDesign** — design a voice from text (the offline generation phase only).
- **CustomVoice** — built-in named speakers; does NOT accept the clone-prompt API.
- **Base** — `create_voice_clone_prompt` / `generate_voice_clone`. **This is what we serve.**

We design the 11 voices offline (VoiceDesign → ref clip → `create_voice_clone_prompt`), commit
the resulting clone prompts as `voices/assets/clone_prompts/<voice_id>.pt`, and at runtime the
custom server loads them and calls `generate_voice_clone(voice_clone_prompt=...)` on **Base**.

## Open risks to validate on a real GPU smoke test
1. **`torch.load` portability** — the clone prompts were `torch.save`d in the design batch and
   are `torch.load(..., weights_only=False)`d here (qwen_tts must be importable so
   `VoiceClonePromptItem` resolves). Cross-process load is unverified on real tensors.
2. **Output sample rate** — server assumes the model returns 24000 Hz (no-op resample path);
   confirm `generate_voice_clone` returns sr==24000, else the linear-resample fallback engages.
3. **No flash-attn** — `from_pretrained` is called WITHOUT `attn_implementation` (flash-attn is
   not installed); confirm the model loads on the default attention kernel on L4.
4. **Cold-start / startup_timeout** — model load + 11 prompt loads must finish within the
   15-min `startup_timeout`; the `@app.on_event("startup")` load also runs once per container.
5. **Cascade `speak_as` handler not built yet** — lives in the cascade worktree; this worktree
   is the model + serve half.
