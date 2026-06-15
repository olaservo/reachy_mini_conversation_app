# DM character voices — generation & handoff

This directory holds everything to produce the **11 designed character voices** for the
whimsical TTRPG dungeon master, using `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` (+ the `-Base`
checkpoint for cloning). It is a **one-time, GPU-only batch**; the dev box has no GPU, so we
run it on Modal and commit the resulting assets.

```
voices/
  character-voices.md    # SOURCE OF TRUTH: voice_id + description + sample line (11 voices)
  generate_voices.py     # design -> reference clip -> reusable clone prompt -> assets
  README.md              # this file
  assets/
    voices.json          # manifest: voice_id -> {description, sample_line, ref_clip, clone_prompt_path, sample_rate}
    ref_clips/<id>.wav   # the designed reference clip per voice
    clone_prompts/<id>.pt# the reusable voice-clone prompt (torch.save blob) per voice
```

`character-voices.md` is the single editable source. `generate_voices.py` parses the
`voice_id` / `description` / `sample line` straight out of it (narrator bullet block + the two
markdown tables), so changing a sample line there changes what gets generated.

## The workflow each voice runs (per the Qwen3-TTS model card)

1. `generate_voice_design(text=<sample line>, language="English", instruct=<description>)`
   → a reference waveform that embodies the natural-language description.
2. `create_voice_clone_prompt(ref_audio=(wav, sr), ref_text=<sample line>)`
   → a **reusable** speaker prompt (extract features once, reuse forever).
3. Save the clone prompt (`clone_prompts/<id>.pt`) + the reference clip (`ref_clips/<id>.wav`).
   At runtime the cascade calls `generate_voice_clone(..., voice_clone_prompt=<that prompt>)`.

## Dry-run (laptop-safe — no GPU, no model downloads)

Validate the spec parse and see exactly what will be generated:

```bash
python voices/generate_voices.py --dry-run
```

## Run path (chosen): Modal L4 batch

We extend the existing TTS Modal scaffold rather than introduce HF Jobs, so the voice batch
shares the same image conventions, `hf-cache` volume, and `huggingface` secret as the serve app.

```bash
# one-time secret (shared with the serve app):
modal secret create huggingface HF_TOKEN=hf_xxxxxxxx

# run the 11-voice batch on an L4; assets are written back into voices/assets/ locally:
modal run modal/qwen3_tts_modal.py::design_voices_main

# regenerate just one (incremental top-up of voices.json):
modal run modal/qwen3_tts_modal.py::design_voices_main --only npc_raider
```

`design_voices_main` (a `@app.local_entrypoint`) calls the GPU function `design_voices`
(`gpu="L4"`), which `sys.path`-imports `generate_voices.py` (shipped into the container via
`add_local_dir`), runs `run_batch()`, and returns every file under `assets/` as bytes; the
local entrypoint then writes them into `voices/assets/`. **Commit those assets.**

### Cost / time
11 voices × (1 design pass + 1 clone-prompt extraction) on two 1.7B checkpoints. On an L4
(~$0.80/GPU-hr; verify at modal.com/pricing) the batch is a few minutes including cold start —
**well under $1**, one-time. Scale-to-zero means $0 idle afterward.

## ⚠️ API signatures to verify on the first real run

These are taken from the model card + `QwenLM/Qwen3-TTS` but were **not** executed here (no GPU).
Each is flagged inline in `generate_voices.py`:

| Flag | What to confirm |
|---|---|
| `FLAG[import]` | `from qwen_tts import Qwen3TTSModel`, and the pip package name (`qwen-tts`) in `voicedesign_image`. |
| `FLAG[design]` | `generate_voice_design(text=, language=, instruct=)` kwargs + that it returns `(wavs, sr)` with `wavs[0]` a waveform. Some builds may use `instruction=`. |
| `FLAG[clone]` | `create_voice_clone_prompt(ref_audio=(wav, sr), ref_text=)` arg names (also accepts a path/URL/base64). Lives on the **Base** checkpoint. |
| `FLAG[serialize]` | HOW the clone prompt persists. It is opaque (likely tensors); we use `torch.save`/`torch.load`. If it is a plain dict of numpy arrays, `np.savez` also works — but the consumer MUST load it the same way. |
| `FLAG[runtime]` | Whether the cascade server wants the prompt **inline** (`VOICE_PROMPTS`) or as a **registered named speaker** (CustomVoice precompute). See handoff below. |

If any signature differs, fix it in `generate_voices.py` and re-run; nothing else changes.

## Consumer handoff → the cascade Qwen3-TTS provider

The consumer is `reachy-dm-cascade/src/reachy_mini_conversation_app/cascade/tts/qwen3_tts.py`
(a sibling worktree — **do not edit it from here**). It already declares the matching roster:

* `ROSTER_VOICE_IDS` — the same 11 `voice_id`s this batch generates.
* `VOICE_PROMPTS: dict[str, str]` — currently empty; an optional `voice_id -> inline clone
  prompt` map. When a `voice_id` is present here, the provider forwards the prompt to the
  server via `extra_body={"voice_clone_prompt": ...}`. Empty ⇒ the server resolves voices by
  **name** (the default assumption).

There are two ways to register the assets, matching the open `FLAG[runtime]` decision in
`modal/README.md`:

1. **Named-speaker (preferred, CustomVoice path).** Precompute/register the 11 clone prompts as
   named speakers on the Qwen3-TTS serve container (the vllm-omni `precompute_custom_voice.py`
   path referenced in `modal/README.md`), keyed by `voice_id`. Then `VOICE_PROMPTS` stays empty
   and `speak_as(voice_id, text)` resolves the voice by name. The committed `clone_prompts/<id>.pt`
   files are the inputs to that registration step.

2. **Inline-prompt fallback.** If the server accepts a clone prompt per request instead, populate
   `VOICE_PROMPTS` in `qwen3_tts.py` from this batch's output. Because `clone_prompt` is a tensor
   blob (stored at `clone_prompt_path`, not JSON), the inline value must be whatever **wire form**
   the serve endpoint accepts (e.g. a base64-encoded blob or a server-side speaker id). Decide the
   wire form when wiring `speak_as`; until then `voices.json` carries `clone_prompt: null` and the
   real prompt lives in the `.pt` file. The loader must `torch.load` it the same way it was saved
   (`FLAG[serialize]`).

Either way, the artifact this worktree owns and commits is the set of **`clone_prompts/<id>.pt`
files + `voices.json`**; how they get registered into the running TTS server is a thin,
documented step on the cascade/serve side.
