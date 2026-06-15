# Qwen3 DM brain on Modal — deploy guide

The cloud GPU server for the **Best Use of Modal** submission. It serves
`Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` through **stock vLLM** as an OpenAI-compatible endpoint,
exposing both the **Responses API** (`/v1/responses`) and `/v1/chat/completions`.

**Architecture seam:** the judged Gradio Space is CPU-only → it connects to a **local
`speech-to-speech` voice loop** (Silero VAD + Parakeet STT + Qwen3-TTS, serving `/v1/realtime`),
which offloads the **language model** to this server over the Responses API. Tool calls
(dice / scene / robot / `speak_as` / camera) round-trip through this endpoint. This server is the
GPU half (the brain); voice + protocol stay near the robot. The same model can run locally for
the off-grid demo — this is the cloud variant.

> Repurposed from the earlier 3-stage Qwen3-Omni server. The brain is a plain text LLM, so it now
> runs on **one H100** with the stock `vllm/vllm-openai` image — no vllm-omni, no 2-GPU stage split.

## Prerequisites
- A Modal account: `pip install modal` then `modal token new`.
- HF token secret (one-time):
  ```bash
  modal secret create huggingface HF_TOKEN=hf_xxxxxxxx
  ```

## Deploy
```bash
# persistent app with a stable https URL
modal deploy modal/qwen_brain_modal.py

# (optional) pre-bake weights into the cache Volume before a demo
modal run modal/qwen_brain_modal.py::download_weights

# iterate with foreground logs
modal serve modal/qwen_brain_modal.py
```

## GPU sizing
| Config | `gpu=` | Notes |
|---|---|---|
| **FP8 (~30 GB)** ← recommended / default | `"H100"` | Fits one 80 GB H100 with plenty of KV headroom at 32k context. Matches the locked 1×H100/FP8 decision. |
| bf16 original (~60 GB) | `"H100"` / `"H200"` | Set `MODEL_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507"`. Still one card, but far less KV headroom — drop `--max-model-len` if it OOMs, or use H200 (141 GB). |
| tiny/local variant | (run local) | `Qwen3-4B-Instruct-2507` for the fully-local Tiny Titan build — no Modal needed. |

## Cold start & keep-warm
- First-ever run downloads the FP8 weights (~30 GB). `HF_HUB_ENABLE_HF_TRANSFER=1` speeds it up;
  after that they're cached in the `hf-cache` Volume so cold containers skip the download.
- `vllm-cache` Volume keeps torch.compile / cudagraph artifacts so warm restarts skip recompile.
- `startup_timeout` and `timeout` are set to 20 min to cover the first-run weight load + cudagraph
  capture. A single-stage text LLM cold-starts much faster than the old 3-stage Omni server.
- **For a live demo:** set `min_containers=1` in the script (one container always warm, no
  mid-session cold start), then **set it back to 0 afterwards** to stop paying.

## Endpoint contract (for the speech-to-speech voice loop)
`modal deploy` prints a stable URL like:
```
https://<workspace>--qwen3-brain-serve.modal.run
```
The voice loop points its Responses-API base URL at that URL **+ `/v1`**:
```bash
# speech-to-speech launch
speech-to-speech --mode realtime --stt parakeet-tdt --tts qwen3 \
  --llm_backend responses-api \
  --model_name Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --responses_api_base_url https://<workspace>--qwen3-brain-serve.modal.run/v1
```
vLLM ignores the API key, so any non-empty string works if the client requires one. Verify the
server directly with:
```bash
curl https://<workspace>--qwen3-brain-serve.modal.run/v1/models
curl https://<workspace>--qwen3-brain-serve.modal.run/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-30B-A3B-Instruct-2507-FP8","input":"Roll for initiative."}'
```

## Cost — serverless, **$250 grant** (NOT the $20k prize pool)
**Modal is serverless / scale-to-zero:** with `min_containers=0` you pay **$0 while idle**, only
per-second while the GPU is actually running. A single `H100` ≈ **$10/GPU-hr** (verify at
modal.com/pricing), so $250 ≈ **~25 hours of *actual* runtime** (cumulative, not wall-clock) —
roughly double the old 2-GPU Omni budget. Two things burn it:
- **`min_containers ≥ 1`** keeps a GPU up (defeats scale-to-zero). Use it **only** for the live
  demo window (1–2 hr ≈ $10–20), then back to 0.
- **A long `scaledown_window`** is a paid idle tail after each burst — set to 5 min here for cheap
  dev; raise it for the demo to avoid mid-session re-cold-starts.

Dev iterations (scale-to-zero) ≈ $1–3 each.

## Open risks to validate on a real run
1. **Tool-call passthrough** — the core WS1+WS3 risk: app `session.update` tools → speech-to-speech
   → brain Responses-API tool calls → results round-trip. Confirm early (`--tool-call-parser
   qwen3_coder` is set; verify Qwen emits parseable calls and vLLM streams them over `/v1/responses`).
2. **Responses API on this vLLM** — `/v1/responses` is served by recent stock vLLM; confirm the
   pinned `v0.23.0` image exposes it (fallback: route the voice loop at `/v1/chat/completions`).
3. **`web_server` has no readiness probe** — the first request after a cold start blocks for the
   full model load; the voice loop needs a long client timeout (or a warm-up ping after deploy).
4. **Image version** — pinned `vllm/vllm-openai:v0.23.0` (newest stable on Docker Hub). Confirm it
   serves Qwen3-30B-A3B-Instruct-2507-FP8; the `download_weights` prebake uses a light CPU image,
   so only `serve` pulls the GPU image.
5. **FP8 KV headroom** — `--max-model-len 32768` is conservative for a single-user turn-based DM.
   Raise it if long sessions truncate; lower it (or drop to bf16 on H200) if you hit KV OOM.
