# Qwen3-Omni on Modal — deploy guide

The cloud GPU server for the **Best Use of Modal** submission. It serves
`Qwen/Qwen3-Omni-30B-A3B-Instruct` through **vllm-omni** as an OpenAI-compatible endpoint.

**Architecture seam:** the judged Gradio Space is CPU-only → the **WS1 CPU adapter** dials
this server's `/v1/chat/completions` (tools + audio out + `speaker` + image in). This server
is the GPU half; the adapter is the protocol half (WS1). Same model can also run locally for
the off-grid demo — this is the cloud variant.

## Prerequisites
- A Modal account: `pip install modal` then `modal token new`.
- HF token secret (one-time):
  ```bash
  modal secret create huggingface HF_TOKEN=hf_xxxxxxxx
  ```

## Deploy
```bash
# persistent app with a stable https URL
modal deploy modal/qwen_omni_modal.py

# (optional) pre-bake weights into the cache Volume before a demo
modal run modal/qwen_omni_modal.py::download_weights

# iterate with foreground logs
modal serve modal/qwen_omni_modal.py
```

## GPU sizing
| Config | `gpu=` | Notes |
|---|---|---|
| **bf16, default YAML (2-GPU split)** ← recommended | `"H100:2"` / `"H200:2"` | Matches the repo's *verified* topology: Thinker on cuda:0, Talker+Code2Wav on cuda:1. No tuning gamble. |
| bf16, forced single GPU | `"H200"` (141GB) | Needs `--stage-overrides` to collapse stages onto cuda:0 (see commented block in the script). H100-80 is risky (OOM in Code2Wav/KV). |
| FP8 / AWQ (~32GB) | single `"H100"` / `"A100-80GB"` | Cheapest, but a working FP8/AWQ checkpoint of all 3 Omni stages is **unconfirmed** — validate first (see risks). |

> ⚠️ Correction to earlier notes: the bundled `qwen3_omni_moe.yaml` is verified on **2 GPUs**,
> not 1×80GB. Single-card is only realistic on an H200 (141GB) with stage overrides.

## Cold start & keep-warm
- First-ever run downloads ~60GB (bf16). `HF_HUB_ENABLE_HF_TRANSFER=1` speeds it up; after that
  it's cached in the `hf-cache` Volume so cold containers skip the download.
- `vllm-cache` Volume keeps torch.compile / cudagraph artifacts so warm restarts skip recompile.
- 3-stage init (Thinker + Talker + Code2Wav + cudagraph capture) adds startup time → `startup_timeout`
  and `timeout` are set to 20 min for the first run.
- **For a live demo:** set `min_containers=1` in the script (one container always warm, no
  mid-session cold start), then **set it back to 0 afterwards** to stop paying.

## Endpoint contract (for the WS1 adapter)
`modal deploy` prints a stable URL like:
```
https://<workspace>--qwen3-omni-voice-serve.modal.run
```
The adapter points the OpenAI SDK at that URL **+ `/v1`**:
```bash
# adapter .env
OMNI_BASE_URL=https://<workspace>--qwen3-omni-voice-serve.modal.run/v1
OMNI_API_KEY=EMPTY        # vllm ignores it; any non-empty string for the OpenAI SDK
OMNI_MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct
```
Call `POST /v1/chat/completions` with `modalities:["text","audio"]`, `tools`, and
`extra_body={"speaker": "chelsie"}`. (Don't route `/v1/realtime` through Modal — the adapter
design avoids it.)

## Cost — serverless, **$250 grant** (NOT the $20k prize pool)
**Modal is serverless / scale-to-zero:** with `min_containers=0` you pay **$0 while idle**, only
per-second while a GPU is actually running. `H100:2` ≈ **$20/GPU-hr** (H100 ≈ $10/GPU-hr; verify
at modal.com/pricing), so $250 ≈ **~12.5 hours of *actual* runtime** (cumulative, not wall-clock)
— plenty for dev + a demo if you scale to zero. Two things burn it:
- **`min_containers ≥ 1`** keeps a GPU up (defeats scale-to-zero). Use it **only** for the live
  demo window (1–2 hr ≈ $20–40), then back to 0. It's needed for the demo because the 30B
  3-stage cold start is multi-minute — too slow to scale-to-zero between turns of a live session.
- **A long `scaledown_window`** is a paid idle tail after each burst — set to 5 min here for cheap
  dev; raise it for the demo to avoid mid-session re-cold-starts.

Dev iterations (scale-to-zero) ≈ $3–7 each. Cheaper GPU options: `gpu="H200"` single-card
(~half, needs stage overrides) or a validated FP8 checkpoint on 1×H100.

## Open risks to validate on a real run
1. **Single-GPU bf16 fit is unproven** — repo only verifies 2-/3-GPU. Validate `gpu="H200"` single-card before relying on it.
2. **FP8/AWQ for Qwen3-Omni is unconfirmed** — the ~32GB number assumes a working 3-stage quant. Don't assume the stock `-Instruct` repo serves FP8.
3. **Image version** — using `vllm/vllm-omni:v0.22.0` (newest published; no v0.23.0 image yet).
   Confirm 0.22.0 serves Qwen3-Omni-30B-A3B; fallback is the commented pip build (vllm 0.23.0).
   The `download_weights` prebake uses a light CPU image, so only `serve` pulls the GPU image.
4. **`web_server` has no readiness probe** — first request after cold start blocks for the full model load; the adapter needs a long client timeout (or warm-up ping after deploy).
5. **3-stage cold-start time on Modal is unmeasured** — time one cold start, size `startup_timeout`, keep `min_containers=1` for the demo.
6. **`--omni` is vllm-omni-specific** — only present because the image has vllm-omni installed (it patches vllm's `serve`). Confirm `vllm` resolves to the omni-patched CLI.
