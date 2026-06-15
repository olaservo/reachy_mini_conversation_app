# Qwen3-VL vision server on Modal — deploy guide

The cloud GPU server that lets the DM **"read the physical table."** It serves
`Qwen/Qwen3-VL-8B-Instruct` through **stock vLLM** as an OpenAI-compatible endpoint. It is a
**leaf**: you POST it an image + a question, it returns a **text description**. It calls no tools.

**Architecture seam (read `../VISION.md` for the full picture):** the DM brain
(`qwen_brain_modal.py`, `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`) is **text-only**. The app's text
`camera` tool sends a webcam frame to *this* server, gets prose back, and feeds the **prose** (not
the pixels) to the brain. `modal/describe_frame.py` is the client that does the POST.

> ⚠️ This server is the ONLY component that ever sees raw images. Do not wire raw frames into the
> text brain (do not use the cascade's `see_image_through_camera` multimodal re-injection path).

## Prerequisites
- A Modal account: `pip install modal` then `modal token new`.
- HF token secret — **shared with the brain app**, create once:
  ```bash
  modal secret create huggingface HF_TOKEN=hf_xxxxxxxx
  ```

## Deploy
```bash
# persistent app with a stable https URL
modal deploy modal/qwen_vl_modal.py

# (optional) pre-bake weights into the cache Volume before a demo
modal run modal/qwen_vl_modal.py::download_weights

# iterate with foreground logs
modal serve modal/qwen_vl_modal.py
```

## GPU sizing & cost vs the brain
An 8B VL model does **not** need an H100. bf16 weights are ~16 GB; add the ViT vision encoder and
a small KV cache for one image + short prompt.

| Config | `gpu=` | VRAM fit | Rough $/GPU-hr* | Notes |
|---|---|---|---|---|
| **8B, L40S 48 GB** ← default | `"L40S"` | comfortable headroom for encoder activation spikes | **~$1.9** | Recommended. Zero OOM risk, still ~5× cheaper than the brain's H100. |
| 8B, A10G/L4 24 GB | `"A10G"` / `"L4"` | tight but workable for single-image prompts | ~$1.1 / ~$0.8 | Cheaper; less headroom. Drop `--max-model-len` if it OOMs. |
| **4B (tiny-titan swap)** | `"L4"` / `"A10G"` | ~8 GB weights, easy fit | ~$0.8 | Set `MODEL_NAME="Qwen/Qwen3-VL-4B-Instruct"`. Cheaper + lower quality. |

\* Verify current rates at modal.com/pricing. For reference the brain runs on an `H100` ≈ **$10/GPU-hr**;
this server is far cheaper, so table-reads barely dent the $250 Modal grant. Both apps are
**serverless / scale-to-zero** (`min_containers=0` ⇒ **$0 while idle**, billed per-second only while
a frame is actually being processed). A whole demo's worth of table-reads is well under $1.

Cost levers (same as the brain):
- **`min_containers ≥ 1`** keeps the GPU warm (defeats scale-to-zero). Use it **only** during the
  live demo window so a between-turns table-read doesn't cold-start, then set back to 0.
- **`scaledown_window`** is a paid idle tail after each burst — 5 min here for dev; raise for demo.

## Cold start & keep-warm
- First-ever run downloads the VL weights (~16 GB for 8B) via **Xet** (`huggingface_hub[hf_xet]` +
  explicit `HF_TOKEN`; the legacy `hf_transfer` backend is deprecated/inert and stalls). After that
  they're cached in the `hf-cache` Volume (shared with the brain) so cold containers skip the
  download. `vllm-cache` keeps compile/cudagraph artifacts so warm restarts skip recompile.
- `startup_timeout`/`timeout` are 20 min to cover first-run load + cudagraph capture.
- `web_server` has **no readiness probe** — the first request after a cold start blocks for the full
  model load. The camera-tool client (`describe_frame`, default `timeout=60s`) should use a generous
  timeout, or warm the endpoint with a ping right after deploy.

## Endpoint contract
`modal deploy` prints a stable URL like:
```
https://<workspace>--qwen3-vl-serve.modal.run
```
This server exposes OpenAI-compatible **`/v1/chat/completions`** accepting **`image_url`** content
(a base64 JPEG data URL) and returning a **text** description. The client (`describe_frame.py`)
points its `base_url` at that URL **+ `/v1`**; pass it via the **`VL_BASE_URL`** env var
(see `../VISION.md`).

Request shape (what `describe_frame` sends):
```jsonc
{
  "model": "Qwen/Qwen3-VL-8B-Instruct",
  "messages": [
    {"role": "system", "content": "You are the eyes of a tabletop RPG dungeon master..."},
    {"role": "user", "content": [
      {"type": "text", "text": "Describe the current tabletop."},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<...>"}}
    ]}
  ],
  "max_tokens": 512,
  "temperature": 0.2
}
```
Verify the server directly:
```bash
curl https://<workspace>--qwen3-vl-serve.modal.run/v1/models
# or run the helper against a real frame:
python modal/describe_frame.py frame.jpg https://<workspace>--qwen3-vl-serve.modal.run/v1
```

## vLLM flags — ✅ verified on a real run (2026-06-15)
Deployed + smoke-tested against `https://olahungerford--qwen3-vl-serve.modal.run`; a generated
test frame (red circle upper-left, blue square lower-right) was described correctly with positions.
1. **Qwen3-VL support in `vllm/vllm-openai:v0.23.0`** — ✅ registers and serves
   `Qwen/Qwen3-VL-8B-Instruct` (`/v1/models` OK).
2. **`--limit-mm-per-prompt` format** — ✅ the JSON-dict form `'{"image": 1}'` is accepted.
3. **`--trust-remote-code`** — ✅ NOT needed (natively supported).
4. **No tool-call parser** — intentional. This is a leaf vision server, not a tool-caller.

> Cold start: like the brain, the first-ever start runs torch.compile + warmup (minutes);
> subsequent cold starts are faster (compile cache persists in `vllm-cache`). Subsequent runs of the
> client should still allow a generous timeout (no readiness probe).
