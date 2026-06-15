"""Serve the **vision** model Qwen3-VL-8B-Instruct on Modal (OpenAI-compatible vLLM).

This is the *leaf* "read the table" server for the Build Small DM. It is NOT the brain.
The DM brain (`Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`, served by `qwen_brain_modal.py`) is a
**text-only** LLM. To let the brain "see" the tabletop we run Qwen3-VL here as a SEPARATE
endpoint that takes an image and returns a **text description**. The app's text `camera` tool
sends a JPEG frame here (see `modal/describe_frame.py`), gets back prose, and that prose — not
the raw image — is what flows back into the text brain.

⚠️  Do NOT route raw images into the brain. This server is the only thing that ever sees pixels.
    It is a leaf: it returns text and calls no tools (no `--tool-call-parser` here, unlike the brain).

Mirrors the brain Modal app (same image base, Volumes, secret, scale-to-zero, download_weights),
differing only in: model, GPU (cheaper — 8B VL doesn't need an H100), and multimodal serve flags.

Deploy:   modal deploy modal/qwen_vl_modal.py
Iterate:  modal serve  modal/qwen_vl_modal.py     (foreground logs)
Secret:   modal secret create huggingface HF_TOKEN=hf_xxx   (one-time; shared with the brain)

See modal/README.md for GPU sizing, cost vs the brain, cold-start, and the endpoint contract.
"""

import subprocess

import modal


# Qwen3-VL-8B-Instruct: bf16 weights ~16 GB + a ViT vision encoder + KV cache. Well under the
# 32B param cap. For the Tiny-Titan / cheaper swap use "Qwen/Qwen3-VL-4B-Instruct" (~8 GB) which
# fits an even smaller card (L4 24 GB) at lower cost and quality.
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
VLLM_PORT = 8000
MINUTES = 60

# --- Image -------------------------------------------------------------------
# Same stock vLLM OpenAI-compatible image as the brain. ✅ VERIFIED 2026-06-15: v0.23.0 registers
# Qwen3-VL and serves Qwen3-VL-8B-Instruct (model loaded, /v1/models OK) with the multimodal
# `/v1/chat/completions` image_url path working — no --trust-remote-code needed.
image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.23.0", add_python="3.12")
    .entrypoint([])  # the image's entrypoint IS `vllm serve`; clear it so we launch our own cmd
    .env({"VLLM_USE_V1": "1"})
)

# Lightweight CPU image just for pre-baking weights — no need for the heavy GPU image.
# hf_xet gives the modern fast Xet transfer (the weights are Xet-backed; legacy hf_transfer is
# deprecated/inert and trickled the brain's shards into a 30-min timeout). Do NOT set
# HF_XET_HIGH_PERFORMANCE=1 — on the brain it left hung threads after commit. Default Xet is fast.
download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_xet]")
)

# --- Caches & secrets --------------------------------------------------------
# Same named Volumes as the brain so weights/compile caches are shared across both apps.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)      # model weights
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)  # compile/cudagraph cache
hf_secret = modal.Secret.from_name("huggingface")  # must contain HF_TOKEN

app = modal.App("qwen3-vl")


@app.function(
    image=image,
    # GPU choice: L40S (48 GB). An 8B bf16 VL model (~16 GB weights) + ViT encoder + KV cache fits
    # an A10G/L4 (24 GB) for short single-image prompts, but L40S gives comfortable headroom for the
    # vision encoder's activation spikes and longer multimodal context with zero OOM risk — and it's
    # still far cheaper than the brain's H100 (~$1.9/GPU-hr vs ~$10). Cheaper/tiny tradeoff: drop to
    # gpu="A10G" (or "L4") for the 4B variant, or to shave cost on the 8B at the price of tighter VRAM.
    gpu="L40S",
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    secrets=[hf_secret],
    timeout=20 * MINUTES,  # covers the slow first-run weight load + cudagraph capture
    # Serverless: min_containers=0 ⇒ $0 while idle. scaledown_window is the paid idle tail after the
    # last request — short (5 min) for cheap dev; raise it for a live demo so a table-read between
    # turns doesn't re-cold-start.
    scaledown_window=5 * MINUTES,
    min_containers=0,  # scale-to-zero. Set to 1 ONLY during the demo window, then back to 0.
)
@modal.concurrent(max_inputs=16)  # one server handles many concurrent table-read requests
@modal.web_server(port=VLLM_PORT, startup_timeout=20 * MINUTES)
def serve():
    """Launch the vLLM OpenAI-compatible server for Qwen3-VL (Modal proxies VLLM_PORT)."""
    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        # NO --tool-call-parser / --enable-auto-tool-choice: this is a leaf vision server that
        # returns a text description. Tool calling belongs to the brain, not here.
        #
        # Multimodal: cap images per prompt to 1 (the camera tool sends a single frame). This bounds
        # the multimodal cache and rejects accidental multi-image payloads.
        "--limit-mm-per-prompt",
        '{"image": 1}',
        # ✅ VERIFIED 2026-06-15: v0.23.0 accepts the JSON-dict form '{"image": 1}' (server started
        #    and served a single-image request fine).
        #
        # Modest context: a single table-read is one image + a short prompt + a short description.
        # 16k is plenty and keeps the KV cache small on the L40S; raise only if descriptions truncate.
        "--max-model-len",
        "16384",
        # ✅ VERIFIED 2026-06-15: loads WITHOUT --trust-remote-code on v0.23.0 (natively supported).
        # ⚠️ OPTIONAL: Qwen3-VL supports mrope/large images; if you want to bound vision tokens you
        #    can add `--mm-processor-kwargs '{"max_pixels": ...}'`. Left off here — confirm defaults
        #    are sane for a ~1MP webcam frame before tuning.
    ]
    subprocess.Popen(cmd)  # non-blocking; Modal proxies VLLM_PORT once vllm is up


@app.function(
    image=download_image,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[hf_secret],
    timeout=60 * MINUTES,  # generous headroom; the 8B VL weights are smaller than the brain's 30 GB
)
def download_weights():
    """Pre-bake the VL weights into the hf-cache Volume before a demo (optional one-off).

    Run with:  modal run modal/qwen_vl_modal.py::download_weights
    """
    import os

    from huggingface_hub import snapshot_download

    # Pass the token explicitly to dodge the anonymous-HF rate limit (the public-repo warning is
    # benign). Log presence (not the value) so a missing/misnamed secret is obvious.
    token = os.environ.get("HF_TOKEN")
    print(f"HF_TOKEN present in env: {bool(token)}")
    snapshot_download(MODEL_NAME, token=token)
    hf_cache.commit()
    print(f"Cached {MODEL_NAME} into the hf-cache volume.")
