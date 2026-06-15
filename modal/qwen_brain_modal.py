"""Serve the DM **brain** Qwen3-30B-A3B-Instruct-2507 on Modal (OpenAI-compatible vLLM).

This is the cloud GPU server for the Build Small "Best Use of Modal" submission. It is the
LLM half of the speech-to-speech cascade: the local `speech-to-speech` voice loop (Silero VAD
+ Parakeet STT + Qwen3-TTS) offloads the language model here over the **Responses API**
(`/v1/responses`), and tool calls (dice / scene / robot / speak_as / camera) round-trip through
this endpoint. The judged Gradio Space stays CPU-only and connects out through the voice loop.

Repurposed from the earlier 3-stage Qwen3-Omni server: the brain is a plain text LLM, so this
runs on **one H100** with stock vLLM (FP8 ~30 GB weights) — no vllm-omni, no multi-GPU split.

Deploy:   modal deploy modal/qwen_brain_modal.py
Iterate:  modal serve  modal/qwen_brain_modal.py     (foreground logs)
Secret:   modal secret create huggingface HF_TOKEN=hf_xxx   (one-time)

See modal/README.md for GPU sizing, cost, cold-start, and the Responses-API endpoint contract.
"""

import subprocess

import modal


# FP8 fits comfortably on a single 80 GB H100 (~30 GB weights, rest for KV cache) and matches
# the locked "1xH100/FP8" decision. For the bf16 original swap to "Qwen/Qwen3-30B-A3B-Instruct-2507"
# (~60 GB — still 1xH100 but far less KV headroom). Tiny/local variant uses Qwen3-4B-Instruct-2507.
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
VLLM_PORT = 8000
MINUTES = 60

# --- Image -------------------------------------------------------------------
# Stock vLLM OpenAI-compatible server image (ships the `/v1/responses` + `/v1/chat/completions`
# endpoints and the qwen3_coder tool-call parser). v0.23.0 is the newest stable tag on Docker Hub.
image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.23.0", add_python="3.12")
    .entrypoint([])  # the image's entrypoint IS `vllm serve`; clear it so we launch our own cmd
    .env({"VLLM_USE_V1": "1"})
)

# Lightweight CPU image just for pre-baking weights — no need for the heavy GPU image.
# hf_xet gives the modern fast Xet transfer (the FP8 shards are Xet-backed; legacy hf_transfer is
# deprecated/inert). NB: do NOT set HF_XET_HIGH_PERFORMANCE=1 — it pulled one ~95 MB/s burst then
# the high-perf client deadlocked (flat network/CPU until timeout). Default Xet is fast and stable.
download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_xet]")
)

# --- Caches & secrets --------------------------------------------------------
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)      # model weights
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)  # compile/cudagraph cache
hf_secret = modal.Secret.from_name("huggingface")  # must contain HF_TOKEN

app = modal.App("qwen3-brain")


@app.function(
    image=image,
    gpu="H100",  # single card: the FP8 30B brain is a plain text LLM, no multi-GPU stage split.
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    secrets=[hf_secret],
    timeout=20 * MINUTES,  # covers the slow first-run weight load + cudagraph capture
    # Serverless: with min_containers=0 you pay $0 while idle. scaledown_window is the paid
    # idle tail after the last request — short (5 min) for cheap dev; raise it for a live demo
    # so you don't re-cold-start between turns.
    scaledown_window=5 * MINUTES,
    min_containers=0,  # scale-to-zero. Set to 1 ONLY during the demo window, then back to 0.
)
@modal.concurrent(max_inputs=16)  # one server handles many concurrent sessions
@modal.web_server(port=VLLM_PORT, startup_timeout=20 * MINUTES)
def serve():
    """Launch the vLLM OpenAI-compatible server for the DM brain (Modal proxies VLLM_PORT)."""
    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        # Tool calling for the DM (dice / scene / robot / speak_as / camera). qwen3_coder is the
        # parser Qwen recommends for Qwen3's Hermes-style tool calls.
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        # Instruct-2507 is non-thinking by default (no <think> blocks); pass the kwarg anyway so
        # the contract is explicit and survives a base-model swap. Harmless no-op for this model.
        "--default-chat-template-kwargs",
        '{"enable_thinking": false}',
        # The DM is single-user + turn-based and doesn't need the model's native 262k context.
        # Cap it to keep the KV cache modest on one H100; raise if long sessions truncate.
        "--max-model-len",
        "32768",
    ]
    subprocess.Popen(cmd)  # non-blocking; Modal proxies VLLM_PORT once vllm is up


@app.function(
    image=download_image,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[hf_secret],
    timeout=60 * MINUTES,  # ~30 GB FP8 pull; Xet is fast but leave generous headroom
)
def download_weights():
    """Pre-bake the brain weights into the hf-cache Volume before a demo (optional one-off).

    Run with:  modal run modal/qwen_brain_modal.py::download_weights
    """
    import os

    from huggingface_hub import snapshot_download

    # The repo is public, but an authenticated pull avoids the anonymous rate limit that stalled
    # the first attempt. Log presence (not the value) so a missing/misnamed secret is obvious.
    token = os.environ.get("HF_TOKEN")
    print(f"HF_TOKEN present in env: {bool(token)}")
    snapshot_download(MODEL_NAME, token=token)
    hf_cache.commit()
    print(f"Cached {MODEL_NAME} into the hf-cache volume.")
