"""Serve Qwen3-Omni-30B-A3B on Modal via vllm-omni (OpenAI-compatible endpoint).

This is the cloud GPU server for the Build Small "Best Use of Modal" submission. The WS1
CPU adapter dials this server's `/v1/chat/completions` (tools + audio out + `speaker` +
image in); the judged Gradio Space stays CPU-only and connects out through that adapter.

Deploy:   modal deploy modal/qwen_omni_modal.py
Iterate:  modal serve  modal/qwen_omni_modal.py     (foreground logs)
Secret:   modal secret create huggingface HF_TOKEN=hf_xxx   (one-time)

See modal/README.md for GPU sizing, cost, cold-start, and the adapter endpoint contract.
"""

import subprocess

import modal

MODEL_NAME = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
VLLM_PORT = 8091
MINUTES = 60

# --- Image -------------------------------------------------------------------
# Lowest-risk path: start from the official vllm-omni image (CUDA + torch + kernels
# already pinned together). It has NO default entrypoint, so we clear it and launch
# `vllm serve --omni` ourselves in serve().
image = (
    modal.Image.from_registry("vllm/vllm-omni:v0.23.0", add_python="3.12")
    .entrypoint([])  # drop any inherited entrypoint
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # faster weight download
            "VLLM_USE_V1": "1",
        }
    )
)

# Fallback image (pip build) if the registry pull is a problem — slower, kernel coupling risk:
# image = (
#     modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12")
#     .entrypoint([])
#     .uv_pip_install("vllm==0.23.0", extra_options="--torch-backend=auto")
#     .uv_pip_install("vllm-omni==0.23.0", "hf_transfer")
# )

# --- Caches & secrets --------------------------------------------------------
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)      # model weights
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)  # compile/cudagraph cache
hf_secret = modal.Secret.from_name("huggingface")  # must contain HF_TOKEN

app = modal.App("qwen3-omni-voice")


@app.function(
    image=image,
    # Default vllm-omni YAML (qwen3_omni_moe.yaml) is verified on 2 GPUs:
    # stage 0 (Thinker) on cuda:0, stages 1+2 (Talker+Code2Wav) on cuda:1.
    # For a single card use "H200" (141GB) + --stage-overrides (see README) — risky on H100-80.
    gpu="H100:2",
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    secrets=[hf_secret],
    timeout=20 * MINUTES,  # covers the slow first-run weight load + 3-stage init
    # Serverless: with min_containers=0 you pay $0 while idle. scaledown_window is the paid
    # idle tail after the last request — short (5 min) for cheap dev; raise it for a live demo
    # so you don't re-cold-start (multi-minute) between turns.
    scaledown_window=5 * MINUTES,
    min_containers=0,  # scale-to-zero. Set to 1 ONLY during the demo window, then back to 0.
)
@modal.concurrent(max_inputs=16)  # one server handles many concurrent sessions/voices
@modal.web_server(port=VLLM_PORT, startup_timeout=20 * MINUTES)
def serve():
    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--omni",  # vllm-omni: auto-loads the bundled qwen3_omni_moe.yaml stage config
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        # Single-GPU squeeze (use with gpu="H200"); collapse stages onto cuda:0:
        # "--stage-overrides",
        # '{"1":{"gpu_memory_utilization":0.3,"devices":"0"},'
        # '"2":{"gpu_memory_utilization":0.1,"devices":"0"}}',
    ]
    subprocess.Popen(cmd)  # non-blocking; Modal proxies VLLM_PORT once vllm is up


@app.function(
    image=image,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[hf_secret],
    timeout=30 * MINUTES,
)
def download_weights():
    """Optional one-off: pre-bake the 30B weights into the hf-cache Volume before a demo.

    Run with:  modal run modal/qwen_omni_modal.py::download_weights
    """
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_NAME)
    hf_cache.commit()
    print(f"Cached {MODEL_NAME} into the hf-cache volume.")
