"""Serve Qwen3-TTS-12Hz-1.7B on Modal via vllm-omni (OpenAI-compatible speech endpoint).

The per-character voice synthesizer for the DM. The WS1 adapter's `speak_as(voice_id, text)`
handler dials this server's `/v1/audio/speech` to render an in-character line in a designed
voice (vs. Omni's 3 built-in speakers Ethan/Chelsie/Aiden). Omni stays the brain + default
narration; this is the optional second model that unlocks the 11-voice roster in
`voices/character-voices.md`.

NOTE — local is the preferred home-use target (this Modal app is the hackathon-cloud variant).
1.7B is tiny (~3.4GB bf16): at home, run the SAME `vllm serve` on the machine beside the
robot/adapter and point the adapter at `http://localhost:8091/v1` — localhost call, $0, lowest
latency, and it keeps the Modal $250 grant for the 30B Omni (which actually needs cloud GPU).
See modal/README.md → "Local deployment (home use)".

Deploy:   modal deploy modal/qwen3_tts_modal.py
Iterate:  modal serve  modal/qwen3_tts_modal.py
Secret:   modal secret create huggingface HF_TOKEN=hf_xxx   (one-time; shared with the Omni app)
"""

import subprocess
from pathlib import Path

import modal

# VoiceDesign = design a voice from a natural-language description (the generation phase in
# voices/character-voices.md). Runtime per-character playback may instead serve CustomVoice
# (register precomputed/designed speakers by name) or Base (clone from a reference). See README
# "Which TTS variant" — kept as VoiceDesign here to match the model the spec designs with.
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
VLLM_PORT = 8091
MINUTES = 60

# Same vllm-omni image as the Omni app (it patches vllm's `serve` with `--omni`).
image = (
    modal.Image.from_registry("vllm/vllm-omni:v0.22.0", add_python="3.12")
    .entrypoint([])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1"})
)

download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Reuse the Omni app's caches + secret so weights/compile artifacts are shared.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

app = modal.App("qwen3-tts-voices")


@app.function(
    image=image,
    # 1.7B + code2wav fits comfortably on one small card. L4 (24GB) is cheapest; A10G also fine.
    gpu="L4",
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    secrets=[hf_secret],
    timeout=10 * MINUTES,
    scaledown_window=5 * MINUTES,
    min_containers=0,  # scale-to-zero; set to 1 ONLY during the demo window, then back to 0.
)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--omni",  # vllm-omni: auto-resolves the bundled qwen3_tts.yaml stage config
        "--trust-remote-code",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--gpu-memory-utilization",
        "0.9",
        # If the bundled TTS YAML does not auto-resolve in this image, pass it explicitly:
        # "--deploy-config", "vllm_omni/deploy/qwen3_tts.yaml",
    ]
    subprocess.Popen(cmd)  # non-blocking; Modal proxies VLLM_PORT once vllm is up


@app.function(
    image=download_image,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[hf_secret],
    timeout=15 * MINUTES,
)
def download_weights():
    """Optional one-off: pre-bake the 1.7B weights into the hf-cache Volume.

    Run with:  modal run modal/qwen3_tts_modal.py::download_weights
    """
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_NAME)
    hf_cache.commit()
    print(f"Cached {MODEL_NAME} into the hf-cache volume.")


# --------------------------------------------------------------------------- #
# ONE-TIME voice-design batch (offline asset generation, NOT the serve path).
#
# Generates the 11 character voices from voices/character-voices.md using the
# VoiceDesign + Base checkpoints, then returns the assets (ref clips, clone
# prompts, voices.json) so they can be written into the local worktree and
# committed. This is the GPU half of voices/generate_voices.py -- that script's
# `run_batch()` does the actual design->clone->save work; here we just run it on
# a Modal L4 and ship the bytes back.
#
# The vllm-omni serving image does NOT necessarily expose the `qwen_tts` Python
# package used by generate_voices.py (it serves via vLLM, not the HF model class),
# so this entrypoint uses a dedicated transformers/qwen_tts image.  FLAG[import]:
# confirm the exact pip package providing `qwen_tts` / `Qwen3TTSModel`.
#
# Run:   modal run modal/qwen3_tts_modal.py::design_voices
#        (writes voices/assets/** back into your local worktree)
# --------------------------------------------------------------------------- #
_LOCAL_VOICES_DIR = Path(__file__).resolve().parent.parent / "voices"

voicedesign_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1")  # soundfile runtime dep
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "soundfile",
        "numpy",
        # FLAG[import]: the package that ships `from qwen_tts import Qwen3TTSModel`.
        # Pin once verified (model card / QwenLM/Qwen3-TTS README).
        "qwen-tts",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Ship the generator + the source-of-truth markdown into the container.
    .add_local_dir(_LOCAL_VOICES_DIR, remote_path="/root/voices")
)


@app.function(
    image=voicedesign_image,
    gpu="L4",  # 2x 1.7B checkpoints (~7GB) + wavegen fit comfortably on 24GB.
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[hf_secret],
    timeout=30 * MINUTES,
)
def design_voices(only: list[str] | None = None) -> dict[str, bytes]:
    """Run the 11-voice design batch on GPU; return {relpath: bytes} of assets.

    The caller (`design_voices.local`/`.remote` via `modal run`) writes these
    back into the local `voices/` tree. ~11 voices on an L4 is a few minutes;
    well under a dollar (L4 ~ $0.80/GPU-hr).
    """
    import sys

    sys.path.insert(0, "/root/voices")
    import generate_voices as gv  # the script above, shipped via add_local_dir

    gv.run_batch(only=only)

    # Collect every generated asset under /root/voices/assets as bytes.
    assets_root = Path("/root/voices/assets")
    out: dict[str, bytes] = {}
    for p in assets_root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(assets_root).as_posix()
            out[rel] = p.read_bytes()
    print(f"design_voices: produced {len(out)} asset files")
    return out


@app.local_entrypoint()
def design_voices_main(only: str = ""):
    """Local driver: run the GPU batch, then write assets into voices/assets/.

    Usage:
        modal run modal/qwen3_tts_modal.py::design_voices_main
        modal run modal/qwen3_tts_modal.py::design_voices_main --only npc_raider
    """
    only_list = [s for s in only.split(",") if s] or None
    assets = design_voices.remote(only=only_list)

    local_assets = _LOCAL_VOICES_DIR / "assets"
    for rel, data in assets.items():
        dest = local_assets / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        print(f"wrote {dest}")
    print(f"Done. {len(assets)} files under {local_assets}")
