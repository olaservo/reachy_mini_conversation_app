"""Serve the DM's per-character voices on Modal via a CUSTOM FastAPI server (Qwen3-TTS).

The per-character voice synthesizer for the DM. The cascade TTS provider calls this
server's `/v1/audio/speech` to render an in-character line in a designed voice (the
11-voice roster in `voices/character-voices.md`), beyond the cascade default narrator.

⚠️ WHY A CUSTOM SERVER (not vllm-omni): vLLM-Omni only supports **offline** inference for
Qwen3-TTS — there is no online `/v1/audio/speech` server for the TTS stages yet. The old
`vllm serve --omni` path here was a dead end. Instead we serve `voices/tts_server.py`, a
FastAPI app that wraps the transformers `qwen_tts` voice-clone API around the clone
prompts we already generated and committed under `voices/assets/clone_prompts/`. The
`voice` request param is a roster id (e.g. `gm_narrator`/`npc_raider`) resolved to a
committed clone prompt server-side. Serving uses the **Base** checkpoint (cloning lives
only on -Base); the offline design batch below uses VoiceDesign + Base.

NOTE — local is the preferred home-use target (this Modal app is the hackathon-cloud variant).
1.7B is tiny (~3.4GB bf16): at home, run the SAME FastAPI server on the machine beside the
robot/cascade (`cd voices && uvicorn tts_server:app --port 8091`) and point the cascade at
`http://localhost:8091/v1` — localhost call, $0, lowest latency, keeping the Modal grant for
the 30B brain. See modal/README.md → "Serving".

Deploy:   modal deploy modal/qwen3_tts_modal.py
Iterate:  modal serve  modal/qwen3_tts_modal.py
Secret:   modal secret create huggingface HF_TOKEN=hf_xxx   (one-time; shared with the brain app)
"""

import os
import subprocess
from pathlib import Path

import modal

# Serving + cloning REQUIRE the Base checkpoint (create/generate_voice_clone live only on
# -Base). The offline design batch additionally pulls VoiceDesign (see DESIGN_MODEL_NAME).
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DESIGN_MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
SERVE_PORT = 8091
MINUTES = 60

_LOCAL_VOICES_DIR = Path(__file__).resolve().parent.parent / "voices"

download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# --- Serve image: our FastAPI app + transformers qwen_tts (NOT vllm-omni) ----------------
# apt: sox/libsox + ffmpeg = qwen_tts/torchaudio audio I/O; libsndfile1 = soundfile runtime.
# pip: torch stack + qwen-tts (ships `from qwen_tts import Qwen3TTSModel`) + fastapi/uvicorn.
# NOTE: flash-attn is intentionally NOT installed (tts_server passes no attn_implementation).
serve_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("sox", "libsox-dev", "libsox-fmt-all", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch",
        "torchaudio",
        "transformers",
        "accelerate",
        "soundfile",
        "numpy",
        "hf_transfer",
        "qwen-tts",
        "fastapi",
        "uvicorn[standard]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Ship the clone prompts + tts_server.py into the container at /root/voices.
    .add_local_dir(_LOCAL_VOICES_DIR, remote_path="/root/voices")
)

# Reuse the brain app's HF cache + secret so weights are shared.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

app = modal.App("qwen3-tts-voices")


@app.function(
    image=serve_image,
    # 1.7B in bf16 (~3.4GB) fits comfortably on one small card. L4 (24GB) is cheapest.
    gpu="L4",
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[hf_secret],
    timeout=15 * MINUTES,
    scaledown_window=5 * MINUTES,
    min_containers=0,  # scale-to-zero; set to 1 ONLY during the demo window, then back to 0.
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=SERVE_PORT, startup_timeout=15 * MINUTES)
def serve():
    """Launch our CUSTOM FastAPI Qwen3-TTS server (Modal proxies SERVE_PORT).

    vLLM-Omni has no online TTS server, so we run `voices/tts_server.py` (FastAPI) which
    wraps the qwen_tts voice-clone API. Non-blocking like the brain app: Popen + return.
    """
    env = {
        **os.environ,
        "QWEN_TTS_MODEL": MODEL_NAME,
        "CLONE_PROMPTS_DIR": "/root/voices/assets/clone_prompts",
        "QWEN_TTS_DEVICE": "cuda:0",
    }
    cmd = [
        "uvicorn",
        "tts_server:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(SERVE_PORT),
    ]
    # cwd=/root/voices so `tts_server:app` imports; env points at the shipped clone prompts.
    subprocess.Popen(cmd, cwd="/root/voices", env=env)  # non-blocking; Modal proxies once up


@app.function(
    image=download_image,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[hf_secret],
    timeout=15 * MINUTES,
)
def download_weights():
    """Optional one-off: pre-bake the 1.7B weights into the hf-cache Volume.

    Caches BOTH the Base (serving + cloning) and VoiceDesign (offline design) checkpoints
    so a prebake covers serving and the design batch.

    Run with:  modal run modal/qwen3_tts_modal.py::download_weights
    """
    from huggingface_hub import snapshot_download

    for name in (MODEL_NAME, DESIGN_MODEL_NAME):
        snapshot_download(name)
        print(f"Cached {name} into the hf-cache volume.")
    hf_cache.commit()


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
# Uses its own transformers/qwen_tts image (same deps as serve_image, plus it ships
# generate_voices.py + character-voices.md). The design batch needs BOTH the VoiceDesign
# and Base checkpoints; serving needs only Base.
#
# Run:   modal run modal/qwen3_tts_modal.py::design_voices
#        (writes voices/assets/** back into your local worktree)
# --------------------------------------------------------------------------- #
voicedesign_image = (
    modal.Image.debian_slim(python_version="3.12")
    # libsndfile1 = soundfile runtime; sox/libsox + ffmpeg = qwen_tts/torchaudio audio I/O
    # (qwen_tts errors with "SoX could not be found!" without these).
    .apt_install("libsndfile1", "sox", "libsox-dev", "libsox-fmt-all", "ffmpeg")
    .pip_install(
        "torch",
        "torchaudio",  # qwen_tts audio backend
        "transformers",
        "accelerate",
        "soundfile",
        "numpy",
        "hf_transfer",  # required because HF_HUB_ENABLE_HF_TRANSFER=1 is set below
        # ships `from qwen_tts import Qwen3TTSModel` (verified: pip package `qwen-tts`).
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
