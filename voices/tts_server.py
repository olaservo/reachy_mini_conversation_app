"""Custom OpenAI-compatible Qwen3-TTS serving microservice for the DM's per-character voices.

What this is
------------
A small FastAPI app that exposes ``POST /v1/audio/speech`` (OpenAI Audio Speech API
shape) backed by the HuggingFace ``qwen_tts`` ``Qwen3TTSModel`` voice-clone path. It
renders the DM's 11-voice roster (``gm_narrator``, ``npc_raider``, ...) by looking up a
pre-generated, committed voice-clone prompt per ``voice`` id and running
``generate_voice_clone`` on the **Base** checkpoint.

Why it exists (instead of vLLM-Omni)
-------------------------------------
vLLM-Omni only supports **offline** inference for Qwen3-TTS — it does NOT yet expose an
online ``/v1/audio/speech`` server for the TTS stages. The old
``modal/qwen3_tts_modal.py::serve`` that did ``vllm serve --omni`` was therefore a dead
end. We serve the voices ourselves by wrapping the transformers ``qwen_tts`` API around
the clone prompts we already generated and committed under ``voices/assets/``.

The cascade contract this satisfies
------------------------------------
The cascade TTS provider (reachy-dm-cascade
``.../cascade/tts/qwen3_tts.py``) calls::

    client.audio.speech.with_streaming_response.create(
        model=..., voice="<voice_id>", input="<text>",
        response_format="pcm", extra_body=...,
    )

and reads ``response.iter_bytes()`` expecting **raw int16 little-endian PCM, mono,
24000 Hz**. ``voice`` is a roster id (e.g. ``gm_narrator`` / ``npc_raider``); it is
resolved here to a committed clone prompt, so the cascade's ``VOICE_PROMPTS`` map can
stay empty. Unknown extra fields in the body (OpenAI ``extra_body`` flattens into the
JSON, e.g. a possible ``voice_clone_prompt``) are tolerated and ignored.

Runs identically local or on Modal — everything is configured from env vars:
  * ``QWEN_TTS_MODEL``   - HF model id / path (default ``Qwen/Qwen3-TTS-12Hz-1.7B-Base``).
                           Cloning REQUIRES the **Base** checkpoint.
  * ``CLONE_PROMPTS_DIR`` - dir of ``<voice_id>.pt`` clone prompts (default
                           ``voices/assets/clone_prompts`` relative to repo root; accepts
                           an absolute path; on Modal it is
                           ``/root/voices/assets/clone_prompts``).
  * ``QWEN_TTS_DEVICE``  - device_map for ``from_pretrained`` (default ``cuda:0``).
  * ``DEFAULT_VOICE``    - fallback voice id when ``voice`` is missing/unknown (default
                           ``gm_narrator``).
  * ``QWEN_TTS_LANGUAGE`` - default synthesis language (default ``English``).

Run locally (near the robot)::

    cd voices
    QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base uvicorn tts_server:app --port 8091

The heavy deps (torch / qwen_tts) are imported lazily inside the model-load path, NOT at
module import, so this file can be imported (and the FastAPI app constructed / inspected)
on a machine without torch present. The model is loaded on FastAPI startup; generation is
serialized under a module-level lock because the model is not thread-safe.
"""

from __future__ import annotations

import io
import logging
import os
import struct
import threading
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger("qwen3_tts_server")
logging.basicConfig(level=logging.INFO)

# --- Configuration (env-driven; read at import so it is cheap and torch-free) ----------
QWEN_TTS_MODEL = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
QWEN_TTS_DEVICE = os.environ.get("QWEN_TTS_DEVICE", "cuda:0")
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "gm_narrator")
QWEN_TTS_LANGUAGE = os.environ.get("QWEN_TTS_LANGUAGE", "English")

# Default clone-prompts dir = voices/assets/clone_prompts relative to THIS file's repo,
# i.e. <repo>/voices/assets/clone_prompts (this file lives in <repo>/voices/).
_DEFAULT_CLONE_PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "clone_prompts"
)
CLONE_PROMPTS_DIR = os.environ.get("CLONE_PROMPTS_DIR", _DEFAULT_CLONE_PROMPTS_DIR)

TARGET_SAMPLE_RATE = 24000  # Qwen3-TTS-12Hz emits 24 kHz; the cascade expects exactly this.
MAX_NEW_TOKENS = 2048
PCM_CHUNK_BYTES = 4096  # streaming chunk size for the raw-PCM response

# --- Lazy/global model state (populated on startup, NOT at import) ----------------------
_model = None  # qwen_tts.Qwen3TTSModel
_prompts: Dict[str, Any] = {}  # voice_id -> List[VoiceClonePromptItem]
_gen_lock = threading.Lock()  # the model is not thread-safe; serialize generation
_load_lock = threading.Lock()


def _load_clone_prompts() -> Dict[str, Any]:
    """torch.load every ``<voice_id>.pt`` in CLONE_PROMPTS_DIR into {voice_id: prompt_items}.

    qwen_tts must be importable so the ``VoiceClonePromptItem`` dataclass resolves during
    unpickling (``weights_only=False``).
    """
    import torch
    import qwen_tts  # noqa: F401  (ensures VoiceClonePromptItem is importable for unpickling)

    prompts: Dict[str, Any] = {}
    if not os.path.isdir(CLONE_PROMPTS_DIR):
        logger.error("CLONE_PROMPTS_DIR does not exist: %s", CLONE_PROMPTS_DIR)
        return prompts
    for fname in sorted(os.listdir(CLONE_PROMPTS_DIR)):
        if not fname.endswith(".pt"):
            continue
        voice_id = fname[: -len(".pt")]
        path = os.path.join(CLONE_PROMPTS_DIR, fname)
        try:
            prompts[voice_id] = torch.load(path, weights_only=False)
            logger.info("Loaded clone prompt for voice '%s' from %s", voice_id, path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load clone prompt %s: %s", path, exc)
    return prompts


def _ensure_loaded() -> None:
    """Load the model + clone prompts once (idempotent). Called on startup / first request."""
    global _model, _prompts
    if _model is not None:
        return
    with _load_lock:
        if _model is not None:
            return
        import torch
        from qwen_tts import Qwen3TTSModel

        logger.info(
            "Loading Qwen3-TTS model '%s' on device_map=%s ...",
            QWEN_TTS_MODEL,
            QWEN_TTS_DEVICE,
        )
        # NO attn_implementation kwarg: flash-attn is not installed in the serve image;
        # passing "flash_attention_2" would fail to load. Default (eager/sdpa) is fine.
        model = Qwen3TTSModel.from_pretrained(
            QWEN_TTS_MODEL,
            device_map=QWEN_TTS_DEVICE,
            dtype=torch.bfloat16,
        )
        prompts = _load_clone_prompts()
        _model = model
        _prompts = prompts
        logger.info(
            "Qwen3-TTS server ready: model=%s, %d voices loaded (%s)",
            QWEN_TTS_MODEL,
            len(prompts),
            ", ".join(sorted(prompts)) or "<none>",
        )


# --- Audio helpers ----------------------------------------------------------------------
def _float_wav_to_pcm16le(wav) -> bytes:
    """Convert a float32 waveform in [-1, 1] to raw int16 little-endian PCM bytes."""
    import numpy as np

    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    np.clip(arr, -1.0, 1.0, out=arr)
    int16 = (arr * 32767.0).astype("<i2")  # little-endian int16
    return int16.tobytes()


def _resample_if_needed(wav, sr: int):
    """Resample a float waveform to TARGET_SAMPLE_RATE if the model's sr differs.

    Expected to be a no-op (Qwen3-TTS-12Hz already emits 24 kHz); kept as a safety net.
    """
    if sr == TARGET_SAMPLE_RATE:
        return wav
    logger.warning("Resampling TTS output from %d Hz -> %d Hz", sr, TARGET_SAMPLE_RATE)
    import numpy as np

    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    n_out = int(round(len(arr) * TARGET_SAMPLE_RATE / float(sr)))
    if n_out <= 0:
        return arr
    # Simple linear resample (dependency-free); fidelity is fine for speech playback.
    x_old = np.linspace(0.0, 1.0, num=len(arr), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, arr).astype(np.float32)


def _pcm_to_wav_bytes(pcm: bytes, sample_rate: int = TARGET_SAMPLE_RATE) -> bytes:
    """Wrap raw mono int16 PCM in a minimal WAV container (courtesy ``response_format=wav``)."""
    try:
        import soundfile as sf  # type: ignore
        import numpy as np

        samples = np.frombuffer(pcm, dtype="<i2")
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()
    except Exception:  # noqa: BLE001 — fall back to a hand-rolled WAV header
        n = len(pcm)
        header = b"RIFF" + struct.pack("<I", 36 + n) + b"WAVE"
        header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        header += b"data" + struct.pack("<I", n)
        return header + pcm


def _synthesize_pcm(voice: str, text: str, language: str) -> bytes:
    """Resolve ``voice`` -> clone prompt and synthesize ``text`` to raw int16 LE PCM @ 24 kHz."""
    _ensure_loaded()

    voice_id = voice or DEFAULT_VOICE
    if voice_id not in _prompts:
        logger.warning(
            "Unknown voice '%s' (loaded: %s) — falling back to DEFAULT_VOICE '%s'",
            voice_id,
            ", ".join(sorted(_prompts)) or "<none>",
            DEFAULT_VOICE,
        )
        voice_id = DEFAULT_VOICE
    if voice_id not in _prompts:
        raise KeyError(
            f"No clone prompt for voice '{voice}' and DEFAULT_VOICE '{DEFAULT_VOICE}' "
            f"is also unavailable. Loaded voices: {sorted(_prompts)}"
        )

    prompt_items = _prompts[voice_id]
    with _gen_lock:  # model is not thread-safe; serialize generation
        wavs, sr = _model.generate_voice_clone(
            text=text,
            language=language or QWEN_TTS_LANGUAGE,
            voice_clone_prompt=prompt_items,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    wav = wavs[0]
    wav = _resample_if_needed(wav, int(sr))
    return _float_wav_to_pcm16le(wav)


# --- FastAPI app ------------------------------------------------------------------------
app = FastAPI(title="Qwen3-TTS DM voices", version="1.0")


class SpeechRequest(BaseModel):
    """OpenAI Audio Speech request. Extra fields (e.g. extra_body's voice_clone_prompt) ok."""

    model_config = {"extra": "allow"}

    model: Optional[str] = None
    voice: Optional[str] = None
    input: str = ""
    response_format: str = "pcm"
    language: Optional[str] = None


@app.on_event("startup")
def _startup() -> None:
    """Load the model + clone prompts when the server boots (so the first request is warm)."""
    try:
        _ensure_loaded()
    except Exception as exc:  # noqa: BLE001 — log; do not crash the worker on startup
        logger.exception("Model failed to load on startup: %s", exc)


@app.post("/v1/audio/speech")
def create_speech(req: SpeechRequest):
    """Synthesize ``input`` in the roster ``voice`` and stream raw PCM (cascade contract).

    Default ``response_format="pcm"`` -> StreamingResponse of raw int16 LE PCM, mono,
    24 kHz, in PCM_CHUNK_BYTES chunks (media_type ``audio/pcm``). ``response_format="wav"``
    is supported as a courtesy and returns a complete WAV file.
    """
    text = (req.input or "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "input text is empty"})

    fmt = (req.response_format or "pcm").lower()
    try:
        pcm = _synthesize_pcm(req.voice or DEFAULT_VOICE, text, req.language or QWEN_TTS_LANGUAGE)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Synthesis failed: %s", exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})

    if fmt == "wav":
        wav_bytes = _pcm_to_wav_bytes(pcm, TARGET_SAMPLE_RATE)
        return StreamingResponse(iter([wav_bytes]), media_type="audio/wav")

    # Default + "pcm": raw int16 LE PCM, chunked for low time-to-first-audio.
    def _iter_pcm():
        for i in range(0, len(pcm), PCM_CHUNK_BYTES):
            yield pcm[i : i + PCM_CHUNK_BYTES]

    return StreamingResponse(_iter_pcm(), media_type="audio/pcm")


@app.get("/v1/models")
def list_models():
    """OpenAI-compatible model list (single served TTS model)."""
    return {
        "object": "list",
        "data": [{"id": QWEN_TTS_MODEL, "object": "model", "owned_by": "qwen-tts"}],
    }


@app.get("/health")
def health():
    """Liveness + the voice ids currently loaded (empty until the model finishes loading)."""
    return {"status": "ok", "voices": sorted(_prompts)}
