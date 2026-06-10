"""Shared WAV parsing and resampling utilities for ASR providers."""

from __future__ import annotations
import io
import wave
import logging

import numpy as np
import numpy.typing as npt


logger = logging.getLogger(__name__)


def _read_wav(wav_bytes: bytes) -> tuple[int, int, int, bytes]:
    """Parse WAV bytes and return (sample_rate, channels, sample_width, raw_frames)."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.getframerate(), wf.getnchannels(), wf.getsampwidth(), wf.readframes(wf.getnframes())


def _resample(audio: npt.NDArray[np.float32], orig_sr: int, target_sr: int) -> npt.NDArray[np.float32]:
    """Resample float32 audio using librosa (high-quality sinc/FFT)."""
    import librosa

    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def wav_to_float32(wav_bytes: bytes, target_sr: int) -> npt.NDArray[np.float32]:
    """Parse WAV bytes into a float32 array at *target_sr*.

    Handles int16/int32 input, stereo-to-mono, and resampling via librosa.
    """
    sr, channels, sw, frames = _read_wav(wav_bytes)

    if sw == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sw} bytes")

    if channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)

    if sr != target_sr:
        audio = _resample(audio, sr, target_sr)

    return audio


def pcm_to_wav(pcm_data: bytes, sample_rate: int) -> bytes:
    """Wrap raw PCM int16 mono bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


def wav_to_pcm_int16(wav_bytes: bytes, target_sr: int) -> bytes:
    """Parse WAV bytes and return raw PCM int16 at *target_sr*."""
    sr, channels, sw, pcm = _read_wav(wav_bytes)

    if sw != 2:
        raise ValueError(f"Unsupported sample width: {sw} bytes (expected 2)")

    audio = np.frombuffer(pcm, dtype=np.int16)

    if channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)

    if sr != target_sr:
        audio_f = audio.astype(np.float32) / 32768.0
        resampled = _resample(audio_f, sr, target_sr)
        audio = np.clip(resampled * 32768.0, -32768, 32767).astype(np.int16)

    return audio.tobytes()
