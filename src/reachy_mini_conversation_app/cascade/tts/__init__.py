"""TTS provider exports (base class eager; providers are loaded dynamically).

Concrete providers (Kokoro/OpenAI/ElevenLabs/Gradium/Qwen3TTS) are normally
instantiated dynamically by ``provider_factory`` from the ``cascade.yaml`` catalog,
so they are NOT imported at module load (that would force optional heavy deps).
``Qwen3TTS`` is additionally exposed here via lazy ``__getattr__`` so it can be
imported by name (``from ...tts import Qwen3TTS``) without breaking the no-deps load.
"""

from __future__ import annotations
from typing import Any

from .base import TTSProvider


__all__ = ["TTSProvider", "Qwen3TTS"]


def __getattr__(name: str) -> Any:
    """Lazily import optional providers only when explicitly requested."""
    if name == "Qwen3TTS":
        from .qwen3_tts import Qwen3TTS

        return Qwen3TTS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
