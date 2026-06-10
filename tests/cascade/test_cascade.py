"""Unit tests for the cascade backend's custom logic."""

from __future__ import annotations
import io
import json
import wave
import asyncio
from typing import Any, AsyncIterator

import numpy as np
import pytest

from reachy_mini_conversation_app.cascade import pipeline
from reachy_mini_conversation_app.cascade.llm import LLMChunk, LLMProvider
from reachy_mini_conversation_app.cascade.config import CascadeConfig, set_config
from reachy_mini_conversation_app.cascade.handler import SPEAK_TOOL_SPEC, _pcm_to_wav, _to_chat_tool_specs
from reachy_mini_conversation_app.cascade.pipeline import PipelineContext
from reachy_mini_conversation_app.cascade.turn_result import PipelineResult
from reachy_mini_conversation_app.cascade.transcript_analysis import KeywordAnalyzer


def test_pcm_to_wav_roundtrips_mono_16k() -> None:
    """_pcm_to_wav wraps int16 PCM in a valid mono WAV at the given rate."""
    pcm = (np.arange(800, dtype=np.int16))
    data = _pcm_to_wav(pcm, 16000)
    with wave.open(io.BytesIO(data), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000
        assert wav.getnframes() == 800


def test_to_chat_tool_specs_includes_speak() -> None:
    """Realtime specs convert to Chat Completions format; speak survives."""
    chat = _to_chat_tool_specs([SPEAK_TOOL_SPEC])
    assert chat == [
        {
            "type": "function",
            "function": {
                "name": "speak",
                "description": SPEAK_TOOL_SPEC["description"],
                "parameters": SPEAK_TOOL_SPEC["parameters"],
            },
        }
    ]


class _StubLLM(LLMProvider):
    """Minimal LLMProvider; only parse_tool_call (from the base) is exercised."""

    async def generate(
        self, messages: list[dict[str, Any]], tools: Any = None, temperature: float = 0.7
    ) -> AsyncIterator[LLMChunk]:
        return
        yield  # pragma: no cover - makes this an (empty) async generator


class _CaptureSpeech:
    def __init__(self) -> None:
        self.said: list[str] = []

    async def speak(self, text: str) -> None:
        self.said.append(text)


def test_speak_is_intercepted_not_dispatched() -> None:
    """A `speak` tool call is turned into TTS locally, never sent to the tool registry."""
    spoken = _CaptureSpeech()
    ctx = PipelineContext(
        llm=_StubLLM(),
        tts=None,  # type: ignore[arg-type]
        speech_output=spoken,
        conversation_history=[],
        tool_specs=[],
        deps=None,  # type: ignore[arg-type]
        result=PipelineResult(),
    )
    tool_call = {
        "id": "1",
        "type": "function",
        "function": {"name": "speak", "arguments": json.dumps({"message": "hello there"})},
    }
    asyncio.run(pipeline.execute_tool_calls([tool_call], ctx))

    assert spoken.said == ["hello there"]
    # The tool result must be recorded in history for the LLM.
    assert ctx.conversation_history[-1]["role"] == "tool"
    assert ctx.conversation_history[-1]["name"] == "speak"


def test_keyword_analyzer_literal_and_glob() -> None:
    """KeywordAnalyzer matches literal substrings and glob patterns on tokens."""
    analyzer = KeywordAnalyzer({"dance": ["danc*"], "greet": ["hello"]})
    matches = asyncio.run(analyzer.analyze("well hello, let us dance", is_final=True))
    assert "greet" in matches
    assert matches["dance"] == ["dance"]


# A self-contained cascade config catalog covering one provider per stage.
_CONFIG = {
    "asr": {
        "provider": "whisper_openai",
        "providers": {"whisper_openai": {"module": "whisper_openai", "class": "WhisperOpenAIASR", "streaming": False, "requires": ["OPENAI_API_KEY"]}},
    },
    "llm": {
        "provider": "gpt-4o-mini",
        "providers": {"gpt-4o-mini": {"module": "openai", "class": "OpenAILLM", "requires": ["OPENAI_API_KEY"], "model": "gpt-4o-mini"}},
    },
    "tts": {
        "provider": "needs_extra",
        "providers": {
            "needs_extra": {
                "module": "kokoro",
                "class": "KokoroTTS",
                "requires": [],
                "import_check": "definitely_not_installed_pkg",
                "install_extra": "cascade_kokoro",
            }
        },
    },
}


def test_missing_dependency_raises_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Selecting a provider whose dependency is absent yields a clear install hint."""
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    set_config(None)  # force reload
    monkeypatch.setattr("reachy_mini_conversation_app.cascade.config._load_cascade_config", lambda: _CONFIG)
    with pytest.raises(RuntimeError, match="cascade_kokoro"):
        CascadeConfig()
    set_config(None)
