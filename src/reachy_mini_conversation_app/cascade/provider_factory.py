"""Factory functions for initializing cascade providers (ASR, LLM, TTS)."""

from __future__ import annotations
import os
import logging
import importlib
from typing import Any, Dict

from reachy_mini_conversation_app.prompts import CASCADE_EXTRA_INSTRUCTIONS, get_session_instructions
from reachy_mini_conversation_app.cascade.asr import ASRProvider
from reachy_mini_conversation_app.cascade.llm import LLMProvider
from reachy_mini_conversation_app.cascade.tts import TTSProvider
from reachy_mini_conversation_app.cascade.config import get_config
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.cascade.transcript_analysis import (
    NoOpTranscriptManager,
    TranscriptAnalysisManager,
    get_profile_reactions,
)


logger = logging.getLogger(__name__)


def init_provider(provider_type: str, extra_kwargs: Dict[str, Any] | None = None) -> Any:
    """Initialize a provider (ASR/LLM/TTS) from cascade.yaml config.

    Args:
        provider_type: One of "asr", "llm", "tts"
        extra_kwargs: Additional kwargs to pass to provider constructor

    Returns:
        Initialized provider instance

    """
    config = get_config()

    # All API keys that any provider might need
    api_key_map = {
        "OPENAI_API_KEY": config.OPENAI_API_KEY,
        "DEEPGRAM_API_KEY": config.DEEPGRAM_API_KEY,
        "GEMINI_API_KEY": config.GEMINI_API_KEY,
        "ELEVENLABS_API_KEY": config.ELEVENLABS_API_KEY,
        "GRADIUM_API_KEY": config.GRADIUM_API_KEY,
    }

    # Get provider name, info, and settings using dynamic attribute access
    name = getattr(config, f"{provider_type}_provider")
    info = getattr(config, f"get_{provider_type}_provider_info")(name)
    kwargs = getattr(config, f"get_{provider_type}_settings")(name)

    # Add API key (validated at config load time)
    requires = info["requires"]
    if len(requires) == 1:
        kwargs["api_key"] = api_key_map[requires[0]]
    elif requires:
        raise ValueError(f"Multi-key providers not supported: {requires}")

    # Allow an env var to override the OpenAI-compatible LLM base URL without editing
    # cascade.yaml. Handy when the endpoint is dynamic (e.g. a freshly-deployed Modal
    # vLLM URL serving the Qwen brain). Mirrors the CASCADE_*_PROVIDER override pattern.
    if provider_type == "llm" and info.get("module") == "openai":
        env_base_url = os.getenv("CASCADE_LLM_BASE_URL")
        if env_base_url:
            kwargs["base_url"] = env_base_url

    # Same override for the Qwen3-TTS provider: point at a deployed (Modal) /v1/audio/speech
    # endpoint without editing cascade.yaml's localhost default — useful on a dev box with no
    # local TTS server. Verified: the Modal endpoint resolves all 11 roster voices by name.
    if provider_type == "tts" and info.get("module") == "qwen3_tts":
        env_base_url = os.getenv("CASCADE_TTS_BASE_URL")
        if env_base_url:
            kwargs["base_url"] = env_base_url

    # Merge extra kwargs if provided
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    # Dynamic import and instantiate
    module = importlib.import_module(f"reachy_mini_conversation_app.cascade.{provider_type}.{info['module']}")
    ProviderClass = getattr(module, info["class"])

    # Log with provider-specific details
    extra_info = f", streaming={info['streaming']}" if "streaming" in info else ""
    logger.info(f"Initializing {provider_type.upper()}: {name} (location={info['location']}{extra_info})")

    return ProviderClass(**kwargs)


def init_asr_provider() -> ASRProvider:
    """Initialize ASR provider from cascade.yaml config."""
    return init_provider("asr")  # type: ignore[no-any-return]


def cascade_system_instructions(memory_manager: Any | None = None) -> str:
    """Resolve the active profile's instructions plus the cascade-specific speak guidance.

    When ``memory_manager`` is provided, the persistent-memory index block is injected
    into the system prompt via ``get_session_instructions``.
    """
    return get_session_instructions(memory_manager) + CASCADE_EXTRA_INSTRUCTIONS


def init_llm_provider(memory_manager: Any | None = None) -> LLMProvider:
    """Initialize LLM provider from cascade.yaml config."""
    return init_provider(  # type: ignore[no-any-return]
        "llm", {"system_instructions": cascade_system_instructions(memory_manager)}
    )


def init_tts_provider() -> TTSProvider:
    """Initialize TTS provider from cascade.yaml config."""
    return init_provider("tts")  # type: ignore[no-any-return]


def init_transcript_analysis(deps: ToolDependencies) -> TranscriptAnalysisManager | NoOpTranscriptManager:
    """Build the transcript analysis manager from the active profile's reactions."""
    reactions = get_profile_reactions()
    if not reactions:
        logger.info("No profile reactions configured, transcript analysis disabled")
        return NoOpTranscriptManager()
    return TranscriptAnalysisManager(reactions=reactions, deps=deps)
