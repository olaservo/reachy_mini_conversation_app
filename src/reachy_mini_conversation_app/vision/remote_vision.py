"""Remote Qwen3-VL vision processor — turns a camera frame into a TEXT description.

Drop-in replacement for the local SmolVLM2 ``VisionProcessor`` (same
``process_image(frame, question) -> str`` contract) that offloads to the Qwen3-VL
Modal server (``modal/qwen_vl_modal.py``) instead of running a model locally.

Why text, not the raw image: the DM brain (Qwen3-30B-A3B-2507) is TEXT-ONLY, so the
``camera`` tool must hand it a string. With a vision_processor set, the tool returns
``{"image_description": <text>}`` (see ``tools/camera.py``); only the string ever
reaches the brain — the frame stops here. This is deliberately NOT the cascade's
multimodal ``see_image_through_camera`` path (which re-injects the image to the LLM).

Selected by setting the ``VL_BASE_URL`` env var (see ``utils.initialize_camera_and_vision``).
Mirrors the deploy-side helper ``modal/describe_frame.py`` (kept separate so the app
never imports from the modal/ deploy scripts).
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from reachy_mini_conversation_app.camera_frame_encoding import encode_bgr_frame_as_jpeg


logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# System framing that focuses the VL model on what a DM cares about. The tool's own
# `question` is forwarded as the user prompt.
DEFAULT_SYSTEM_PROMPT = (
    "You are the eyes of a tabletop RPG dungeon master looking down at the play area. "
    "Describe the tabletop concisely and factually: visible dice and their face/number values, "
    "miniatures or tokens and their positions relative to each other and the map/grid, any cards "
    "or character sheets and what they show, and notable changes or anything unusual. "
    "Report only what you can actually see. Do not invent rules or narrate the story."
)
DEFAULT_USER_PROMPT = "Describe the current tabletop."

# vLLM is keyless; the OpenAI client just needs any non-empty api_key.
_DUMMY_API_KEY = "EMPTY"


class RemoteVisionProcessor:
    """Describe camera frames via a remote Qwen3-VL OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str = _DUMMY_API_KEY,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tokens: int = 512,
        temperature: float = 0.2,
        timeout: float = 60.0,
    ) -> None:
        """Bind to the VL server.

        Args:
            base_url: the server's OpenAI base URL ending in `/v1`
                (e.g. https://<ws>--qwen3-vl-serve.modal.run/v1).
            model: model id the server serves (must match qwen_vl_modal.py's MODEL_NAME).
            api_key: any non-empty string (vLLM is keyless).
            system_prompt / max_tokens / temperature / timeout: framing + request knobs.
        """
        # Lazy import so the module loads even where `openai` isn't installed.
        from openai import OpenAI

        self.base_url = base_url
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = OpenAI(base_url=base_url, api_key=api_key or _DUMMY_API_KEY, timeout=timeout)
        logger.info("RemoteVisionProcessor -> %s (model=%s)", base_url, model)

    def process_image(self, frame: Any, question: str | None = None) -> str:
        """Encode a BGR frame and return the VL model's text description.

        Returns a plain string in all cases (a request failure yields a short
        ``(vision unavailable: ...)`` string rather than raising, so a transient VL
        outage degrades gracefully instead of breaking the realtime turn).
        """
        jpeg_bytes = encode_bgr_frame_as_jpeg(frame)
        data_url = f"data:image/jpeg;base64,{base64.b64encode(jpeg_bytes).decode('ascii')}"
        prompt = (question or "").strip() or DEFAULT_USER_PROMPT

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as e:  # noqa: BLE001 — degrade gracefully, keep the turn alive
            logger.error("RemoteVisionProcessor request to %s failed: %s", self.base_url, e)
            return f"(vision unavailable: {e})"

        text = (response.choices[0].message.content or "").strip()
        return " ".join(text.split())
