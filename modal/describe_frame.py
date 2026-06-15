"""Turn a camera JPEG frame into a text tabletop description via the Qwen3-VL Modal server.

This is the dependency-light helper the app's **text** `camera` tool calls. It base64-encodes the
JPEG into an OpenAI `image_url` data URL, POSTs an OpenAI-compatible `/v1/chat/completions` request
to the Qwen3-VL server (deployed by `qwen_vl_modal.py`), and returns the model's text reply.

The raw image never reaches the text-only DM brain — only the returned string does. Keep this
module importable and testable WITHOUT a GPU (only depends on `openai`, already an app dep).

Usage (from the camera tool / a remote VisionProcessor):

    from modal.describe_frame import describe_frame
    text = describe_frame(jpeg_bytes, base_url="https://<ws>--qwen3-vl-serve.modal.run/v1")

Quick manual test against a live endpoint:

    python modal/describe_frame.py path/to/frame.jpg https://<ws>--qwen3-vl-serve.modal.run/v1
"""

from __future__ import annotations

import base64

# Default tabletop-reading prompt. The camera tool's own `question` (e.g. "what did the d20 land
# on?") is passed as the user prompt; this is the system framing that focuses the VL model on the
# things a DM cares about.
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


def _data_url(jpeg_bytes: bytes) -> str:
    """Encode raw JPEG bytes as an OpenAI image_url data URL."""
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def describe_frame(
    jpeg_bytes: bytes,
    base_url: str,
    prompt: str = DEFAULT_USER_PROMPT,
    *,
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_tokens: int = 512,
    temperature: float = 0.2,
    timeout: float = 60.0,
    api_key: str = _DUMMY_API_KEY,
) -> str:
    """Send one JPEG frame to the Qwen3-VL server and return a text description.

    Args:
        jpeg_bytes: the camera frame, already JPEG-encoded.
        base_url:   the VL server's OpenAI base URL, ending in `/v1`
                    (e.g. https://<ws>--qwen3-vl-serve.modal.run/v1). See VL_BASE_URL env var.
        prompt:     the user question/instruction about the frame (the camera tool forwards its
                    own `question` here).
        model:      model id the server is serving (must match MODEL_NAME in qwen_vl_modal.py).
        system_prompt: system framing that focuses the VL model on tabletop details.
        max_tokens / temperature / timeout: generation + request knobs.
        api_key:    any non-empty string (vLLM is keyless).

    Returns:
        The model's text description (single line, whitespace-normalized).
    """
    # Imported lazily so the module imports cleanly even where `openai` isn't installed (tests).
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _data_url(jpeg_bytes)}},
                ],
            },
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    text = (response.choices[0].message.content or "").strip()
    return " ".join(text.split())


if __name__ == "__main__":  # pragma: no cover - manual smoke test only
    import sys

    if len(sys.argv) < 3:
        print(
            "usage: python modal/describe_frame.py <frame.jpg> <base_url> [prompt]",
            file=sys.stderr,
        )
        raise SystemExit(2)

    frame_path, base = sys.argv[1], sys.argv[2]
    user_prompt = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_USER_PROMPT
    with open(frame_path, "rb") as fh:
        data = fh.read()
    print(describe_frame(data, base_url=base, prompt=user_prompt))
