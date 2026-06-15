# Vision — letting the text-only DM brain "read the table"

## The constraint
The DM **brain** is `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` — a **text-only** LLM, served on Modal
by `modal/qwen_brain_modal.py`. It cannot accept image tokens. So we can never hand it a raw frame.

## The architecture
```
  Reachy webcam
       │  BGR frame
       ▼
  camera tool (text variant)         ── app: src/reachy_mini_conversation_app/tools/camera.py
       │  JPEG bytes + question
       ▼
  describe_frame()                   ── modal/describe_frame.py  (OpenAI image_url POST)
       │  /v1/chat/completions (base64 image_url)
       ▼
  Qwen3-VL-8B server on Modal        ── modal/qwen_vl_modal.py   (the slow GPU deploy, this task)
       │  TEXT description ("a d20 showing 17, two minis north of the bridge…")
       ▼
  camera tool returns {"image_description": "<text>"}
       │  text only — NO pixels
       ▼
  DM brain (text-only)               ── reasons over the description, narrates, calls more tools
```
The frame becomes **text** at the Qwen3-VL server. Only that text ever reaches the brain.

## Why `see_image_through_camera` must NOT be used with our brain
The cascade pipeline has **two** camera paths:

- **WRONG for us — multimodal re-injection:** `cascade/pipeline.py:253` (`see_image_through_camera`).
  After that tool runs, the pipeline appends the raw image back into the conversation as a `user`
  message with `{"type": "image", "image": <bytes>}` and re-calls the LLM
  (`cascade/pipeline.py:311`–`326`). That feeds **pixels to the LLM** — fine for a multimodal model,
  but our brain is text-only, so it would error or silently drop the image. The realtime path does
  the equivalent re-injection at `base_realtime.py:610`–`631` (posts an `input_image` content part).
  **Do not enable this path for the Qwen3 text brain.**

- **RIGHT for us — text path:** the plain `camera` tool. Its result is *sanitized* before going to
  the model (`base_realtime.py:171`: if a `camera` result carries `b64_im`, the bytes are stripped
  and replaced with `{"image_attached": true}`), and the tool itself already prefers returning
  **text** when a vision processor is present (`camera.py:47`–`57`: it calls
  `vision_processor.process_image(frame, question) -> str` and returns `{"image_description": <text>}`).
  This is the seam we use — we just point that vision processor at the Qwen3-VL Modal server instead
  of the bundled local SmolVLM2.

## Precise integration point (not yet wired — specified here)
`tools/camera.py` already branches on `deps.vision_processor`:

```python
if deps.vision_processor is not None:
    vision_result = await asyncio.to_thread(deps.vision_processor.process_image, frame, question)
    return {"image_description": vision_result}          # ← TEXT to the brain. This is our path.
...
return {"b64_im": ...}                                    # ← only when NO processor (raw image; avoid)
```

The bundled `VisionProcessor` (`vision/local_vision.py`) runs SmolVLM2 **locally** and exposes
`process_image(frame: ndarray(BGR), prompt: str) -> str`. To use the Modal Qwen3-VL server instead,
add a tiny **remote** processor with the same one-method shape that JPEG-encodes the frame and calls
`describe_frame`:

```python
# e.g. vision/remote_vision.py (NOT created here — needs the live endpoint to test end-to-end)
from reachy_mini_conversation_app.camera_frame_encoding import encode_bgr_frame_as_jpeg
from modal.describe_frame import describe_frame   # or vendor this helper into the package

class RemoteVisionProcessor:
    def __init__(self, base_url: str):
        self.base_url = base_url
    def process_image(self, frame, prompt: str) -> str:
        jpeg = encode_bgr_frame_as_jpeg(frame)         # same encoder camera.py uses
        return describe_frame(jpeg, base_url=self.base_url, prompt=prompt)
```

Then in `utils.initialize_camera_and_vision` (called from `main.py:157`), when a `VL_BASE_URL` is
set, construct `RemoteVisionProcessor(VL_BASE_URL)` and return it as `vision_processor` instead of
initializing the local SmolVLM2. **No other code changes** — `camera.py`, the result sanitizer, and
the brain all already speak the `image_description` text contract.

> We deliberately do **not** rewire the app here: doing it without the live endpoint risks a
> merge/runtime conflict and can't be tested. The seam above is exact and additive.

## Env var
- **`VL_BASE_URL`** — the Qwen3-VL server's OpenAI base URL ending in `/v1`, e.g.
  `https://<workspace>--qwen3-vl-serve.modal.run/v1` (from `modal deploy modal/qwen_vl_modal.py`).
  `describe_frame(base_url=...)` and the `RemoteVisionProcessor` read this. Mirrors how the brain
  uses `CASCADE_LLM_BASE_URL`. (Local SmolVLM2 keeps using `LOCAL_VISION_MODEL`; the two are
  mutually exclusive — prefer remote when `VL_BASE_URL` is set.)

## Files
- `modal/qwen_vl_modal.py` — the Modal deploy (Qwen3-VL-8B on L40S, OpenAI-compatible vLLM, leaf).
- `modal/describe_frame.py` — JPEG → `image_url` POST → text. Importable/testable without a GPU.
- `modal/README.md` — deploy/serve commands, GPU sizing + cost vs the brain, endpoint contract.

## Open questions / to validate on a real run
1. **vLLM Qwen3-VL support / flags** — confirm `vllm/vllm-openai:v0.23.0` loads
   `Qwen/Qwen3-VL-8B-Instruct`; confirm `--limit-mm-per-prompt '{"image": 1}'` format; confirm
   `--trust-remote-code` is unnecessary. (Flagged in `qwen_vl_modal.py` comments + README.)
2. **Latency budget** — VL cold start (~model load) + per-frame inference must fit the DM's turn
   pacing; keep `min_containers=1` during the demo. Measure end-to-end frame→text.
3. **Description quality / prompt** — tune `DEFAULT_SYSTEM_PROMPT` in `describe_frame.py` against
   real dice/minis; consider returning structured-ish text (dice values first) for easier brain use.
4. **Frame size / token cost** — webcam frames may be large; consider downscaling before encode or
   bounding vision tokens via `--mm-processor-kwargs` (flagged in the serve cmd).
5. **Where to import `describe_frame`** — `modal/` isn't part of the installed package. For runtime
   use, vendor the helper into `src/reachy_mini_conversation_app/vision/` (it has no Modal deps).
6. **4B tiny-titan variant** — validate `Qwen/Qwen3-VL-4B-Instruct` quality for the all-≤4B build.
