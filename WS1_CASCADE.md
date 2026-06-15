# WS1 — Wiring the Qwen brain into the in-app cascade backend

This documents how the Reachy Mini conversation app's **native cascade backend**
(`src/reachy_mini_conversation_app/cascade/`) is pointed at our DM **brain**
`Qwen/Qwen3-30B-A3B-Instruct-2507`, served as an **OpenAI-compatible vLLM endpoint on Modal**
(deployed in a separate worktree). We adopt the cascade's `openai` LLM provider and aim it at the
Modal `base_url` — no external `huggingface/speech-to-speech` process required.

Each model in the stack is < 32B total params (Qwen3-30B-A3B = 30.5B total).

## What changed in this worktree

| File | Change | Why |
|------|--------|-----|
| `cascade/llm/openai.py` | `__init__` now accepts `base_url: Optional[str]` and makes `api_key` optional (defaults to a dummy `"EMPTY"` when unset). `AsyncOpenAI(api_key=api_key or "EMPTY", base_url=base_url)`. Log line notes the base_url. | Lets the OpenAI provider target a self-hosted vLLM endpoint (the Modal Qwen brain) instead of api.openai.com. vLLM ignores the key but the SDK requires a non-empty string. |
| `cascade/provider_factory.py` | Added an optional `CASCADE_LLM_BASE_URL` env override (only applied when the active LLM provider's `module == "openai"`). | The Modal URL is dynamic per deploy; this avoids editing `cascade.yaml` each time. Mirrors the existing `CASCADE_*_PROVIDER` env-override convention. |

No changes were needed in `config.py` or `base.py`:
- `base_url` is **not** in `LLM_METADATA_KEYS` (`config.py:31`), so any `base_url:` field placed
  under a provider entry in `cascade.yaml` flows automatically through `get_llm_settings()` →
  `init_provider()` kwargs → the constructor. The existing config pattern already supports it.
- A provider declared with `requires: []` gets **no** `api_key` injected by
  `init_provider` (`provider_factory.py:52-53`), which is now fine because `api_key` is optional.

### Threading recap (how a setting reaches the constructor)
`cascade.yaml` provider entry → `CascadeConfig.get_llm_settings(name)` strips metadata keys →
`provider_factory.init_provider("llm", …)` builds `kwargs` (+ optional `CASCADE_LLM_BASE_URL`
override, + `system_instructions`) → `OpenAILLM(**kwargs)`.

## Recommended `cascade.yaml` config

> NOTE: do NOT hand-edit the `tts:` section / `cascade/tts/` — a parallel agent owns TTS.
> The snippets below are what to set for `asr:` and `llm:`. TTS is left to that work.

### ASR — Windows reality check
The default `asr.provider` is `parakeet_mlx_progressive`, which is **Apple-Silicon/MLX only**
(`hardware: apple_silicon`, `import_check: mlx_audio`). Config validation hard-fails on
non-Darwin/arm64 (`config.py:204-210`). **This dev machine is Windows 11**, so MLX and the CUDA
NeMo providers (`parakeet_nemo_progressive`, `nemotron`, which need a CUDA GPU) are out for
local dev.

Windows-viable ASR providers in the catalog:
- **`whisper_openai`** — cloud batch (`requires: OPENAI_API_KEY`). Simplest; works with the
  cascade's batch turn path today. **Recommended default for Windows dev.**
- **`deepgram`** — cloud streaming (`requires: DEEPGRAM_API_KEY`). Lower latency / true streaming;
  use if you have a Deepgram key.
- `openai_realtime_asr` — cloud streaming via OpenAI Realtime WS (`requires: OPENAI_API_KEY`).

Recommendation: **`whisper_openai`** for Windows dev (fewest moving parts); switch to `deepgram`
if a key is available and you want lower latency. On the eventual Apple-Silicon robot host you can
flip back to `parakeet_mlx_progressive` for a fully-local ASR.

```yaml
asr:
  provider: whisper_openai   # Windows dev. Use `deepgram` for streaming; `parakeet_mlx_progressive` only on Apple Silicon.
```

### LLM — point `openai` at the Modal Qwen brain
Add a provider entry that reuses the existing `openai` module/class but sets `base_url` and the
Qwen model, and declares `requires: []` so no real OpenAI key is needed:

```yaml
llm:
  provider: qwen_modal
  temperature: 0.7          # Qwen3 tool-calling is steadier below the default 1.0

  providers:
    qwen_modal:
      module: openai
      class: OpenAILLM
      location: cloud
      requires: []                       # vLLM endpoint needs no OpenAI key
      description: "Qwen3-30B-A3B brain on Modal (OpenAI-compatible vLLM)"
      # Settings (passed to OpenAILLM.__init__):
      model: Qwen/Qwen3-30B-A3B-Instruct-2507
      base_url: https://<your-modal-app>--vllm-serve.modal.run/v1   # the /v1 vLLM endpoint
      # Costs left at 0.0 (self-hosted); cost logging stays silent.
```

If the Modal endpoint enforces a token, instead set `requires: [OPENAI_API_KEY]` and put the
token in `OPENAI_API_KEY` (the factory will inject it as `api_key`).

For the **tiny/local variant**, the same entry works against a local vLLM:
`model: Qwen/Qwen3-4B-Instruct-2507`, `base_url: http://localhost:8000/v1`.

## Env vars / config to set

| Var | Purpose |
|-----|---------|
| `CASCADE_LLM_PROVIDER=qwen_modal` | Select the Qwen brain entry without editing `cascade.yaml`'s `provider:`. (existing override) |
| `CASCADE_LLM_BASE_URL=https://<modal>/v1` | **New.** Overrides `base_url` for the active `openai` LLM provider — set the live Modal URL here per deploy. |
| `CASCADE_ASR_PROVIDER=whisper_openai` | Pick a Windows-viable ASR. (existing override) |
| `OPENAI_API_KEY=...` | Needed by `whisper_openai`/`openai_realtime_asr` ASR (and only by the LLM if the Modal endpoint is token-gated). |
| `DEEPGRAM_API_KEY=...` | Only if using `deepgram` ASR. |

Minimal Windows-dev `.env` example:
```
CASCADE_ASR_PROVIDER=whisper_openai
CASCADE_LLM_PROVIDER=qwen_modal
CASCADE_LLM_BASE_URL=https://<your-modal-app>--vllm-serve.modal.run/v1
OPENAI_API_KEY=sk-...        # for Whisper ASR
```
Run the app with the cascade backend via the `--cascade` flag (sets `BACKEND_PROVIDER=cascade`;
see `CASCADE_CODEBASE.md` / `main.build_handler()`).

## Tool-calling assessment (current state)

**The `openai` provider already fully supports streaming tool calls — no new code needed for the
happy path.** Specifics (`cascade/llm/openai.py`):
- Request sends `tools` + `tool_choice="auto"` when tools are present (`openai.py:114-116`).
- Streamed tool-call deltas are accumulated by index (id / name / incremental `arguments`)
  (`openai.py:156-177`) and emitted as `LLMChunk(type="tool_call", …)` on `finish_reason`
  (`openai.py:180-183`).
- `LLMProvider.parse_tool_call` (`llm/base.py:49-72`) parses the standard OpenAI tool-call shape
  into `(call_id, name, args_dict)`.
- The pipeline (`cascade/pipeline.py`) executes calls via `execute_tool_calls` →
  `dispatch_tool_call` (shared registry; `speak` intercepted locally), appends `role:"tool"`
  results, and **re-invokes the LLM** so it can react to results
  (`pipeline.py:186-194`, `execute_tool_calls` at `pipeline.py:197-306`).

So app → cascade → brain tool round-trips work for **any OpenAI-compatible endpoint that emits
OpenAI-style streamed `tool_calls`**. The remaining requirement is purely **server-side**: vLLM
must be launched with tool-call parsing enabled.

### What's needed for the Modal vLLM serving the Qwen brain
Launch vLLM (≥0.21) with auto tool choice + the Qwen parser so it emits OpenAI `tool_calls`:
```
--enable-auto-tool-choice --tool-call-parser hermes
```
(Per project brief WS1, `qwen3_coder` is the documented parser for the `Qwen3-30B-A3B-2507`
coder-style tool calling; `hermes` is the common Qwen3 chat parser. **Confirm which parser the
deployed model card / vLLM version expects** — see risks.) Without `--enable-auto-tool-choice`
vLLM returns the tool call as plain text and the cascade will see no `tool_call` chunk
(it would auto-`speak` the raw JSON instead — `pipeline.py:173-181`).

### Tool-passthrough verification plan
1. **Endpoint smoke test (no app):** `curl`/python `chat.completions.create(..., stream=True,
   tools=[roll_dice spec], tool_choice="auto")` against `base_url`. Confirm the stream contains
   `delta.tool_calls` chunks with a function name + JSON `arguments` (not text). This isolates the
   server config from the app.
2. **In-app round-trip:** start the cascade backend with the `dm` profile (its `tools.txt` enables
   the MCP tools incl. `roll_dice` from the fallout-helper server on :3001). Speak/inject a prompt
   that forces a roll ("roll a d20 for my attack").
3. **Confirm via logs (DEBUG):** look for, in order — `LLM tool call: …` (pipeline.py:143) →
   `Executing tool: roll_dice(...)` (pipeline.py:213) → `Tool result: …` (pipeline.py:230) →
   the re-invoke line `No speak in tool calls — re-invoking LLM…` (pipeline.py:193) → a final
   `speak` with the narrated result. The `_log_prompt` dump (pipeline.py:18) shows the tool spec
   reached the model and the `role:"tool"` result was fed back.
4. **Pass criterion:** the assistant's spoken turn reflects the actual rolled number returned by
   the MCP tool (proves the result round-tripped, not a hallucination).

## Open questions / risks

1. **vLLM tool-call parser choice** (highest risk). The brain must be served with
   `--enable-auto-tool-choice --tool-call-parser <hermes|qwen3_coder>`. Wrong/missing parser →
   tool calls arrive as plain text, the cascade never sees a `tool_call`, and it auto-speaks raw
   JSON. This is **owned by the Modal worktree**; verify with step 1 above before blaming the app.
2. **`temperature` source.** The pipeline always passes `get_config().llm_temperature` (the
   top-level `llm.temperature`, default 1.0) to `generate` (`pipeline.py:135`), ignoring any
   per-provider temperature. 1.0 can destabilize Qwen tool-calling; recommend setting
   `llm.temperature: 0.7` (snippet above). Note this is a **shared** field — confirm the TTS-owner
   isn't relying on it (they shouldn't be; it's LLM-only).
3. **`stream_options.include_usage`** (`openai.py:111`) — most vLLM builds support it, but a few
   older ones reject unknown params. If the stream errors on open, drop it; cost stays at 0 anyway
   for the self-hosted brain.
4. **Image/multimodal messages.** `OpenAILLM._convert_messages_for_openai` emits `image_url`
   data-URLs (`openai.py:60-67`). The text-only Qwen3-30B-A3B brain will reject image parts if the
   `see_image_through_camera` tool feeds a frame back. Vision is meant to be a separate Qwen3-VL
   **tool returning text** (per brief), so keep camera output textual; do not route raw frames to
   the brain.
5. **Connectivity assumption.** The Modal endpoint may not be reachable yet (deployed in another
   worktree). All wiring is config-only; set `CASCADE_LLM_BASE_URL` once the URL exists. The app
   fails cleanly at first LLM call if the URL is wrong (pipeline retries twice, then speaks a
   fallback — `pipeline.py:104-119`).
6. **No env var for ASR/LLM model name.** Model is yaml-only (by design). The dynamic bit (URL)
   is covered by `CASCADE_LLM_BASE_URL`; model changes need a yaml edit or a new provider entry.
```
