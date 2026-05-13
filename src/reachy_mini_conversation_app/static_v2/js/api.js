/**
 * Single source of truth for HTTP calls to the settings backend.
 *
 * Every fetch lives in this module so that:
 *   - View code can be reviewed without having to grep for ``fetch(`` calls.
 *   - We keep one place to add timeouts, retries, error normalisation, etc.
 *
 * All endpoints are served by ``console.LocalStream._init_settings_ui_if_needed``
 * (existing routes) and ``static_v2`` is purely additive on top of them - we do
 * not introduce any new endpoint here besides the SSE stream consumed elsewhere.
 */

const DEFAULT_TIMEOUT_MS = 8000;

class HttpError extends Error {
  constructor(status, body, message) {
    super(message || `HTTP ${status}`);
    this.status = status;
    this.body = body;
  }
}

/**
 * Wrap ``fetch`` with a timeout and JSON decoding. Raises ``HttpError`` on
 * non-2xx responses so callers can handle failures consistently.
 */
async function request(method, url, { body, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      method,
      signal: controller.signal,
      headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
    const text = await response.text();
    let json = null;
    if (text) {
      try {
        json = JSON.parse(text);
      } catch {
        json = { raw: text };
      }
    }
    if (!response.ok) {
      throw new HttpError(response.status, json, json?.error || response.statusText);
    }
    return json;
  } finally {
    clearTimeout(timer);
  }
}

/* ---------------------------------------------------------------------------
 * Status / readiness
 * ------------------------------------------------------------------------- */

/** Snapshot of which backend is active and whether credentials are present. */
export const getStatus = () => request("GET", "/status");

/** Whether the backend has finished loading tools (used to gate the UI). */
export const getReady = () => request("GET", "/ready");

/* ---------------------------------------------------------------------------
 * Backend selection & credentials
 * ------------------------------------------------------------------------- */

/**
 * Save a backend choice (and optionally its credentials) to the instance .env.
 *
 * ``payload`` mirrors ``console.BackendPayload``:
 *   { backend, api_key?, hf_mode?, hf_host?, hf_port? }
 */
export const saveBackendConfig = (payload) =>
  request("POST", "/backend_config", { body: payload });

/** Persist an OpenAI API key without changing the active backend. */
export const saveOpenAiKey = (key) =>
  request("POST", "/openai_api_key", { body: { openai_api_key: key } });

/** Validate (without persisting) that an OpenAI API key actually works. */
export const validateOpenAiKey = (key) =>
  request("POST", "/validate_api_key", {
    body: { openai_api_key: key },
    timeoutMs: 12000, // upstream call to api.openai.com, can be slower
  });

/* ---------------------------------------------------------------------------
 * Personalities
 * ------------------------------------------------------------------------- */

/** List all personality choices, plus current and startup selections. */
export const listPersonalities = () => request("GET", "/personalities");

/** Read a personality's instructions, tools, voice, etc. */
export const loadPersonality = (name) =>
  request("GET", `/personalities/load?name=${encodeURIComponent(name)}`);

/**
 * Save a personality (creates it under ``user_personalities/`` if new).
 *
 * ``payload`` shape:
 *   { name, instructions, tools_text, voice }
 */
export const savePersonality = (payload) =>
  request("POST", "/personalities/save", { body: payload });

/**
 * Apply a personality at runtime. Pass ``persist: true`` to also use it on
 * next startup. Backend rejects with 403 when ``LOCKED_PROFILE`` is set.
 */
export const applyPersonality = (name, { persist = false } = {}) =>
  request("POST", "/personalities/apply", { body: { name, persist } });

/* ---------------------------------------------------------------------------
 * Voice
 * ------------------------------------------------------------------------- */

/** Voices available for the active backend. */
export const listVoices = () => request("GET", "/voices");

/** Currently selected voice on the live realtime session. */
export const getCurrentVoice = () => request("GET", "/voices/current");

/** Apply a voice to the live session. Some backends require a session restart. */
export const applyVoice = (voice) =>
  request("POST", "/voices/apply", { body: { voice } });

export { HttpError };
