/** HTTP client for all calls to the settings backend. */

const DEFAULT_TIMEOUT_MS = 8000;

class HttpError extends Error {
  constructor(status, body, message) {
    super(message || `HTTP ${status}`);
    this.status = status;
    this.body = body;
  }
}

/** fetch with timeout and JSON decoding; throws HttpError on non-2xx. */
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

export const getStatus = () => request("GET", "/status");
export const getReady = () => request("GET", "/ready");

export const saveBackendConfig = (payload) =>
  request("POST", "/backend_config", { body: payload });
export const saveOpenAiKey = (key) =>
  request("POST", "/openai_api_key", { body: { openai_api_key: key } });
export const validateOpenAiKey = (key) =>
  request("POST", "/validate_api_key", {
    body: { openai_api_key: key },
    timeoutMs: 12000, // upstream call to api.openai.com, can be slower
  });

export const listPersonalities = () => request("GET", "/personalities");
export const loadPersonality = (name) =>
  request("GET", `/personalities/load?name=${encodeURIComponent(name)}`);
export const savePersonality = (payload) =>
  request("POST", "/personalities/save", { body: payload });
export const applyPersonality = (name, { persist = false } = {}) =>
  request("POST", "/personalities/apply", { body: { name, persist } });

export const listVoices = () => request("GET", "/voices");
export const getCurrentVoice = () => request("GET", "/voices/current");
export const applyVoice = (voice) =>
  request("POST", "/voices/apply", { body: { voice } });

export { HttpError };
