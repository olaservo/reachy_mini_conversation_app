/**
 * Conversation orb - vanilla port of the orb in
 * ``reachy_mini_minimal_conversation`` (its visual reference) plus the
 * SSE-driven state machine that maps backend activity reasons onto a small
 * set of visual buckets.
 *
 * Design notes
 * ------------
 * Every indicator (mic, bars, dots, voice wave, spinner, error) is rendered
 * once at construction time and stacked in the same CSS grid cell. The
 * CSS rules in ``style.css`` show the right one based on the
 * ``data-state`` attribute we set on the root, with a small opacity +
 * scale transition. We never replace child nodes when the state changes,
 * which:
 *
 *   - eliminates the "indicator pops in/out" jank we had with the
 *     previous DOM-replacing implementation;
 *   - keeps the orb a self-contained widget: no virtual DOM, no
 *     reconciliation logic, just a single attribute swap per transition;
 *   - matches what the React orb in the mobile app does
 *     (``ConversationOrb.tsx``), which made this exact tradeoff for
 *     the same reason.
 *
 * State mapping
 * -------------
 * The backend emits ~9 different activity reasons (``user_speech_started``,
 * ``response_created``, ``assistant_audio_delta``, ``tool_call_received``,
 * ...). The user only needs four high-level cues:
 *
 *   idle       | nothing is happening
 *   listening  | the user is speaking
 *   thinking   | the user finished, the model is preparing a response
 *   speaking   | the model is producing audio
 *
 * Concentrating that mapping in ``mapActivityToState`` keeps both the SSE
 * client and the DOM render dumb and easy to reason about.
 *
 * No audio reactivity
 * -------------------
 * Unlike the mobile orb, the conversation app routes audio entirely through
 * Python (the browser is a control plane). We have no per-frame mic level on
 * the JS side, so the orb is animated purely through CSS keyframes and the
 * ``data-state`` attribute swaps palettes / indicators.
 */

import { h } from "./ui.js";
import { GLOW_BY_STATE, ORB_STATES } from "./constants.js";

/**
 * Map a backend activity reason (see ``base_realtime._mark_activity``) to one
 * of the four visual buckets above. Unknown reasons keep the previous state
 * (returned as ``null``) so we never flicker on noise.
 *
 * @param {string} reason
 * @returns {string|null} new ORB_STATES value, or null to keep the current one
 */
export function mapActivityToState(reason) {
  switch (reason) {
    case "user_speech_started":
    case "user_transcription_delta":
      return ORB_STATES.LISTENING;

    case "user_speech_stopped":
    case "user_transcription_completed":
    case "response_created":
    case "tool_call_received":
    case "tool_result_ready":
      return ORB_STATES.THINKING;

    case "assistant_audio_delta":
      return ORB_STATES.SPEAKING;

    case "assistant_transcript_done":
      return ORB_STATES.IDLE;

    default:
      return null;
  }
}

/**
 * Time of inactivity after which the orb auto-falls back to ``idle``.
 *
 * The backend never explicitly emits an ``idle`` event after a turn ends, so
 * this timeout is the safety net that brings the orb back to its breathing
 * state once the assistant goes quiet.
 */
const IDLE_FALLBACK_MS = 1500;

/**
 * Build the orb DOM and return a controller exposing setState + dispose.
 *
 * The DOM mirrors the structure of ``circle`` in the minimal conversation
 * app (see ``index.html`` / ``style.css`` there). Every indicator is mounted
 * statically and shown / hidden via CSS based on ``data-state``.
 */
export function createOrb({ initialState = ORB_STATES.IDLE } = {}) {
  let currentState = initialState;
  let idleTimer = null;

  const indicator = h(
    "span",
    { class: "convo-orb__indicator", "aria-hidden": "true" },
    micIcon(),
    spinnerIndicator(),
    barsIndicator(),
    thinkingDotsIndicator(),
    voiceWaveIcon(),
    errorIcon()
  );

  const root = h(
    "div",
    {
      class: "convo-orb",
      dataset: { state: currentState },
      "aria-label": "Conversation status",
      style: { "--glow": GLOW_BY_STATE[currentState] },
    },
    h("span", { class: "convo-orb__glow", "aria-hidden": "true" }),
    h("span", { class: "convo-orb__ring", "aria-hidden": "true" }),
    h("span", { class: "convo-orb__ring-outer", "aria-hidden": "true" }),
    h("span", { class: "convo-orb__core" }, indicator)
  );

  /** Update the orb to reflect a new visual state. */
  function setState(nextState) {
    if (!Object.values(ORB_STATES).includes(nextState)) return;
    if (nextState === currentState) {
      // Same state, just refresh the idle fallback timer (e.g. continued
      // ``assistant_audio_delta`` events while speaking).
      bumpIdleTimer(nextState);
      return;
    }
    currentState = nextState;
    root.dataset.state = nextState;
    root.style.setProperty("--glow", GLOW_BY_STATE[nextState]);
    bumpIdleTimer(nextState);
  }

  /**
   * Schedule (or cancel) the auto-return-to-idle timer.
   *
   * Only ``listening``, ``thinking`` and ``speaking`` are transient states
   * that should decay; ``idle``, ``connecting`` and ``error`` are sticky and
   * must be cleared explicitly.
   */
  function bumpIdleTimer(state) {
    if (idleTimer != null) {
      clearTimeout(idleTimer);
      idleTimer = null;
    }
    const transient =
      state === ORB_STATES.LISTENING ||
      state === ORB_STATES.THINKING ||
      state === ORB_STATES.SPEAKING;
    if (!transient) return;
    idleTimer = setTimeout(() => {
      idleTimer = null;
      setState(ORB_STATES.IDLE);
    }, IDLE_FALLBACK_MS);
  }

  /** Drive the orb from a raw backend activity reason. */
  function applyActivity(reason) {
    const next = mapActivityToState(reason);
    if (next != null) setState(next);
  }

  /** Stop any pending timer. Call before detaching the DOM node. */
  function dispose() {
    if (idleTimer != null) {
      clearTimeout(idleTimer);
      idleTimer = null;
    }
  }

  return { root, setState, applyActivity, dispose };
}

/* ----------------------------------------------------------------------------
 * Indicators
 * --------------------------------------------------------------------------
 * Each helper returns the static DOM for one indicator. They all live in the
 * same CSS grid cell inside ``.convo-orb__indicator`` and are toggled via
 * CSS rules in ``style.css`` (``.convo-orb[data-state="..."] .ind-X``).
 *
 * Inline SVG markup is intentional: every icon ships with the JS bundle,
 * inherits ``currentColor`` (the per-state ``--glow``), and stays a single
 * file to audit. ``html`` is the explicit escape hatch ``ui.h`` exposes for
 * trusted, hand-authored HTML.
 * -------------------------------------------------------------------------- */

function barsIndicator() {
  return h(
    "span",
    { class: "ind ind-bars" },
    h("span", { class: "bar" }),
    h("span", { class: "bar" }),
    h("span", { class: "bar" }),
    h("span", { class: "bar" }),
    h("span", { class: "bar" })
  );
}

function thinkingDotsIndicator() {
  return h(
    "span",
    { class: "ind ind-thinking" },
    h("span", { class: "dot" }),
    h("span", { class: "dot" }),
    h("span", { class: "dot" })
  );
}

function spinnerIndicator() {
  return h("span", { class: "ind ind-spinner" });
}

function voiceWaveIcon() {
  return h("span", {
    class: "ind ind-voice",
    html: `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M3 10v4a1 1 0 0 0 1 1h3l5 4V5L7 9H4a1 1 0 0 0-1 1z" fill="currentColor" stroke="none"/>
        <path class="wave wave-1" d="M16 8a5 5 0 0 1 0 8"/>
        <path class="wave wave-2" d="M19 5a9 9 0 0 1 0 14"/>
      </svg>`,
  });
}

function micIcon() {
  return h("span", {
    class: "ind ind-mic",
    html: `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <rect x="9" y="2" width="6" height="12" rx="3" fill="currentColor" stroke="none"/>
        <path d="M5 10a7 7 0 0 0 14 0"/>
        <line x1="12" y1="19" x2="12" y2="22"/>
        <line x1="8" y1="22" x2="16" y2="22"/>
      </svg>`,
  });
}

function errorIcon() {
  return h("span", {
    class: "ind ind-error",
    html: `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <circle cx="12" cy="12" r="9"/>
        <line x1="12" y1="8" x2="12" y2="13"/>
        <line x1="12" y1="16" x2="12" y2="16"/>
      </svg>`,
  });
}
