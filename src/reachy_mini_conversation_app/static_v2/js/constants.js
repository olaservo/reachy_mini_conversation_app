/**
 * Shared constants and mappings for the modern web UI.
 *
 * This module is intentionally side-effect free so it can be imported by any
 * other module (views, components, helpers) without ordering concerns.
 */

/**
 * Backend identifiers, mirroring the values accepted by ``POST /backend_config``
 * on the Python side. Keep these strings in sync with ``config.py``.
 */
export const BACKENDS = Object.freeze({
  HUGGINGFACE: "huggingface",
  OPENAI: "openai",
  GEMINI: "gemini",
});

/**
 * Profile name used by the backend to mean "no custom profile".
 *
 * Comes from ``headless_personality.DEFAULT_OPTION`` and is the literal string
 * the backend's ``/personalities`` endpoint returns at index 0.
 */
export const BUILT_IN_DEFAULT_OPTION = "(built-in default)";

/**
 * Maps a profile name (as listed by ``/personalities``) to its illustration.
 *
 * The avatar pool was extracted from the ``feat/new-ui`` branch and lives in
 * ``static_v2/avatars/``. Profiles without a dedicated illustration fall back
 * to ``default.svg`` via ``avatarFor`` below.
 */
export const AVATAR_BY_PROFILE = Object.freeze({
  bored_teenager: "bored-teenager.svg",
  captain_circuit: "captain-circuit.svg",
  chess_coach: "chess-coach.svg",
  cosmic_kitchen: "cosmic-kitchen.svg",
  default: "default.svg",
  hype_bot: "hype-bot.svg",
  mad_scientist_assistant: "mad-scientist.svg",
  mars_rover: "mars-rover.svg",
  nature_documentarian: "nature-doc.svg",
  noir_detective: "noir-detective.svg",
  sorry_bro: "sorry-bro.svg",
  time_traveler: "time-traveler.svg",
  victorian_butler: "victorian-butler.svg",
});

/**
 * Default tool set assigned to profiles created via the "Custom" card.
 *
 * Mirrors the ``tools.txt`` file shipped with every built-in profile on
 * ``main`` (e.g. ``profiles/default/tools.txt``). Without these, a freshly
 * created profile would run with zero capabilities (no dance, no emotions,
 * no head tracking) and the robot would feel broken on first use.
 *
 * Users who want a leaner profile can edit ``tools.txt`` directly on disk -
 * the v1 modern UI deliberately does not surface per-tool toggles.
 */
export const DEFAULT_TOOLS = Object.freeze([
  "dance",
  "stop_dance",
  "play_emotion",
  "stop_emotion",
  "camera",
  "idle_do_nothing",
  "head_tracking",
  "move_head",
]);

/**
 * Visual conversation states surfaced by the orb.
 *
 * Distinct from the raw activity reasons emitted by the backend SSE stream
 * (see ``base_realtime._mark_activity``). The mapping from raw reasons to
 * these visual buckets lives in ``orb.js``.
 */
export const ORB_STATES = Object.freeze({
  IDLE: "idle",
  CONNECTING: "connecting",
  LISTENING: "listening",
  THINKING: "thinking",
  SPEAKING: "speaking",
  ERROR: "error",
});

/**
 * Per-state glow color, written as ``--glow`` inline on the orb element.
 *
 * The palette mirrors ``reachy_mini_minimal_conversation``'s orb so the
 * three surfaces (mobile orb, minimal conversation orb, this one) read as
 * the same widget across the product.
 */
export const GLOW_BY_STATE = Object.freeze({
  [ORB_STATES.IDLE]: "#34d399",       // ready / breathing
  [ORB_STATES.CONNECTING]: "#facc15", // negotiating
  [ORB_STATES.LISTENING]: "#22d3ee",  // user speaks
  [ORB_STATES.THINKING]: "#f59e0b",   // model composes
  [ORB_STATES.SPEAKING]: "#8b7dff",   // model speaks
  [ORB_STATES.ERROR]: "#ff6a75",
});

/**
 * Hash routes recognised by ``router.js``.
 *
 * Centralised so view code never assembles route strings by hand.
 */
export const ROUTES = Object.freeze({
  HOME: "#/",
  TALK: "#/talk",
  SETTINGS: "#/settings",
});

/**
 * Resolve a profile name to its avatar URL.
 *
 * @param {string} profileName Name returned by ``/personalities`` (without
 *   the ``user_personalities/`` prefix).
 * @returns {string} URL relative to the page root.
 */
export function avatarFor(profileName) {
  const file = AVATAR_BY_PROFILE[profileName] || AVATAR_BY_PROFILE.default;
  return `/static/avatars/${file}`;
}
