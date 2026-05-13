/**
 * Talk view: conversation orb + state caption.
 *
 * The view subscribes to the SSE event stream on mount and unsubscribes on
 * unmount via the AbortSignal carried by the router context. The orb is the
 * sole source of UI state - this module just glues SSE -> orb and renders
 * a small caption below.
 *
 * No audio is captured or played in the browser: audio I/O happens entirely
 * inside the Python process talking to the realtime model. This view is a
 * read-only "watch what the robot is doing" surface.
 */

import { listPersonalities } from "../api.js";
import { BUILT_IN_DEFAULT_OPTION, ORB_STATES } from "../constants.js";
import { subscribeConversationEvents } from "../conversation-events.js";
import { createOrb, mapActivityToState } from "../orb.js";
import { h, prettifyProfileName } from "../ui.js";

/** Human-readable label below the orb for each visual state. */
const CAPTION_BY_STATE = Object.freeze({
  [ORB_STATES.IDLE]: "Tap to talk - the robot is listening passively.",
  [ORB_STATES.CONNECTING]: "Connecting to the conversation event stream…",
  [ORB_STATES.LISTENING]: "Listening…",
  [ORB_STATES.THINKING]: "Thinking…",
  [ORB_STATES.SPEAKING]: "Speaking…",
  [ORB_STATES.ERROR]: "Conversation event stream disconnected.",
});

/**
 * @param {{ outlet: HTMLElement, signal: AbortSignal }} ctx
 */
export async function mountTalkView({ outlet, signal }) {
  const orb = createOrb({ initialState: ORB_STATES.CONNECTING });
  const caption = h("p", { class: "talk__caption" }, CAPTION_BY_STATE[ORB_STATES.CONNECTING]);
  const profileBadge = h("span", { class: "talk__profile" }, "");

  // The back affordance lives in the global header (see main.js +
  // index.html); it shows up automatically when this route is active.
  // Keeping it out of the view body lets the orb have the whole stage
  // and avoids two competing "back" anchors (one per view, one global).
  const view = h(
    "section",
    { class: "view view--talk" },
    profileBadge,
    h("div", { class: "talk__orb-wrap" }, orb.root),
    caption
  );
  outlet.replaceChildren(view);

  // Best-effort lookup of the active profile name to label the surface.
  fetchActiveProfileLabel().then((label) => {
    if (signal.aborted || !label) return;
    profileBadge.textContent = label;
  });

  // Drive the orb from the SSE stream. ``mapActivityToState`` lives on the
  // orb so it can decide which states are worth animating; this view just
  // forwards the raw reasons.
  const subscription = subscribeConversationEvents({
    onReady: () => {
      orb.setState(ORB_STATES.IDLE);
      caption.textContent = CAPTION_BY_STATE[ORB_STATES.IDLE];
    },
    onActivity: (reason) => {
      const next = mapActivityToState(reason);
      if (next == null) return;
      orb.setState(next);
      caption.textContent = CAPTION_BY_STATE[next] || "";
    },
    onError: () => {
      // ``EventSource`` will keep retrying on its own; we just hint the user.
      orb.setState(ORB_STATES.ERROR);
      caption.textContent = CAPTION_BY_STATE[ORB_STATES.ERROR];
    },
  });

  signal.addEventListener("abort", () => {
    subscription.close();
    orb.dispose();
  });
}

async function fetchActiveProfileLabel() {
  try {
    const data = await listPersonalities();
    const current = data?.current;
    if (!current || current === BUILT_IN_DEFAULT_OPTION) return "Default personality";
    return prettifyProfileName(current);
  } catch {
    return "";
  }
}
