/**
 * Talk view: conversation orb driven by the SSE activity stream.
 * Audio I/O runs entirely in Python; this view is a read-only status surface.
 */

import { listPersonalities } from "../api.js";
import { BUILT_IN_DEFAULT_OPTION, ORB_STATES } from "../constants.js";
import { createOrb, mapActivityToState } from "../orb.js";
import { h, prettifyProfileName } from "../ui.js";

const SSE_ENDPOINT = "/conversation_events";

/** Human-readable label below the orb for each visual state. */
const CAPTION_BY_STATE = Object.freeze({
  [ORB_STATES.IDLE]: "Tap to talk - the robot is listening passively.",
  [ORB_STATES.CONNECTING]: "Connecting to the conversation event stream…",
  [ORB_STATES.LISTENING]: "Listening…",
  [ORB_STATES.THINKING]: "Thinking…",
  [ORB_STATES.SPEAKING]: "Speaking…",
  [ORB_STATES.ERROR]: "Conversation event stream disconnected.",
});

export async function mountTalkView({ outlet, signal }) {
  const orb = createOrb({ initialState: ORB_STATES.CONNECTING });
  const caption = h("p", { class: "talk__caption" }, CAPTION_BY_STATE[ORB_STATES.CONNECTING]);
  const profileBadge = h("span", { class: "talk__profile" }, "");

  const view = h(
    "section",
    { class: "view view--talk" },
    profileBadge,
    h("div", { class: "talk__orb-wrap" }, orb.root),
    caption
  );
  outlet.replaceChildren(view);

  fetchActiveProfileLabel().then((label) => {
    if (signal.aborted || !label) return;
    profileBadge.textContent = label;
  });

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
      // EventSource auto-reconnects; we surface the error so callers can show a hint.
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

function subscribeConversationEvents({ onActivity, onReady, onError } = {}) {
  if (typeof onActivity !== "function") {
    throw new TypeError("subscribeConversationEvents: onActivity is required");
  }

  const source = new EventSource(SSE_ENDPOINT);

  source.addEventListener("activity", (ev) => {
    const reason = (ev.data || "").trim();
    if (reason) onActivity(reason);
  });

  if (typeof onReady === "function") {
    source.addEventListener("ready", () => onReady());
  }

  if (typeof onError === "function") {
    source.addEventListener("error", (err) => onError(err));
  }

  return {
    close() {
      source.close();
    },
  };
}
