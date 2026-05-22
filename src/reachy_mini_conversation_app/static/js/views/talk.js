/**
 * Talk view: conversation orb driven by the SSE activity stream.
 * Audio I/O runs entirely in Python; this view is a read-only status surface.
 */

import { listPersonalities } from "../api.js";
import { BUILT_IN_DEFAULT_OPTION, ORB_STATES } from "../constants.js";
import { createOrb, mapActivityToState } from "../orb.js";
import { consumePendingApply } from "../pending-apply.js";
import { setPersonality } from "../personality-badge.js";
import { h, prettifyProfileName } from "../ui.js";

const SSE_ENDPOINT = "/conversation_events";

/** Human-readable label below the orb for each visual state. */
const CAPTION_BY_STATE = Object.freeze({
  [ORB_STATES.IDLE]: "Ready — just speak to Reachy.",
  [ORB_STATES.CONNECTING]: "Connecting to the conversation event stream…",
  [ORB_STATES.LISTENING]: "Listening…",
  [ORB_STATES.THINKING]: "Thinking…",
  [ORB_STATES.SPEAKING]: "Speaking…",
  [ORB_STATES.ERROR]: "Conversation event stream disconnected.",
});

export async function mountTalkView({ outlet, signal }) {
  const pending = consumePendingApply();
  const caption = h("p", { class: "talk__caption" }, CAPTION_BY_STATE[ORB_STATES.CONNECTING]);
  const orb = createOrb({
    initialState: ORB_STATES.CONNECTING,
    onStateChange: (state) => {
      caption.textContent = CAPTION_BY_STATE[state] || "";
    },
  });

  const view = h(
    "section",
    { class: "view view--talk" },
    h("div", { class: "talk__orb-wrap" }, orb.root),
    caption
  );
  outlet.replaceChildren(view);

  // While the apply is in flight, show a context-aware caption but keep
  // the orb visually in CONNECTING. When there is no pending apply (deep
  // link to /talk), refresh the header badge from the backend instead.
  if (pending) {
    caption.textContent = `Applying "${prettifyProfileName(pending.name)}"…`;
    try {
      await pending.promise;
    } catch (error) {
      if (signal.aborted) return;
      orb.setState(ORB_STATES.ERROR);
      caption.textContent = `Failed to apply personality: ${error?.message || error}`;
      return;
    }
    if (signal.aborted) return;
    // Reset to the generic CONNECTING caption; SSE ``ready`` will flip
    // the orb to IDLE on the next tick.
    caption.textContent = CAPTION_BY_STATE[ORB_STATES.CONNECTING];
  } else {
    fetchActivePersonality().then((name) => {
      if (signal.aborted) return;
      if (name) setPersonality(name);
    });
  }

  const subscription = subscribeConversationEvents({
    onReady: () => {
      orb.setState(ORB_STATES.IDLE);
    },
    onActivity: (reason) => {
      const next = mapActivityToState(reason);
      if (next == null) return;
      orb.setState(next);
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

async function fetchActivePersonality() {
  try {
    const data = await listPersonalities();
    const current = data?.current;
    if (!current || current === BUILT_IN_DEFAULT_OPTION) return null;
    return current;
  } catch {
    return null;
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
