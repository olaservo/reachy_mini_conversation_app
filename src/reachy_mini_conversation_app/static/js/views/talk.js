/**
 * Talk view: conversation orb driven by the RPC activity stream.
 * Audio I/O runs entirely in Python; the orb doubles as the mic toggle.
 * Robot stays live, tapping the orb only mutes or unmutes the user's mic.
 * In push-to-talk mode the tap is the latch instead: open the mic, tap again to send.
 */

import {
  LISTEN_MODES,
  applyPersonality,
  getListenState,
  getMicState,
  listPersonalities,
  setListenLatched,
  setMicMuted,
  subscribe,
} from "../api.js";
import { createConversationPanel } from "../components/conversation-panel.js";
import { createLanguagePicker } from "../components/language-picker.js";
import { ORB_STATES } from "../constants.js";
import { createOrb, mapActivityToState } from "../orb.js";
import { consumePendingApply } from "../pending-apply.js";
import { setPersonality } from "../personality-badge.js";
import { h, prettifyProfileName } from "../ui.js";

const CAPTION_BY_STATE = Object.freeze({
  [ORB_STATES.MUTED]: "Muted",
  [ORB_STATES.IDLE]: "Ready",
  [ORB_STATES.CONNECTING]: "Connecting to the backend...",
  [ORB_STATES.LISTENING]: "Listening",
  [ORB_STATES.THINKING]: "Thinking",
  [ORB_STATES.SPEAKING]: "Speaking",
  [ORB_STATES.ERROR]: "Connection error",
});

const PTT_CAPTIONS = Object.freeze({
  closed: "Tap, or hold an antenna, to talk",
  latched: "Listening. Tap to send",
  held: "Listening. Let go of the antenna to send",
});

export async function mountTalkView({ outlet, signal }) {
  const pending = consumePendingApply();
  const micStatePromise = getMicState().catch((error) => {
    console.warn("Failed to load microphone state", error);
    return null;
  });
  const listenStatePromise = getListenState().catch((error) => {
    console.warn("Failed to load listen state", error);
    return null;
  });
  let muted = false;
  let listen = { mode: LISTEN_MODES.ALWAYS_ON, enabled: true, latched: false };
  let micReady = false;
  let togglePending = false;
  let activePersonality = null;
  let subscription = null;

  const caption = h(
    "p",
    { class: "talk__caption", role: "status", "aria-live": "polite" },
    CAPTION_BY_STATE[ORB_STATES.CONNECTING]
  );
  const defaultAction = document.querySelector('[data-component="default-personality-action"]');
  if (defaultAction) {
    defaultAction.hidden = true;
    defaultAction.addEventListener("click", onSetDefault);
  }
  const languageRoot = document.querySelector('[data-component="language-picker"]');
  const languagePicker = languageRoot
    ? createLanguagePicker({
        root: languageRoot,
        signal,
        onStatus: (message) => {
          caption.textContent = message;
        },
      })
    : null;
  const orb = createOrb({
    initialState: ORB_STATES.CONNECTING,
    onStateChange: (state) => {
      // The orb's own idle fallback lands on IDLE; a closed gate or a mute rests elsewhere.
      if (state === ORB_STATES.IDLE && restingState() !== ORB_STATES.IDLE) {
        orb.setState(restingState());
        return;
      }
      caption.textContent = captionFor(state);
    },
  });
  orb.root.disabled = true;
  orb.root.addEventListener("click", onMicTap);
  syncMicAria();
  const conversationPanel = createConversationPanel({
    toggle: document.querySelector('[data-component="conversation-toggle"]'),
    signal,
  });

  signal.addEventListener("abort", cleanup, { once: true });

  const view = h(
    "section",
    { class: "view view--talk" },
    h("div", { class: "talk__stage" }, h("div", { class: "talk__orb-wrap" }, orb.root), caption),
    conversationPanel.root
  );
  outlet.replaceChildren(view);

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
    // The activity subscription will flip the orb to its resting state next tick.
    caption.textContent = CAPTION_BY_STATE[ORB_STATES.CONNECTING];
    void refreshPersonalityState();
  } else {
    // Deep link to /talk with no pending apply: refresh the header badge.
    void refreshPersonalityState();
  }

  const [micState, listenState] = await Promise.all([micStatePromise, listenStatePromise]);
  if (micState) muted = Boolean(micState.muted);
  if (listenState) listen = listenState;
  if (signal.aborted) return;
  micReady = true;
  orb.root.disabled = false;
  syncMicAria();

  subscription = subscribeConversationEvents({
    // Re-sync mic state after subscribing: another tab may have toggled it.
    onReady: async () => {
      if (!togglePending) {
        try {
          const [mic, gate] = await Promise.all([getMicState(), getListenState()]);
          muted = Boolean(mic?.muted);
          if (gate) listen = gate;
        } catch {
          // keep the last known state
        }
      }
      if (signal.aborted) return;
      orb.setState(restingState());
      caption.textContent = captionFor(restingState());
      syncMicAria();
    },
    onActivity: (reason) => {
      if (muted) return;
      const next = mapActivityToState(reason);
      if (next == null) return;
      orb.setState(next);
    },
    // The gate moves under us: an antenna hold, a key, or another tab's latch.
    onListen: (gate) => {
      listen = gate;
      if (signal.aborted) return;
      orb.setState(restingState());
      caption.textContent = captionFor(restingState());
      syncMicAria();
    },
  });

  function cleanup() {
    subscription?.close();
    conversationPanel.dispose();
    orb.dispose();
    if (defaultAction) {
      defaultAction.hidden = true;
      defaultAction.removeEventListener("click", onSetDefault);
    }
  }

  function pushToTalk() {
    return listen.mode === LISTEN_MODES.PUSH_TO_TALK;
  }

  function restingState() {
    if (muted) return ORB_STATES.MUTED;
    if (pushToTalk() && !listen.enabled) return ORB_STATES.MUTED;
    return ORB_STATES.IDLE;
  }

  function captionFor(state) {
    if (pushToTalk() && !muted) {
      if (state === ORB_STATES.MUTED) return PTT_CAPTIONS.closed;
      if (state === ORB_STATES.IDLE) return listen.latched ? PTT_CAPTIONS.latched : PTT_CAPTIONS.held;
    }
    return CAPTION_BY_STATE[state] || "";
  }

  async function onMicTap() {
    if (!micReady || togglePending) return;
    togglePending = true;
    try {
      if (pushToTalk()) {
        listen = await setListenLatched(!listen.latched);
      } else {
        const data = await setMicMuted(!muted);
        muted = Boolean(data?.muted);
      }
    } catch (error) {
      if (!signal.aborted) {
        caption.textContent = `Failed to toggle the microphone: ${error?.message || error}`;
      }
      return;
    } finally {
      togglePending = false;
    }
    if (signal.aborted) return;
    orb.setState(restingState());
    // setState skips unchanged states, so set the caption explicitly
    caption.textContent = captionFor(restingState());
    syncMicAria();
  }

  async function refreshPersonalityState() {
    // The language can be saved per personality, so it follows the active one.
    void languagePicker?.refresh();
    const personalityState = await fetchPersonalityState();
    if (signal.aborted || personalityState == null) return;
    activePersonality = personalityState.current;
    setPersonality(personalityState.current);
    const shouldHide = personalityState.locked || personalityState.current === personalityState.startup;
    if (defaultAction) {
      defaultAction.hidden = shouldHide;
    }
  }

  async function onSetDefault() {
    if (!defaultAction || !activePersonality) return;
    defaultAction.disabled = true;
    caption.textContent = `Saving "${prettifyProfileName(activePersonality)}" as default...`;
    try {
      await applyPersonality(activePersonality, { persist: true });
      if (signal.aborted) return;
      defaultAction.hidden = true;
      caption.textContent = `"${prettifyProfileName(activePersonality)}" will be used at startup.`;
    } catch (error) {
      if (!signal.aborted) {
        caption.textContent = `Failed to save default: ${error?.message || error}`;
      }
    } finally {
      defaultAction.disabled = false;
    }
  }

  function syncMicAria() {
    if (!micReady) {
      orb.root.setAttribute("aria-pressed", "false");
      orb.root.setAttribute("aria-label", "Loading microphone state");
      return;
    }
    if (pushToTalk() && !muted) {
      orb.root.setAttribute("aria-pressed", String(listen.latched));
      orb.root.setAttribute("aria-label", listen.latched ? "Send and stop listening" : "Start listening");
      return;
    }
    orb.root.setAttribute("aria-pressed", String(!muted));
    orb.root.setAttribute("aria-label", muted ? "Unmute microphone" : "Mute microphone");
  }
}

async function fetchPersonalityState() {
  try {
    const data = await listPersonalities();
    const current = data?.current;
    if (!current) return null;
    return {
      current,
      startup: data?.startup || "default",
      locked: Boolean(data?.locked),
    };
  } catch {
    return null;
  }
}

function subscribeConversationEvents({ onActivity, onReady, onListen } = {}) {
  if (typeof onActivity !== "function") {
    throw new TypeError("subscribeConversationEvents: onActivity is required");
  }

  // Activity reasons now arrive as conversation.activity notifications over the
  // /rpc WebSocket; the shared client (api.js) owns reconnection.
  const unsubscribe = subscribe("conversation.activity", (params) => {
    const reason = (params?.reason || "").trim();
    if (reason) onActivity(reason);
  });
  const unsubscribeListen =
    typeof onListen === "function"
      ? subscribe("conversation.listen", (params) => {
          if (params?.mode) onListen(params);
        })
      : () => {};

  // The socket connects lazily, so schedule the initial mic and orb sync.
  if (typeof onReady === "function") Promise.resolve().then(onReady);

  return {
    close() {
      unsubscribe();
      unsubscribeListen();
    },
  };
}
