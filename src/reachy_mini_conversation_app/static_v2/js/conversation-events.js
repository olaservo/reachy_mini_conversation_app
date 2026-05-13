/**
 * Server-Sent Events client for the conversation activity stream.
 *
 * The backend exposes ``GET /conversation_events`` (see
 * ``console._conversation_events_stream``) which emits two kinds of frames:
 *
 *   event: ready    -> sent once when the connection is established.
 *   event: activity -> ``data: <reason>`` for every activity transition
 *                       (``user_speech_started``, ``response_created``,
 *                       ``assistant_audio_delta``, ...). The exact set lives
 *                       in ``base_realtime._mark_activity`` call sites.
 *
 * Browsers' built-in ``EventSource`` already handles auto-reconnect and the
 * SSE wire format, so this module is a thin wrapper that exposes a
 * lifecycle-friendly API:
 *
 *   const sub = subscribeConversationEvents({
 *     onActivity: (reason) => console.log(reason),
 *     onReady:    ()       => console.log("connected"),
 *     onError:    (err)    => console.warn(err),
 *   });
 *   // ... later ...
 *   sub.close();
 */

const ENDPOINT = "/conversation_events";

/**
 * @param {object} options
 * @param {(reason: string) => void} options.onActivity
 * @param {() => void} [options.onReady]
 * @param {(error: Event) => void} [options.onError]
 * @returns {{ close: () => void }}
 */
export function subscribeConversationEvents({ onActivity, onReady, onError } = {}) {
  if (typeof onActivity !== "function") {
    throw new TypeError("subscribeConversationEvents: onActivity is required");
  }

  const source = new EventSource(ENDPOINT);

  source.addEventListener("activity", (ev) => {
    const reason = (ev.data || "").trim();
    if (reason) onActivity(reason);
  });

  if (typeof onReady === "function") {
    source.addEventListener("ready", () => onReady());
  }

  if (typeof onError === "function") {
    // ``EventSource`` fires ``error`` on transient drops too; the browser will
    // auto-reconnect after the ``retry: 2000`` hint we send. We bubble the
    // event up so callers can render a discreet "reconnecting" hint, but we
    // do not try to reopen ourselves.
    source.addEventListener("error", (err) => onError(err));
  }

  return {
    close() {
      source.close();
    },
  };
}
