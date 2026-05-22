/**
 * Cross-view hand-off for an in-flight personality apply.
 *
 * Set by ``home.js`` right before it navigates to ``/talk``, then awaited
 * (and cleared) by ``talk.js`` so the orb can stay in CONNECTING with a
 * dedicated caption until the backend acknowledges the switch. Keeping
 * this state at the module scope avoids polluting ``window`` and survives
 * the AbortController teardown between views.
 */

let pending = null;

/**
 * Record an in-flight personality apply.
 *
 * @param {{ name: string, promise: Promise<unknown> }} entry
 */
export function setPendingApply(entry) {
  if (!entry || typeof entry.promise?.then !== "function") {
    pending = null;
    return;
  }
  pending = entry;
  // Attach a no-op catch on a derived chain so a rejected apply never
  // becomes an ``unhandledrejection`` if the consumer (talk view) is
  // never mounted - e.g. the user navigates away before /talk loads.
  // The original ``entry.promise`` is untouched so ``consumePendingApply``
  // callers can still ``await`` and observe the rejection themselves.
  entry.promise.catch(() => {});
}

/**
 * Read and clear the in-flight apply, if any. Single-shot by design:
 * the consumer (talk view) owns the lifecycle from here on.
 *
 * @returns {{ name: string, promise: Promise<unknown> } | null}
 */
export function consumePendingApply() {
  const entry = pending;
  pending = null;
  return entry;
}
