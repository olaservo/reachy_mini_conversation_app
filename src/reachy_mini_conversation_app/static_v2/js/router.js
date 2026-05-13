/**
 * Minimal hash router.
 *
 * Hash-based routing avoids any need for HTML5 ``History`` API server-side
 * support: every internal link uses ``#/foo``, the browser does not refetch,
 * and we listen to ``hashchange`` to swap the active view.
 *
 * Usage:
 *
 *   const router = createRouter({
 *     "#/":         () => mountHome(outlet),
 *     "#/talk":     () => mountTalk(outlet),
 *     "#/settings": () => mountSettings(outlet),
 *   }, { fallback: "#/" });
 *   router.start();
 *
 * The route handler is called every time the matching route becomes active.
 * It receives a ``ViewContext`` carrying the outlet element and a "signal"
 * that fires when the view is being unmounted (so the handler can clean up
 * timers, SSE subscriptions, event listeners, ...).
 */

/**
 * @typedef {object} ViewContext
 * @property {HTMLElement} outlet  The container the view should populate.
 * @property {AbortSignal}  signal  Aborted right before the next view mounts.
 * @property {string}      route   The matched route key (e.g. "#/talk").
 */

export function createRouter(routes, { fallback = "#/", outlet } = {}) {
  if (!outlet) throw new Error("createRouter: outlet is required");
  let currentController = null;
  let lastRoute = null;

  function resolve() {
    const raw = window.location.hash || fallback;
    return Object.prototype.hasOwnProperty.call(routes, raw) ? raw : fallback;
  }

  function dispatch() {
    const route = resolve();
    if (route === lastRoute) return;

    // Tear down the previous view first.
    if (currentController) {
      currentController.abort();
      currentController = null;
    }
    outlet.replaceChildren();

    lastRoute = route;
    currentController = new AbortController();
    /** @type {ViewContext} */
    const ctx = { outlet, signal: currentController.signal, route };
    try {
      routes[route](ctx);
    } catch (error) {
      // A view crashing should not nuke the whole app: render a tiny error
      // surface and let the user navigate away.
      console.error("Route handler failed for", route, error);
      outlet.replaceChildren(renderRouteError(route, error));
    }
  }

  function renderRouteError(route, error) {
    const div = document.createElement("div");
    div.className = "route-error";
    div.textContent = `Failed to render ${route}: ${error?.message || error}`;
    return div;
  }

  return {
    start() {
      window.addEventListener("hashchange", dispatch);
      // Normalise on boot: if the hash is missing or unknown, replace it so
      // the URL reflects the actual view we are about to render.
      const target = resolve();
      if (window.location.hash !== target) {
        window.location.replace(target);
      }
      dispatch();
    },
    /** Programmatic navigation, e.g. after applying a personality. */
    navigate(route) {
      if (window.location.hash === route) return;
      window.location.hash = route;
    },
  };
}
