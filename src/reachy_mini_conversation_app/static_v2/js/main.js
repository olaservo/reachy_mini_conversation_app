/**
 * Modern web UI bootstrap.
 *
 * Wires the persistent shell (logo + settings gear) to the hash router and
 * mounts views on demand. Keeping this file intentionally tiny: every view
 * lives in its own module under ``views/`` and owns its own teardown via the
 * router's per-view AbortSignal.
 */

import { ROUTES } from "./constants.js";
import { createRouter } from "./router.js";
import { $, h } from "./ui.js";
import { mountHomeView } from "./views/home.js";
import { mountTalkView } from "./views/talk.js";
import { mountSettingsView } from "./views/settings.js";

function boot() {
  const outlet = $("#view-outlet");
  if (!outlet) {
    console.error("static_v2: #view-outlet missing from index.html");
    return;
  }

  const router = createRouter(
    {
      // Home needs ``navigate`` so a personality click can transition to
      // the Talk view straight after applying. Talk and Settings only
      // navigate via the shell header (back button + gear), so they do
      // not need a router reference of their own.
      [ROUTES.HOME]: (ctx) => mountHomeView({ ...ctx, navigate: router.navigate }),
      [ROUTES.TALK]: (ctx) => mountTalkView(ctx),
      [ROUTES.SETTINGS]: (ctx) => mountSettingsView(ctx),
    },
    { fallback: ROUTES.HOME, outlet }
  );

  // The settings gear lives in the shell ``<header>`` (see index.html) and
  // is wired up here so navigation goes through the same router as the rest
  // of the app rather than via a raw ``<a href="#/settings">``. Clicking the
  // gear toggles between Settings and Home so users can use the same control
  // to enter and leave the settings panel.
  const gear = $('[data-action="open-settings"]');
  if (gear) {
    gear.addEventListener("click", () => {
      const onSettings = window.location.hash === ROUTES.SETTINGS;
      router.navigate(onSettings ? ROUTES.HOME : ROUTES.SETTINGS);
    });
  }

  // Brand link returns to home, regardless of which view is currently active.
  const brand = $('[data-action="go-home"]');
  if (brand) {
    brand.addEventListener("click", (event) => {
      event.preventDefault();
      router.navigate(ROUTES.HOME);
    });
  }

  // The Back button lives in the header (next to the brand) so it has a
  // consistent place across views instead of floating inside the page
  // content. It is only meaningful when the user is "deeper" than the
  // home grid - currently just on the Talk view - so we toggle ``hidden``
  // accordingly. ``data-action`` keeps the wiring in this single file.
  const back = $('[data-action="go-back"]');
  if (back) {
    back.addEventListener("click", () => router.navigate(ROUTES.HOME));
  }

  // Reflect the active route on the header controls. The gear becomes
  // "active" on Settings (rotated icon + tinted bg) and the back button
  // shows up on Talk.
  function syncHeaderForRoute() {
    const route = window.location.hash;
    if (gear) {
      const onSettings = route === ROUTES.SETTINGS;
      gear.classList.toggle("is-active", onSettings);
      gear.setAttribute("aria-label", onSettings ? "Close settings" : "Open settings");
    }
    if (back) {
      back.hidden = route !== ROUTES.TALK;
    }
  }
  window.addEventListener("hashchange", syncHeaderForRoute);
  syncHeaderForRoute();

  router.start();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot, { once: true });
} else {
  boot();
}
