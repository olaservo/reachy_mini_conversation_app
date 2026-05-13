/** Bootstrap: wire the shell header to the hash router and mount views on demand. */

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
      [ROUTES.HOME]: (ctx) => mountHomeView({ ...ctx, navigate: router.navigate }),
      [ROUTES.TALK]: (ctx) => mountTalkView(ctx),
      [ROUTES.SETTINGS]: (ctx) => mountSettingsView(ctx),
    },
    { fallback: ROUTES.HOME, outlet }
  );

  const gear = $('[data-action="open-settings"]');
  if (gear) {
    gear.addEventListener("click", () => {
      const onSettings = window.location.hash === ROUTES.SETTINGS;
      router.navigate(onSettings ? ROUTES.HOME : ROUTES.SETTINGS);
    });
  }

  const brand = $('[data-action="go-home"]');
  if (brand) {
    brand.addEventListener("click", (event) => {
      event.preventDefault();
      router.navigate(ROUTES.HOME);
    });
  }

  const back = $('[data-action="go-back"]');
  if (back) {
    back.addEventListener("click", () => router.navigate(ROUTES.HOME));
  }

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
