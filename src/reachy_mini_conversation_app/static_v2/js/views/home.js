/**
 * Home view: grid of personality cards + "Custom" card.
 *
 * Flow:
 *   1. Fetch ``/personalities`` to build the card grid (one per profile,
 *      avatar resolved via ``avatarFor``).
 *   2. Click on a card -> apply the personality (POST /personalities/apply)
 *      then navigate to ``#/talk`` so the user can start speaking.
 *   3. Click on the "Custom" card -> open a modal asking for a name and
 *      instructions, save it as a new profile (with the default tool set
 *      so the robot keeps its full expressivity), then apply + navigate.
 *
 * The view owns no global state. All async work is cancelled via the
 * AbortSignal provided by the router so a fast tab-switch never leaks
 * pending fetches.
 */

import {
  applyPersonality,
  listPersonalities,
  savePersonality,
} from "../api.js";
import {
  AVATAR_BY_PROFILE,
  BUILT_IN_DEFAULT_OPTION,
  DEFAULT_TOOLS,
  ROUTES,
  avatarFor,
} from "../constants.js";
import { $, clear, h, prettifyProfileName } from "../ui.js";
import { openCustomProfileModal } from "../components/profile-modal.js";

/**
 * @param {{ outlet: HTMLElement, signal: AbortSignal, navigate: (route: string) => void }} ctx
 */
export async function mountHomeView({ outlet, signal, navigate }) {
  const view = h(
    "section",
    { class: "view view--home" },
    h(
      "header",
      { class: "view-header" },
      h("h1", { class: "view-title" }, "Choose a personality"),
      h(
        "p",
        { class: "view-subtitle" },
        "Pick how Reachy Mini should think and talk. Tap a card to start a conversation."
      )
    ),
    h("div", { class: "personality-grid", role: "list" }, h("p", { class: "muted" }, "Loading…"))
  );
  outlet.replaceChildren(view);

  const grid = $(".personality-grid", view);
  const status = h("p", { class: "view-status", role: "status", "aria-live": "polite" });
  view.appendChild(status);

  let personalities;
  try {
    personalities = await listPersonalities();
  } catch (error) {
    if (signal.aborted) return;
    clear(grid);
    grid.appendChild(renderError("Could not list personalities", error));
    return;
  }
  if (signal.aborted) return;

  const choices = (personalities?.choices || []).filter((name) => name !== BUILT_IN_DEFAULT_OPTION);
  const current = personalities?.current;
  const lockedTo = personalities?.locked ? personalities.locked_to : null;

  clear(grid);
  for (const name of choices) {
    grid.appendChild(
      buildPersonalityCard({
        name,
        isActive: name === current,
        disabled: Boolean(lockedTo) && name !== lockedTo,
        onSelect: () => handleSelection(name),
      })
    );
  }
  // The Custom card is always last, never disabled by a profile lock (the
  // lock concerns the active profile, not the ability to author new ones on
  // disk; the user just won't be able to apply the new profile until the
  // lock is lifted).
  grid.appendChild(buildCustomCard({ onClick: handleCustomClick }));

  if (lockedTo) {
    status.textContent = `Profile locked to "${lockedTo}" by REACHY_MINI_LOCKED_PROFILE; switching is disabled.`;
    status.classList.add("is-warning");
  }

  async function handleSelection(name) {
    status.classList.remove("is-warning", "is-error");
    status.textContent = `Applying "${prettifyProfileName(name)}"…`;
    try {
      await applyPersonality(name, { persist: false });
      if (signal.aborted) return;
      navigate(ROUTES.TALK);
    } catch (error) {
      if (signal.aborted) return;
      status.textContent = `Failed to apply: ${error?.message || error}`;
      status.classList.add("is-error");
    }
  }

  async function handleCustomClick() {
    const created = await openCustomProfileModal({ signal });
    if (!created || signal.aborted) return;
    status.classList.remove("is-warning", "is-error");
    status.textContent = `Saving "${created.name}"…`;
    try {
      const saveResult = await savePersonality({
        name: created.name,
        instructions: created.instructions,
        tools_text: DEFAULT_TOOLS.join("\n"),
        // Voice is left unset on purpose: the backend falls back to the
        // current backend's default voice, and the user can fine-tune it
        // from the Settings view if they want.
        voice: "",
      });
      if (signal.aborted) return;
      const newName = saveResult?.value || created.name;
      await applyPersonality(newName, { persist: false });
      if (signal.aborted) return;
      navigate(ROUTES.TALK);
    } catch (error) {
      if (signal.aborted) return;
      status.textContent = `Failed to create profile: ${error?.message || error}`;
      status.classList.add("is-error");
    }
  }
}

function buildPersonalityCard({ name, isActive, disabled, onSelect }) {
  const hasAvatar = Object.prototype.hasOwnProperty.call(AVATAR_BY_PROFILE, stripUserPrefix(name));
  return h(
    "button",
    {
      type: "button",
      class: ["personality-card", isActive && "is-active", disabled && "is-disabled"],
      role: "listitem",
      disabled: disabled ? "disabled" : null,
      "aria-pressed": isActive ? "true" : "false",
      "aria-label": `Use personality ${prettifyProfileName(name)}`,
      onClick: disabled ? undefined : onSelect,
    },
    h(
      "span",
      { class: "personality-card__avatar" },
      h("img", {
        src: avatarFor(stripUserPrefix(name)),
        alt: "",
        loading: "lazy",
        // Decorative for screen readers (the card label below already
        // names the personality).
        "aria-hidden": "true",
        class: !hasAvatar ? "personality-card__avatar--fallback" : null,
      })
    ),
    h("span", { class: "personality-card__name" }, prettifyProfileName(name)),
    isActive && checkBadge()
  );
}

/** Discreet "active" badge, top-right of the card. */
function checkBadge() {
  const badge = h("span", { class: "personality-card__badge", "aria-hidden": "true" });
  badge.innerHTML = `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
      <polyline points="20 6 9 17 4 12"/>
    </svg>`;
  return badge;
}

function buildCustomCard({ onClick }) {
  return h(
    "button",
    {
      type: "button",
      class: "personality-card personality-card--custom",
      role: "listitem",
      "aria-label": "Create a custom personality",
      onClick,
    },
    h("span", { class: "personality-card__plus", "aria-hidden": "true" }, "+"),
    h("span", { class: "personality-card__name" }, "Custom"),
    h("span", { class: "personality-card__hint" }, "Write your own prompt")
  );
}

function stripUserPrefix(name) {
  return name.replace(/^user_personalities\//, "");
}

function renderError(label, error) {
  return h(
    "div",
    { class: "view-error" },
    h("p", null, label),
    h("p", { class: "muted small" }, error?.message || String(error))
  );
}
