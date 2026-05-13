/**
 * Tiny vanilla DOM helpers to keep view code declarative.
 *
 * No framework, no virtual DOM. ``h`` builds an element from a tag name plus
 * an optional attribute object plus children. ``$`` and ``$$`` are short
 * aliases for query selectors scoped to a root.
 *
 * Children can be:
 *   - ``string`` / ``number`` / ``boolean``  -> text node
 *   - ``Node``                                -> appended as is
 *   - ``Array``                                -> flattened
 *   - ``null`` / ``undefined`` / ``false``    -> ignored (handy for conditional UI)
 *
 * Special attributes:
 *   - ``class``        : CSS class (string or array of strings)
 *   - ``style``        : inline style as object ``{ color: "red" }``
 *   - ``dataset``      : ``{ foo: "bar" }`` -> ``data-foo="bar"``
 *   - ``on<Event>``    : ``onClick: fn`` -> ``element.addEventListener("click", fn)``
 *   - everything else  : set as attribute via ``setAttribute``
 */
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value == null || value === false) continue;
    if (key === "class") {
      el.className = Array.isArray(value) ? value.filter(Boolean).join(" ") : String(value);
    } else if (key === "style" && typeof value === "object") {
      Object.assign(el.style, value);
    } else if (key === "dataset" && typeof value === "object") {
      for (const [dk, dv] of Object.entries(value)) {
        if (dv != null) el.dataset[dk] = String(dv);
      }
    } else if (key.startsWith("on") && typeof value === "function") {
      el.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === "html") {
      // Escape hatch for trusted HTML (e.g. inline SVG icons we author).
      el.innerHTML = String(value);
    } else {
      el.setAttribute(key, String(value));
    }
  }
  appendChildren(el, children);
  return el;
}

/** Append a (possibly nested) list of children, ignoring falsy values. */
function appendChildren(parent, children) {
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false || child === true) continue;
    if (child instanceof Node) {
      parent.appendChild(child);
    } else {
      parent.appendChild(document.createTextNode(String(child)));
    }
  }
}

/** ``document.querySelector`` shortcut, scoped to ``root`` (default ``document``). */
export function $(selector, root = document) {
  return root.querySelector(selector);
}

/** ``document.querySelectorAll`` shortcut returning a real ``Array``. */
export function $$(selector, root = document) {
  return Array.from(root.querySelectorAll(selector));
}

/** Replace all children of ``parent`` with ``newChildren``. */
export function clear(parent) {
  while (parent.firstChild) parent.removeChild(parent.firstChild);
}

/**
 * Format a profile name for display.
 *
 * Backend names are snake_case (e.g. ``mad_scientist_assistant``) or prefixed
 * with ``user_personalities/`` for user-created profiles. We strip the prefix
 * and convert to Title Case for human-friendly card labels.
 */
export function prettifyProfileName(name) {
  const stripped = name.replace(/^user_personalities\//, "");
  return stripped
    .split(/[_-]/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}
