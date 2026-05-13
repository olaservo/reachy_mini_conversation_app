#!/usr/bin/env python3
"""Standalone preview server for ``static_v2/``.

Why this script exists
----------------------
The full conversation app needs the realtime backend (OpenAI/Gemini/HF), the
ReachyMini SDK, gradio, fastrtc, ... to even boot. When iterating on the
modern UI (visuals, navigation, orb states, modal flow), spinning up that
whole stack is overkill.

This script:
  - serves the ``static_v2/`` bundle (HTML + JS modules + CSS + SVG avatars)
    on http://localhost:7860 with the same URL layout as the real app
    (``/``, ``/static/...``);
  - implements a tiny in-memory mock of the HTTP endpoints the UI calls
    (``/personalities``, ``/personalities/load``, ``/personalities/save``,
    ``/personalities/apply``, ``/status``, ``/voices``, ``/voices/current``,
    ``/voices/apply``, ``/backend_config``);
  - emits a fake conversation activity stream on ``/conversation_events``
    that loops through the ``listening -> thinking -> speaking -> idle``
    cycle so the orb can be visually validated end-to-end.

It uses only the Python standard library (``http.server``) so it works in
any minimal env, including a freshly cloned checkout with no venv.

Run:
    python3 dev_serve_v2.py             # serves on http://localhost:7860
    python3 dev_serve_v2.py --port 8080 # custom port

The real backend behaviour (and SSE driven by actual realtime events) only
lives inside ``console.LocalStream``; this script is purely a UI dev aid and
is intentionally not bundled with the package.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


STATIC_ROOT = Path(__file__).parent / "src" / "reachy_mini_conversation_app" / "static_v2"

# ----------------------------------------------------------------------------
# Mock state
#
# We keep a single shared snapshot mirroring the shape returned by the real
# endpoints. Mutating endpoints (apply / save / backend_config) update this
# snapshot in place so subsequent GETs reflect the change, just like the
# real backend persists into the .env / profile files.
# ----------------------------------------------------------------------------

_LOCK = threading.Lock()

_MOCK_PROFILES = [
    "(built-in default)",
    "bored_teenager",
    "captain_circuit",
    "chess_coach",
    "cosmic_kitchen",
    "default",
    "hype_bot",
    "mad_scientist_assistant",
    "mars_rover",
    "nature_documentarian",
    "noir_detective",
    "sorry_bro",
    "time_traveler",
    "victorian_butler",
]

_state = {
    "current_profile": "(built-in default)",
    "startup_profile": "(built-in default)",
    "backend_provider": "openai",
    "active_backend": "openai",
    "voice": "alloy",
    "user_profiles": [],  # names created via the modal
}

_MOCK_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

# Synthetic activity loop emitted on ``/conversation_events`` so the orb's
# state machine can be exercised end-to-end without a real conversation.
_FAKE_CYCLE = [
    ("user_speech_started", 1.5),
    ("user_speech_stopped", 0.6),
    ("response_created", 0.8),
    ("assistant_audio_delta", 1.2),
    ("assistant_audio_delta", 0.4),
    ("assistant_transcript_done", 2.5),
]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _all_profiles() -> list[str]:
    with _LOCK:
        return list(_MOCK_PROFILES) + [
            f"user_personalities/{n}" for n in _state["user_profiles"]
        ]


def _status_payload() -> dict:
    with _LOCK:
        return {
            "active_backend": _state["active_backend"],
            "backend_provider": _state["backend_provider"],
            "has_key": True,
            "has_openai_key": True,
            "has_gemini_key": False,
            "has_hf_session_url": True,
            "has_hf_ws_url": False,
            "has_hf_connection": True,
            "hf_connection_mode": "deployed",
            "hf_direct_host": None,
            "hf_direct_port": None,
            "can_proceed": True,
            "can_proceed_with_openai": True,
            "can_proceed_with_gemini": False,
            "can_proceed_with_hf": True,
            "requires_restart": False,
        }


# ----------------------------------------------------------------------------
# HTTP handler
# ----------------------------------------------------------------------------


class DevHandler(BaseHTTPRequestHandler):
    # Quieter logs: only show non-200 lines, otherwise the activity stream
    # spams the terminal.
    def log_message(self, fmt: str, *args) -> None:  # noqa: D401, ANN001
        msg = fmt % args
        if " 200 " in msg or " 304 " in msg:
            return
        sys.stderr.write(f"[dev_serve_v2] {self.address_string()} - {msg}\n")

    # ---- Routing ---------------------------------------------------------

    def do_HEAD(self) -> None:  # noqa: N802
        """Reuse the GET pipeline but suppress the response body.

        Browsers and CLIs (curl -I, link checkers, devtools preflights)
        occasionally probe assets with HEAD before downloading them. Without
        this method ``BaseHTTPRequestHandler`` returns 501, which both
        pollutes the log and fails some clients. The ``_head_only`` flag is
        consumed by the body-writing helpers below.
        """
        self._head_only = True
        try:
            self.do_GET()
        finally:
            self._head_only = False

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        query = parse_qs(urlsplit(self.path).query)

        if path == "/":
            return self._serve_file(STATIC_ROOT / "index.html", "text/html; charset=utf-8")
        if path.startswith("/static/"):
            return self._serve_static(path[len("/static/") :])
        if path == "/favicon.ico":
            return self._send(204)

        if path == "/status":
            return self._send_json(200, _status_payload())
        if path == "/ready":
            return self._send_json(200, {"ready": True})
        if path == "/personalities":
            with _LOCK:
                current = _state["current_profile"]
                startup = _state["startup_profile"]
            return self._send_json(
                200,
                {
                    "choices": _all_profiles(),
                    "current": current,
                    "startup": startup,
                    "locked": False,
                    "locked_to": None,
                },
            )
        if path == "/personalities/load":
            name = (query.get("name") or [""])[0]
            return self._send_json(
                200,
                {
                    "instructions": f"# Mock instructions for {name}",
                    "tools_text": "",
                    "voice": _state["voice"],
                    "uses_default_voice": True,
                    "available_tools": [],
                    "enabled_tools": [],
                },
            )
        if path == "/voices":
            return self._send_json(200, _MOCK_VOICES)
        if path == "/voices/current":
            with _LOCK:
                return self._send_json(200, {"voice": _state["voice"]})
        if path == "/conversation_events":
            return self._serve_sse()

        self._send(404, b"not found")

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        body = self._read_json_body()

        if path == "/personalities/apply":
            name = str(body.get("name", "")) if body else ""
            with _LOCK:
                if name:
                    _state["current_profile"] = name
                    if body.get("persist"):
                        _state["startup_profile"] = name
            return self._send_json(200, {"ok": True, "value": name})
        if path == "/personalities/save":
            name = str(body.get("name", "")) if body else ""
            if not name:
                return self._send_json(400, {"ok": False, "error": "invalid_name"})
            with _LOCK:
                if name not in _state["user_profiles"]:
                    _state["user_profiles"].append(name)
            full = f"user_personalities/{name}"
            return self._send_json(
                200,
                {"ok": True, "value": full, "choices": _all_profiles()},
            )
        if path == "/backend_config":
            backend = str(body.get("backend", "")) if body else ""
            if backend:
                with _LOCK:
                    _state["backend_provider"] = backend
                    _state["active_backend"] = backend
            return self._send_json(
                200,
                {
                    "ok": True,
                    "message": "Saved (mock).",
                    **_status_payload(),
                },
            )
        if path == "/voices/apply":
            voice = str(body.get("voice", "")) if body else ""
            if voice:
                with _LOCK:
                    _state["voice"] = voice
            return self._send_json(200, {"ok": True, "status": f"Voice changed to {voice}."})

        self._send(404, b"not found")

    # ---- Helpers ---------------------------------------------------------

    def _serve_static(self, rel_path: str) -> None:
        target = (STATIC_ROOT / rel_path).resolve()
        if STATIC_ROOT.resolve() not in target.parents and target != STATIC_ROOT.resolve():
            return self._send(403, b"forbidden")
        if not target.is_file():
            return self._send(404, b"not found")
        return self._serve_file(target, _guess_mime(target))

    def _serve_file(self, path: Path, mime: str) -> None:
        try:
            data = path.read_bytes()
        except OSError:
            return self._send(404, b"not found")
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self._write_body(data)

    def _serve_sse(self) -> None:
        """Emit a synthetic activity stream so the orb can be exercised live."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        # HEAD on a stream endpoint just confirms the headers are servable.
        if getattr(self, "_head_only", False):
            return
        try:
            self.wfile.write(b"retry: 2000\n\n")
            self.wfile.write(b"event: ready\ndata: connected\n\n")
            self.wfile.flush()
            while True:
                for reason, sleep_s in _FAKE_CYCLE:
                    self.wfile.write(f"event: activity\ndata: {reason}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(sleep_s)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _send(self, status: int, body: bytes = b"", mime: str = "text/plain") -> None:
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self._write_body(body)

    def _send_json(self, status: int, payload: dict | list) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self._write_body(body)

    def _write_body(self, data: bytes) -> None:
        """Write the response body unless we are servicing a HEAD request.

        Centralising this lets every helper above (``_serve_file``,
        ``_send``, ``_send_json``) build full responses identically for GET
        and HEAD; only the final body emission differs.
        """
        if getattr(self, "_head_only", False):
            return
        self.wfile.write(data)

    def _read_json_body(self) -> dict | None:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return None
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode())
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None


_MIMES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".json": "application/json",
}


def _guess_mime(path: Path) -> str:
    return _MIMES.get(path.suffix.lower(), "application/octet-stream")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Dev preview server for static_v2/")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    if not (STATIC_ROOT / "index.html").exists():
        sys.exit(f"static_v2/index.html not found at {STATIC_ROOT}")

    server = ThreadingHTTPServer((args.host, args.port), DevHandler)
    print(f"Serving {STATIC_ROOT} on http://{args.host}:{args.port}")
    print("Mock backend active. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down…")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
