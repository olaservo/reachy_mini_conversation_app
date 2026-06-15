#!/usr/bin/env python3
"""Audition the DM character voices against the running Qwen3-TTS endpoint.

For each roster voice it POSTs a line to ``/v1/audio/speech`` (response_format=wav)
and saves ``voices/auditions/<voice_id>.wav`` so you can listen. By default it uses
each voice's own ``sample_line`` from ``assets/voices.json``; pass ``--text`` to make
them all say the same line instead.

Stdlib only (urllib) — no pip installs, runs on the laptop. The first request
cold-starts the Modal L4 (~40s); the rest are fast while the container stays warm.

Examples:
    python voices/audition_voices.py                      # all 11, each its sample line
    python voices/audition_voices.py gm_narrator marvin   # just these two
    python voices/audition_voices.py --text "Hello there, traveler."   # same line, all voices
    python voices/audition_voices.py --url http://localhost:8091  # a local tts_server
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

VOICES_DIR = Path(__file__).resolve().parent
MANIFEST = VOICES_DIR / "assets" / "voices.json"
OUT_DIR = VOICES_DIR / "auditions"
DEFAULT_URL = "https://olahungerford--qwen3-tts-voices-serve.modal.run"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("voice_ids", nargs="*", help="Voice ids to audition (default: all in voices.json).")
    ap.add_argument("--text", help="Say this exact line for every voice (default: each voice's sample_line).")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"TTS server base URL (default: {DEFAULT_URL}).")
    ap.add_argument("--timeout", type=int, default=900, help="Per-request timeout seconds (default 900, for cold start).")
    args = ap.parse_args(argv)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    voices = manifest["voices"]
    ids = args.voice_ids or list(voices)
    unknown = [v for v in ids if v not in voices]
    if unknown:
        print(f"Unknown voice ids: {unknown}\nAvailable: {list(voices)}", file=sys.stderr)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    endpoint = args.url.rstrip("/") + "/v1/audio/speech"
    print(f"Auditioning {len(ids)} voice(s) -> {endpoint}\n(first call cold-starts the GPU; please wait)\n")

    failures = 0
    for vid in ids:
        text = args.text or voices[vid]["sample_line"]
        body = json.dumps({"voice": vid, "input": text, "response_format": "wav"}).encode("utf-8")
        req = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                data = resp.read()
            dest = OUT_DIR / f"{vid}.wav"
            dest.write_bytes(data)
            print(f"  [ok]   {vid:<16} {len(data):>8} bytes -> {dest.relative_to(VOICES_DIR)}   “{text[:48]}...”")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  [FAIL] {vid:<16} {exc}", file=sys.stderr)

    print(f"\nDone. {len(ids) - failures}/{len(ids)} saved under {OUT_DIR}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
