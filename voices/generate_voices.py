#!/usr/bin/env python3
"""Batch-generate the 11 DM character voices with Qwen3-TTS-12Hz-1.7B-VoiceDesign.

This is a ONE-TIME, GPU-only batch. It reads the voice roster (voice_id +
natural-language description + in-character sample line) straight out of the
source-of-truth markdown ``voices/character-voices.md``, then for each voice runs
the documented VoiceDesign -> voice-clone workflow and writes reusable, committed
assets under ``voices/assets/``:

    voices/assets/
      voices.json                 # manifest: voice_id -> {description, sample_line, ...}
      ref_clips/<voice_id>.wav     # the designed reference clip
      clone_prompts/<voice_id>.pt  # the reusable voice-clone prompt (torch.save)

The cascade TTS provider (reachy-dm-cascade .../cascade/tts/qwen3_tts.py) consumes
``voices.json`` -> its ``VOICE_PROMPTS`` map / registered speakers. See
``voices/README.md`` for the run command and the consumer handoff.

================================================================================
QWEN3-TTS VOICEDESIGN API  (verified against the model card + QwenLM/Qwen3-TTS,
but the EXACT signatures still need a smoke-test on a GPU host -- see FLAGS)
================================================================================
    from qwen_tts import Qwen3TTSModel        # FLAG[import]: confirm pkg/class name

    # VoiceDesign checkpoint -- design a voice from a NL instruction + text:
    design = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", device_map="cuda:0",
        dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    )
    wavs, sr = design.generate_voice_design(            # FLAG[design]: confirm kwargs
        text=<sample line>, language="English", instruct=<description>,
    )   # wavs[0] is a float32/float waveform numpy array, sr the sample rate

    # Base checkpoint -- build + reuse a clone prompt from that reference clip:
    base = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0", dtype=torch.bfloat16,
    )
    prompt = base.create_voice_clone_prompt(           # FLAG[clone]: confirm kwargs
        ref_audio=(wavs[0], sr), ref_text=<sample line>,
    )   # `prompt` is an opaque structured object (speaker features); we torch.save it.
    wavs2, sr2 = base.generate_voice_clone(            # (runtime path, not run here)
        text="...", language="English", voice_clone_prompt=prompt,
    )

FLAGS to verify before a real run (do NOT trust blindly):
  * FLAG[import]   package import name (`qwen_tts`) and class (`Qwen3TTSModel`).
  * FLAG[design]   generate_voice_design kwargs / return shape ((wavs, sr) tuple,
                   wavs a list of waveforms). Does it need `language=`? Model card
                   shows it; we pass it. Some builds may use `instruction=`.
  * FLAG[clone]    create_voice_clone_prompt arg names: `ref_audio` as a
                   (numpy, sr) tuple + `ref_text`. Docs also allow a path/URL/b64.
  * FLAG[serialize] HOW the clone prompt serializes. It is an opaque object (likely
                   tensors). We use torch.save / torch.load as the safe default. If
                   it is a plain dict of numpy arrays, np.savez would also work. The
                   CONSUMER must load it the SAME way -- see voices/README.md.
  * FLAG[runtime]  whether the cascade server wants the clone prompt inline
                   (VOICE_PROMPTS map) or registered as a named speaker via the
                   CustomVoice checkpoint's precompute path. See README "Handoff".

NO GPU / NO heavy deps on the dev box: this module is import-light at top level and
only touches torch / qwen_tts / soundfile INSIDE the GPU entrypoints, so it can be
ast-parsed and its markdown parser unit-poked on a laptop without installing them.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# Paths (relative to this file so it runs from any cwd / on a GPU host).
# --------------------------------------------------------------------------- #
VOICES_DIR = Path(__file__).resolve().parent
SPEC_MD = VOICES_DIR / "character-voices.md"
ASSETS_DIR = VOICES_DIR / "assets"
REF_CLIPS_DIR = ASSETS_DIR / "ref_clips"
CLONE_PROMPTS_DIR = ASSETS_DIR / "clone_prompts"
MANIFEST_PATH = ASSETS_DIR / "voices.json"

LANGUAGE = "English"
DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

# Authoritative ordering / membership check (mirrors ROSTER_VOICE_IDS in the
# cascade's qwen3_tts.py). The parser must yield exactly these 11 ids.
EXPECTED_VOICE_IDS = (
    "gm_narrator",
    "augusta_byron",
    "tommy_doyle",
    "bailey_bigsmile",
    "old_tallman",
    "hazel_johnson",
    "marvin",
    "npc_raider",
    "npc_settler",
    "npc_merchant",
    "npc_overseer",
)


# --------------------------------------------------------------------------- #
# 1. Source of truth: parse voice_id / description / sample line from the spec.
#    The narrator is a bullet block; pregens + NPCs are markdown tables. This
#    keeps a SINGLE editable source (the .md) instead of a duplicated table.
# --------------------------------------------------------------------------- #
def _clean(cell: str) -> str:
    """Strip surrounding whitespace and a single pair of wrapping double-quotes."""
    s = cell.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].strip()
    return s


def parse_spec(md_path: Path = SPEC_MD) -> "list[dict]":
    """Return [{voice_id, description, sample_line}, ...] parsed from the markdown.

    Handles two shapes:
      * Narrator bullet block: ``- **voice_id:** `gm_narrator` `` then
        ``- **description:** ...`` then ``- **sample line:** "..."`` (multi-line).
      * Tables whose header row contains ``voice_id`` and ``sample line``; the
        ``voice_id`` cell is wrapped in backticks, the sample line in quotes.
    """
    text = md_path.read_text(encoding="utf-8")
    voices: "dict[str, dict]" = {}

    # --- Narrator bullet block -------------------------------------------------
    vid_m = re.search(r"\*\*voice_id:\*\*\s*`([^`]+)`", text)
    desc_m = re.search(r"\*\*description:\*\*\s*(.+?)(?=\n- \*\*|\n\n)", text, re.S)
    sample_m = re.search(r"\*\*sample line:\*\*\s*(.+?)(?=\n\n|\n##)", text, re.S)
    if vid_m and desc_m and sample_m:
        voices[vid_m.group(1).strip()] = {
            "voice_id": vid_m.group(1).strip(),
            "description": _clean(re.sub(r"\s+", " ", desc_m.group(1))),
            "sample_line": _clean(re.sub(r"\s+", " ", sample_m.group(1))),
        }

    # --- Tables ----------------------------------------------------------------
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        # Need a backticked voice_id in col 0, plus a description + sample line.
        # Pregen rows: | voice_id | character | description | sample line |
        # NPC rows:    | voice_id | use       | description | sample line |
        if len(cells) < 4:
            continue
        m = re.match(r"`([^`]+)`", cells[0])
        if not m:
            continue  # header / separator / non-voice row
        vid = m.group(1).strip()
        description = _clean(cells[2])
        sample_line = _clean(cells[3])
        if vid and description and sample_line:
            voices[vid] = {
                "voice_id": vid,
                "description": description,
                "sample_line": sample_line,
            }

    # Order by the canonical roster; surface any drift loudly.
    parsed_ids = set(voices)
    expected = set(EXPECTED_VOICE_IDS)
    missing = expected - parsed_ids
    extra = parsed_ids - expected
    if missing or extra:
        raise ValueError(
            f"Spec parse mismatch vs ROSTER. missing={sorted(missing)} "
            f"extra={sorted(extra)}. Check {md_path}."
        )
    return [voices[v] for v in EXPECTED_VOICE_IDS]


# --------------------------------------------------------------------------- #
# 2. GPU workflow. Imports heavy deps lazily so the file stays laptop-safe.
# --------------------------------------------------------------------------- #
def _load_models():
    """Load the VoiceDesign + Base checkpoints. GPU only.  FLAG[import]."""
    import torch  # noqa: F401  (lazy)
    from qwen_tts import Qwen3TTSModel  # FLAG[import]

    design = Qwen3TTSModel.from_pretrained(
        DESIGN_MODEL,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    base = Qwen3TTSModel.from_pretrained(
        BASE_MODEL,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    return design, base


def generate_one(design, base, voice: dict) -> dict:
    """Design -> reference clip -> reusable clone prompt for a single voice.

    Returns the manifest entry (without the shared description/sample fields,
    which the caller merges in). GPU only.
    """
    import soundfile as sf
    import torch

    vid = voice["voice_id"]
    text = voice["sample_line"]
    instruct = voice["description"]

    # (a) Design the voice from its NL description.            FLAG[design]
    wavs, sr = design.generate_voice_design(
        text=text,
        language=LANGUAGE,
        instruct=instruct,
    )
    ref_wav = wavs[0]

    ref_clip_path = REF_CLIPS_DIR / f"{vid}.wav"
    sf.write(str(ref_clip_path), ref_wav, sr)

    # (b) Build a reusable clone prompt from that clip.        FLAG[clone]
    clone_prompt = base.create_voice_clone_prompt(
        ref_audio=(ref_wav, sr),
        ref_text=text,
    )

    # (c) Persist the clone prompt as a committed asset.       FLAG[serialize]
    #     torch.save handles tensors/objects; the CONSUMER must torch.load it.
    clone_prompt_path = CLONE_PROMPTS_DIR / f"{vid}.pt"
    torch.save(clone_prompt, str(clone_prompt_path))

    return {
        "ref_clip": str(ref_clip_path.relative_to(VOICES_DIR)).replace("\\", "/"),
        "clone_prompt_path": str(clone_prompt_path.relative_to(VOICES_DIR)).replace("\\", "/"),
        "sample_rate": int(sr),
    }


def run_batch(only: Optional["list[str]"] = None) -> dict:
    """Generate all (or a subset of) voices and write voices.json. GPU only."""
    roster = parse_spec()
    if only:
        roster = [v for v in roster if v["voice_id"] in set(only)]
        if not roster:
            raise SystemExit(f"No roster voices matched --only {only}")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    REF_CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    CLONE_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    design, base = _load_models()

    # Merge with an existing manifest so --only does incremental top-ups.
    manifest: dict = {"model": DESIGN_MODEL, "language": LANGUAGE, "voices": {}}
    if MANIFEST_PATH.exists():
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    manifest.setdefault("voices", {})

    for voice in roster:
        vid = voice["voice_id"]
        print(f"[voice] designing {vid} ...")
        result = generate_one(design, base, voice)
        manifest["voices"][vid] = {
            "description": voice["description"],
            "sample_line": voice["sample_line"],
            **result,
            # `clone_prompt` is stored out-of-line at clone_prompt_path (a tensor
            # blob is not JSON-serializable). Consumer loads it via torch.load.
            "clone_prompt": None,
        }
        print(f"[voice]   wrote {result['ref_clip']} + {result['clone_prompt_path']}")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[voice] manifest -> {MANIFEST_PATH}")
    return manifest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional["list[str]"] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse the spec and print the roster WITHOUT loading models or GPU. "
        "Safe on a laptop; use to validate character-voices.md edits.",
    )
    ap.add_argument(
        "--only",
        nargs="+",
        metavar="VOICE_ID",
        help="Generate only these voice_ids (incremental top-up of voices.json).",
    )
    args = ap.parse_args(argv)

    if args.dry_run:
        roster = parse_spec()
        print(f"Parsed {len(roster)} voices from {SPEC_MD.name}:")
        for v in roster:
            print(f"  - {v['voice_id']}")
            print(f"      desc:   {v['description']}")
            print(f"      sample: {v['sample_line']}")
        return 0

    run_batch(only=args.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
