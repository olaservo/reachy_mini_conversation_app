# WS3 — DM Mode (profile + tools)

The `dm` profile turns the conversation app into a tabletop Game Master: local-skill
knowledge in the system prompt + deployed MCP tools for dice, sheets, choices, lighting,
and robot expression.

## Files
- `profiles/dm/instructions.txt` — GM persona (Fallout 2d20), the **six pregen roster**
  (by exact slug), action-resolution rules, lighting/performance guidance, guardrails.
- `profiles/dm/tools.txt` — robot tools + `ttrpg__*` (engine) + `hass__*` (lighting).

## Setup
1. **TTRPG engine** (the fallout-helper MCP server, already running on :3001):
   ```
   reachy-mini-conversation-app mcp-servers add ttrpg http://127.0.0.1:3001 --profile dm
   ```
   Exposes: `ttrpg__roll_dice`, `ttrpg__present_player_choice`, `ttrpg__show_character_sheet`.
2. **Story lighting** (optional, for reactive scenes):
   ```
   reachy-mini-conversation-app mcp-servers add hass http://homeassistant.local:8123/api/mcp \
     --token-env HA_ACCESS_TOKEN --profile dm
   ```
3. Select the `dm` profile (UI or `LOCKED_PROFILE=dm`).

## Pregen characters (served by the ttrpg server, read from its skill files)
`augusta-byron` (Vault Dweller scientist, L1) · `tommy-doyle` (Survivor gambler, L2) ·
`bailey-bigsmile` (Ghoul wanderer, L3) · `old-tallman` (Super Mutant philosopher, L2) ·
`hazel-johnson` (Brotherhood Field Scribe, L1) · `marvin` (Mister Handy, L2).
The DM must show/consult the real sheet via `ttrpg__show_character_sheet` — never invent stats.

## "Local skills" approach
- **Core rules** (2d20 resolution loop) live in `instructions.txt` — enough to run play.
- **Detailed content** (full sheets, the "Machine Frequency" adventure: acts, NPCs, stat
  blocks) lives in the ttrpg server's skill files / `skill://` resources and the
  `agent-skills-ttrpg-demo` repo. The DM pulls specifics via tools (e.g.
  `show_character_sheet`) rather than preloading rulebooks into a realtime prompt.
- TODO: if the model needs more adventure grounding, add a lightweight `lookup_rule` /
  `get_scene` tool (or expose the skill resources) instead of bloating instructions.

## Coordination with WS1 (see `everything/home_tech/reachy/build-small/WS1-qwen-omni-research.md`)
- The app sends `tools.txt` entries as **OpenAI-shaped function tools** in `session.update`;
  tool calls route through the WS1 **adapter → Omni `/v1/chat/completions`**. So WS3's tools
  only work once that adapter handles the tool round-trip (WS1 is prototyping tool-calls first).
- **NPC voices** = the DM calls **`speak_as(voice_id, text)`** — an app-native tool the **WS1
  adapter** intercepts: it synthesizes the line in that character's designed voice (Qwen3-TTS
  voice-clone prompt) and streams it as output audio, instead of routing to the default Omni
  speaker. WS1 must implement this handler, add `speak_as` to the app's native tool
  definitions, and register the designed voices. Voice IDs: see `voices/character-voices.md`.
- **Tool-calling reliability** is the shared risk. Fallback: route GM logic through
  `Qwen/Qwen3-30B-A3B-Instruct-2507` if Omni's tool-calling is weak.
- **Vision**: the `camera` tool injects frames in-band to Omni (absorbs the Qwen3-VL role) —
  consistent with the DM reading a physical sheet/dice.

## Open WS3 items
- [ ] Live-test the tool loop end-to-end once the WS1 voice path + adapter are up.
- [ ] Tune which `hass__*` lights/scenes exist on the actual HA instance.
- [ ] Decide on a rule/adventure retrieval tool vs. instruction summary.
- [ ] Confirm tool count/latency is acceptable with the full `dm` tool set.
