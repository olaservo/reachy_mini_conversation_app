# WS4 — Story-reactive lighting + robot expression

The DM drives the room: LIFX lights shift with the scene and Reachy reacts to story beats.
Judges won't have your lights, so this is a **local / demo-video** feature — not something
the deployed Space needs to run.

## How it works
- The fork ships a `smart_home` profile + HA MCP endpoint; the `dm` profile (`profiles/dm`)
  already enables `hass__GetLiveContext`, `hass__HassLightSet`, `hass__HassTurnOn/Off`.
- LIFX bulbs appear as `light.*` entities in HA, so the DM controls them through those tools.
- Two ways to set a mood:
  1. **Direct (default):** the DM calls `hass__HassLightSet` with a palette color + brightness
     (`lighting/scene-palette.md`). Simplest, no extra HA config.
  2. **Declarative scenes:** import `lighting/ha-scenes.yaml` (with your entity_ids) and the DM
     activates `scene.dm_<mood>` — smooth transitions via `scene.turn_on`.
- Robot: `lighting/robot-beats.md` maps story beats to `play_emotion` intents + dance/move_head.

## Files
- `lighting/scene-palette.md` — moods → color/brightness + when to trigger.
- `lighting/robot-beats.md` — story beats → robot motion (real `play_emotion` intents).
- `lighting/ha-scenes.yaml` — optional HA scenes template (replace entity_ids).
- `profiles/dm/instructions.txt` — ATMOSPHERE section, aligned to the palette.

## Optional richer path — MCP events/triggers
For reactions driven by *pushed events* (lights/robot responding on their own rather than via
DM tool calls), the HA events bridge in `everything/_mcp/__events` /
`events-triggers-resources/ha-events-bridge` can push `ha.state_changed`-style events. A nice
"the lights react by themselves" beat, but not required for the core demo.

## Verify (needs your HA + LIFX + robot)
- [ ] `hass__GetLiveContext` returns your LIFX `light.*` entities.
- [ ] Each palette color reads well on camera; tune rgb/brightness.
- [ ] (If using scenes) `ha-scenes.yaml` imported with real entity_ids; `scene.dm_*` activate.
- [ ] `play_emotion` intents and a `dance` fire on the robot.
- [ ] A story beat triggers light + emotion in sync with narration.
