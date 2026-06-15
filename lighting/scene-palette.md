# Scene palette — story-reactive lighting

Named moods the DM sets via `hass__HassLightSet` on real scene beats. The **color name** +
**brightness** are what the HA `HassLightSet` intent reliably accepts; `rgb` is for precise
control through the HA scenes template (`ha-scenes.yaml`). Keep changes to genuine beats —
not every line.

| scene | HassLightSet `color` | rgb (scenes) | brightness | when |
|---|---|---|---|---|
| `combat` | red | `[210, 25, 25]` | 90% | fights, ambushes, imminent violence |
| `dread` | red | `[120, 10, 10]` | 25% | horror, a body, something terribly wrong |
| `explore` | blue | `[40, 90, 160]` | 40% | tense ruins, caves, sneaking, the unknown |
| `settlement` | orange | `[255, 150, 60]` | 70% | towns, campfires, safe rest, friendly NPCs |
| `radiation` | green | `[80, 200, 70]` | 55% | rads, toxic zones, sickly tech glow |
| `arcane` | purple | `[150, 60, 220]` | 50% | mystery, visions, eerie/uncanny moments |
| `neutral` | white | `[255, 220, 180]` | 60% | default / between scenes / out-of-character |

## Usage notes
- Call `hass__GetLiveContext` once at session start to learn the actual `light.*` entities,
  then target those.
- Prefer a short transition (~1s) for atmosphere; instant is fine for a hard combat cut.
  Transitions are applied when activating the HA scenes (`scene.turn_on … transition: 1`);
  the plain `HassLightSet` intent may snap instantly — acceptable on camera.
- Tune the rgb/brightness to how your specific LIFX bulbs read **on camera** for the demo —
  saturated colors and lower brightness usually film better.
- These names are also the HA scene suffixes in `ha-scenes.yaml` (`scene.dm_combat`, …) if you
  prefer activating declarative scenes over per-bulb `HassLightSet` calls.
