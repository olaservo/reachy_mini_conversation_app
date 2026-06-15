# Robot expression — story beats → motion

The DM punctuates the story with the robot. `play_emotion` takes a **compact intent** (the
enum below), `dance` takes a move name from the runtime `AVAILABLE_MOVES` enum, `move_head`
aims attention, `head_tracking` "listens" to the active player. Use these as accents on real
beats — not every line.

## Story beat → `play_emotion(emotion=…)` intent
| beat | intent(s) |
|---|---|
| Critical success / heroic moment | `success`, `excited` |
| Solid success | `success`, `yes` |
| Failure / botched roll / complication | `displeased`, `irritated` |
| Grim setback / loss | `downcast`, `sad`, `lonely` |
| Combat begins / threat looms | `attentive`, `anxious` |
| Sudden danger / jump scare | `scared`, `surprised` |
| Eerie / arcane reveal | `amazed`, `confused` |
| NPC greets the party | `greeting`, `welcoming` |
| Friendly / warm exchange | `loving`, `grateful`, `helpful` |
| Comic beat / banter | `happy` |
| DM weighing a tricky ruling | `thinking`, `uncertain` |
| Hostile dismissal | `go_away`, `angry` |
| Victory celebration | `dance`, `excited` |
| Session wind-down / farewell | `goodbye`, `tired` |
| Robot/tech malfunction flavor | `electric`, `dying` |

`emotion="random"` is a safe fallback when no intent clearly fits.

## Other motion
- **`dance`** — short celebratory or NPC-performance beats (victory, a bard NPC, a tense
  release). Pick a move from the enum the tool exposes; `stop_dance` to end early.
- **`move_head`** — direct attention (left/right/up/down/front) toward a speaker, an exit,
  or an off-screen threat to sell the moment.
- **`head_tracking`** — keep on so Reachy "listens" to whoever is acting.
- Pair with lighting: e.g. combat → `scene=combat` + `play_emotion=attentive`; eerie reveal
  → `scene=arcane` + `play_emotion=amazed`.

## Notes
- Emotion intents are resolved by the app to curated recorded moves (`play_emotion.py`),
  so the DM only needs the intent word, not a move ID.
- Keep performance sparing — over-animating undercuts the dramatic ones.
