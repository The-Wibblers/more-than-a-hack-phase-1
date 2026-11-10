# SkillCorner Open Data

The one dataset from outside the challenge match that we use. `possession/touch_possession.py` learns from it how a
sequence of touches turns into possession (writeup.md, section 5).

| | |
|---|---|
| Games | 20 (A-League 2024/25) |
| Tracking | from the broadcast, 10 Hz, off-screen players extrapolated |
| Events | dynamic events (player possessions, pressing, runs) and phases of play |
| Licence | MIT |
| Host | [GitHub, SkillCorner/opendata](https://github.com/SkillCorner/opendata) |

RULES.md allows external data as long as the organisers can reach it; a public host with an open licence guarantees
that.

## Getting it

```bash
python -m external.download skillcorner    # ~1.8 GB, no account
python -m external.build skillcorner       # writes data/external/games/skillcorner-<id>/
```

Everything lands in `data/external/`, which is gitignored. The viewer shows the built games at `/games`.

## What each game becomes

One bundle per game (`__init__.py` has the format): metres from the pitch's top-left corner with y down, one frame
index running through the periods, object 0 the ball, and events on the same frame index. SkillCorner's events are
exact to the frame (median 0.2 m from the tracked actor). Its event coordinates are in the attacking team's frame; the
build turns them back into the pitch's.
