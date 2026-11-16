# Tactical frame viewer

A Next.js app for stepping through the match one video frame at a time, with everything the pipeline produces drawn
on it. Every frame is its own page, `/frame/<n>` (`n` = video frame = submission row, 0–199318).

```bash
cd viewer
npm install
npm run dev          # http://localhost:3000  -> redirects to /frame/11185 (1st-half kickoff)
```

It reads the match from `../data/` (or `$MTAH_DATA_DIR`), the public ground truth from `../evaluation/gt/`, and our
outputs from `../data/ballcv/` and `../evaluation/results/`. Each file is re-read when it changes, so a rerun of
`ballcv/run_match.py` shows up on the next page load. The first request loads the tracking (~200 MB) into memory in
under a second; after that each frame is a slice.

The frame image comes from **ffmpeg** (`FFMPEG_PATH` overrides the binary): `/api/frame-image/<n>` returns frame `n`
as a JPEG. The video is constant 25 fps from t = 0 with a keyframe every 2 s, so a seek to `n/25` is frame-exact and
takes ~80 ms. Frames are extracted on demand, with no disk cache.

## What each page shows

| Panel | Source |
|---|---|
| The video frame with our players, our ball and the pitch lines projected on it | the mp4, `player_track.py`'s tracks, `ownball.py`'s ball, the homography |
| Top-down view: players by team with 1 s heading arrows, our ball and its trail, the assistant referees on their touchlines with their flag when it is up | the same, plus `flag_self.py` (via `viewer_export.py`) |
| Period, match clock, graded or not | DATA.md frame boundaries |
| Ground-truth possession, or "no label" outside the public window 11185–18684 | `evaluation/gt/` |
| Our possession, with a ✓ / ✗ against the ground truth | `possession_touch_model.csv` (the touch model, which we submit); the timeline also shows `possession_touches.csv` (the old control rule) |
| The ball played: who is on the ball now, or the last touch (player, team, body part), and what was played to the next player (ground pass, lofted ball; completed / lost) | `data/ballcv/BAR-ATH_touches.json` |
| Touches within ±30 s: every touch, bounce and deflection, with interception / block / keeper-claim badges (click to jump) | same |
| The ±30 s timeline: ground truth, our prediction, old rule, stoppages, set-piece tags, assistant flag, broadcast state, referee whistle, touches, on the ball (each player's spell), ball motion (flight / roll), balls played (from one player's spell to the next) | all of the above, `stoppages.py`, the Smart Tagging, `alignment/` |
| Players sorted by distance to the ball; the homography, its validity and reprojection error | tracks, homography |

**Keys:** ← → one frame · Shift ±1 s · Option/Alt ±10 s. The frame box jumps to any frame, and the match bar jumps
anywhere.

## SkillCorner games: `/games`

Every SkillCorner game built by `python -m external.build skillcorner` (`external/README.md`) gets a page at
`/games/<id>?f=<frame>`, with the challenge match listed first. Each page shows:

- a top-down pitch with the players (jersey, name, heading), the ball and its trail;
- the whole-match bar, with goals, shots and cards;
- a ±30 s event strip and list, which can be filtered by type;
- SkillCorner's possession and ball-in-play;
- the players by distance to the ball.

The keys are the same.

## Which slot is which team

The tracking metadata (`BAR-ATH_tactical_atd.xml`) maps track slots to teams: slots 1–24 are Barcelona (H) and 25–46
are Athletic (A). Slots are tracker IDs, not shirt numbers.
