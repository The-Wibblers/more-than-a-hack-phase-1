# Who has the ball: our own ball, players and officials, then touches

**More than a Hack 2027, Phase 1: technical document.**

We answer the question from the picture rather than from the provided ball. We find the ball, the players and the
assistant referees in the tactical video ourselves, with detectors that teach themselves from the match, **without a
single hand label** (section 2 walks through a training run). A *touch* is a break in the ball's motion at a player's
body. Possession is decided from the sequence of touches, and dead balls from set-piece tags, the referee's
whistle, the ball leaving the pitch and the players' pace.

On the public five minutes:

| | macroF1 | spellF1 | possessions (truth: 17) |
|---|---|---|---|
| **Our possession rules** (submitted) | **0.972** | **0.827** | 13 |
| `baseline_smooth125` (the best published baseline) | 0.581 | 0.424 | 16 |

The sections follow RULES.md: what we used and why it is inspectable, how the labels come from algorithms, what we
did with the ground truth, how we replaced the provided data, the possession rules, and reproduction.

---

## 1. Any method, provided it is accessible and explained

Everything in the pipeline is in this repository. There are no pre-trained models, commercial APIs or LLMs anywhere
in it. Every network is small, defined in `ballcv/`, and trained here from scratch.

| Asset | How we use it |
|---|---|
| Tactical video | Everything we detect: the ball, the players and their kits, the assistant referees and their flags |
| Tactical tracking (ATD) | Players: a seed for our detectors, and identities for our tracker. The ATD ball seeds the ball detector and flags balls flying over a player. It never places a touch |
| Tactical homography | Refined against the painted lines and mowing stripes (`refine_h.py`), then used everywhere |
| Smart Tagging | The set-piece tags: which restart a stoppage waits for, and who takes it |
| Broadcast video | Put on the tactical clock by aligning the two feeds. Its **audio** gives the referee's whistle; its **picture**, a second view of the ball, settles touches the tactical camera cannot see in depth (section 4). Live frames only |
| A product photo of the match ball, the Puma Orbita ([public URL](https://www.prodirectsport.us/products/puma-orbita-laliga-1-24-25-fifa-quality-pro-puma-white-multicolor-puma-white-multicolor-1008765)) | One constant: the ball's mean colour, so the ball detector's seed looks for whitish blobs. `python ballcv/selftrain.py orbita` measures it again from the photo, which is not committed |

**Tried and dropped:** Metrica's broadcast ball, fused with the tactical one (it lags and is smoothed: for the second view we detect the ball ourselves); IDSSE Bundesliga data, whose
possession is labelled per touch, not per spell as here; learned possession models on tracking features; a classical
CV ball finder and earlier ball networks; image embeddings of players and of the ball; a supervised assistant-referee
classifier; a min-cost-flow player tracker; and a self-trained touch detector (`touch_self.py`). That last one is kept
in the repository but not used: its seed taught it that "near the ball" means "touched it". The physical touch rule
in section 4 was already good enough for possession.

## 2. The labels must come from an algorithm

**No hand labels were used anywhere in the pipeline: none to train a model, none to validate one.** Nobody marked a
ball, a player, a kit, a flag or a touch on a single frame. The submitted file is written by `ballcv/run_match.py` from
the models' outputs, and nothing in it was corrected by a person. The only labels we ever looked at are the
organisers' public five minutes, as a final check (section 3).

Each detector **teaches itself** the same way:

1. **Round 0, the seed.** A label is made only where two independent sources agree, for example the tracking and
   simple colour rules on the picture. Rather fewer labels than wrong ones.
2. **Rounds 1 to K.** Model *k* is trained from scratch on every label so far. It then labels fresh frames where it
   is at least `conf_min(r)` sure. That bar starts at 0.8 and drops 0.015 a round, and it is the same schedule for
   every detector: there is no ground truth to tune it on.
3. **Every round is checked without labels**, on held-out seed frames. For players, for example: are the seed
   players found, how many detections have no tracked body near them, and does a team ever come out with more than
   eleven?

| Detector | Seed: the two sources that agree | Label-free check |
|---|---|---|
| Ball (`selftrain.py`, 18 rounds) | Metrica's ball **and** a lone ball-sized, round, whitish blob within a few metres of it | the seed's held-out balls |
| Kits (`kit_team.py`) | Metrica's team on a mature, unoccluded track **and** an uncrowded crop | validation accuracy per kit |
| Players (`player_det.py`) | Metrica's player alone in the picture **and** the kit model reading his slot's kit | found 98.0%, 0.48 extra detections a frame, a team over eleven on 4.5% of frames |
| Assistants (`assistant_self.py`) | lime shirt and socks on the touchline **and** level with the offside line from the tracking | the seed's held-out assistants; agreement with the offside line |
| Flags (`flag_self.py`) | one flag-sized orange-red patch in the assistant's hand, moving with him over neighbouring frames | found, steadiness, and raised flags while play is on |

**A training run, step by step: the ball detector** (`python ballcv/selftrain.py run`). The numbers are from the
run that made the committed `ballcv/models/ball.pt` (`self/rounds.csv` and each round's `log.csv`).

1. **Round 0, the seed.** 15,000 frames are tried, spread over both halves. None is from the public five minutes,
   and none is a still frame (players standing about, as in a stoppage). A frame is labelled only where Metrica's
   ball and the picture agree: the only ball-sized, round, whitish blob within 3 m of Metrica's ball, no more than
   1.5 m from it, with nothing else bright within a metre. Metrica alone drifts onto players, and the picture alone
   finds socks and kit, but both are rarely wrong at the same spot. The seed is strict, so it fires on only
   **565 frames**. Each of them also gives "not a ball" examples: every other blob in the frame at least three ball
   widths away.
2. **Round 1.** A network is trained from scratch on those 565 frames, with every fifth one held out. Those held-out
   frames are seed labels too: the validation asks whether the network finds the ball where Metrica and the picture
   agreed, on frames it never trained on. It does so in 94.3% of them after the first epoch, and in 97.8% at the
   epoch with the lowest validation loss, which is the one kept. It puts its top peak on a "not a ball" crop under
   0.7% of the time. Model 1 then looks at 3,000 fresh frames and keeps a label wherever it is at least 80% sure:
   **234 new balls**.
3. **Rounds 2 to 17.** Each round trains a new network from scratch on every label so far. Before it starts, the
   previous model is scored on the new validation set, as the baseline to beat. The bar for a new label comes down
   a little each round as the models improve. Labels are kept one per 2 s moment, so the model does not learn twenty
   copies of one picture, and every fourth round the same moments are opened again for the better model. That is why
   the new labels jump at rounds 4, 8, 12 and 16 (801, 1,071, 1,214 and 1,312).
4. **The model in use.** Model 17 was trained on **7,225 labelled frames**, 13 times what the seed gave. Its held-out
   frames are every fifth of all those labels, most of them made by models 1 to 16. It finds the ball on 99.7% of
   them, with 0.4% false alarms, and its validation loss is 0.075 against model 1's 0.221. It is copied to
   `ballcv/models/ball.pt`.

These scores measure agreement with labels the pipeline made itself, never with the truth. They show that each model
learned its labels and generalised to frames it did not see. They cannot show that those labels were right. That is
what the seed's caution is for (two independent sources, both wrong at the same spot, is rare). It is also what the
later checks look for. Without any labels, we check that the ball's path through the detections is physical
(section 4). Last of all, the organisers' public five minutes check possession as a whole, and no detector ever
trained or validated on those frames.

The other detectors run the same loop with their own seeds (the table above), so every label any of them learned
from was made by an algorithm: first by two sources agreeing, then by the models themselves.

**The ball and the touches.** The ball detector finds up to eight candidates a frame. Dynamic programming picks the
smoothest physical path through them, in camera-stabilised pixels: the tactical camera only pans, tilts and zooms
about a fixed centre, so any two frames are related by an exact homography, whatever the ball's height. A touch is a
break in that path, found by comparing fits of the motion just before and just after each frame (section 4).

**The players.** `player_track.py` keeps Metrica's 46 slots as identities. It moves each slot to the detector's
player of the same kit (one-to-one, nearest overall). Where nobody matches, a slot keeps Metrica's position, or
follows the player hiding it, or coasts for a moment. Every path then goes through one Kalman smoother whose noise is
measured on the match: about 0.02 m for the picture, about 0.09 m for Metrica, and accelerations of about 1 m/s².
This puts back the players Metrica loses in duels. In the public window, a Barcelona player hidden behind an Athletic
one at the far touchline had been tracked as Athletic, and every touch there went to the wrong team.

## 3. No private ground truth

Only the public five minutes exist for us, and **nothing is fitted or selected on them**. Every frame in that window
is kept out of every detector's training, and no threshold or weight was tuned on it: the constants are physical, or
measured on the match itself without labels (the referee's whistle, for one, is learned from its tagged kick-offs).
We use the window as a correctness check: when it scored worse, we looked for the error in the mechanism, not for a
better constant.

## 4. The provided data as a starting point: what we replaced

The provided data is automatic and uncorrected, and its gaps are where possession changes: the ball is missing on 41%
of graded frames, and players are lost behind each other exactly in duels. So we rebuilt the ball, the players, the
camera and the officials (section 2), and judged each against physics:

- **Touch or bounce** (`ownball.py events`). At each break in the ball's motion:
  - **A touch**, if the ball is at a player's body column: within 0.7 m always; otherwise within a reach of
    0.5 m + 0.05 m per m/s of velocity change (at most 1.5 m), with the change pushing the ball away from him.
  - **A bounce**, if the velocity change points straight up the image's local vertical while the ball is falling.
    The vertical comes from the camera at every point of the picture, so the direction of gravity is known there.
  - **Not a touch**, if the ball is high and passing over a player: Metrica's ball goes straight through the moment.
  - **Settled by a second camera** (`broadcast_check.py`), when a break looks like a bounce but a player's body is
    over the ball in the picture. Taking a dropping ball on the chest changes its motion the same way a bounce does,
    and the tactical camera looks down, so a ball bouncing several metres behind a player sits on his body in the
    picture. The broadcast camera stands about 14 m away and is zoomed about three times further in; there the two
    are tens of pixels apart. We run our own ball detector on the live broadcast picture, shrunk by the two cameras'
    focal lengths to the scale it learned on, and intersect its ray with our tactical ball's ray. Two lines in space
    do not meet by chance, so a frame counts only where they meet, within the same 0.25 m that `ball3d.py` asks of a
    flight. The ball's place in 3D then decides: within the player's reach it is his touch, otherwise not. Where
    the broadcast cannot see the ball, the picture's closeness stands. Of 191 such breaks over the match, the broadcast
    sees the ball at 82, and puts it beyond the player's reach at 41.
- **The ball in the air** (`ball3d.py`). Between two breaks the ball either rolls or flies on a gravity arc. Fitting
  the arc to the rays through the ball's pixel gives its height, so a ball in the air past a line is not called out.
- **The whistle** (`alignment/detect_whistles.py`). Nothing in the detector says what the referee's whistle sounds
  like: no band, no pitch, no number of tones. It is learned from the match's own kick-offs, which the Laws of the
  Game make him whistle and the smart tags mark. Across the whole audible band, the detector looks for tonal sound: a
  line narrow in frequency, louder than the same frequency a moment before, and held. Then it asks which frequencies
  stand out more in the 25 s after a kick-off tag than in any other 25 s of the match. Significance is judged
  against random sets of six windows, so that neighbouring frequencies are not counted as separate tests. His
  whistle at each kick-off is where those frequencies peak; his tones are the lines it holds at most kick-offs.
  On BAR-ATH that is two tones, 3836 and 4266 Hz, found at five of the six kick-offs; the sixth tag has only a faint
  pip. Commentary, crowd and the mix's constant hum all fail the test. A whistle is then his tones sounding together;
  a run too short to be a whistle is kept as a pip, which only times a stoppage that other evidence shows.

  An earlier version found his tones without the tags, as the two pitches the match's whistle sounds hold most
  often. The two agree on 39 of that version's 45 whistles, score and all. But a constant hum in the mix pulled one of
  its tones 130 Hz low, so five other sounds passed as his. Two of them made false stoppages: a keeper holding the
  ball while players walked back, and advantage played on at the edge of the box. In the broadcast, play is live in
  both. Learning the tones from the kick-offs removes both, about 36 s of play, and changes nothing else. Its two hits
  in the public window are stoppages, 0.3 s and 0.6 s before the labelled dead ball.

## 5. The possession rules

CHALLENGE.md asks for a change of possession only on *controlled* possession, and for `D` whenever the ball is not in
play. Here is how we drew those lines.

We think of every cue as **evidence**, and the evidence comes in two kinds. Some tells us **that** something
happened: play stopped, or the ball changed hands. Some tells us **when** it happened, to the frame. Few cues do both
well. A set-piece tag is certain that play stopped, but it is dated loosely, often seconds off. The players slowing
down can be timed to the frame, but on its own it proves nothing, since a player can stop running for any reason. So
we **chain** evidence: the first link says the event happened, and the later links narrow down when. The order of
play constrains the chain. A goal is scored only in play, a restart has to put the ball back in play, and a ball
cannot be dead and live again at once.

**Dead ball** (`stoppages.py`). The evidence:

| evidence | says **that** play stopped | says **when** |
|---|---|---|
| Set-piece and goal tags (Metrica) | yes, and which restart, and who takes it | loosely: seconds off |
| The referee's stoppage whistle | yes | to a fraction of a second |
| Players setting up a restart (one at a corner arc, or off the pitch over a touchline) | yes: players leave the pitch only while the ball is dead | no |
| Our ball crossing a line, and staying off the pitch | yes, if it is not in the air (`ball3d.py`) | to the frame |
| Our ball seen in the goal, for a tagged goal | with the tag | to the frame |
| The referee's pip (a whistle too short to stand alone) | no | yes, once something else says play stopped |
| The broadcast cutting away to a close-up or replays | weakly | roughly: play had stopped by then |
| Nobody running for over 20 s (the three fastest players under a jog) | weakly: only with replays or a restart whistle | no |
| The players easing off (the biggest drop in their pace) | no | yes, inside a window the other evidence gives |
| The assistant's flag | no (see below) | only for an out something else found |

So a tagged corner whose ball we lost goes **corner tag → nobody running → players stopped**. The tag says a corner
is coming, the stretch with nobody running says play had stopped before it, and the players' pace dropping picks the
frame. A goal goes **whistle → ball in the goal**: the whistle says play stopped, and our ball crossing the goal
line, a moment before it, dates the stop. Over the match, our 89 stoppages are timed by the ball leaving the pitch
(44), a whistle (20), the players' pace (13), the flag (3), the ball in the goal (3), the broadcast (4), and the
referee's pip and the half's first frame (one each). The viewer shows each stoppage's chain, in order.

**When play restarted** follows the same idea. Where the ball itself can be trusted, it decides: the placed ball
struck, for a goal kick or a corner, or the kick after the referee's restart whistle, for a free kick. That covers 46
of the 89 stoppages. Everywhere else, the players' pace rising out of its lull times the restart. A restart has to
lead to play: if play stops again within a few seconds, or before anyone ran, it was the same stoppage. **Who has
the ball after** is the team named on the tag, or otherwise the team of the player at the ball when it restarts.

**The flag is the weakest evidence we have.** The flag detector (`flag_self.py`) taught itself, like the others, but
it has much less to learn from. The assistants raise their flags only a handful of times in a match, and we have no
hand labels to measure the detector against. It reports 92 raises over the match. On the near touchline 14% of them
follow the ball going out, against 5% by chance, and on the far touchline none do. So a raise never makes a stoppage
by itself. It only times an out that a tag or a set-up already says happened, and it does so for 3 of the 89.

**Live play: the rules, which we submit** (`ownball.control_roles`, then `possession/predict.py`). Here too the
evidence chains. A touch by the team not in possession says the ball *may* have changed hands. That team's next
play on the ball says it did. The start of its stretch on the ball says when. A touch takes the ball only if it is
controlled:

| case | change of possession? |
|---|---|
| the ball reached him **loose** (a bounce or a deflection since the last touch) | yes: nobody had it. But not in a **duel**, with a player of the team in possession within reach too: who touched it is in doubt, and the other team's touch is as likely a block |
| his team's **next play on the ball is theirs again** (its next touch, or a stretch on the ball, whichever comes first), or nothing follows | yes: an interception or tackle that kept it |
| a **goalkeeper's claim** | yes, at once. A claim we cannot see (the ball vanishes, and is next seen in his hands) is dated when the ball, at the speed it was last seen at, reaches him |
| a lone touch **between two of the other team's** | no: a block, a poke or a deflection of a pass |
| a head-height "touch" of a ball **flying over** him | no: not a touch at all |

So a contested or loose ball stays with the last team that had it until someone controls it, as CHALLENGE.md asks,
and an aerial duel won without control by that team does not change possession.

A player can have the ball for seconds with touches too gentle to break its motion, or take it while we cannot see
it. So we also ask who the ball is *with*. A player is *on the ball* from a frame it is at his feet (within
`CONTACT_M`, at foot height) while he stays the player nearest to it on the ground, for at least `PLACED_S`. We compare
on the ground because the picture puts a ball running on behind a player on his body. A ball he controls moves with
him; one he blocks, or that runs past him, leaves him, and someone else is soon nearer. Being on the ball changes
nothing by itself, since a ball rolling past a standing player looks the same. But a change of
possession is dated from the start of the taking team's stretches on the ball just before its touch: not before the
previous touch, the start of the half, or a dead ball. A player's touches and stretches on the ball make up his
*spell*, and a ball played goes from one player's spell to the next player's.

## 6. Reproduction

**One command.** The trained models are committed in `ballcv/models/` (the ball, kit, player, assistant and flag
detectors), so regenerating the submission needs no training:

```bash
pip install -r requirements.txt          # Python 3.12; ffmpeg on the PATH
# the match data unpacked into ./data (data/README.md)
python reproduce.py                      # -> evaluation/results/possession_touches.csv, validated
python reproduce.py --train              # the same, after training every model again from nothing
```

`reproduce.py` runs the steps in order. Each is skipped when its output is already there, so a stopped run carries
on where it left off:

| step | script | writes |
|---|---|---|
| 1. the tracking, as arrays | `ballcv/cache_tracking.py` | `data/ballcv/work/align/tactical.npz`, `broadcast.npz` |
| 2. broadcast frame → tactical frame | `alignment/align_broadcast.py` | `data/broadcast/BAR-ATH_broadcast_to_tactical.csv` |
| 3. the referee's whistles, learned from the kick-off tags | `alignment/detect_whistles.py` | `data/broadcast/BAR-ATH_broadcast_whistles.csv`, `_whistle_pips.csv` |
| 4. the homography, snapped to the painted lines | `ballcv/refine_h.py fit` | `work/align/homography_refined.npy` |
| 5. the ball: candidates, then their best path | `ballcv/ownball.py detect`, `link` | `work/align/fullmatch/` |
| 6. the players: detected, then tracked | `ballcv/player_det.py detect`, `player_track.py run` | `work/align/players/` |
| 7. the assistants' flags | `ballcv/flag_self.py signals` | `work/align/flags/` |
| 8. touches, stoppages, possession, and the viewer's files | `ballcv/run_match.py` | `data/ballcv/*.json`, `evaluation/results/possession_touches.csv` |

With `--train`, the five detectors are trained before step 5 (`selftrain.py`, `kit_team.py`, `player_det.py`,
`assistant_self.py` and `flag_self.py`, each `run`). Each copies its last model into `ballcv/models/` and skips the
rounds it has already finished. Then steps 5 to 7 are made again with the new models.

**Where everything lives.** The repository holds the code and the trained models. Everything made from the match
lives under `data/`, which is gitignored like the match data itself (`MTAH_DATA_DIR` moves it elsewhere).
`games.py` and `ballcv/paths.py` are the only places that spell out a path.

```
evaluation/results/possession_touches.csv     the submission: our possession, one label per tactical frame
ballcv/models/                                the trained detectors: ball, kit, players, assistants, flags (committed)
data/
├─ tactical/                                  as provided (the video may keep its download name, ..._tactical-001.mp4)
│  ├─ BAR-ATH_tactical.mp4
│  ├─ BAR-ATH_tactical_atd.xml, _atd.txt      Metrica's tracking
│  ├─ BAR-ATH_tactical_homography.csv
│  └─ BAR-ATH_tactical_smart_tagging.xml      the set-piece and goal tags
├─ broadcast/                                 as provided, plus what we derive from the broadcast:
│  ├─ BAR-ATH_broadcast.mp4, _atd.*, _homography.csv
│  ├─ BAR-ATH_broadcast_to_tactical.csv       step 2: the tactical frame and picture kind of every broadcast frame
│  ├─ BAR-ATH_broadcast_red_fraction.npy      step 2's scan of the video, cached
│  └─ BAR-ATH_broadcast_whistles.csv, _whistle_pips.csv     step 3
├─ ballcv/                                    our outputs, read by possession/predict.py and the viewer
│  ├─ BAR-ATH_touches.json                    the ball's path, touches, bounces, balls played, spells
│  ├─ BAR-ATH_stoppages.json                  dead balls, each with its chain of evidence
│  ├─ BAR-ATH_players_tracked.f32, .json      our player tracks, for the viewer
│  ├─ BAR-ATH_assistants.json, _flags.json    the assistants and their flags, for the viewer
│  └─ work/align/                             caches and intermediate files
│     ├─ tactical.npz, broadcast.npz          step 1
│     ├─ homography_refined.npy               step 4
│     ├─ fullmatch/                           step 5: ball candidates per 10,000 frames, and the linked track
│     ├─ players/                             step 6: detections (dets/) and tracks.npz; training rounds r0, r1, ...
│     ├─ flags/, assistants/, kit/, self/     step 7, and the training rounds of the flag, assistant, kit and ball detectors
│     ├─ ball3d.npz, broadcast_ball3d.json    step 8: the ball's height, and the broadcast's view of it, cached
│     └─ touches/, motion_*.json              the unused touch detector (touch_self.py), not part of the pipeline
└─ orbita-ball.png                            optional: the ball's product photo, only to measure its colour again
```

`cd viewer && npm install && npm run dev` opens the viewer (`viewer/README.md`), where every one of these outputs can
be checked frame by frame.
