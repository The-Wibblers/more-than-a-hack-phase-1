# Who has the ball: our own ball, players and officials, then touches

**More than a Hack 2027, Phase 1: technical document.**

We answer the question from the picture rather than from the provided ball. We find the ball, the players and the
assistant referees in the tactical video ourselves, with detectors that teach themselves from the match, without a
single hand label. A *touch* is a break in the ball's motion at a player's body. Possession is decided from the
sequence of touches, and dead balls from set-piece tags, the referee's whistle, the ball leaving the pitch and the
players' pace.

On the public five minutes:

| | macroF1 | spellF1 | possessions (truth: 17) |
|---|---|---|---|
| **Touch model** (submitted) | **0.970** | **0.827** | 13 |
| Possession rule on the same touches | 0.965 | 0.800 | 13 |
| `baseline_smooth125` (the best published baseline) | 0.581 | 0.424 | 16 |

The sections follow RULES.md: what we used and why it is inspectable, how the labels come from algorithms, what we
did with the ground truth, how we replaced the provided data, the possession rules, reproduction, and limitations.

---

## 1. Any method, provided it is accessible and explained

Everything in the pipeline is in this repository. There are no pre-trained models, commercial APIs or LLMs anywhere
in it. Every network is small, defined in `ballcv/`, and trained here from scratch.

| Asset | How we use it |
|---|---|
| Tactical video | Everything we detect: the ball, the players and their kits, the assistant referees and their flags |
| Tactical tracking (ATD) | Players: a seed for our detectors, and identities for our tracker. The ATD ball seeds the ball detector, flags balls flying over a player, and gives the nearest-player fallback where no touch has happened yet. It never places a touch |
| Tactical homography | Refined against the painted lines and mowing stripes (`refine_h.py`), then used everywhere |
| Smart Tagging | The set-piece tags: which restart a stoppage waits for, and who takes it |
| Broadcast video | Put on the tactical clock by aligning the two feeds. Its **audio** gives the referee's whistle; its **picture**, a second view of the ball, settles touches the tactical camera cannot see in depth (section 4). Live frames only |
| A product photo of the match ball, the Puma Orbita ([public URL](https://www.prodirectsport.us/products/puma-orbita-laliga-1-24-25-fifa-quality-pro-puma-white-multicolor-puma-white-multicolor-1008765)) | One constant: the ball's mean colour, so the ball detector's seed looks for whitish blobs. `python ballcv/selftrain.py orbita` measures it again from the photo, which is not committed |
| SkillCorner Open Data (MIT, 20 A-League games) | Teaches the possession model how touches turn into possession (section 5) |

**Tried and dropped:** Metrica's broadcast ball, fused with the tactical one (it lags and is smoothed: for the second view we detect the ball ourselves); IDSSE Bundesliga data, whose
possession is labelled per touch, not per spell as here; learned possession models on tracking features; a classical
CV ball finder and earlier ball networks; image embeddings of players and of the ball; a supervised assistant-referee
classifier; a min-cost-flow player tracker; and a self-trained touch detector (`touch_self.py`). That last one is kept
in the repository but not used: its seed taught it that "near the ball" means "touched it". The physical touch rule
in section 4 was already good enough for possession.

## 2. The labels must come from an algorithm

The submitted file is written by `ballcv/run_match.py` from the models' outputs. No person labels or corrects
anything, and **no hand-labelled data is used to train or to validate any model**. RULES.md would allow a little for
training or validation; we built none.

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
measured on the match itself without labels. We use the window as a correctness check: when it scored worse, we
looked for the error in the mechanism, not for a better constant. The possession model learns from SkillCorner alone.

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
- **The whistle** (`alignment/detect_whistles.py`). This is a pea-whistle detector on the broadcast audio: two or
  more flat, narrow tones that start and stop together. It found both kickoffs. Its two hits in the public window are
  stoppages, 0.3 s and 0.6 s before the labelled dead ball.

## 5. The possession rules

CHALLENGE.md asks for a change of possession only on *controlled* possession, and for `D` whenever the ball is not in
play. Here is how we drew those lines.

**Dead ball** (`stoppages.py`). A stoppage is found from four sources:

1. **That play stopped:** a set-piece tag, a stoppage whistle, or players setting up a restart (one standing at a
   corner arc, or off the pitch over a touchline).
2. **When it stopped:** the whistle; otherwise the ball leaving the pitch; otherwise the players easing off. The
   assistant's flag only confirms an out the ball already shows, because the flag model is not reliable enough to
   create one.
3. **When it restarted:** the placed ball struck, for goal kicks and corners; the referee's restart whistle, for free
   kicks; otherwise the players' pace rising out of its lull.
4. **Who has the ball after:** the team named on the tag; otherwise the team of the player at the ball when it
   restarts.

**Live play: the rule** (`ownball.control_roles`). A touch by the team not in possession takes the ball only if it
is controlled:

| case | change of possession? |
|---|---|
| the ball reached him **loose** (a bounce or a deflection since the last touch) | yes: nobody had it |
| his team's **next touch is theirs again**, or no touch follows | yes: an interception or tackle that kept it |
| a **goalkeeper's claim** | yes, at once. A claim we cannot see (the ball vanishes, and is next seen in his hands) is dated when the ball, at the speed it was last seen at, reaches him |
| a lone touch **between two of the other team's** | no: a block, a poke or a deflection of a pass |
| a head-height "touch" of a ball **flying over** him | no: not a touch at all |

So a contested or loose ball stays with the last team that had it until someone controls it, as CHALLENGE.md asks,
and an aerial duel won without control by that team does not change possession.

**Live play: the touch model, which we submit** (`possession/touch_possession.py`). The control rule is one fixed
reading of "controlled". The touch model learns it instead, from the 20 SkillCorner games:

- **Input:** home or away touches every 0.1 s.
- **Output:** who has the ball, from SkillCorner's phases of play. Their median possession lasts 7.7 s, against 7.6 s
  on BAR-ATH.
- **Training:** a small bidirectional GRU on 60 s windows. Its touches are degraded at random every time (up to 30%
  missed, up to one false touch a minute), so it never relies on a perfect touch stream.
- **Where it decides:** only inside gaps between touches no longer than the longest gap SkillCorner ever shows within
  one possession, 25.5 s. Beyond that it says nothing new, and the rule's last touch stands.
- **Dead balls:** it takes them from the rule's file, unchanged.

On the 4 held-out SkillCorner test games, accuracy on live steps is:

| touches | last touch | control rule | touch model |
|---|---|---|---|
| all of SkillCorner's | 0.934 | 0.931 | **0.948** |
| 15% missed | 0.912 | 0.907 | **0.931** |
| 30% missed, 1 false a minute | 0.881 | 0.885 | **0.911** |

On the public five minutes, the gain over the rule is in the timing of transitions (transition F1 0.83 against 0.78),
which lifts spellF1 from 0.800 to 0.827.

## 6. Reproduction

The trained models are committed in `ballcv/models/`: the ball, kit, player, assistant and flag detectors, and the
touch model. Regenerating the submission from them needs no training:

```bash
pip install -r requirements.txt          # Python 3.12; ffmpeg on the PATH
# the match data unpacked into ./data (README.md)

# caches and alignment
python ballcv/cache_tracking.py                       # the ATD as arrays
python alignment/align_broadcast.py                   # broadcast frame -> tactical frame
python alignment/detect_whistles.py                   # the referee's whistles
python ballcv/refine_h.py fit                         # the homography snapped to the painted lines

# the ball, the players and the officials, by the committed models
for s in $(seq 0 10000 190000); do python ballcv/ownball.py detect $s $((s + 10000 < 199319 ? s + 10000 : 199319)); done
python ballcv/ownball.py link
python ballcv/player_det.py detect 11185 190197
python ballcv/player_track.py run 11185 190197
python ballcv/flag_self.py signals 11185 190197

# touches, stoppages, possession -> evaluation/results/possession_touch_model.csv
python ballcv/run_match.py
python evaluation/validate_submission.py evaluation/results/possession_touch_model.csv
```

To train the models again from nothing, before the detection steps (each `run` copies its last model into
`ballcv/models/`, and skips the rounds it has already finished, so a stopped run carries on where it left off):

```bash
python ballcv/selftrain.py run                        # the ball
python ballcv/kit_team.py run                         # the kits
python ballcv/player_det.py run                       # the players
python ballcv/assistant_self.py run                   # the assistant referees
python ballcv/flag_self.py run                        # their flags
python -m external.download skillcorner && python -m external.build skillcorner
python possession/touch_possession.py build && python possession/touch_possession.py train   # the touch model
```

`run_match.py` skips steps whose output is already newer than their inputs. Every cache lives under `data/`, which is
gitignored like the match data itself. `cd viewer && npm install && npm run dev` opens the viewer
(`viewer/README.md`), where every one of these outputs can be checked frame by frame.

## 7. Limitations, and what we would do next

- **Training is not seeded.** Training again gives slightly different models: round-to-round differences are about
  1-2 points on the label-free checks. The committed models are what reproduce the submitted file exactly.
- **Close pairs.** In about 12% of moments where two players stand within a metre of each other, the detector sees
  one. The tracker then holds both on Metrica's numbers, which is right only as often as Metrica is.
- **Mostly one camera.** The second view is used only to settle bounce against touch, and only where the broadcast
  is live and shows the ball. It saw the ball at 82 of the 191 breaks it was asked about; the rest keep the picture's call. Elsewhere a
  ball in the air is placed by its gravity arc, and a single camera can mistake a fast roll for a flight. Before
  11593 the two cameras show the ball rolling where the arc fit had it flying. Running the broadcast detection on
  every live frame, to give the ball's height everywhere, is the clearest next step.
- **The flags are noisy.** On the near touchline, 15% of raises follow a ball going out, against 5% by chance; on
  the far line, none do. So the flag only confirms.
- **Unwhistled, untagged stoppages** depend on the ball being seen leaving the pitch, or on the players slowing down.
- **The touch model learned from another league** and from broadcast tracking at 10 Hz. More open data with
  possession labelled per spell would help most.
- **The check is small.** The public window has 17 possessions, so differences of a few hundredths of spellF1
  between our two methods are inside the noise EVALUATION.md describes.
