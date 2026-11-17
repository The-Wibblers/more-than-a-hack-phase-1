# Findings for Ansun — possession/stoppage review

Living doc from Joel's frame-by-frame review of the BAR-ATH viewer. Each item has the frame(s), the
diagnosis, the root cause, and a proposed lever. **Frame numbers are tactical video frames.**

_Last updated: 2026-11-16. Current base: your `db0ca27` + Joel's keeper fixes (uncommitted on `joel/phase1`)._

---

## Environment Joel has set up (so you know what's verifiable on his side)

- **Caches in place:** your `work/` bundle at `data/ballcv/work/` (ball track, player tracks, flags,
  assistants, ball3d, `tactical.npz`). Gitignored. Video symlinked `BAR-ATH_tactical-001.mp4 → .mp4`.
- **Deps installed** into `.venv`: `scipy`, `opencv-python`, `torch 2.15` (MPS available). Whistles+pips
  regenerated locally (`detect_whistles.py`).
- **Both loops now run locally.** Rule: `ownball.py events` (~1m35s) → `stoppages.py` → `predict.py`.
  Model: `touch_possession.py public` (now that `touch_games.npz` is in `data/possession/`).
- **Current 5-min scores** on `db0ca27` + keeper fixes: **rule spellF1 0.8267 / macroF1 0.9741**,
  **model 0.7742**.

**Guardrail on every change:** the 5-min public rule score must not drop below **0.8267**.

---

## 1. Keeper claims — THREE patterns, all now handled (Joel's fix, rule-verified)

The keeper takes the ball in three ways; only the first was caught before.

| pattern | frame | what the keeper does | detector |
|---|---|---|---|
| vanish | — | ball disappears ≥1 s near him | `keeper_claims` (yours) |
| **held visible ball** | **40457** | catches a shot, holds it visible ~2.7 s at ≤0.5 m | **`keeper_holds`** (new) |
| **quick aerial catch** | **77507** | plucks it from the air, holds ~1 s, throws it out | **catch case in `keeper_claims`** (new) |

- **40457:** ball sits 0.14–0.49 m from keeper 46 (A) in his box for ~2.7 s, never vanishing ≥1 s →
  `keeper_claims` missed it. Fix `keeper_holds()`: a claim when the ball sits within `CLAIM_HOLD_REACH_M`
  (1 m) of a keeper in his box for ≥ `CLAIM_HOLD_S` (1 s).
- **77507:** keeper catches from the air — ball reaches **0.45 m** then **lost 0.96 s** (77514→77538), then
  thrown away. Near-miss on both: `keeper_claims` needs a ≥1 s vanish (this is 0.96 s), `keeper_holds` needs
  ≥1 s within 1 m (this is ~1 frame). Fix: a **catch** case — ball within `CATCH_REACH_M` (0.6 m, at his
  hands) of a keeper in his box then lost ≥ `CATCH_GAP_S` (0.4 s) is a claim. Flipped possession to A at
  77517 (was 77614, **3.9 s late**).

**Verification (rule):** keeper claims 51 → 165. **5-min rule score unchanged at 0.8267**, **0 frames changed
in the scored window** both times. Match-wide ~7.3k frames re-attributed.

**For you:** these three live in Joel's forked `ownball.py` — worth folding into your keeper logic coherently,
then regenerating the **model** (which also reads these touches).

---

## 2. Ball out over the goal line in a `flight` segment — ✅ RESOLVED by `db0ca27`

**Frame 41659 → corner 42747.** Was 30 s late (`speed drop`, 42398). Your **broadcast cut-away** detection
now catches it: the stoppage runs **41664 → 42746, `timed_by='broadcast cut'`** — exactly the corroborated-
stoppage idea. Verified on Joel's machine. (Original cause: ball crossed on its last tracked frame in a
`flight` segment with no ball3d arc, `sure`=False; and the out was 43.6 s before the corner, past `OUT_S`.)

---

## 3. Broadcast close-up / replay as a "play stopped" signal — ✅ now in `db0ca27`

Extractable from `broadcast_to_tactical.csv` (`kind`, `pitch_view`). Measured: close-up **65.5% dead** vs
wide pitch **18.6%** (base 35%). You added it as the broadcast cut-away signal. **Caveat for tuning:** 34.5%
of close-ups are live play, and it lags the out ~3 s — strongest as a corroborator, not alone.

---

## 4. Ball-boy's second ball defeats the "ball came back" check — ✅ RESOLVED by `db0ca27`

**Frame 43056 (21:15).** A ball boy threw on a replacement ball, which the tracker followed back inside, so
the stoppage was ~3 s late. Now **`D` from 43049** on your latest. (Joel's earlier read that broadcast would
fix it was wrong — the close-up came too late; your fix catches it another way.)

---

## 5. Home touches read as team-less bounces — possession stuck on the wrong team

**Frame 79023.** The ball is **headed by Home, then kicked by another Home player**, but both are detected as
**team-less `bounce`s** — `79024` (nearest home slot 15) and `79060` (nearest home slot 6). So the last
*real* touch stays **A** (slot-A control at 78923) and possession never flips to **H**. The whole stretch
78950–79174 reads `A` when it should be `H` from ~79023.

**Root cause:** touch-vs-bounce / team attribution — a header and a driven kick off Home players classified
as bounces, not `H` touches, even though the nearest bodies are Home.

**Lever (ownball):** when a motion-break's nearest body is a player right at the ball, prefer a team touch
over a team-less bounce — especially headers (high break) off a known player.

---

## 6. Disallowed quick restart splits one dead ball into three

**Frames 79173 → 79599 (should be one dead ball).** Foul + out at ~79173 (referee **pip 79191, live**),
players try a **quick free kick (~79287)**, ref blows it back and makes them **retake (~79599)**. The whole
stretch is dead. The pipeline instead gives:

| frames | pipeline | truth |
|---|---|---|
| 79175–79286 | `D` ✓ | dead |
| **79287–79395** | **`H`** ✗ | dead (quick restart read as live) |
| 79396–79623 | `D` ✓ (end ~right, true ~79599) | dead |

**Two levers (stoppages):**
1. A restart quickly followed by another stoppage signal (the ref blowing a quick restart back) isn't a real
   restart — **merge into one continuous dead ball**. Adjacent to your "quick throws" work.
2. The 79396 whistle is flagged **`kind='replay'`** (heard over replay footage) — a replay whistle shouldn't
   **start** a stoppage.

---

## 7. Corner restart timed late — tag + touch ignored, waits for the arc-strike

**Frame 80913 (corner tag) / 80921 (corner struck) → stoppage ends 80991.** The dead ball runs **~2.8 s too
long**: 80921–80991 reads `D` *after* the corner is already taken.

- Smart-tag: **Corner Kick (Barcelona/H), start 80913**.
- Our touch: **H control at 80921** — the corner being struck.
- Stoppage: `start 80065 … end 80991, restart 80992` (`ball exit` start; restart from a late fallback).

**Root cause:** a corner's restart is timed by detecting the **placed ball struck at the corner arc**; that
needs the ball seen lying at the arc first, which didn't happen, so it fell back to a later "players resume"
(~80992). The **tag (80913) and the H corner-touch (80921) aren't used** to end the dead ball.

**Lever:** when a corner is tagged and a touch lands at/just after the tag start, time the restart from that,
rather than waiting on the arc-strike detection.

---

## Pattern notes

- **Stoppage boundaries** (#2, #4, #6): pinning when play stops/restarts when the ball is near a line, a
  second ball appears, or a quick restart is disallowed. Corroboration (ball3d arc + deceleration + broadcast
  cut + whistle/pip, disregarding replay whistles) is the robust answer. You've knocked out #2 and #4.
- **Touch detection** (#1, #5): catches off keepers, and headers/kicks off outfield players, misread as
  team-less bounces. Both change the touch sequence, so they feed **rule and model** alike.
