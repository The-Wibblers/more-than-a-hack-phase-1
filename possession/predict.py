"""Possession for every frame, the rule: our touches and our stoppages.

  1. Dead ball. A frame is D inside a stoppage of ballcv/stoppages.py (data/ballcv/BAR-ATH_stoppages.json): from
     when play stopped -- a referee's stoppage whistle, our ball leaving the pitch, a goal, half-time -- until the
     restart it waits for (a throw-in, a goal kick, a corner, a free kick, a kick-off), or the players resuming. The
     team taking the restart has the ball from then on, until the touches say otherwise. Dead frames are D whatever
     the tracking says: players carry the ball around during a stoppage, and the ground truth does not count it.

  2. Touches. Live frames take the team in controlled possession from our own ball tracker's touches
     (data/ballcv/BAR-ATH_touches.json: the ball's motion breaks at a player's body): an opponent's block or tackle
     only changes it when his team's next touch is theirs again, and a change is dated from the taking team being on
     the ball, where it was. A half starts with the team that makes its first touch (the kick-off).

Frames outside the two halves are X, which is never scored.

    python possession/predict.py                      # -> evaluation/results/possession_touches.csv
    python evaluation/validate_submission.py evaluation/results/possession_touches.csv
"""

import argparse
import csv
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, half_video_blocks, require, source_frames  # noqa: E402
from baseline import HOME, AWAY, NOBODY, team_of_each_track, write_submission  # noqa: E402

# Params (read by ballcv/stoppages.py)
BALL_D = 0.22                        # the ball's diameter (m): Law 2's 68-70 cm round
OUT_M = BALL_D / 2                   # it is out when wholly over a line (Law 9): its centre a radius past
RESUME_RISE = 0.5                    # m/s rise of the players' median speed that ends a stoppage (they resume)
WHISTLE_CAP = 60 * 25                # a stoppage whistle holds at most this long without a resume signal
SPEED_HALF = 25                      # players' median speed is averaged over +/- this many frames
DATA_DIR = os.path.join(ROOT, "data")
WHISTLES_CSV = os.path.join(DATA_DIR, "broadcast", "{game}_broadcast_whistles.csv")
TOUCHES_JSON = os.environ.get("TOUCHES_OUT") or os.path.join(DATA_DIR, "ballcv", "{game}_touches.json")
STOPPAGES_JSON = os.environ.get("STOPPAGES_OUT") or os.path.join(DATA_DIR, "ballcv", "{game}_stoppages.json")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")


def read_atd(raw_path, n_frames, n_slots):
    """-> ball (n, 2) normalised, players (n, slots, 2) normalised, speeds (n, slots); NaN where missing."""
    ball = np.full((n_frames, 2), np.nan)
    players = np.full((n_frames, n_slots, 2), np.nan)
    speeds = np.full((n_frames, n_slots), np.nan)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1                     # frame_id counts from 1, video frames from 0
            if not 0 <= f < n_frames:
                continue
            values = parts[2].split(",")
            if values[0] != "NaN" and values[1] != "NaN":
                ball[f] = float(values[0]), float(values[1])
            for slot, group in enumerate(parts[1].split(";")[:n_slots]):
                x, y, s = group.split(",")
                if x != "NaN" and y != "NaN":
                    players[f, slot] = float(x), float(y)
                if s != "NaN":
                    speeds[f, slot] = float(s)
    return ball, players, speeds


def match_atd(game=GAME):
    """-> Metrica's ball (n, 2), players (n, slots, 2), speeds (n, slots) for every tactical frame (read_atd)."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    print("reading {} ...".format(os.path.basename(raw_path)))
    return read_atd(raw_path, source_frames("tactical", game), len(team_of_each_track(meta_path)))


def read_whistles(path):
    """-> (stoppage frames, restart frames) on the tactical timeline; 'unclear' whistles are ignored."""
    if not os.path.exists(path):
        print("no whistles at {}".format(os.path.relpath(path, ROOT)))
        return [], []
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    return ([int(r["tactical_frame"]) for r in rows if r["role"] == "stoppage"],
            [int(r["tactical_frame"]) for r in rows if r["role"] == "restart"])


def read_touches(path):
    """-> (first, last) covered frames, [(frame, HOME|AWAY, touch frame)] possession changes and [(frame, HOME|AWAY)]
    the touch spells (a player's run of touches) in order.

    The changes are the touch pipeline's controlled-possession sequence: an opponent's block or tackle does not
    change possession unless his team's next touch is theirs again. A change is dated from the taking team being on
    the ball, before the touch that took it, where they were."""
    with open(path) as handle:
        data = json.load(handle)
    changes = sorted((e["f"], HOME if e["team"] == "H" else AWAY, e.get("touch", e["f"])) for e in data["possession"])
    spells = [(s["f0"], HOME if s["team"] == "H" else AWAY) for s in data["spells"] if s["team"] in ("H", "A")]
    return tuple(data["frames"]), changes, spells


def last_touch_codes(n, covered, touches):
    """Team of the last touch on every covered frame; NOBODY before the first touch and outside the coverage."""
    codes = np.full(n, NOBODY, dtype=np.int8)
    first, last = covered
    i, current = 0, NOBODY
    for f in range(first, last + 1):
        while i < len(touches) and touches[i][0] <= f:
            current = touches[i][1]
            i += 1
        codes[f] = current
    return codes


def player_speed(speeds):
    """Median player speed per frame, averaged over +/- SPEED_HALF frames (cumulative sums: O(n))."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)              # frames with no tracked player at all
        med = np.nanmedian(speeds, axis=1)
    med = np.where(np.isfinite(med), med, 0.0)
    n = len(med)
    c = np.concatenate([[0.0], np.cumsum(med)])
    i = np.arange(n)
    lo, hi = np.clip(i - SPEED_HALF, 0, n), np.clip(i + SPEED_HALF + 1, 0, n)
    return (c[hi] - c[lo]) / (hi - lo)


def read_stoppages(path, n):
    """-> (dead frames, [(restart frame, HOME|AWAY)]) from ballcv/stoppages.py: every stoppage from when play stopped
    until its own restart, and the team taking each restart (the team taking a set piece has the ball)."""
    with open(path) as handle:
        stoppages = json.load(handle)["stoppages"]
    dead = np.zeros(n, bool)
    for st in stoppages:
        dead[st["start"]:st["end"] + 1] = True
    return dead, [(st["restart"], HOME if st["restart_team"] == "H" else AWAY)
                  for st in stoppages if st["restart"] is not None and st.get("restart_team") in ("H", "A")]


def stoppage_restarts(path):
    """[(frame, HOME|AWAY)]: each restart and the team taking it, from ballcv/stoppages.py (read_stoppages)."""
    with open(path) as handle:
        n = 1 + max(st["end"] for st in json.load(handle)["stoppages"])
    return read_stoppages(path, n)[1]


def predict(game=GAME):
    """-> (labels array of 'H'/'A'/'D'/'X', per-frame info dict for reporting)."""
    n = source_frames("tactical", game)
    dead, restarts = read_stoppages(STOPPAGES_JSON.format(game=game), n)
    covered, changes, spells = read_touches(TOUCHES_JSON.format(game=game))
    # a change dated from the taking team being on the ball (before its touch): not while the ball was dead
    changes = [(max([f] + [g + 1 for g in np.flatnonzero(dead[f:t]) + f]), code) for f, code, t in changes]
    # the touches' changes run through the stoppages; after a restart its team has the ball, and the touches that follow
    # it decide as in play: the other team's touch takes it when that team's next touch is theirs again (up to the
    # touches' own next change, or play stopping again)
    spells.sort()
    for r, code in restarts:
        stops = np.flatnonzero(dead[r + 1:])
        upto = min([f for f, _c in changes if f > r] + [r2 for r2, _c in restarts if r2 > r] + [covered[1] + 1]
                   + ([r + 1 + int(stops[0])] if len(stops) else []))
        after = [(f, c) for f, c in spells if r < f < upto]
        taken = next((f for (f, c), (_f2, c2) in zip(after, after[1:]) if c != code and c2 == c), None)
        if taken is not None:
            changes.append((taken, HOME if code == AWAY else AWAY))
    changes = sorted(changes + [r for r in restarts if covered[0] <= r[0] <= covered[1]])   # a restart: its team's
    codes = last_touch_codes(n, covered, changes)
    in_half = np.zeros(n, bool)
    for _, first, last in half_video_blocks(game):
        in_half[first:last + 1] = True
        opening = next((code for f, code in changes if first <= f <= last), NOBODY)   # the kick-off's team
        codes[first:last + 1][codes[first:last + 1] == NOBODY] = opening
    labels = np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D")).astype("<U1")
    labels[dead] = "D"
    labels[~in_half] = "X"
    return labels, {"graded": int(in_half.sum()), "dead": int((dead & in_half).sum())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", help="output CSV (default: evaluation/results/possession_touches.csv)")
    args = parser.parse_args()

    labels, info = predict()
    path = write_submission(args.out or os.path.join(RESULTS_DIR, "possession_touches.csv"), labels)
    g = info["graded"]
    print("graded frames {}: judged dead {:.1%}".format(g, info["dead"] / g))
    print("labels: " + "  ".join("{} {}".format(k, int(np.sum(labels == k))) for k in "HADX"))
    print("-> {}".format(os.path.relpath(path, ROOT)))
