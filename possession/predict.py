"""Possession for every frame: our touches, a dead-ball judgement, and the nearest player where there are no touches.

Builds on starter/baseline.py, which gives the ball to the team of the nearest tracked player and says D
wherever the tracking has no ball. Two additions:

  1. Dead-ball judgement. A frame is D inside a stoppage of ballcv/stoppages.py
     (data/ballcv/BAR-ATH_stoppages.json): from when play stopped -- a referee's stoppage whistle, or our ball
     past a touchline or goal line (Metrica's ball agreeing) -- until the restart it waits for: a throw-in, a
     goal kick, a corner, or a free kick (on a restart whistle if one is heard); if that restart is never seen,
     until the players speed up again. The team taking the restart (the player at the ball) has the ball from
     then on, until the touches say otherwise. Without that file (or with --no-stoppages): a stoppage whistle until a
     restart whistle or the players speed up again (their median speed rises RESUME_RISE above its low point
     since the whistle), at most WHISTLE_CAP; or Metrica's ball more than OUT_M past a
     line, until it is seen back inside the pitch or a restart whistle sounds.
     Dead frames are D whatever the tracking says: the tracking keeps naming a team while players carry the
     ball around during a stoppage, and the ground truth does not.

  2. Touches. Where our own ball tracker has computed touches (data/ballcv/BAR-ATH_touches.json: the ball's
     motion breaks at a player's body, in camera-stabilised pixels, no Metrica ball), the label is the team in
     controlled possession: an opponent's block or tackle only changes it when his team's next touch is theirs
     again. No nearest-player guess and no smoothing there. Elsewhere the nearest-player labels are used.

Nearest-player frames are smoothed exactly like the baseline (majority vote over +/- WINDOW frames); dead frames
are D; frames outside the two halves are X, which is never scored.

    python possession/predict.py                      # -> evaluation/results/possession_touches.csv
    python possession/predict.py --no-touches         # -> possession_nearest_player.csv (nearest player only)
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
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import HOME, AWAY, NOBODY, team_of_each_track, write_submission  # noqa: E402

# Params
WINDOW = 50                          # majority vote half-width, frames (the published floor's window)
OUT_M = 0.5                          # ball this far past a line -> out of play
RESUME_RISE = 0.5                    # m/s rise of the players' median speed that ends a whistle stoppage
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


def read_whistles(path):
    """-> (stoppage frames, restart frames) on the tactical timeline; 'unclear' whistles are ignored."""
    if not os.path.exists(path):
        print("no whistles at {} -- out-of-play judgement only".format(os.path.relpath(path, ROOT)))
        return [], []
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    return ([int(r["tactical_frame"]) for r in rows if r["role"] == "stoppage"],
            [int(r["tactical_frame"]) for r in rows if r["role"] == "restart"])


def read_touches(path):
    """-> (first, last) covered frames and [(frame, HOME|AWAY)] possession changes, or None without a touch file.

    The changes are the touch pipeline's controlled-possession sequence: an opponent's block or tackle does not
    change possession unless his team's next touch is theirs again. Older files without it fall back to every touch."""
    if not os.path.exists(path):
        print("no touches at {} -- nearest player everywhere".format(os.path.relpath(path, ROOT)))
        return None
    with open(path) as handle:
        data = json.load(handle)
    events = data.get("possession") or [e for e in data["events"] if e["kind"] == "touch"]
    changes = sorted((e["f"], HOME if e["team"] == "H" else AWAY) for e in events)
    return tuple(data["frames"]), changes


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


def whistle_dead(n, stops, restarts, speed):
    dead = np.zeros(n, bool)
    restart_set = set(restarts)
    for w in stops:
        low = np.inf
        for f in range(w, min(n, w + WHISTLE_CAP)):
            if f > w and f in restart_set:
                break
            low = min(low, speed[f])
            if f > w + SPEED_HALF and speed[f] > low + RESUME_RISE:
                break
            dead[f] = True
    return dead


def out_of_play(ball, restarts, pitch):
    """Latched from the ball > OUT_M past a line until it is seen > OUT_M back inside, or a restart whistle."""
    length, width = pitch
    x, y = ball[:, 0] * length, ball[:, 1] * width
    with np.errstate(invalid="ignore"):
        past = np.maximum.reduce([-x, x - length, -y, y - width])     # metres past the nearest line
    restart_set = set(restarts)
    out = np.zeros(len(ball), bool)
    state = False
    for f in range(len(ball)):
        if f in restart_set:
            state = False
        if np.isfinite(past[f]):
            if past[f] > OUT_M:
                state = True
            elif past[f] < -OUT_M:
                state = False
        out[f] = state
    return out


def nearest_team(ball, players, team_of_track, pitch):
    length, width = pitch
    d = np.linalg.norm((players - ball[:, None, :]) * [length, width], axis=2)
    d = np.where(np.isfinite(d), d, np.inf)
    best = np.argmin(d, axis=1)
    known = np.isfinite(ball[:, 0]) & np.isfinite(d).any(axis=1)
    return np.where(known, team_of_track[best], NOBODY).astype(np.int8)


def smooth(codes, window):
    """Majority vote over +/- window frames among frames with an answer; NOBODY where none has one."""
    if window <= 0:
        return codes.copy()
    home = np.concatenate([[0], np.cumsum(codes == HOME)])
    away = np.concatenate([[0], np.cumsum(codes == AWAY)])
    i = np.arange(len(codes))
    lo, hi = np.maximum(0, i - window), np.minimum(len(codes), i + window + 1)
    h, a = home[hi] - home[lo], away[hi] - away[lo]
    out = np.full(len(codes), NOBODY, dtype=np.int8)
    seen = (h + a) > 0
    out[seen] = np.where(h[seen] >= a[seen], HOME, AWAY)
    return out


def stoppage_dead(path, n):
    """Dead frames from ballcv/stoppages.py: every stoppage from when play stopped (whistle, ball out) until its own
    restart (throw-in, goal kick, corner, free kick) -- or None without the file."""
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        stoppages = json.load(handle)["stoppages"]
    dead = np.zeros(n, bool)
    for st in stoppages:
        dead[st["start"]:st["end"] + 1] = True
    return dead


def stoppage_restarts(path):
    """[(frame, HOME|AWAY)]: each restart and the team taking it (the player at the ball), from ballcv/stoppages.py --
    the team taking a set piece has the ball."""
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        stoppages = json.load(handle)["stoppages"]
    return [(st["restart"], HOME if st["restart_team"] == "H" else AWAY)
            for st in stoppages if st["restart"] is not None and st.get("restart_team") in ("H", "A")]


def match_state(game=GAME, use_stoppages=True):
    """-> (team_of_track, Metrica ball, players, dead, stoppage whistles) for every tactical frame; dead = stoppage whistle
    or ball out of play (see the module docstring)."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    pitch = GAMES[game]["pitch"]
    n = source_frames("tactical", game)
    team_of_track = team_of_each_track(meta_path)
    print("reading {} ...".format(os.path.basename(raw_path)))
    metrica, players, speeds = read_atd(raw_path, n, len(team_of_track))
    stops, restarts = read_whistles(WHISTLES_CSV.format(game=game))

    dead = stoppage_dead(STOPPAGES_JSON.format(game=game), n) if use_stoppages else None
    if dead is None:
        dead = whistle_dead(n, stops, restarts, player_speed(speeds)) | out_of_play(metrica, restarts, pitch)
    return team_of_track, metrica, players, dead, stops


def predict(game=GAME, window=WINDOW, use_touches=True, use_stoppages=True):
    """-> (labels array of 'H'/'A'/'D'/'X', per-frame info dict for reporting)."""
    n = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]
    team_of_track, metrica, players, dead, stops = match_state(game, use_stoppages)
    ball = np.where(dead[:, None], np.nan, metrica)                 # no possession while dead

    codes = smooth(nearest_team(ball, players, team_of_track, pitch), window)
    touched = np.zeros(n, bool)
    touch_file = read_touches(TOUCHES_JSON.format(game=game)) if use_touches else None
    if touch_file:
        covered, changes = touch_file
        if use_stoppages:                                           # a restart: the team taking it has the ball
            changes = sorted(changes + [r for r in stoppage_restarts(STOPPAGES_JSON.format(game=game))
                                        if covered[0] <= r[0] <= covered[1]])
        by_touch = last_touch_codes(n, covered, changes)
        touched = by_touch != NOBODY
        codes = np.where(touched, by_touch, codes)                  # the last touch wins wherever we have one
    labels = np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D")).astype("<U1")
    labels[dead] = "D"
    in_half = np.zeros(n, bool)
    for _, first, last in half_video_blocks(game):
        in_half[first:last + 1] = True
    labels[~in_half] = "X"

    gap = ~np.isfinite(metrica[:, 0])
    info = {
        "graded": int(in_half.sum()),
        "metrica": int((~gap & in_half).sum()),
        "dead": int((dead & in_half).sum()),
        "stoppage_whistles": len(stops),
        "by_touch": int((touched & in_half).sum()),
    }
    return labels, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-touches", action="store_true", help="ignore our touches (nearest player everywhere)")
    parser.add_argument("--window", type=int, default=WINDOW, help="majority vote half-width in frames")
    parser.add_argument("--no-stoppages", action="store_true",
                        help="dead ball from whistles + out of play, not from ballcv/stoppages.py (each stoppage until its own restart)")
    parser.add_argument("--out", help="output CSV (default: evaluation/results/possession_<variant>.csv)")
    args = parser.parse_args()

    labels, info = predict(window=args.window, use_touches=not args.no_touches, use_stoppages=not args.no_stoppages)
    name = "possession_touches" if not args.no_touches else "possession_nearest_player"
    path = write_submission(args.out or os.path.join(RESULTS_DIR, name + ".csv"), labels)
    g = info["graded"]
    print("graded frames {}: Metrica ball {:.1%}, judged dead {:.1%}".format(g, info["metrica"] / g, info["dead"] / g))
    print("labelled from the last touch: {:.1%} of graded frames (the rest: nearest player)".format(info["by_touch"] / g))
    print("labels: " + "  ".join("{} {}".format(k, int(np.sum(labels == k))) for k in "HADX"))
    print("-> {}".format(os.path.relpath(path, ROOT)))
