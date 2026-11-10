"""Align every broadcast frame to the tactical timeline, and flag the replays to disregard.

    python alignment/align_broadcast.py     # writes data/broadcast/BAR-ATH_broadcast_to_tactical.csv

The broadcast recording is not frame-aligned to the tactical one. Two signals are combined:

* Tracking. Both ATDs report positions in the same normalised pitch coordinates, so a broadcast
  frame and the tactical frame showing the same moment have the same players in the same places.
  This gives the live offsets to the frame.
* Replay wipes. Every replay starts and ends with a full-screen LaLiga-red logo wipe (~4 frames of
  RGB 236,55,66). They are found by decoding the video at 32x18, once; the result is cached next
  to the output, so later runs only take the tracking time.

What it finds on BAR-ATH
------------------------
* Within each half the broadcast runs on the tactical clock: broadcast frame + 4014 in the 1st
  half, + 26920 in the 2nd. Any drift is at most one frame over a half -- the resolution of the
  tracking itself. The on-screen clocks of the two feeds agree to the second.
* Half-time is edited out: the file is the 1st-half recording followed by the 2nd-half recording,
  joined during aerial stadium shots that have no pitch tracking.
* Replays REPLACE live footage rather than being inserted, so they never shift the timeline. The
  on-screen match clock keeps running through them while the picture shows an earlier moment, so
  broadcast data from a replay is about the wrong moment and must be disregarded. A stretch between
  two wipes is a replay unless the tracking shows it matching live play.

Columns
-------
    tactical_frame  tactical frame at the same instant. Blank only inside the half-time join.
    kind            live    use it: the picture is the live moment at tactical_frame
                    replay  disregard: the picture is an earlier moment (clock still says now)
                    wipe    disregard: the LaLiga-red transition into or out of a replay
                    break   disregard: the half-time join, no tactical frame
    pitch_view      1 when the broadcast homography is valid (the pitch is in shot)
    cost            on live frames, metres between broadcast and tactical players (+ball); ~0.6 is a
                    good match, 3-5 is unrelated, blank when not measurable. A high cost on a live
                    frame usually means a rough broadcast homography, not a wrong frame.
"""

import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require  # noqa: E402

# Params
CLIP_M = 5.0  # a broadcast player further than this from every tactical player counts as 5 m
MIN_PLAYERS = 4  # fewer visible broadcast players than this and a frame is not scored at all
SAMPLE_EVERY = 250  # broadcast frames between samples in the offset search (10 s)
SAMPLE_MIN_PLAYERS = 8
COARSE_STEP = 5  # tactical frames between candidates in a brute-force search
CONFIDENT_COST = 1.0  # a brute-force match this good...
CONFIDENT_RATIO = 2.0  # ...and this much better than the best match elsewhere is trusted
OFFSET_REFINE = 10  # frames either side of a coarse offset to test exactly
LIVE_MAX_COST = 1.5  # smoothed cost at or below this means the picture is live
SMOOTH_FRAMES = 13
CHUNK = 4000

WIPE_RGB = (236, 55, 66)
WIPE_RGB_TOLERANCE = 60  # sum of absolute channel differences
WIPE_FRAME_RED = 0.5  # a frame this red is part of a wipe...
WIPE_PEAK_RED = 0.7  # ...and a wipe must peak this red (a VAR-review frame peaks at ~0.5)
WIPE_MERGE = 15  # red frames this close together are one wipe
SCAN_SIZE = (32, 18)
SCAN_JOBS = 8

REPLAY_LIVE_SHARE = 0.5  # between two wipes, below this share of live-matching frames is a replay
REPLAY_MAX_BLIND_S = 90  # with too little tracking to judge, a stretch up to this long is a replay

OUT_NAME = "BAR-ATH_broadcast_to_tactical.csv"
SCAN_CACHE = "BAR-ATH_broadcast_red_fraction.npy"


def load_source(source):
    """Players (n, slots, 2) and ball (n, 2) in metres, team per slot, homography valid flag."""
    meta, raw, homography = require(["atd_meta", "atd_raw", "homography"], source=source)
    length, width = GAMES[GAME]["pitch"]
    scale = np.array([length, width], np.float32)

    with open(raw) as handle:
        lines = [line for line in handle.read().split("\n") if line]
    n_slots = len(lines[0].split(":")[1].split(";"))
    players = np.full((len(lines), n_slots, 2), np.nan, np.float32)
    ball = np.full((len(lines), 2), np.nan, np.float32)
    for i, line in enumerate(lines):
        frame_id, part_players, part_ball = line.split(":")
        assert int(frame_id) == i + 1, "ATD frame ids must run 1..n"  # video_frame = frame_id - 1
        values = np.array(part_players.replace(";", ",").split(","), dtype=np.float32)
        players[i] = values.reshape(n_slots, 3)[:, :2]
        ball[i] = np.array(part_ball.split(",")[:2], dtype=np.float32)

    with open(meta) as handle:
        xml = handle.read()
    team_ids = re.findall(r'<Team\b[^>]*\bid="([^"]+)"', xml)
    slot_teams = re.findall(r'<Player\b[^>]*\bteamId="([^"]+)"', xml)
    team = np.array([team_ids.index(t) for t in slot_teams], np.int8)

    valid = pd.read_csv(homography, usecols=["valid"])["valid"].values.astype(bool)
    return players * scale, ball * scale, team, valid[: len(lines)]


class Matcher:
    """Cost of putting broadcast frame b on tactical frame t."""

    def __init__(self, tactical, broadcast):
        self.tp, self.tb, self.tteam, _ = tactical
        self.bp, self.bb, self.bteam, self.bvalid = broadcast
        self.tp_far = np.where(np.isfinite(self.tp), self.tp, 1e6).astype(np.float32)
        self.tb_far = np.where(np.isfinite(self.tb), self.tb, 1e6).astype(np.float32)
        self.same_team = self.bteam[:, None] == self.tteam[None, :]
        self.n_tactical, self.n_broadcast = len(self.tp), len(self.bp)
        self.n_seen = np.isfinite(self.bp[..., 0]).sum(1)

    def scorable(self, b):
        return self.bvalid[b] & (self.n_seen[b] >= MIN_PLAYERS)

    def pairs(self, bfs, tfs):
        """Cost per (b, t) pair, NaN where the broadcast frame is not scorable."""
        bfs, tfs = np.asarray(bfs), np.clip(np.asarray(tfs), 0, self.n_tactical - 1)
        out = np.full(len(bfs), np.nan, np.float32)
        for lo in range(0, len(bfs), CHUNK):
            b, t = bfs[lo:lo + CHUNK], tfs[lo:lo + CHUNK]
            bp = self.bp[b]
            d = np.sqrt(((bp[:, :, None, :] - self.tp_far[t][:, None, :, :]) ** 2).sum(-1))
            d = np.where(self.same_team[None], d, np.inf).min(-1)
            seen = np.isfinite(bp[..., 0])
            total = np.where(seen, np.minimum(d, CLIP_M), 0).sum(1)
            count = seen.sum(1).astype(np.float32)
            ball_seen = np.isfinite(self.bb[b, 0])
            d_ball = np.sqrt(((np.nan_to_num(self.bb[b]) - self.tb_far[t]) ** 2).sum(-1))
            total += np.where(ball_seen, 2 * np.minimum(d_ball, CLIP_M), 0)  # the ball counts twice
            count += 2 * ball_seen
            ok = self.scorable(b)
            out[lo:lo + CHUNK][ok] = total[ok] / count[ok]
        return out

    def search(self, b, lo=0, hi=None):
        """Best tactical frame for broadcast frame b in [lo, hi): (frame, cost, cost elsewhere)."""
        hi = self.n_tactical if hi is None else min(hi, self.n_tactical)
        candidates = np.arange(max(lo, 0), hi, COARSE_STEP)
        if len(candidates) == 0:
            return None
        pts, ok = self.bp[b], np.isfinite(self.bp[b, :, 0])
        acc = np.zeros(len(candidates), np.float32)
        for p, k in zip(pts[ok], self.bteam[ok]):
            team_pts = self.tp_far[candidates][:, self.tteam == k]
            acc += np.minimum(np.sqrt(((team_pts - p) ** 2).sum(-1)).min(1), CLIP_M)
        n = ok.sum()
        if np.isfinite(self.bb[b, 0]):
            acc += 2 * np.minimum(np.sqrt(((self.tb_far[candidates] - self.bb[b]) ** 2).sum(-1)), CLIP_M)
            n += 2
        acc /= n
        i = int(np.argmin(acc))
        # refine around the coarse winner at full frame resolution
        fine = np.arange(candidates[i] - COARSE_STEP, candidates[i] + COARSE_STEP + 1)
        fine = fine[(fine >= max(lo, 0)) & (fine < hi)]
        fine_cost = self.pairs(np.full(len(fine), b), fine)
        j = int(np.nanargmin(fine_cost))
        elsewhere = acc[np.abs(candidates - candidates[i]) > 50]
        return int(fine[j]), float(fine_cost[j]), float(elsewhere.min()) if len(elsewhere) else np.inf


def find_offsets(m):
    """The two live offsets (one per half), from brute-force matches of sampled broadcast frames."""
    samples = [b for b in range(0, m.n_broadcast, SAMPLE_EVERY)
               if m.bvalid[b] and m.n_seen[b] >= SAMPLE_MIN_PLAYERS]
    found = []
    for b in samples:
        t, cost, elsewhere = m.search(b)
        if cost <= CONFIDENT_COST and elsewhere / cost >= CONFIDENT_RATIO:
            found.append((b, t - b))
    found = np.array(found)
    print("offset search: {} samples, {} confident".format(len(samples), len(found)))

    # The two most common offsets (to within a second) are the halves; everything else is replays.
    bins = np.round(found[:, 1] / 25).astype(int)
    values, counts = np.unique(bins, return_counts=True)
    top = values[np.argsort(-counts)[:2]]
    halves = []
    for v in top:
        rows = found[bins == v]
        coarse = int(np.median(rows[:, 1]))
        b_lo, b_hi = int(rows[:, 0].min()), int(rows[:, 0].max())
        bfs = np.arange(b_lo, b_hi + 1)
        bfs = bfs[m.scorable(bfs)]
        totals = [np.nanmean(m.pairs(bfs, bfs + o)) for o in range(coarse - OFFSET_REFINE, coarse + OFFSET_REFINE + 1)]
        exact = coarse - OFFSET_REFINE + int(np.nanargmin(totals))
        halves.append((b_lo, b_hi, exact, len(rows)))
    halves.sort()
    for b_lo, b_hi, offset, n in halves:
        print("  live offset {:+d}  (broadcast {}..{}, {} samples)".format(offset, b_lo, b_hi, n))
    return halves




def runs(mask):
    """(start, end_inclusive) of each True run."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def tool(name):
    return os.environ.get("{}_PATH".format(name.upper())) or shutil.which(name) or "/opt/homebrew/bin/" + name


def video_frame_count(video):
    """Frames in the video file itself. The broadcast ATD has one record more than the video."""
    out = subprocess.run([tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_frames", "-of", "csv=p=0", video], check=True, capture_output=True, text=True)
    return int(out.stdout.strip())


def red_fraction(video, cache):
    """Share of each frame's pixels that are wipe red, from a 32x18 decode. Cached."""
    n_frames = video_frame_count(video)
    if os.path.exists(cache):
        red = np.load(cache)
        if len(red) == n_frames:
            return red
    ffmpeg = tool("ffmpeg")
    width, height = SCAN_SIZE
    fps = GAMES[GAME]["fps"]
    step = -(-n_frames // SCAN_JOBS)

    def scan(start):
        count = min(step, n_frames - start)
        # n / fps has at most two decimals at 25 fps, so the input seek is frame-exact
        raw = subprocess.run(
            [ffmpeg, "-nostdin", "-v", "error", "-ss", "{:.2f}".format(start / fps), "-i", video,
             "-frames:v", str(count), "-vf", "scale={}:{}".format(width, height),
             "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
            check=True, capture_output=True).stdout
        frames = np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3).astype(np.int16)
        assert len(frames) == count, "decoded {} frames from {}, expected {}".format(len(frames), start, count)
        return (np.abs(frames - np.array(WIPE_RGB)).sum(-1) < WIPE_RGB_TOLERANCE).mean((1, 2))

    print("scanning the broadcast video for replay wipes ({} parallel decodes)...".format(SCAN_JOBS))
    with ThreadPoolExecutor(SCAN_JOBS) as pool:
        red = np.concatenate(list(pool.map(scan, range(0, n_frames, step)))).astype(np.float32)
    np.save(cache, red)
    return red


def find_wipes(red):
    """(first, last) frame of each replay wipe."""
    wipes = []
    for f in np.flatnonzero(red >= WIPE_FRAME_RED):
        if wipes and f - wipes[-1][1] <= WIPE_MERGE:
            wipes[-1][1] = f
        else:
            wipes.append([f, f])
    return [(int(s), int(e)) for s, e in wipes if red[s:e + 1].max() >= WIPE_PEAK_RED]


def align():
    print("loading tracking...")
    m = Matcher(load_source("tactical"), load_source("broadcast"))
    (_, _, off1, _), (_, _, off2, _) = find_offsets(m)
    every = np.arange(m.n_broadcast)
    scorable = m.scorable(every)

    # The half-time join: the last frame that matches live under the 1st-half offset, and the first
    # after it that matches under the 2nd.
    cost1 = pd.Series(m.pairs(every, every + off1)).rolling(SMOOTH_FRAMES, center=True, min_periods=1).median().values
    cost2 = pd.Series(m.pairs(every, every + off2)).rolling(SMOOTH_FRAMES, center=True, min_periods=1).median().values
    last1 = int(np.flatnonzero(cost1 <= LIVE_MAX_COST).max())
    first2 = int(np.flatnonzero((cost2 <= LIVE_MAX_COST) & (every > last1)).min())
    print("half-time join: 1st half matches up to broadcast {}, 2nd half from {}".format(last1, first2))

    live_t = np.full(m.n_broadcast, -1, np.int64)
    live_t[: last1 + 1] = every[: last1 + 1] + off1
    live_t[first2:] = every[first2:] + off2
    in_break = (every > last1) & (every < first2)
    live_match = np.where(every <= last1, cost1, cost2) <= LIVE_MAX_COST

    # Replays: the stretch between two wipes, unless the tracking says it is live.
    video, = require(["video"], source="broadcast")
    red = red_fraction(video, os.path.join(os.path.dirname(video), SCAN_CACHE))
    if len(red) < m.n_broadcast:  # the ATD can carry one record more than the video has frames
        red = np.concatenate([red, np.zeros(m.n_broadcast - len(red), red.dtype)])
    wipes = find_wipes(red)
    print("replay wipes: {}".format(len(wipes)))

    kind = np.full(m.n_broadcast, "live", dtype=object)
    for s, e in wipes:
        kind[s:e + 1] = "wipe"
    replays = []
    fps = GAMES[GAME]["fps"]
    for (_, e0), (s1, _) in zip(wipes, wipes[1:]):
        frames = np.arange(e0 + 1, s1)
        frames = frames[~in_break[frames]]
        if len(frames) == 0:
            continue
        judged = scorable[frames].sum()
        if judged >= 2 * SMOOTH_FRAMES:
            is_replay = live_match[frames][scorable[frames]].mean() < REPLAY_LIVE_SHARE
        else:
            is_replay = len(frames) <= REPLAY_MAX_BLIND_S * fps
        if is_replay:
            kind[frames] = "replay"
            replays.append(frames)
    kind[in_break] = "break"

    cost = np.full(m.n_broadcast, np.nan, np.float32)
    live = kind == "live"
    cost[live] = m.pairs(every[live], live_t[live])
    return every, live_t, kind, m.bvalid, cost, (off1, off2, last1, first2), replays


def clock(t):
    """Approximate match clock of a tactical frame (from each half's kickoff), for printing."""
    fps = GAMES[GAME]["fps"]
    first, second = 11185, 113742  # kickoff video frames, DATA.md / starter/load_smart_tagging.py
    s = (t - first) / fps if t < second - 5 * 60 * fps else 45 * 60 + (t - second) / fps
    return "{:d}:{:02d}".format(int(s // 60), int(s % 60)) if s >= 0 else "-"


def main():
    every, live_t, kind, view, cost, (off1, off2, last1, first2), replays = align()
    out_dir = os.path.dirname(require(["atd_raw"], source="broadcast")[0])
    out_path = os.path.join(out_dir, OUT_NAME)
    pd.DataFrame({
        "broadcast_frame": every,
        "tactical_frame": pd.array(np.where(live_t >= 0, live_t, pd.NA), dtype="Int64"),
        "kind": kind,
        "pitch_view": view.astype(np.int8),
        "cost": np.round(cost, 3),
    }).to_csv(out_path, index=False)

    fps = GAMES[GAME]["fps"]
    print("\nreplays to disregard: {}".format(len(replays)))
    for frames in replays:
        print("  broadcast {:6d}-{:6d}  {:5.1f} s  at {}".format(
            frames[0], frames[-1], len(frames) / fps, clock(live_t[frames[0]])))
    print("\nframes by kind:")
    for k, n in pd.Series(kind).value_counts().items():
        print("  {:<7} {:>7}  {:5.1f}%".format(k, n, 100 * n / len(kind)))
    live = kind == "live"
    good = np.isfinite(cost) & live
    print("live frames with a measurable match: {}, median cost {:.2f} m".format(good.sum(), np.median(cost[good])))
    print("\nlive offsets: 1st half {:+d}, 2nd half {:+d}; half-time join between broadcast {} and {}".format(off1, off2, last1, first2))
    print("wrote {}".format(out_path))


if __name__ == "__main__":
    main()
