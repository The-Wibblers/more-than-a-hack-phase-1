"""Find the referee's whistle in the broadcast audio, and put each one on the tactical timeline.

    python alignment/detect_whistles.py     # writes data/broadcast/BAR-ATH_broadcast_whistles.csv (and _whistle_pips.csv)

Needs data/broadcast/BAR-ATH_broadcast_to_tactical.csv, so run alignment/align_broadcast.py first.

The tactical video has no audio; the broadcast does, and a whistle marks most stoppages and
restarts. See writeup.md for how the method was arrived at and how well it does.

What a whistle looks like
-------------------------
A pea whistle is two to three flat, parallel tones between ~3.3 and 5.5 kHz (on BAR-ATH typically
~3.8, 4.25 and 4.7 kHz) held for 0.2-1.5 s. Nothing else in the mix looks like that:

* Commentary is below ~3.5 kHz, and its harmonics bend with the voice instead of staying flat.
* Crowd noise is broadband, so no single frequency stands out from its neighbours.
* The mix carries a constant narrowband line near 3.6-3.8 kHz. It is flat and narrow, but it never
  starts or stops, so it is removed by requiring a line to be louder than the same frequency a
  couple of seconds either side.

Per STFT column (16 ms hop) the score is built in four steps:

1. prominence  dB above the median of the ±160 Hz around it            (narrow in frequency)
2. novelty     dB above the median of the same bin over ±2 s           (it has an onset)
3. line        min(1, 2), then a ~0.24 s median along time            (held at one pitch)
4. score       the strongest line at least 250 Hz from the strongest  (two tones at once)

A whistle is a run of columns with score >= SCORE_MIN, runs closer than MERGE_S joined, at least
MIN_DURATION_S long -- and the referee's: most of its columns on one of his two tones, the two pitches the match's
whistle sound holds most often (fans' whistles are pitched anywhere in the band, and jump between pitches).

Columns
-------
    start_s, end_s      seconds into the broadcast video
    broadcast_frame     broadcast video frame of the start
    tactical_frame      tactical frame of the start (and _end, of the end), via the alignment
    kind                the broadcast picture at the start: live / replay / wipe / break. Audio in
                        a replay is not reliably live, so treat non-live whistles with suspicion.
                        In the half-time break the alignment has no tactical frame; the 2nd-half
                        offset is extrapolated backwards so the 2nd-half kickoff whistle (which
                        falls just inside the break) still lands on the timeline.
    score               peak score, dB. ~12 is the threshold, 20+ is a clear whistle
    peak_hz             median pitch of the strongest tone
    speed_before        mean tactical player speed (m/s) over the SPEED_BEFORE_S before the whistle
    speed_after         ...and over SPEED_AFTER_S after it
    role                stoppage  players slow down: play was live and the whistle stopped it
                        restart   players speed up: play was dead and the whistle restarted it
                                  (kickoffs, some free kicks and corners)
                        unclear   speed changes by less than ROLE_DELTA either way, e.g. a
                                  second whistle inside a stoppage. Also when there is no
                                  tracking (the half-time break).

Stoppage or restart
-------------------
A referee whistles both to stop play and to restart it, and the sound is the same. Whether players
slow down or speed up around the whistle tells the two apart. Players run at 2-4 m/s before a
stoppage and are near 1 m/s after it; before a restart they are standing.
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d, median_filter
from scipy.signal import stft

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from align_broadcast import OUT_NAME as ALIGNMENT_NAME, clock, tool  # noqa: E402
from games import GAMES, GAME, require  # noqa: E402

# Params
SAMPLE_RATE = 16000
NPERSEG = 2048  # 7.8 Hz bins: whistle tones are a few bins wide
HOP = 256  # 16 ms
BAND_HZ = (3300, 5500)
PROMINENCE_BINS = 41  # ±160 Hz
NOVELTY_COLS = 251  # ±2 s
HOLD_COLS = 15  # ~0.24 s
SECOND_TONE_HZ = 250
SCORE_MIN = 12.0
MERGE_S = 0.3
MIN_DURATION_S = 0.15
CHUNK_S = 120  # STFT in chunks, each padded by PAD_S so the filters see across the seams
PAD_S = 6
SPEED_BEFORE_S = (-3, 0)  # seconds around the whistle start
SPEED_AFTER_S = (2, 6)  # skips the first 2 s, while players are still reacting
ROLE_DELTA = 0.3  # m/s

OUT_NAME = "BAR-ATH_broadcast_whistles.csv"
PIPS_NAME = "BAR-ATH_broadcast_whistle_pips.csv"


def load_audio(video):
    """Mono float32 audio at SAMPLE_RATE, decoded by ffmpeg."""
    raw = subprocess.run(
        [tool("ffmpeg"), "-nostdin", "-v", "error", "-i", video, "-vn", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
        check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


def whistle_score(audio):
    """(time in s, score in dB, pitch of the strongest tone in Hz) per STFT column."""
    lo, hi = BAND_HZ
    chunk, pad = CHUNK_S * SAMPLE_RATE, PAD_S * SAMPLE_RATE
    times, scores, pitches = [], [], []
    for c0 in range(0, len(audio), chunk):
        a = max(0, c0 - pad)
        f, t, z = stft(audio[a:c0 + chunk + pad], SAMPLE_RATE, nperseg=NPERSEG, noverlap=NPERSEG - HOP,
                       boundary=None, padded=False)
        band = (f >= lo - 300) & (f <= hi + 300)  # margin so the prominence median has neighbours
        db = 20 * np.log10(np.abs(z[band]) + 1e-9)
        prominence = db - median_filter(db, size=(PROMINENCE_BINS, 1))
        novelty = db - median_filter(db, size=(1, NOVELTY_COLS))
        line = median_filter(np.minimum(prominence, novelty), size=(1, HOLD_COLS))

        inside = (f[band] >= lo) & (f[band] <= hi)
        line, freqs = line[inside], f[band][inside]
        strongest = line.argmax(0)
        far = np.abs(freqs[:, None] - freqs[strongest][None, :]) >= SECOND_TONE_HZ
        second = np.where(far, maximum_filter1d(line, 3, axis=0), -np.inf).max(0)

        t = t + a / SAMPLE_RATE
        keep = (t >= c0 / SAMPLE_RATE) & (t < (c0 + chunk) / SAMPLE_RATE)
        times.append(t[keep])
        scores.append(second[keep])
        pitches.append(freqs[strongest][keep])
    return np.concatenate(times), np.concatenate(scores), np.concatenate(pitches)


def find_whistles(t, score, pitch):
    """-> (whistles, pips), each [(start_s, end_s, peak score, median pitch)]. Whistles: runs of whistle-like sound held
    for MIN_DURATION_S, and of those, the referee's. Pips: his too, but shorter -- alone a pip may be anything; where
    something else says play stopped (ballcv/stoppages.py), it is his whistle, and says when. He blows the same whistle all match, while the crowd's whistles scatter over the band:
    his two tones are the two pitches the match's whistle sound holds most often (each counting the columns within the
    prominence width of it; the second at least SECOND_TONE_HZ from the first), and a run is his when most of its
    columns sit on one of them."""
    edges = np.diff(np.concatenate(([0], (score >= SCORE_MIN).astype(np.int8), [0])))
    merged = []
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1):
        if merged and t[s] - t[merged[-1][1]] < MERGE_S:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    hop = HOP / SAMPLE_RATE                                      # a run of columns lasts to the end of its last one
    runs = [(s, e) for s, e in merged if t[e] - t[s] + hop >= MIN_DURATION_S]
    short = [(s, e) for s, e in merged if t[e] - t[s] + hop < MIN_DURATION_S]
    held = lambda s, e: pitch[s:e + 1][score[s:e + 1] >= SCORE_MIN]
    width = PROMINENCE_BINS // 2 * SAMPLE_RATE / NPERSEG
    heard = np.concatenate([held(s, e) for s, e in runs])
    grid = np.unique(heard)
    support = np.array([np.sum(np.abs(heard - g) <= width) for g in grid])
    first = grid[np.argmax(support)]
    far = np.abs(grid - first) >= SECOND_TONE_HZ
    tones = np.array([first, grid[far][np.argmax(support[far])]])
    print("the referee's tones: {:.0f} Hz and {:.0f} Hz".format(*tones))
    his = lambda s, e: np.mean(np.min(np.abs(held(s, e)[:, None] - tones[None]), axis=1) <= width) > 0.5
    found = lambda rs: [(t[s], t[e], float(score[s:e + 1].max()), float(np.median(pitch[s:e + 1]))) for s, e in rs if his(s, e)]
    return found(runs), found(short)


def player_speed(atd_raw):
    """Mean speed (m/s) of the tracked players on each tactical frame, NaN where none has one."""
    out = []
    with open(atd_raw) as handle:
        for line in handle:
            if not line.strip():
                continue
            values = np.array([p.split(",")[2] for p in line.split(":")[1].split(";")], dtype=np.float32)
            seen = np.isfinite(values)
            out.append(values[seen].mean() if seen.any() else np.nan)
    return np.array(out, np.float32)


def role(speed, frame, fps):
    """(speed before, speed after, stoppage / restart / unclear) around a whistle at frame."""
    def mean(window):
        lo, hi = frame + window[0] * fps, frame + window[1] * fps
        if lo < 0 or hi > len(speed):
            return np.nan
        return float(np.nanmean(speed[lo:hi])) if np.isfinite(speed[lo:hi]).any() else np.nan
    before, after = mean(SPEED_BEFORE_S), mean(SPEED_AFTER_S)
    if not (np.isfinite(before) and np.isfinite(after)):
        return before, after, "unclear"
    if after < before - ROLE_DELTA:
        return before, after, "stoppage"
    if after > before + ROLE_DELTA:
        return before, after, "restart"
    return before, after, "unclear"


def tactical_frames(alignment):
    """Tactical frame per broadcast frame, with the half-time break extrapolated (see Columns)."""
    tf = alignment["tactical_frame"].to_numpy(dtype="float64", copy=True)
    blank = np.isnan(tf)
    if blank.any() and not blank.all():
        b = np.arange(len(tf))
        after = np.flatnonzero(~blank & (b > np.flatnonzero(blank)[0]))
        if len(after):
            first = after[0]
            gap = blank & (b < first)
            tf[gap] = b[gap] + (tf[first] - first)
    return tf


def main():
    video, = require(["video"], source="broadcast")
    atd_raw, = require(["atd_raw"], source="tactical")
    out_dir = os.path.dirname(video)
    alignment_path = os.path.join(out_dir, ALIGNMENT_NAME)
    if not os.path.exists(alignment_path):
        sys.exit("missing {}; run alignment/align_broadcast.py first".format(alignment_path))
    alignment = pd.read_csv(alignment_path)
    to_tactical = tactical_frames(alignment)
    kinds = alignment["kind"].values
    fps = GAMES[GAME]["fps"]

    print("decoding broadcast audio...")
    audio = load_audio(video)
    print("scoring {:.0f} s of audio...".format(len(audio) / SAMPLE_RATE))
    whistles, pips = find_whistles(*whistle_score(audio))
    print("loading tactical player speeds...")
    speed = player_speed(atd_raw)

    rows = []
    for start, end, peak, hz in whistles:
        b0 = min(int(round(start * fps)), len(alignment) - 1)
        b1 = min(int(round(end * fps)), len(alignment) - 1)
        before, after, kind_of_whistle = role(speed, int(to_tactical[b0]), fps)
        rows.append({
            "start_s": round(start, 2),
            "end_s": round(end, 2),
            "broadcast_frame": b0,
            "tactical_frame": int(to_tactical[b0]),
            "tactical_frame_end": int(to_tactical[b1]),
            "kind": kinds[b0],
            "score": round(peak, 1),
            "peak_hz": int(round(hz)),
            "speed_before": round(before, 2),
            "speed_after": round(after, 2),
            "role": kind_of_whistle,
        })
    out = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, OUT_NAME)
    out.to_csv(out_path, index=False)
    pip_rows = []
    for start, _end, peak, hz in pips:
        b0 = min(int(round(start * fps)), len(alignment) - 1)
        pip_rows.append({"start_s": round(start, 2), "broadcast_frame": b0, "tactical_frame": int(to_tactical[b0]),
                         "kind": kinds[b0], "score": round(peak, 1), "peak_hz": int(round(hz))})
    pd.DataFrame(pip_rows).to_csv(os.path.join(out_dir, PIPS_NAME), index=False)
    print("referee's pips (shorter than a whistle): {} -> {}".format(len(pip_rows), PIPS_NAME))

    print("\nwhistles: {}".format(len(out)))
    print(pd.crosstab(out["role"], out["kind"]).to_string())
    for r in out.itertuples():
        print("  {:>6}  {:7.2f} s  {:4.2f} s  score {:4.1f}  {:>5} Hz  {:<6}  {:4.2f} -> {:4.2f} m/s  {}".format(
            clock(r.tactical_frame), r.start_s, r.end_s - r.start_s, r.score, r.peak_hz, r.kind,
            r.speed_before, r.speed_after, r.role))
    print("wrote {}".format(out_path))


if __name__ == "__main__":
    main()
