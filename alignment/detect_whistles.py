"""Find the referee's whistle in the broadcast audio, and put each one on the tactical timeline.

    python alignment/detect_whistles.py     # writes data/broadcast/BAR-ATH_broadcast_whistles.csv (and _whistle_pips.csv)

Needs data/broadcast/BAR-ATH_broadcast_to_tactical.csv, so run alignment/align_broadcast.py first, and the tactical
smart tags (its kick-offs).

The tactical video has no audio; the broadcast does, and a whistle marks most stoppages and
restarts. See writeup.md for how the method was arrived at and how well it does.

What a whistle looks like
-------------------------
Nothing here is told what the referee's whistle sounds like: it is learned from the match's kick-offs, which he
always whistles and the smart tags mark. All that is assumed is that a whistle is a tonal sound -- a line narrow in
frequency that starts and is held -- which the rest of the mix is not:

* Commentary harmonics bend with the voice instead of staying flat.
* Crowd noise is broadband, so no single frequency stands out from its neighbours.
* The mix carries constant narrowband lines (one near 3.6-3.8 kHz). They are flat and narrow, but they never
  start or stop, so they are removed by requiring a line to be louder than the same frequency a couple of
  seconds either side.

Per STFT column (16 ms hop), per frequency, the line is built in three steps:

1. prominence  dB above the median of the ±160 Hz around it            (narrow in frequency)
2. novelty     dB above the median of the same bin over ±2 s           (it has an onset)
3. line        min(1, 2), then a ~0.24 s median along time            (held at one pitch)

His whistle, learned from the kick-offs (learn_whistle), over SEARCH_HZ:

4. where       the frequencies whose line stands out more in the KICKOFF_AFTER_S after a kick-off tag than in
               other windows that long: per kick-off, the share of windows at least as high (each frequency widened
               by the prominence width, as his pitch wobbles); the kick-offs combined by Fisher's method; and
               significant where above the 1 - ALPHA quantile of the strongest frequency of NULL_DRAWS random sets
               of as many windows -- so the many neighbouring, correlated frequencies are not counted as many tests
5. his tones   the run, per kick-off window, where those frequencies peak (if it reaches SCORE_MIN), and the lines
               it holds there; his tones are the ones held at at least half the kick-offs -- as many as there are
6. band        his tones, ± the prominence width

The score, per column: the strongest line in the band that sits on another of his tones than the strongest one does
(at least half their smallest spacing from it) -- all of his tones at once, as his whistle sounds them.

A whistle is a run of columns with score >= SCORE_MIN, runs closer than MERGE_S joined, at least
MIN_DURATION_S long -- and the referee's: most of its columns on one of his tones (fans' whistles are pitched
anywhere, and jump between pitches). A shorter run on his tones is a pip.

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

import multiprocessing
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d, median_filter
from scipy.signal import stft
from scipy.stats import chi2

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
SEARCH_HZ = (200, 7800)  # where his whistle is looked for: the audio's 8 kHz, but for its ends
PROMINENCE_BINS = 41  # ±160 Hz
NOVELTY_COLS = 251  # ±2 s
HOLD_COLS = 15  # ~0.24 s
WIDTH_HZ = PROMINENCE_BINS // 2 * SAMPLE_RATE / NPERSEG  # a line's width: the prominence neighbourhood
SCORE_MIN = 12.0
MERGE_S = 0.3
MIN_DURATION_S = 0.15
CHUNK_S = 120  # STFT in chunks, each padded by PAD_S so the filters see across the seams
PAD_S = 6
KICKOFF_AFTER_S = 25  # a kick-off tag leads his whistle by up to ~20 s
NULL_DRAWS = 2000
ALPHA = 0.05
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


def tonal_lines(audio, c0, band):
    """(time in s, line in dB [frequency x column], frequencies) of the chunk of audio starting at sample c0, over
    band (steps 1-3)."""
    lo, hi = band
    chunk, pad = CHUNK_S * SAMPLE_RATE, PAD_S * SAMPLE_RATE
    a = max(0, c0 - pad)
    f, t, z = stft(audio[a:c0 + chunk + pad], SAMPLE_RATE, nperseg=NPERSEG, noverlap=NPERSEG - HOP,
                   boundary=None, padded=False)
    near = (f >= lo - 300) & (f <= hi + 300)  # margin so the prominence median has neighbours
    db = 20 * np.log10(np.abs(z[near]) + 1e-9)
    prominence = db - median_filter(db, size=(PROMINENCE_BINS, 1))
    novelty = db - median_filter(db, size=(1, NOVELTY_COLS))
    line = median_filter(np.minimum(prominence, novelty), size=(1, HOLD_COLS))
    inside = (f[near] >= lo) & (f[near] <= hi)
    t = t + a / SAMPLE_RATE
    keep = (t >= c0 / SAMPLE_RATE) & (t < (c0 + chunk) / SAMPLE_RATE)
    return t[keep], line[inside][:, keep], f[near][inside]


def whistle_score(audio, band, second_hz):
    """(time in s, score in dB, pitch of the strongest tone in Hz) per STFT column: the strongest line in band at least
    second_hz from the strongest."""
    times, scores, pitches = [], [], []
    for c0 in range(0, len(audio), CHUNK_S * SAMPLE_RATE):
        t, line, freqs = tonal_lines(audio, c0, band)
        strongest = line.argmax(0)
        far = np.abs(freqs[:, None] - freqs[strongest][None, :]) >= second_hz
        second = np.where(far, maximum_filter1d(line, 3, axis=0), -np.inf).max(0)
        times.append(t)
        scores.append(second)
        pitches.append(freqs[strongest])
    return np.concatenate(times), np.concatenate(scores), np.concatenate(pitches)


def kickoff_tags(smart_tagging):
    """Tactical frames of the tagged kick-offs (starting a half or after a goal)."""
    out = []
    for i in ET.parse(smart_tagging).getroot().iter("instance"):
        kind = {l.findtext("group"): l.findtext("text") for l in i.findall("label")}.get("Type") or ""
        if i.findtext("code") == "SET PIECES" and kind.startswith("Kick Off"):
            out.append(round(float(i.findtext("start")) * 25))
    return sorted(out)


_AUDIO = None


def _search_chunk(c0, window, kickoffs):
    """For one chunk over SEARCH_HZ: {window index: (columns, peak line per frequency)} of the match cut into windows
    of `window` columns, and {kick-off: [(column, line)]} of the columns in each kick-off's window."""
    t, line, _ = tonal_lines(_AUDIO, c0, SEARCH_HZ)
    col = np.round((t - NPERSEG / 2 / SAMPLE_RATE) * SAMPLE_RATE / HOP).astype(int)
    line = line.astype(np.float32)
    tiles = {}
    for k in np.unique(col // window):
        m = col // window == k
        tiles[k] = (int(m.sum()), line[:, m].max(1))
    near = {i: [(c, line[:, j]) for j, c in enumerate(col) if k0 <= c < k0 + window] for i, k0 in enumerate(kickoffs)}
    return tiles, near


def learn_whistle(audio, kickoffs_s):
    """His tones (Hz), learned from the kick-offs at kickoffs_s (broadcast seconds); steps 4-5."""
    global _AUDIO
    _AUDIO = audio
    hop = HOP / SAMPLE_RATE
    window = int(round(KICKOFF_AFTER_S / hop))
    kickoffs = [int(np.ceil((k - NPERSEG / 2 / SAMPLE_RATE) / hop)) for k in kickoffs_s]   # first column at the tag
    with multiprocessing.get_context("fork").Pool() as pool:
        parts = pool.starmap(_search_chunk, [(c0, window, kickoffs)
                                             for c0 in range(0, len(audio), CHUNK_S * SAMPLE_RATE)])
    freqs = tonal_lines(audio[:NPERSEG * 4], 0, SEARCH_HZ)[2]
    tiles = {}
    for part, _ in parts:
        for k, (n, peak) in part.items():
            m, p = tiles.get(k, (0, peak))
            tiles[k] = (m + n, np.maximum(p, peak))
    peak = np.array([p for k, (n, p) in sorted(tiles.items()) if n == window])   # windows x frequencies
    segs = [np.array([l for _c, l in sorted(sum((near[i] for _, near in parts), []), key=lambda x: x[0])])
            for i in range(len(kickoffs))]                                             # per kick-off: columns x freqs

    w = PROMINENCE_BINS // 2
    widen = lambda x: maximum_filter1d(x, 2 * w + 1, axis=-1)
    P = widen(peak)
    K = widen(np.array([s.max(0) for s in segs]))
    share = lambda x: (1 + (P[None] >= x[:, None]).sum(1)) / (1 + len(P))     # share of windows at least as high
    fisher = -2 * np.log(share(P))
    stat = -2 * np.log(share(K)).sum(0)
    rng = np.random.default_rng(0)
    null = [fisher[rng.choice(len(P), len(K), replace=False)].sum(0).max() for _ in range(NULL_DRAWS)]
    where = widen((stat > np.quantile(null, 1 - ALPHA)).astype(np.uint8)) > 0

    held, heard = [], 0
    for k, seg in zip(kickoffs_s, segs):
        s = seg[:, where].max(1) if where.any() else np.zeros(len(seg))
        c = int(s.argmax())
        if s[c] < SCORE_MIN:
            print("  kick-off at {:7.1f} s: no whistle".format(k))
            continue
        a, e = c, c
        while a > 0 and s[a - 1] >= SCORE_MIN:
            a -= 1
        while e < len(s) - 1 and s[e + 1] >= SCORE_MIN:
            e += 1
        profile = seg[a:e + 1].max(0)
        lines = [freqs[j] for j in range(len(profile)) if where[j] and profile[j] >= SCORE_MIN
                 and profile[j] == profile[max(0, j - w):j + w + 1].max()]
        heard += 1
        held += lines
        print("  kick-off at {:7.1f} s: whistle {:+.2f} s, {:.2f} s long, lines at {} Hz".format(
            k, a * hop, (e - a + 1) * hop, [int(x) for x in lines]))
    held = np.sort(held)
    groups = np.split(held, np.flatnonzero(np.diff(held) > WIDTH_HZ) + 1) if len(held) else []
    return np.array([np.median(g) for g in groups if len(g) >= heard / 2])


def find_whistles(t, score, pitch, tones):
    """-> (whistles, pips), each [(start_s, end_s, peak score, median pitch)]. Whistles: runs of whistle-like sound held
    for MIN_DURATION_S, and of those, the referee's: most of its columns on one of his tones. Pips: his too, but shorter
    -- alone a pip may be anything; where something else says play stopped (ballcv/stoppages.py), it is his whistle,
    and says when."""
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
    his = lambda s, e: np.mean(np.min(np.abs(held(s, e)[:, None] - tones[None]), axis=1) <= WIDTH_HZ) > 0.5
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
    atd_raw, smart_tagging = require(["atd_raw", "smart_tagging"], source="tactical")
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
    print("learning his whistle from the kick-offs...")
    kickoffs_s = [int(np.nanargmin(np.abs(to_tactical - f))) / fps for f in kickoff_tags(smart_tagging)]
    tones = learn_whistle(audio, kickoffs_s)
    band = (tones.min() - WIDTH_HZ, tones.max() + WIDTH_HZ)
    second_hz = np.diff(tones).min() / 2 if len(tones) > 1 else 0
    print("his tones: {} Hz; band {:.0f}-{:.0f} Hz; a second tone {:.0f} Hz from the first".format(
        [int(round(x)) for x in tones], *band, second_hz))
    print("scoring {:.0f} s of audio...".format(len(audio) / SAMPLE_RATE))
    whistles, pips = find_whistles(*whistle_score(audio, band, second_hz), tones)
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
