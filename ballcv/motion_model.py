"""A physical motion model of the ball, fitted to our own track. Contacts are where the physics stops explaining it.

Between contacts the ball is in one of two modes, each with known physics:
  roll    on the ground: a nearly constant velocity (friction slows it a little)
  flight  in the air under gravity: the point straight below it moves in a line, its height is a parabola
The tactical camera turns about a fixed centre CT, so a ball at height h is seen at the ground point g where the ray
from CT through the ball meets the pitch, and the point below the ball is B = g + (CT - g) h / CT_z. With g known
from the pixel, a flight is LINEAR in its six unknowns (x0, y0, vx, vy, h0, vh), like a roll in its six
(x0, y0, vx, vy, ax, ay): every candidate stretch is scored by weighted linear least squares.

Every decision is a model comparison scored the same way: misfit in units of the calibrated measurement noise (chi^2,
i.e. -2 log likelihood) plus ln(N) for every extra free parameter (BIC, N = observations in the track):
  contact    the track is cut into the cheapest sequence of stretches (dynamic programming); a new stretch has six
             parameters, so a cut is made only where it saves more than 6 ln N. Each cut is a contact, timed at the
             first frame on the new path. Dropouts are only frames without data, so a contact next to one is found.
  two?       one contact between two stretches means their paths meet somewhere in the gap between them; if they
             can't (within the noise) the ball made a jump of its own in between -- e.g. a pass blocked a frame
             later -- which costs one more stretch: two contacts.
  outlier    a lone detection off the path: smooth path vs. that point free (2 parameters) vs. a contact there.
There is no threshold to tune. What the model learns -- the measurement noise and how a rolling ball slows down -- is
calibrated from stretches away from every player (no contact possible), and again once the model has found contacts.

  fit      calibrate -> work/align/motion_model.json
  breaks   cut the track with the calibrated model -> work/align/motion_breaks.json (ownball.py events, BREAKS=model)
"""
import json, sys
import numpy as np

import ownball as ob
from ownball import CT, G, L, W

FPS = 25.0
MAX_GAP = ob.MAX_SKIP      # a longer dropout splits the track into separate stretches
MAX_SEG = 250              # longest stretch scored (track points, 10 s); far longer than any free run of the ball
MIN_FLIGHT = 4             # points needed to fit a flight (two of its six unknowns are the height's)
H_FLOOR = -0.3             # a flight may not go further below the ground than this (m)
H_CEIL = 30.0
FREE_M = 2.0               # calibration: the ball at least this far from every player ...
CLEAN_MIN = 15             # ... for at least this many consecutive tracked frames, no break within CLEAN_PAD
CLEAN_PAD = 4
STRETCH_PARAMS = 6         # free parameters of a stretch (either mode)
POINT_PARAMS = 2           # ... of a lone detection that is off the path (its own position)
MODEL_JSON = ob.S + 'motion_model.json'
BREAKS_JSON = ob.S + 'motion_breaks.json'


# ---------------------------------------------------------------- data
def load(track_json=None):
    from cameras import load_h
    M, _, _ = load_h('tactical')
    track = {int(k): np.array(v) for k, v in json.load(open(track_json or ob.WORK + '/track.json')).items()}
    fr = np.array(sorted(track))
    uv = np.array([track[f][:2] for f in fr])
    Mi = np.linalg.inv(M[fr])
    q = np.einsum('nij,nj->ni', Mi, np.c_[uv, np.ones(len(fr))])
    g = q[:, :2] / q[:, 2:] * [L, W]                                      # ground point of the ray, metres
    pp = np.array([ob.ppm(M[f], p) for f, p in zip(fr, uv)])
    return M, track, fr, uv, g, pp


def stretches(fr):
    """index ranges [a, b) of runs of track points with no dropout longer than MAX_GAP."""
    cut = np.flatnonzero(np.diff(fr) > MAX_GAP + 1) + 1
    edges = np.r_[0, cut, len(fr)]
    return list(zip(edges[:-1], edges[1:]))


def rows(mode, t, g):
    """design rows (n, 2, 6) and targets (n, 2) of each point for one mode, t in seconds from the stretch start."""
    n = len(t); A = np.zeros((n, 2, 6)); y = g.copy()
    A[:, 0, 0] = A[:, 1, 1] = 1; A[:, 0, 2] = A[:, 1, 3] = t
    if mode == 'roll':                                                     # g = x0 + v t + a t^2 / 2
        A[:, 0, 4] = A[:, 1, 5] = t * t / 2
    else:                                                                  # g = x0 + v t - k (h0 + vh t - G t^2 / 2)
        k = (CT[:2] - g) / CT[2]
        A[:, :, 4] = -k; A[:, :, 5] = -k * t[:, None]
        y = g - k * G * t[:, None] ** 2 / 2
    return A, y


RIDGE = np.eye(6) * 1e-6                                                   # keeps too-short fits solvable


def roll_prior(chord, dt, prm):
    """learned prior on a roll's acceleration, for stretches whose ends are chord (m, 2) apart after dt (m,) s:
    slowing down along the direction of travel by roll_k x its speed, little acceleration sideways.
    -> terms (m, 6, 6), (m, 6), (m,) added to the normal equations."""
    m = len(dt); ln = np.linalg.norm(chord, axis=1)
    d = np.where(ln[:, None] > 1e-9, chord / np.maximum(ln, 1e-9)[:, None], [1.0, 0.0]); n = np.c_[-d[:, 1], d[:, 0]]
    speed = ln / np.maximum(dt, 1 / FPS)
    ia, ip = 1 / prm['roll_sd_along'] ** 2, 1 / prm['roll_sd_perp'] ** 2
    R = np.zeros((m, 6, 6)); R[:, 4:, 4:] = ia * np.einsum('ni,nj->nij', d, d) + ip * np.einsum('ni,nj->nij', n, n)
    mean = -prm['roll_k'] * speed
    bp = np.zeros((m, 6)); bp[:, 4:] = d * (mean * ia)[:, None]
    return R, bp, mean ** 2 * ia


def scan(t, g, wt, prm, i, jmax):
    """fits of every stretch starting at point i and ending at i..jmax-1 -> per mode (cost (m,), theta (m, 6))."""
    out = {}
    tt = t[i:jmax] - t[i]
    for mode in ('roll', 'flight'):
        A, y = rows(mode, tt, g[i:jmax])
        w = wt[i:jmax, None]
        Q = np.cumsum(np.einsum('nki,nk,nkj->nij', A, np.repeat(w, 2, 1), A), 0) + RIDGE
        b = np.cumsum(np.einsum('nki,nk,nk->ni', A, np.repeat(w, 2, 1), y), 0)
        Y = np.cumsum(np.einsum('nk,nk,nk->n', y, np.repeat(w, 2, 1), y), 0)
        if mode == 'roll':
            R, bp, Yp = roll_prior(g[i:jmax] - g[i], tt, prm)
            Q, b, Y = Q + R, b + bp, Y + Yp
        th = np.linalg.solve(Q, b[..., None])[..., 0]
        cost = Y - np.einsum('ni,ni->n', b, th)
        if mode == 'flight':
            m = np.arange(len(tt))
            h0 = th[:, 4]; h1 = th[:, 4] + th[:, 5] * tt - G * tt ** 2 / 2
            tv = np.clip(th[:, 5] / G, 0, tt); hv = th[:, 4] + th[:, 5] * tv - G * tv ** 2 / 2
            bad = (m + 1 < MIN_FLIGHT) | (np.minimum(h0, h1) < H_FLOOR) | (hv > H_CEIL)
            cost = np.where(bad, np.inf, cost)
        else:
            cost = np.where(np.arange(len(tt)) < 1, np.inf, cost)          # at least two points
        out[mode] = (np.maximum(cost, 0), th)
    return out


def weights(pp, prm):
    return (pp / prm['noise_px']) ** 2                                     # 1 / sigma^2, sigma in metres


# ---------------------------------------------------------------- the model at a time
def ground_at(seg, t):
    """(ground ray point g (2,), height h) of a fitted stretch at t seconds after its first point."""
    th = seg['theta']
    if seg['mode'] == 'roll':
        return np.array([th[0] + th[2] * t + th[4] * t * t / 2, th[1] + th[3] * t + th[5] * t * t / 2]), 0.0
    B = np.array([th[0] + th[2] * t, th[1] + th[3] * t]); h = th[4] + th[5] * t - G * t * t / 2
    return CT[:2] + (B - CT[:2]) * CT[2] / (CT[2] - h), float(h)


def pixel_at(M, f, seg, t):
    gg, _ = ground_at(seg, t); q = M[f] @ [gg[0] / L, gg[1] / W, 1.0]
    return q[:2] / q[2]


def pixel_vel(M, f, seg, t):
    """image velocity (px / frame, in frame f's pixels) of a fitted stretch at t."""
    d = 0.5 / FPS
    return (pixel_at(M, f, seg, t + d) - pixel_at(M, f, seg, t - d)) / (2 * d * FPS)


# ---------------------------------------------------------------- cutting the track
def bic(n_obs, k):
    return k * float(np.log(n_obs))


def best_cost(t, g, wt, prm):
    """chi^2 of the better mode fitted to these points as one stretch."""
    fit = scan(t, g, wt, prm, 0, len(t))
    return min(fit[m][0][-1] for m in fit)


def drop_outliers(fr, g, wt, prm, n_obs):
    """mask of track points to keep. Around each point (6 frames either side) three explanations are compared: one
    smooth path, the point off the path (a bad detection), or a contact at it -- the cheapest wins."""
    keep = np.ones(len(fr), bool)
    t = fr / FPS
    for i in range(len(fr)):
        win = [j for j in range(max(0, i - 6), min(len(fr), i + 7)) if abs(fr[j] - fr[i]) <= 6]
        left, right = [j for j in win if j < i], [j for j in win if j > i]
        if len(left) < 3 or len(right) < 3:
            continue
        c = lambda idx: best_cost(t[idx], g[idx], wt[idx], prm)
        smooth = c(win)
        outlier = c(left + right) + bic(n_obs, POINT_PARAMS)
        contact = min(c(left) + c([i] + right), c(left + [i]) + c(right)) + bic(n_obs, STRETCH_PARAMS)
        if outlier < min(smooth, contact):
            keep[i] = False
    return keep


def cut(fr, g, wt, prm, lam):
    """-> list of fitted stretches dict(i0, i1 (track point indices, inclusive), mode, theta, cost) covering fr."""
    segs = []
    for a, b in stretches(fr):
        n = b - a; best = np.full(n + 1, np.inf); best[0] = 0.0; back = [None] * (n + 1)
        t = (fr[a:b] - fr[a]) / FPS
        fits = []
        for i in range(n):
            fits.append(scan(t, g[a:b], wt[a:b], prm, i, min(n, i + MAX_SEG)))
            for mode, (cost, th) in fits[i].items():
                m = len(cost); c = best[i] + lam + cost
                jj = np.arange(i + 1, i + 1 + m)
                better = c < best[jj]
                for k in np.flatnonzero(better):
                    best[jj[k]] = c[k]; back[jj[k]] = (i, mode, k)
        j = n; out = []
        while j > 0:
            i, mode, k = back[j]
            cost, th = fits[i][mode]
            other = fits[i]['flight' if mode == 'roll' else 'roll'][0][k]
            out.append(dict(i0=a + i, i1=a + i + k, mode=mode, theta=th[k], cost=float(cost[k]), other=float(other)))
            j = i
        segs += out[::-1]
    return segs


def breaks_from(M, fr, track, segs, prm, n_obs):
    """contacts between consecutive stretches -> dicts (frame, pos px, vin, vout px / frame, in frame fb's pixels)."""
    out = []
    for A, B in zip(segs, segs[1:]):
        fa, fb = fr[A['i1']], fr[B['i0']]
        if fb - fa > MAX_GAP + 1:
            continue                                                       # separate stretches: no contact known
        ta = (fa - fr[A['i0']]) / FPS
        vA = pixel_vel(M, fb, A, ta); vB = pixel_vel(M, fb, B, 0.0)
        pa = ob.warp(ob.H(M, fa, fb), track[fa][:2]); pb = track[fb][:2]
        vt = (pb - pa) / (fb - fa)                                         # the jump between them, px / frame
        # one contact: the two paths meet somewhere between fa and fb. How close do they get (chi^2, both paths'
        # positions carry the measurement noise)? If not close enough to be worth one more stretch: two contacts.
        tb0 = (fb - fr[B['i0']]) / FPS
        taus = np.linspace(0.0, (fb - fa) / FPS, 11)
        miss = min(np.linalg.norm(pixel_at(M, fb, A, ta + u) - pixel_at(M, fb, B, tb0 - (fb - fa) / FPS + u)) for u in taus)
        if miss ** 2 / (2 * prm['noise_px'] ** 2) > bic(n_obs, STRETCH_PARAMS):
            f1 = fa if fb == fa + 1 else fa + 1                            # the first contact: just after fa
            p1 = track[fa][:2] if f1 == fa else ob.warp(ob.H(M, fb, f1), pa + vt)
            out += [(int(f1), p1, vA, vt), (int(fb), pb, vt, vB)]
        else:
            out.append((int(fb), pb, vA, vB))
    return [dict(f=f, pos=[float(x) for x in pos], vin=[float(x) for x in vin], vout=[float(x) for x in vout])
            for f, pos, vin, vout in out]


# ---------------------------------------------------------------- did anything happen at this moment?
SMOOTH_WIN = 8             # frames either side
SMOOTH_PRM = dict(noise_px=0.85, roll_k=0.615, roll_sd_along=1.5, roll_sd_perp=1.0)   # first-round calibration (fit)


def smooth_through(M, track, f):
    """True if one path (a roll or a gravity arc) explains the ball's track SMOOTH_WIN frames either side of f better
    than two paths broken at f, after BIC (a second path costs STRETCH_PARAMS more parameters); None without enough
    track on both sides. Nothing touched the ball at f if its path just carries on."""
    fr = np.array([g for g in range(f - SMOOTH_WIN, f + SMOOTH_WIN + 1) if g in track])
    if (fr < f).sum() < 3 or (fr >= f).sum() < 3:
        return None
    uv = np.array([track[g][:2] for g in fr])
    q = np.einsum('nij,nj->ni', np.linalg.inv(M[fr]), np.c_[uv, np.ones(len(fr))])
    G = q[:, :2] / q[:, 2:] * [L, W]; pp = np.array([ob.ppm(M[g], p) for g, p in zip(fr, uv)])
    t = (fr - fr[0]) / FPS; wt = weights(pp, SMOOTH_PRM)
    one = min(c[-1] for c, _ in scan(t, G, wt, SMOOTH_PRM, 0, len(fr)).values())
    k = int(np.searchsorted(fr, f))
    two = (min(c[-1] for c, _ in scan(t, G, wt, SMOOTH_PRM, 0, k).values())
           + min(c[-1] for c, _ in scan(t, G, wt, SMOOTH_PRM, k, len(fr)).values()))
    return bool(one <= two + bic(2 * len(fr), STRETCH_PARAMS))


def evidence(fr, g, wt, prm, n_obs, f):
    """How sure a contact at frame f is: 2 ln(Bayes factor) of two stretches meeting at f over one smooth path, on the
    track points within 6 frames either side (BIC: the second stretch's parameters paid for). 0 when too few points."""
    t = fr / FPS
    win = np.flatnonzero(np.abs(fr - f) <= 6)
    left, right = win[fr[win] < f], win[fr[win] >= f]
    if len(left) < 3 or len(right) < 3:
        return 0.0
    c = lambda idx: best_cost(t[idx], g[idx], wt[idx], prm)
    return float(c(win) - c(left) - c(right) - bic(n_obs, STRETCH_PARAMS))


# ---------------------------------------------------------------- calibration
def detection_noise(M, fr, uv):
    """measurement noise (px) from the third differences of the track over 4 consecutive frames: zero for any path
    with constant acceleration, so (for iid noise, variance 20 sigma^2) all noise. A robust spread over the whole
    track, independent of where the contacts are: the rare contact in a 4-frame window does not move the median."""
    d3 = []
    for n in range(len(fr) - 3):
        f = fr[n]
        if fr[n + 3] != f + 3:
            continue
        p = [ob.warp(ob.H(M, fr[n + k], f), uv[n + k]) for k in range(4)]
        d3.append(p[3] - 3 * p[2] + 3 * p[1] - p[0])
    d3 = np.array(d3)
    return float(np.mean([1.4826 * np.median(np.abs(d3[:, k] - np.median(d3[:, k]))) for k in range(2)]) / np.sqrt(20))


def free_mask(M, fr, uv):
    """track points with the ball at least FREE_M from every player (no contact possible)."""
    t = np.load(ob.S + 'tactical.npz'); P = t['P'][..., :2]
    ob.STALE = ob.stale_mask(P)
    out = np.zeros(len(fr), bool)
    for n, (f, p) in enumerate(zip(fr, uv)):
        nb = ob.nearest_body(M, P, f, p)
        out[n] = nb is None or nb[0] >= FREE_M
    return out


def clean_runs(fr, free, break_frames):
    """index ranges of clean stretches: consecutive frames, all free, no break within CLEAN_PAD."""
    bf = np.array(sorted(break_frames)) if len(break_frames) else np.array([-10 ** 9])
    ok = free & (np.min(np.abs(fr[:, None] - bf[None, :]), 1) > CLEAN_PAD)
    runs, a = [], None
    for i in range(len(fr) + 1):
        good = i < len(fr) and ok[i] and (a is None or fr[i] == fr[i - 1] + 1)
        if good and a is None:
            a = i
        elif not good and a is not None:
            for s0 in range(a, i - CLEAN_MIN + 1, 50):                     # pieces of at most 50 frames
                runs.append((s0, min(i, s0 + 50)))
            a = i if i < len(fr) and ok[i] else None
    return runs


def calibrate(fr, g, pp, runs, prm):
    """from clean stretches (each one stretch, no contact): the measurement noise and how a roll slows down."""
    loose = dict(prm, roll_k=0.0, roll_sd_along=1e3, roll_sd_perp=1e3)
    chi, along, perp, speed, heights, modes = [], [], [], [], [], []
    for a, b in runs:
        t = (fr[a:b] - fr[a]) / FPS; wt = weights(pp[a:b], prm)
        free = scan(t, g[a:b], wt, loose, 0, b - a)                       # the roll's acceleration unconstrained
        if free['roll'][0][-1] <= free['flight'][0][-1]:
            th = free['roll'][1][-1]; tm = t[-1] / 2
            v = np.array([th[2] + th[4] * tm, th[3] + th[5] * tm]); sp = np.linalg.norm(v); d = v / max(sp, 1e-9)
            speed.append(sp); along.append(th[4:6] @ d); perp.append(th[4:6] @ [-d[1], d[0]])
        whole = scan(t, g[a:b], wt, prm, 0, b - a)
        mode = min(whole, key=lambda m: whole[m][0][-1]); cost, th = whole[mode]
        modes.append(mode); chi.append(cost[-1] / (2 * (b - a) - 6))
        if mode == 'flight':
            tv = np.clip(th[-1, 5] / G, 0, t[-1]); heights.append(th[-1, 4] + th[-1, 5] * tv - G * tv ** 2 / 2)
    speed, along, perp = map(np.array, (speed, along, perp))
    mad = lambda x: 1.4826 * float(np.median(np.abs(x - np.median(x))))
    k = float(np.median(-along[speed > 1] / speed[speed > 1]))
    return dict(chi_dof=round(float(np.median(chi)), 2), roll_k=round(k, 3),
                roll_sd_along=round(max(0.5, mad(along + k * speed)), 2), roll_sd_perp=round(max(0.3, mad(perp)), 2),
                runs=len(runs), roll_share=round(modes.count('roll') / max(1, len(modes)), 2),
                flight_peak_h=[round(float(x), 1) for x in np.percentile(heights, [10, 50, 90])] if heights else None)


LEARNED = ('roll_k', 'roll_sd_along', 'roll_sd_perp')


def cmd_fit():
    M, track, fr, uv, g, pp = load()
    free = free_mask(M, fr, uv)
    breaks = [e['f'] for e in json.load(open(ob.OUT_JSON))['events']]     # to start with: the window detector's
    prm = dict(noise_px=round(detection_noise(M, fr, uv), 3), roll_k=0.0, roll_sd_along=1.0, roll_sd_perp=1.0)
    print('detection noise', prm['noise_px'], 'px')
    for rnd in range(3):
        runs = clean_runs(fr, free, breaks)
        for _ in range(3):
            c = calibrate(fr, g, pp, runs, prm); prm.update({k: c[k] for k in LEARNED})
        n_obs = 2 * len(fr)
        wt = weights(pp, prm); keep = drop_outliers(fr, g, wt, prm, n_obs)
        segs = cut(fr[keep], g[keep], wt[keep], prm, bic(n_obs, STRETCH_PARAMS))
        breaks = [fr[keep][s['i0']] for s in segs[1:]]
        print(f'round {rnd + 1}: {c}  cut price {bic(n_obs, STRETCH_PARAMS):.1f}  outliers dropped {int((~keep).sum())}  '
              f'stretches {len(segs)}', flush=True)
    json.dump(prm, open(MODEL_JSON, 'w'), indent=1)
    print('->', MODEL_JSON, prm)


def run_model(prm=None, track_json=None):
    M, track, fr, uv, g, pp = load(track_json)
    prm = prm or json.load(open(MODEL_JSON))
    n_obs = 2 * len(fr)
    wt = weights(pp, prm); keep = drop_outliers(fr, g, wt, prm, n_obs)
    frk = fr[keep]
    segs = cut(frk, g[keep], wt[keep], prm, bic(n_obs, STRETCH_PARAMS))
    br = breaks_from(M, frk, track, segs, prm, n_obs)
    for b in br:
        b['evidence'] = evidence(frk, g[keep], wt[keep], prm, n_obs, b['f'])
    stretch = []
    for s in segs:
        t1 = (frk[s['i1']] - frk[s['i0']]) / FPS
        if s['mode'] == 'flight':
            th = s['theta']; tv = np.clip(th[5] / G, 0, t1); peak = th[4] + th[5] * tv - G * tv ** 2 / 2
        else:
            peak = 0.0
        kind = s['mode'] if s['other'] - s['cost'] > bic(n_obs, 1) else 'unclear'   # too short to tell the modes apart
        stretch.append(dict(f0=int(frk[s['i0']]), f1=int(frk[s['i1']]), kind=kind,
                            g_frac=1.0 if s['mode'] == 'flight' else 0.0, max_h=round(float(peak), 1)))
    return br, stretch


def cmd_breaks():
    br, stretch = run_model()
    json.dump(dict(breaks=br, segments=stretch), open(BREAKS_JSON, 'w'))
    from collections import Counter
    print(f'{len(br)} breaks, stretches {Counter(s["kind"] for s in stretch)} -> {BREAKS_JSON}')


if __name__ == '__main__':
    {'fit': cmd_fit, 'breaks': cmd_breaks}[sys.argv[1]]()
