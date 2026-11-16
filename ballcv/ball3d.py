"""The ball in three dimensions, from one camera: on the grass, or on a gravity arc.

Our ball is a pixel; where it is on the pitch depends on its height, which one camera cannot see in a single frame. Its
pixel is a ray from the camera (CAMERA, fixed: the camera only turns and zooms) through the point where the refined
homography (refine_h.py) puts that pixel on the grass; the ball is somewhere on that ray. Between two breaks in its
motion (a touch, a bounce, a deflection -- ownball.py events) the ball either rolls on the grass, or flies under gravity:
  p(t) = p0 + v0 t - g t^2 / 2 (up)
Lying on every frame's ray makes the flight a linear least-squares problem in p0, v0 (the residual is each frame's
distance from its ray, in metres). One camera's rays are nearly parallel along a flight, so the arc could slide along
them; the ball's height at the breaks that bound the stretch fixes it (on the grass at a bounce, at the foot, body,
head or hands at a touch -- BREAK_HEIGHT). A stretch is a flight when the arc, so held, runs through every frame's ray (ON_RAYS_M) -- a
ball rolling on the grass does not follow a gravity arc -- and stays above ground; or two arcs do, the ball bouncing
on the grass in between where the track saw no bounce (not where ownball sees the ball rolling). A flight gives the ball's height and true
place in every frame of the stretch -- and where the track has no ball, the arc says where it was. Otherwise the ball is
on the grass, where the ground point is its place.

  python ball3d.py fit      -> work/align/ball3d.npz   per frame: x, y, z (m), kind (0 ground, 1 flight, 2 flight
                                                        filled where the track has no ball), the stretch's residual
  python ball3d.py check    rolling stretches stay down, flights start and end low; against Metrica's ball
"""
import json, sys
import numpy as np

import ownball as ob
from ownball import L, W
import refine_h

OUT = __import__('os').environ.get('BALL3D_OUT') or ob.S + 'ball3d.npz'
FULL = __import__('os').environ.get('OWNBALL_WORK') or ob.S + 'fullmatch'   # the full-match track
CAMERA = np.array([53.13, 114.50, 29.34])        # the tactical camera's centre, m (refined homographies, median)
G_MS2 = 9.81
MIN_FRAMES = 8              # a stretch this long, or longer, is fitted
MAX_GAP = 12                # a track gap longer than this (frames) ends a stretch
ON_RAYS_M = 0.25            # a flight: the anchored arc runs this close to every frame's ray (RMS, m) -- flights do
                            # (median 0.04 m), a rolling ball cannot (median 1 m)
BELOW_M = 0.3               # ... and never dips further than this below the grass
MAX_SPEED_MS = 35.0         # ... and leaves no faster than the hardest kick
MAX_HEIGHT_M = 40.0         # no ball in play goes higher
# the ball's height at a break, m: on the grass at a bounce; at a touch, the part of the body that played it
BREAK_HEIGHT = {('bounce', None): 0.0, ('touch', 'foot'): 0.2, ('touch', 'body'): 1.1, ('touch', 'head'): 1.9,
                ('touch', 'hands'): 1.5}
RAY_SD, HEIGHT_SD = 0.2, 0.3   # how well a ray (m, about a pixel) and a break's height pin the ball down
REACH_SD = 0.7              # at a touch the ball is within about this of the player's tracked place (m)
GRASS_SD = 0.3              # on the grass the ground point is the ball's place, this close (m)
ROLL_GAP = 6                # a flight starting / ending within this many frames of a roll left / landed on the grass there
DRAG_K = 0.012              # air drag on a football, 1/m: deceleration k |v| v (rho Cd A / 2m: 1.2 * 0.25 * 0.038 / 0.86)
DT = 0.01                   # s, integration step


def rays(H, track, frames):
    """unit direction from the camera through each frame's ball pixel."""
    uv = np.array([track[f][:2] for f in frames])
    q = np.einsum('nij,nj->ni', np.linalg.inv(H[frames]), np.c_[uv, np.ones(len(uv))])
    g = np.c_[q[:, :2] / q[:, 2:] * [L, W], np.zeros(len(uv))]
    d = g - CAMERA
    return g, d / np.linalg.norm(d, axis=1, keepdims=True)


def fly(p0, v0, tau, drag=DRAG_K):
    """the ball's place at times tau (s) from p0, v0 at tau 0, under gravity and air drag (deceleration k |v| v)."""
    out = np.empty((len(tau), 3))
    for sign in (1, -1):                                              # forwards and backwards from tau 0
        ts = tau[tau * sign >= 0] * sign
        if not len(ts):
            continue
        p, v, t, k = p0.astype(float).copy(), v0.astype(float).copy() * sign, 0.0, 0
        order = np.argsort(ts); res = np.empty((len(ts), 3))
        for idx in order:
            while t < ts[idx] - 1e-9:
                h = min(DT, ts[idx] - t)
                # drag opposes the motion; run backwards (velocity reversed), it pushes the other way
                a = np.array([0, 0, -G_MS2]) - sign * drag * np.linalg.norm(v) * v
                p = p + v * h + a * h * h / 2; v = v + a * h; t += h
            res[idx] = p
        out[tau * sign >= 0] = res
    return out


def fit_flight(tau, d, anchors, drag=DRAG_K, iters=4):
    """p0, v0 of the flight under gravity and air drag closest to the rays, held to where the ball is known to be at
    its ends ({tau: (place on the ground or None, height, how sure of the place, m)} -- a touch: at the player, at the
    height of the part of the body; a bounce, or the grass it rolled on just before or after: there, on the grass);
    and each frame's distance from its ray (m). One camera's rays are nearly parallel along a flight -- the arc could
    slide along them -- so these anchors are what fix how far away it is. Drag (DRAG_K, a football's) is not free: the
    arc is fitted, the drag it implies taken as known, and the arc fitted again, a few times."""
    ta = np.array(list(anchors)); allt = np.r_[tau, ta]
    D = np.zeros((len(allt), 3)); x = None
    for _ in range(iters):
        A, b = [], []
        for t, u, dd in zip(tau, d, D[:len(tau)]):
            Pp = np.eye(3) - np.outer(u, u)                           # removes the part along the ray
            A.append(Pp @ np.c_[np.eye(3), t * np.eye(3)] / RAY_SD)
            b.append(Pp @ (CAMERA + np.array([0, 0, G_MS2 * t * t / 2]) - dd) / RAY_SD)
        for (t, (xy, h, sd)), dd in zip(anchors.items(), D[len(tau):]):
            A.append(np.r_[0, 0, 1, 0, 0, t][None] / HEIGHT_SD); b.append(np.array([h + G_MS2 * t * t / 2 - dd[2]]) / HEIGHT_SD)
            if xy is not None:
                A.append(np.c_[np.eye(3), t * np.eye(3)][:2] / sd); b.append((np.asarray(xy) - dd[:2]) / sd)
        x = np.linalg.lstsq(np.concatenate(A), np.concatenate(b), rcond=None)[0]
        if not drag:
            break
        free = x[:3] + np.outer(allt, x[3:]) - np.outer(allt * allt / 2, [0, 0, G_MS2])
        D = fly(x[:3], x[3:], allt, drag) - free                      # what drag adds to the drag-free arc
    p = x[:3] + np.outer(tau, x[3:]) - np.outer(tau * tau / 2, [0, 0, G_MS2]) + D[:len(tau)]
    return x[:3], x[3:], p, ray_distance(p, d)


def ray_distance(p, d):
    v = p - CAMERA
    return np.linalg.norm(v - (v * d).sum(1, keepdims=True) * d, axis=1)


def fit_ground(tau, g, d):
    """the smooth path on the grass (quadratic in time) closest to the rays, and each frame's distance from its ray."""
    A = np.c_[np.ones_like(tau), tau, tau * tau]
    cx, cy = (np.linalg.lstsq(A, g[:, k], rcond=None)[0] for k in (0, 1))
    p = np.c_[A @ cx, A @ cy, np.zeros(len(tau))]
    return p, ray_distance(p, d)


def stretches(track, events, starts=()):
    """the track cut at every break, at every start of ownball's own segments (it cuts where the motion changes, a roll
    becoming a flight, even with no event) and at every gap longer than MAX_GAP: [(frames, break before, break after)],
    a break being its event (or None, at a segment start or a gap)."""
    at = {e['f']: e for e in events}; seg = set(starts); fs = sorted(track); out, cur, first = [], [], None
    for f in fs:
        if cur:
            cut = next((at[c] for c in range(cur[-1] + 1, f + 1) if c in at), None)
            if cut is not None or f - cur[-1] > MAX_GAP or any(c in seg for c in range(cur[-1] + 1, f + 1)):
                out.append((cur, first, cut)); cur, first = [], cut
        cur.append(f)
    if cur:
        out.append((cur, first, None))
    return out


def break_height(e):
    return None if e is None else BREAK_HEIGHT.get((e['kind'], e.get('body')))


def arc_ok(p, v0, rf):
    """the arc runs through the rays (a rolling ball does not follow one), stays above the grass, below MAX_HEIGHT_M, and
    leaves no faster than a ball is kicked"""
    return (np.sqrt(np.mean(rf ** 2)) <= ON_RAYS_M and p[:, 2].min() >= -BELOW_M and p[:, 2].max() <= MAX_HEIGHT_M
            and np.linalg.norm(v0) <= MAX_SPEED_MS)


def flight_arcs(fr, g, d, anchors, may_split=True):
    """[(frames, rays, p0, v0, places, rms)]: one arc through the stretch; or, if none fits, two -- the ball bounced on
    the grass in between (where the track shows no bounce): the bounce that best explains the rays, on the grass under
    its pixel. None if neither fits."""
    tau = (fr - fr[0]) / 25.0
    p0, v0, p, rf = fit_flight(tau, d, anchors)
    if arc_ok(p, v0, rf):
        return [(fr, d, p0, v0, p, float(np.sqrt(np.mean(rf ** 2))))]
    if not may_split or len(fr) < 2 * MIN_FRAMES:
        return None
    def halves(k):
        tk = tau[k]; bounce = (g[k, :2], 0.0, GRASS_SD)
        first = {t: v for t, v in anchors.items() if t < tk}; first[tk] = bounce
        second = {t - tk: v for t, v in anchors.items() if t > tk}; second[0.0] = bounce
        return (fr[:k + 1], d[:k + 1], tau[:k + 1], first), (fr[k:], d[k:], tau[k:] - tk, second)
    # the bounce: searched without drag (fast), then both arcs fitted with it
    cost = []
    for k in range(MIN_FRAMES - 1, len(fr) - MIN_FRAMES + 1):
        (_, d1, t1, a1), (_, d2, t2, a2) = halves(k)
        r = np.r_[fit_flight(t1, d1, a1, 0.0)[3], fit_flight(t2, d2, a2, 0.0)[3]]
        cost.append((float(np.sqrt(np.mean(r ** 2))), k))
    _, k = min(cost)
    out = []
    for f_, d_, t_, a_ in halves(k):
        q0, w0, q, r = fit_flight(t_, d_, a_)
        if not arc_ok(q, w0, r):
            return None
        out.append((f_, d_, q0, w0, q, float(np.sqrt(np.mean(r ** 2)))))
    return out


def cmd_fit():
    H = np.load(refine_h.OUT)
    track = {int(k): v for k, v in json.load(open(FULL + '/track.json')).items()}
    ev = json.load(open(ob.OUT_JSON))['events']
    segs = json.load(open(ob.OUT_JSON))['segments']
    pieces = stretches(track, ev, [sg['f0'] for sg in segs])
    rolls = [sg for sg in segs if sg['kind'] == 'roll']
    rolling = np.zeros(len(H), bool)
    for r in rolls:
        rolling[r['f0']:r['f1'] + 1] = True
    # for a stretch starting at frame a: the last frame of a roll ending just before it; ending at b: a roll's first
    roll_end = {a: max((r['f1'] for r in rolls if a - ROLL_GAP <= r['f1'] <= a), default=None) for a in [p[0][0] for p in pieces]}
    roll_start = {b: min((r['f0'] for r in rolls if b <= r['f0'] <= b + ROLL_GAP), default=None) for b in [p[0][-1] for p in pieces]}
    def grass_at(f):                                                        # the ground point of the ball's pixel
        return rays(H, track, np.array([f]))[0][0, :2] if f in track else None
    P = np.load(ob.S + 'tactical.npz')['P'][..., :2] * [L, W]
    def player_at(e):                                                       # the touching player's tracked place
        k = e.get('slot')
        if k is None:
            return None
        xy = P[e['f'], k - 1] if 0 < k <= P.shape[1] else np.array([np.nan, np.nan])
        return xy if np.isfinite(xy).all() else None

    n = len(H); X = np.full((n, 3), np.nan); kind = np.full(n, -1, np.int8); resid = np.full(n, np.nan)
    flights = 0
    for fr, before, after in pieces:
        fr = np.array(fr)
        g, d = rays(H, track, fr)
        X[fr] = g; kind[fr] = 0                                             # on the grass, unless a flight fits better
        heights = {}
        for e in (before, after):
            if break_height(e) is not None:
                if e['kind'] == 'touch':
                    heights[(e['f'] - fr[0]) / 25.0] = (player_at(e), break_height(e), REACH_SD)
                else:                                                       # a bounce: on the grass, under its pixel
                    heights[(e['f'] - fr[0]) / 25.0] = (grass_at(e['f']), 0.0, GRASS_SD)
        for f in (roll_end.get(fr[0]), roll_start.get(fr[-1])):            # left / landed on the grass it rolled on
            if f is not None and (f - fr[0]) / 25.0 not in heights:
                heights[(f - fr[0]) / 25.0] = (grass_at(f), 0.0, GRASS_SD)
        heights = {t: v for t, v in heights.items() if v[0] is not None or v[2] == REACH_SD}
        if len(fr) < MIN_FRAMES or not heights:
            continue                                                        # nothing says how far away a flight was
        arcs = flight_arcs(fr, g, d, heights, may_split=np.mean(rolling[fr]) <= 0.5)   # a roll is no flight that bounced
        if arcs is None:
            _, rg = fit_ground((fr - fr[0]) / 25.0, g, d)
            resid[fr] = np.sqrt(np.mean(rg ** 2))
            continue
        for a_fr, a_d, p0, v0, p, rms in arcs:
            flights += 1
            # each frame: the point on its own ray nearest the arc (keeps the observed pixel exactly)
            v = p - CAMERA; s = (v * a_d).sum(1, keepdims=True); X[a_fr] = CAMERA + s * a_d
            kind[a_fr] = 1; resid[a_fr] = rms
            gaps = np.setdiff1d(np.arange(a_fr[0], a_fr[-1] + 1), a_fr)      # where the track lost it: the arc
            if len(gaps):
                X[gaps] = fly(p0, v0, (gaps - a_fr[0]) / 25.0); kind[gaps] = 2; resid[gaps] = rms
    np.savez_compressed(OUT, X=X.astype(np.float32), kind=kind, resid=resid.astype(np.float32))
    seen = kind >= 0
    print(f'{len(pieces)} stretches, {flights} flights; frames: on the grass {np.sum(kind == 0)}, in flight {np.sum(kind == 1)}, '
          f'filled by an arc {np.sum(kind == 2)} -> {OUT}')


def cmd_check():
    d = np.load(OUT); X, kind = d['X'], d['kind']
    seg = json.load(open(ob.OUT_JSON))['segments']
    rolled = [s for s in seg if s['kind'] == 'roll']; flew = [s for s in seg if s['kind'] == 'flight']
    share = lambda ss: np.mean([np.mean(kind[s['f0']:s['f1'] + 1] == 1) for s in ss])
    print(f"stretches ownball calls rolling: {share(rolled):.0%} of their frames fitted as flight (want ~0); "
          f"called flight: {share(flew):.0%} (want high)")
    z = X[:, 2]; fl = kind >= 1
    print(f'heights in flight: median {np.nanmedian(z[fl]):.2f} m, p90 {np.nanpercentile(z[fl], 90):.2f} m, max {np.nanmax(z[fl]):.1f} m')
    # against Metrica's ball, in flight: is our corrected place closer than the plain ground point?
    sys.path[:0] = [__import__('os').path.join(__import__('paths').REPO, 'possession'), __import__('paths').REPO]
    import predict
    metrica = predict.match_atd()[0] * [L, W]
    H = np.load(refine_h.OUT); track = {int(k): v for k, v in json.load(open(FULL + '/track.json')).items()}
    fs = np.array([f for f in np.flatnonzero(kind == 1) if np.isfinite(metrica[f]).all() and f in track])
    g, _ = rays(H, track, fs)
    raw = np.linalg.norm(g[:, :2] - metrica[fs], axis=1); cor = np.linalg.norm(X[fs, :2] - metrica[fs], axis=1)
    print(f'in flight, distance to Metrica\'s ball: plain ground point median {np.median(raw):.2f} m, '
          f'arc-corrected {np.median(cor):.2f} m ({len(fs)} frames)')
    for f in (24911, 18026, 25418):
        print(f'  frame {f}: kind {kind[f]}, place ({X[f, 0]:.1f}, {X[f, 1]:.1f}) m, height {X[f, 2]:.2f} m')


if __name__ == '__main__':
    {'fit': cmd_fit, 'check': cmd_check}[sys.argv[1]]()
