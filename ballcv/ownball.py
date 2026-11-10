"""Our own ball tracking and touch detection, in camera-stabilised pixels. No Metrica ball anywhere.

The tactical camera pans/tilts/zooms about a fixed centre, so the image-to-image map between any two frames is
the exact homography M[g] @ inv(M[f]) for EVERY point, whatever its height. All motion is compared in one frame's
pixel space, so camera movement drops out and the ball's height never has to be known.

  detect START END   ball candidates (heatmap.py: the ball model's top peaks) for every frame -> cands_START.npz
  link               best path through the candidates (dynamic programming) -> track.json
  events             breaks in the ball's smooth motion -> touches (Metrica PLAYERS only) / bounces, and what kind
                     of ball was played between touches -> data/ballcv/BAR-ATH_touches.json
"""
import bisect, glob, json, sys
import numpy as np

S = __import__('paths').S
WORK = __import__('os').environ.get('OWNBALL_WORK') or S + 'fullmatch'   # candidates and track: the full match
L, W = 105.3, 68.0
CT = np.array([53.2, 115.5, 30.0])         # tactical camera centre
OUT_JSON = __import__('os').environ.get('TOUCHES_OUT') or __import__('paths').DATA + 'ballcv/BAR-ATH_touches.json'
KIT_MODEL = __import__('os').environ.get('KIT_MODEL') or __import__('paths').MODELS + 'kit.pt'   # kit_team.py
PLAYER_MODEL = __import__('os').environ.get('PLAYER_MODEL') or __import__('paths').MODELS + 'players.pt'   # player_det.py
PLAYER_TRACKS = __import__('os').environ.get('PLAYER_TRACKS') or S + 'players/tracks.npz'   # player_track.py

# detection
TOP = 8                 # candidates kept per frame
CAND_LOGIT = -1.0
# linking
REWARD_AT = 0.0         # a candidate adds (logit - REWARD_AT) to the path score
SKIP_COST = 0.12        # per frame with no candidate on the path
MAX_SKIP = 12           # longest dropout bridged directly
VMAX_MPF = 1.6          # fastest ball, metres (at the ball's ground scale) per frame (= 40 m/s)
MOVE_COST = 0.04        # per metre moved (prefers the smoother of two otherwise equal paths)
RESTART_COST = 2.5      # starting a new path segment anywhere (ball reappears after a long absence)
# events
FIT_HALF = 4            # frames each side used to find breaks in the ball's motion (touches 0.3 s apart stay separate)
MIN_SIDE = 3            # points needed on each side
BREAK_MS = 5.0          # velocity change (m/s at the ball's ground scale) that counts as a break
CONTACT_SEARCH = (-2, 2)  # a break is timed at the FIRST frame already on the outgoing path: the first frame closer to
CONTACT_FIT = 5          # the outgoing path (fitted on the frames after it) than to the incoming path (fitted on the
                         # frames before it), searched this far around the detected break. Matches how the user marks
                         # touches and bounces (the contact itself happens between two frames).
MIN_SEG = 5             # points needed to fit a stretch between breaks with p0 + v t + a t^2 / 2
G = 9.81
FLIGHT_G = (0.5, 1.7)   # stretch is a flight when its downward image acceleration is this fraction of gravity
ROLL_G = 0.3            # ... and rolling when its acceleration is below this fraction of gravity
BOUNCE_COS = 0.8        # a break whose velocity change points this squarely up the local vertical is a bounce
BODY_H = 2.0
STRAIGHT_DEG = 5.0         # Metrica's ball turning less than this over +-0.5 s: it went straight on
CONTACT_M = 0.7         # ball this close to a player's column: a touch even if the change looks like a bounce
REACH_BASE_M = 0.5      # otherwise a touch needs the ball within REACH_BASE_M + REACH_PER_MS * dv (an outstretched
REACH_PER_MS = 0.05     # leg is more plausible the harder the ball's motion changes), at most MAX_REACH_M,
MAX_REACH_M = 1.5       # and the velocity change must push the ball away from that player
PLACED_S = 1.0          # a set piece: the ball lies still (moves less than CONTACT_M) this long before it is struck
DRIBBLE_M = 8.0
LONG_M = 30.0
STALE_FRAMES = 12      # a track whose position has not moved for this long is frozen (stale tracklet): it can't touch
CLAIM_GAP_S = 1.0       # keeper claim: the ball vanishes for at least this long ...
CLAIM_REACH_M = 2.0     # ... last seen within this distance of a goalkeeper ...
BOX_DEPTH_M, BOX_HALF_W_M = 16.5, 20.16   # ... who is inside his own penalty area

def H(M, f, g):
    """pixel map frame f -> frame g (exact for a fixed-centre camera)."""
    return M[g] @ np.linalg.inv(M[f])


def warp(Hm, uv):
    q = Hm @ [uv[0], uv[1], 1.0]; return q[:2] / q[2]


def ppm(Mf, uv):
    """pixels per ground metre around image point uv."""
    q = np.linalg.inv(Mf) @ [uv[0], uv[1], 1.0]; g = q[:2] / q[2]
    a = Mf @ [g[0], g[1], 1.0]; b = Mf @ [g[0] + 1 / L, g[1], 1.0]; c = Mf @ [g[0], g[1] + 1 / W, 1.0]
    a, b, c = a[:2] / a[2], b[:2] / b[2], c[:2] / c[2]
    return float((np.linalg.norm(b - a) + np.linalg.norm(c - a)) / 2)


def column(Mf, G):
    """image segment (feet, top) of a player standing at ground point G (m)."""
    out = []
    for h in (0.0, BODY_H):
        Q = CT[:2] + (G - CT[:2]) * CT[2] / (CT[2] - h)
        q = Mf @ [Q[0] / L, Q[1] / W, 1.0]; out.append(q[:2] / q[2])
    return out


# ---------------------------------------------------------------- 1. candidates
def cmd_detect(start, end):
    from cameras import load_h
    from video import read_frames
    import heatmap
    M, _, _ = load_h('tactical'); net = heatmap.load()
    C = np.full((end - start, TOP, 3), np.nan, np.float32)
    for i, img in enumerate(read_frames(start, end - start)):
        for j, p in enumerate(heatmap.peaks(net, img, M[start + i], thresh=CAND_LOGIT, top=TOP)):
            C[i, j] = p
    np.savez_compressed(f'{WORK}/cands_{start}.npz', C=C, start=start)
    print(f'{start}-{end}: {np.isfinite(C[:, 0, 0]).sum()} frames with candidates', flush=True)


def load_cands():
    parts = sorted((int(np.load(p)['start']), np.load(p)['C']) for p in glob.glob(WORK + '/cands_*.npz'))
    start = parts[0][0]; C = np.concatenate([c for _, c in parts]); return start, C


# ---------------------------------------------------------------- 2. linking
def cmd_link():
    from cameras import load_h
    M, _, _ = load_h('tactical')
    start, C = load_cands(); n = len(C)
    nodes = [[(float(u), float(v), float(lg)) for u, v, lg in C[i] if np.isfinite(u)] for i in range(n)]
    score = [np.full(len(nodes[i]), -np.inf) for i in range(n)]
    back = [[None] * len(nodes[i]) for i in range(n)]
    best_upto = -np.inf; best_upto_at = None                       # best path end seen so far (for restarts)
    best_hist = []
    for i in range(n):
        f = start + i
        for j, (u, v, lg) in enumerate(nodes[i]):
            gain = lg - REWARD_AT
            cand = [(gain - RESTART_COST + max(0.0, best_upto), best_upto_at if best_upto > 0 else None)]
            pp = ppm(M[f], (u, v))
            for k in range(1, MAX_SKIP + 1):
                if i - k < 0:
                    break
                Hk = H(M, f - k, f)
                for jj, (u2, v2, _) in enumerate(nodes[i - k]):
                    if not np.isfinite(score[i - k][jj]):
                        continue
                    p2 = warp(Hk, (u2, v2)); dm = np.hypot(u - p2[0], v - p2[1]) / pp
                    if dm > VMAX_MPF * k:
                        continue
                    s = score[i - k][jj] + gain - SKIP_COST * (k - 1) - MOVE_COST * dm
                    cand.append((s, (i - k, jj)))
            s, b = max(cand, key=lambda x: x[0])
            score[i][j] = s; back[i][j] = b
        # paths may only be resumed through a restart after they end: track the best finished score
        if len(nodes[i]):
            j = int(np.argmax(score[i]))
            if score[i][j] > best_upto:
                best_upto, best_upto_at = score[i][j], (i, j)
        best_hist.append(best_upto)
    # backtrack from the best end overall
    track = {}
    at = best_upto_at
    while at is not None:
        i, j = at; u, v, lg = nodes[i][j]; track[start + i] = (u, v, lg); at = back[i][j]
    json.dump({str(f): list(p) for f, p in sorted(track.items())}, open(WORK + '/track.json', 'w'))
    print(f'linked: ball on {len(track)}/{n} frames ({len(track) / n:.0%})')


# ---------------------------------------------------------------- 3. events
def fit_side(M, track, f, frames):
    """positions of `frames` (those tracked) mapped into frame f's pixels -> (pos, vel) at f.

    The motion is fitted as p0 + v t + a t^2 / 2 with the acceleration FIXED to either zero (rolling, or a short
    straight stretch) or gravity as it appears at this spot of the image (in flight), whichever fits better. A ball
    in flight then follows its own curve instead of drifting away from a straight line and faking a break."""
    pts = [(g - f, warp(H(M, g, f), track[g][:2])) for g in frames if g in track]
    if len(pts) < MIN_SIDE:
        return None
    t = np.array([p[0] for p in pts], float); X = np.array([p[1] for p in pts])
    up, pxm = vertical(M[f], X[np.argmin(np.abs(t))])
    gvec = -up * G * pxm / 625.0                                    # gravity, px / frame^2, in frame f's image
    A = np.stack([np.ones_like(t), t], 1)
    best = None
    for acc in (np.zeros(2), gvec):
        Y = X - 0.5 * np.outer(t * t, acc)                          # remove the fixed acceleration, fit the rest linearly
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        res = float(np.sum((A @ coef - Y) ** 2))
        if best is None or res < best[0]:
            best = (res, coef[0], coef[1])
    return best[1], best[2]


def vertical(Mf, pos):
    """(unit image direction of 'up' at image point pos, pixels per vertical metre there)."""
    q = np.linalg.inv(Mf) @ [pos[0], pos[1], 1.0]; Q = q[:2] / q[2] * [L, W]
    feet, top = column(Mf, Q); d = top - feet
    return d / np.linalg.norm(d), float(np.linalg.norm(d) / BODY_H)


def fit_segment(M, track, frames):
    """Constant-acceleration fit of a stretch in its middle frame's pixels -> dict(ref, a, kind, g_frac, side_frac)."""
    fr = [g for g in frames if g in track]
    if len(fr) < MIN_SEG:
        return None
    ref = fr[len(fr) // 2]
    t = np.array([g - ref for g in fr], float); X = np.array([warp(H(M, g, ref), track[g][:2]) for g in fr])
    cu = np.polyfit(t, X[:, 0], 2); cv = np.polyfit(t, X[:, 1], 2)
    acc = np.array([2 * cu[0], 2 * cv[0]])                                   # px / frame^2
    up, pxm = vertical(M[ref], X[len(fr) // 2]); gpx = G * pxm / 625.0       # gravity, px / frame^2
    g_frac = float(-(acc @ up) / gpx)                                         # downward acceleration / gravity
    side = float(abs(acc[0] * up[1] - acc[1] * up[0]) / gpx)
    kind = ('flight' if FLIGHT_G[0] <= g_frac <= FLIGHT_G[1] and side < 0.6 else
            'roll' if np.hypot(g_frac, side) < ROLL_G else 'unclear')
    return dict(f0=fr[0], f1=fr[-1], kind=kind, g_frac=round(g_frac, 2), side_frac=round(side, 2))


def nearest_body(M, P, f, pos):
    """-> (distance m, slot index, height on the body m) of the player column nearest image point pos."""
    b = bodies(M, P, f, pos)
    return b[0][:3] if b else None


def bodies(M, P, f, pos):
    """All player columns, nearest first: [(dist m, slot index, height on the body m, closest image point)]."""
    people = [(s, P[f, s] * [L, W]) for s in range(P.shape[1])
              if np.isfinite(P[f, s, 0]) and not (STALE is not None and STALE[f, s])]
    out = []
    for s, G_ in people:
        feet, top = column(M[f], G_); pp = ppm(M[f], feet)
        ab = top - feet; tt = float(np.clip((pos - feet) @ ab / max(1e-9, ab @ ab), 0, 1)); q = feet + tt * ab
        out.append((float(np.linalg.norm(pos - q) / pp), s, tt * BODY_H, q))
    return sorted(out, key=lambda x: x[0])


STALE = None            # (frames, slots) bool, set by cmd_events: frozen tracklets


def stale_mask(P):
    """True where a slot's position is identical to STALE_FRAMES frames earlier (a frozen, stale tracklet)."""
    out = np.zeros(P.shape[:2], bool)
    same = np.all(np.abs(P[STALE_FRAMES:] - P[:-STALE_FRAMES]) < 1e-6, axis=2) & np.isfinite(P[STALE_FRAMES:, :, 0])
    out[STALE_FRAMES:] = same
    return out


def goal_sides(P, frames):
    """{team: 'left'|'right'} -- the end each team defends, from where its players stand over these frames."""
    xs = {'H': P[frames, :24, 0], 'A': P[frames, 24:, 0]}
    med = {k: np.nanmedian(v) for k, v in xs.items()}
    return {'H': 'left', 'A': 'right'} if med['H'] < med['A'] else {'H': 'right', 'A': 'left'}


def sides_by_half(P):
    """frame -> {team: 'left'|'right'}: the ends defended in that frame's half (the teams swap them at half-time), each
    half's from goal_sides over that half alone; a frame outside both halves takes the nearer half's."""
    import sys
    sys.path.insert(0, __import__('paths').REPO)
    from games import GAME, half_video_blocks
    halves = [(a, b, goal_sides(P, np.arange(a, b + 1, 25))) for _h, a, b in half_video_blocks(GAME)]
    return lambda f: min(halves, key=lambda h: 0 if h[0] <= f <= h[1] else min(abs(f - h[0]), abs(f - h[1])))[2]


ATD_XML = __import__('paths').DATA + 'tactical/BAR-ATH_tactical_atd.xml'


def keeper_slots():
    """Slot indices Metrica's metadata marks position_type = Goalkeeper (Track 24 home, Track 46 away)."""
    import re
    xml = open(ATD_XML).read()
    return [int(i) - 1 for i, body in re.findall(r'<Player id="(\d+)"[^>]*>(.*?)</Player>', xml, re.S)
            if re.search(r'<Name>position_type</Name>\s*<Value>Goalkeeper</Value>', body)]


KEEPERS = None


def box_team(g, sides):
    """Team whose penalty area ground point g (m) is in, or None."""
    if abs(g[1] - W / 2) > BOX_HALF_W_M:
        return None
    if g[0] <= BOX_DEPTH_M:
        return next(t for t, sd in sides.items() if sd == 'left')
    if g[0] >= L - BOX_DEPTH_M:
        return next(t for t, sd in sides.items() if sd == 'right')
    return None


def keeper_team(P, f, s, sides):
    """A goalkeeper track belongs to the team defending the half he is in -- keepers do not cross halfway, while
    Metrica's keeper tracks do swap (#46 follows the HOME keeper at times, a frozen track drifts) -- not to his slot's.
    sides: sides_by_half(P)."""
    return next(t for t, sd in sides(f).items() if sd == ('left' if P[f, s, 0] * L < L / 2 else 'right'))


def goalkeepers_at(P, f, sides):
    """[(slot index, team)] of live, non-frozen goalkeeper tracks at frame f."""
    return [(s, keeper_team(P, f, s, sides)) for s in KEEPERS
            if np.isfinite(P[f, s, 0]) and not (STALE is not None and STALE[f, s])]


def keeper_claims(M, P, track, f0, f1, sides):
    """Touch events for a keeper smothering/catching a ball we cannot see: the ball vanishes for CLAIM_GAP_S or
    longer and is last seen, or first seen again, within CLAIM_REACH_M of a goalkeeper track inside a penalty
    area. The claim goes to the team defending that box. Last seen at the keeper: dated when it vanished. First seen
    again with him: the ball was loose, and reached him as it was going -- dated when the ball, at the ground speed it
    was last seen at (over FIT_HALF frames), covers the distance from where it vanished to where he holds it."""
    out = []
    fs = sorted(track)
    for a, b in zip(fs, fs[1:]):
        if (b - a) / 25.0 < CLAIM_GAP_S:
            continue
        found = None
        for f, how in ((a, 'last seen at the keeper'), (b, 'reappears with the keeper')):
            pos = track[f][:2]
            for k, t in goalkeepers_at(P, f, sides):
                if box_team(P[f, k] * [L, W], sides(f)) is None:
                    continue
                feet, top = column(M[f], P[f, k] * [L, W]); ab = top - feet
                tt = float(np.clip((pos - feet) @ ab / max(1e-9, ab @ ab), 0, 1))
                dist = np.linalg.norm(pos - (feet + tt * ab)) / ppm(M[f], feet)
                if dist <= CLAIM_REACH_M:
                    found = (f, k, t, dist, how); break
            if found:
                break
        if found:
            f, k, t, dist, how = found
            if f == b:                                         # reappears with him: when did it reach him?
                g = lambda h: (lambda q: q[:2] / q[2] * [L, W])(np.linalg.inv(M[h]) @ [track[h][0], track[h][1], 1.0])
                before = [h for h in range(a - FIT_HALF, a + 1) if h in track]
                speed = np.linalg.norm(g(before[-1]) - g(before[0])) / ((before[-1] - before[0]) / 25.0) if len(before) > 1 else 0.0
                if speed > 0:
                    f = int(np.clip(a + round(np.linalg.norm(P[b, k] * [L, W] - g(a)) / speed * 25), a, b))
            uv = track[f][:2] if f in track else column(M[f], P[b, k] * [L, W])[0]   # unseen: where he holds it
            out.append(dict(f=f, kind='touch', slot=k + 1, team=t, dist_m=round(float(dist), 2), body='hands',
                            conf='keeper claim', claim=True, how=how, dv_ms=None,
                            u=round(float(uv[0]), 1), v=round(float(uv[1]), 1)))
    return out


def body_part(h):
    return 'foot' if h < 0.6 else 'body' if h < 1.5 else 'head'


def placed_before(M, track, f):
    """Was the ball placed before the break at f -- lying still (within CONTACT_M of where it lay) for PLACED_S, as
    for a set piece (free kick, goal kick, corner, kick-off)? Needs the ball seen over most of that second."""
    seen = [g for g in range(f - int(PLACED_S * 25), f) if g in track]
    if len(seen) < PLACED_S * 25 / 2:
        return False
    G = []
    for g in seen:
        q = np.linalg.inv(M[g]) @ [track[g][0], track[g][1], 1.0]; G.append(q[:2] / q[2] * [L, W])
    G = np.array(G)
    return bool(np.max(np.linalg.norm(G - np.median(G, 0), axis=1)) < CONTACT_M)


def control_roles(touches, events):
    """Possession: a team has the ball until the other team takes it. The other team takes it when one of its players
    touches it and
      the ball reached him loose -- since the previous touch something else happened to it (a bounce, a deflection
      off nobody): nobody had it, so whoever plays it has it; or
      it came straight off the other team's play, and his team's next touch is theirs again (or no touch follows).
    Otherwise it was only contact with the other team's play: a poke, a ricochet, a block. A keeper claim takes it at
    once. Sets e['role'] on every touch (control, interception, block, keeper claim) and returns the possession
    changes [{f, team, via}], dated at the touch that took it."""
    changes, current = [], None
    others = sorted(x['f'] for x in events if x['kind'] != 'touch')
    for i, e in enumerate(touches):
        if e['team'] == current:
            e['role'] = 'keeper claim' if e.get('claim') else 'control'; continue
        nxt = touches[i + 1] if i + 1 < len(touches) else None
        prev = touches[i - 1]['f'] if i else None
        loose = prev is not None and bisect.bisect_right(others, prev) < bisect.bisect_left(others, e['f'])
        if current is None or e.get('claim') or loose or nxt is None or nxt['team'] == e['team']:
            e['role'] = 'keeper claim' if e.get('claim') else 'control' if current is None else 'interception'
            changes.append(dict(f=e['f'], team=e['team'], via='start' if current is None else e['role']))
            current = e['team']
        else:
            e['role'] = 'block'
    return changes


def window_breaks(M, track, f0, f1):
    """breaks in the ball's motion from short fits either side of every frame -> (frames, {frame: (dv, pos, vin, vout)})."""
    # 1. breaks: local maxima of the velocity change between short fits either side
    dv = {}
    for f in range(f0 + MIN_SIDE, f1 - MIN_SIDE):
        left = fit_side(M, track, f, range(f - FIT_HALF, f))
        right = fit_side(M, track, f, range(f + 1, f + FIT_HALF + 1))
        if left is None or right is None:
            continue
        pos = track[f][:2] if f in track else (left[0] + right[0]) / 2
        if np.linalg.norm(left[0] - right[0]) / ppm(M[f], pos) > 3.0:     # the two sides are not the same ball
            continue
        dv[f] = (np.linalg.norm(right[1] - left[1]) / ppm(M[f], pos) * 25.0, pos, left[1], right[1])
    peaks = sorted(f for f, (d, *_r) in dv.items()
                   if d >= BREAK_MS and not any(dv.get(g, (0,))[0] > d for g in range(f - 4, f + 5) if g != f))
    # time each break at the first frame already on the outgoing path
    breaks, shifts = [], []
    for b in peaks:
        onset = b
        for c in range(b + CONTACT_SEARCH[0], b + CONTACT_SEARCH[1] + 1):
            if c not in track:
                continue
            inc = fit_side(M, track, c, range(c - CONTACT_FIT, c))               # incoming path, predicted at c
            out = fit_side(M, track, c, range(c + 1, c + CONTACT_FIT + 1))       # outgoing path, back to c
            if inc is None or out is None:
                continue
            if np.linalg.norm(track[c][:2] - out[0]) < np.linalg.norm(track[c][:2] - inc[0]):
                onset = c; break
        shifts.append(onset - b)
        dv.setdefault(onset, dv[b])
        breaks.append(onset)
    breaks = sorted(set(breaks))
    from collections import Counter as _C
    print('break timing moved to the first frame on the outgoing path by (frames):', dict(sorted(_C(shifts).items())))
    return breaks, dv


def contact_point(M, track, f):
    """Where the ball was struck at the break f: where its incoming path (fitted on the FIT_HALF frames before f) and its
    outgoing path (the FIT_HALF frames after) come closest -- a ball changes course only where it is played. The moment
    lies between the last frame on the old path and the first on the new (f is the first), one frame of slack either
    side. -> (frame g nearest that moment, the point in g's pixels), or None without both fits."""
    left = fit_side(M, track, f, range(f - FIT_HALF, f)); right = fit_side(M, track, f, range(f + 1, f + FIT_HALF + 1))
    if left is None or right is None:
        return None
    (p_in, v_in), (p_out, v_out) = left, right
    dv = v_in - v_out
    tau = 0.0 if dv @ dv < 1e-9 else float(-(p_in - p_out) @ dv / (dv @ dv))
    tau = min(max(tau, -2.0), 1.0)
    c = (p_in + v_in * tau + p_out + v_out * tau) / 2
    g = int(round(f + tau))
    return g, warp(H(M, f, g), c)


def contact_bodies(M, P, track, f, pos):
    """-> (bodies nearest first, the ball's point they are measured from): at the contact point of the incoming and
    outgoing paths (contact_point); without both paths, at the ball's position on the break frame."""
    cp = contact_point(M, track, f)
    if cp is not None:
        g, c = cp
        return bodies(M, P, g, c), c
    return bodies(M, P, f, pos), pos


def at_player(P, track, f, pos, s, reach):
    """Is the ball within reach (m) of player s's body column, measured in 3D from both cameras (broadcast_check.py) at
    the break f (the ball's pixel pos) or, failing that, the frame before it? None where the broadcast cannot say."""
    import broadcast_check
    for g, uv in ((f, pos), (f - 1, track.get(f - 1))):
        b = broadcast_check.ball_at(g, uv[:2]) if uv is not None else None
        if b is None:
            continue
        xy = P[g, s] * [L, W]
        if not np.isfinite(xy).all():
            xy = P[f, s] * [L, W]
        return bool(np.linalg.norm([b[0] - xy[0], b[1] - xy[1], b[2] - np.clip(b[2], 0, BODY_H)]) <= reach)
    return None


def cmd_events():
    from cameras import load_h
    M, _, _ = load_h('tactical'); t = np.load(S + 'tactical.npz'); P = t['P'][..., :2]
    z = np.load(PLAYER_TRACKS); P = z['P']   # Metrica's slots kept on the picture (player_track.py): its lost and hidden players back in place
    print(f'players: player_track.py over {z["frames"][0]}-{z["frames"][1]}, Metrica elsewhere', flush=True)
    team = ['H'] * 24 + ['A'] * 22
    track = {int(k): np.array(v) for k, v in json.load(open(WORK + '/track.json')).items()}
    fs = sorted(track); f0, f1 = fs[0], fs[-1]
    global STALE, KEEPERS
    STALE = stale_mask(P); KEEPERS = keeper_slots()
    sides = sides_by_half(P)
    breaks, dv = window_breaks(M, track, f0, f1)
    # 2. stretches between breaks (and dropouts), each fitted with constant acceleration
    cuts = sorted(set(breaks) | {g for g in fs if g - 1 not in track} | {g + 1 for g in fs if g + 1 not in track})
    segs = []
    for a, b in zip(cuts, cuts[1:]):
        sg = fit_segment(M, track, range(a, b))
        if sg:
            segs.append(sg)
    seg_at = lambda f: next((sg for sg in segs if sg['f0'] <= f <= sg['f1']), None)
    # 3. classify each break: bounce (velocity change straight up the local vertical while falling) or touch
    events, asked = [], []
    for f in breaks:
        d, pos, vin, vout = dv[f]
        dvec = vout - vin
        up, _ = vertical(M[f], pos)
        cos_up = float(dvec @ up / max(1e-9, np.linalg.norm(dvec)))
        falling = float(vin @ up) < 0
        bounce_like = falling and cos_up >= BOUNCE_COS
        reach = min(MAX_REACH_M, REACH_BASE_M + REACH_PER_MS * d)
        touch = None
        near_c, at = contact_bodies(M, P, track, f, pos)
        for dist, sidx, h, point in near_c:
            if dist > max(reach, CONTACT_M):
                break
            away = at - point
            cos_away = float(dvec @ away / max(1e-9, np.linalg.norm(dvec) * np.linalg.norm(away))) if np.linalg.norm(away) > 1e-6 else 1.0
            if dist <= CONTACT_M or (not bounce_like and cos_away > 0):
                touch = (dist, sidx, h); break
        if touch and bounce_like:                              # the picture cannot tell: the broadcast camera, if it sees
            seen = at_player(P, track, f, pos, touch[1], max(reach, CONTACT_M))
            asked.append(seen)
            if seen is False:
                touch = None                                   # it puts the ball away from him
        before, after = seg_at(f - 1), seg_at(f + 1)
        k_ms = 25.0 / ppm(M[f], pos)
        e = dict(f=f, dv_ms=round(d, 1), v_in=round(float(np.linalg.norm(vin)) * k_ms, 1), v_out=round(float(np.linalg.norm(vout)) * k_ms, 1),
                 u=round(float(pos[0]), 1), v=round(float(pos[1]), 1), conf='break',
                 cos_up=round(cos_up, 2), before=before['kind'] if before else None, after=after['kind'] if after else None)
        if touch:
            tm = keeper_team(P, f, touch[1], sides) if touch[1] in KEEPERS else team[touch[1]]
            e.update(kind='touch', slot=touch[1] + 1, team=tm, dist_m=round(touch[0], 2), body=body_part(touch[2]))
        else:
            nb = nearest_body(M, P, f, pos)
            e.update(kind='bounce' if bounce_like else 'deflection', nearest=nb[1] + 1 if nb else None, dist_m=round(nb[0], 2) if nb else None)
        events.append(e)
    print(f'{len(asked)} breaks that look like bounces, a body over them in the picture: the broadcast camera sees the ball '
          f'at {sum(a is not None for a in asked)}, and puts it beyond his reach at {sum(a is False for a in asked)} -- not '
          f'touches', flush=True)
    events = sorted(events + keeper_claims(M, P, track, f0, f1, sides), key=lambda e: e['f'])
    # a ball high in the air passing over a player reads as a touch at head height (its ray crosses his head). Metrica's
    # ball is smoothed too much to see a quick touch, but a high ball does travel straight: where Metrica's ball goes
    # straight through (STRAIGHT_DEG) a head-height touch, the ball flew past
    Bm = np.load(S + 'tactical.npz')['B'][:, :2] * [L, W]
    def straight(f, k=12):
        if f < k or not np.isfinite(Bm[f - k:f + k + 1]).all():
            return False
        a, b = Bm[f] - Bm[f - k], Bm[f + k] - Bm[f]
        return np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9), -1, 1))) < STRAIGHT_DEG
    flew = [e['f'] for e in events if e['kind'] == 'touch' and e.get('body') == 'head' and straight(e['f'])]
    events = [e for e in events if e['f'] not in flew]
    print(f'{len(flew)} head-height touches dropped: the ball flew past (Metrica straight through)')
    merged = []
    for e in events:
        if merged and e['kind'] == 'touch' and merged[-1]['kind'] == 'touch' and e['slot'] and merged[-1]['slot'] == e['slot'] and e['f'] - merged[-1]['f'] <= 8:
            continue
        merged.append(e)
    touches = [e for e in merged if e['kind'] == 'touch']
    # possession half by half: each half starts afresh with its kick-off; the warm-up, half-time and after the final
    # whistle (the ball still moves about) are not play
    sys.path.insert(0, __import__('paths').REPO)
    from games import GAME, half_video_blocks
    possession = []
    for _half, h0, h1 in half_video_blocks(GAME):
        possession += control_roles([e for e in touches if h0 <= e['f'] <= h1], [e for e in merged if h0 <= e['f'] <= h1])
    # 4. ball played between consecutive touches: lofted if any stretch in between is a flight
    plays = []
    for a, b in zip(touches, touches[1:]):
        Ga, Gb = (P[e['f'], e['slot'] - 1] * [L, W] if e['slot'] else np.full(2, np.nan) for e in (a, b))
        length = float(np.linalg.norm(Gb - Ga)) if np.all(np.isfinite([*Ga, *Gb])) else None
        kinds = [sg['kind'] for sg in segs if sg['f1'] > a['f'] and sg['f0'] < b['f']]
        bounces = sum(1 for e in merged if a['f'] < e['f'] < b['f'] and e['kind'] == 'bounce')
        same = a['slot'] is not None and a['slot'] == b['slot']
        if same and (length or 0) < DRIBBLE_M:
            kind = 'dribble'
        else:
            kind = ('long ' if (length or 0) >= LONG_M else '') + ('lofted ball' if 'flight' in kinds else 'ground pass')
        outcome = 'kept' if same else ('completed' if a['team'] == b['team'] else 'lost')
        plays.append(dict(f0=a['f'], f1=b['f'], from_slot=a['slot'], from_team=a['team'], to_slot=b['slot'], to_team=b['team'],
                          kind=kind, outcome=outcome, length_m=None if length is None else round(length, 1), bounces=bounces,
                          max_h=2.0 if 'flight' in kinds else 0.0))
    restarts = [f for f in breaks if placed_before(M, track, f)]
    json.dump(dict(frames=[f0, f1], track={str(f): [round(float(x), 1) for x in p[:2]] for f, p in track.items()},
                   events=merged, plays=plays, segments=segs, possession=possession, restarts=restarts), open(OUT_JSON, 'w'))
    from collections import Counter
    print(f'roles: {Counter(e.get("role", "outside the halves") for e in touches)};  possession changes: {len(possession)}')
    print(f'{len(touches)} touches, {Counter(e["kind"] for e in merged)}, {len(plays)} balls played; stretches: {Counter(s["kind"] for s in segs)} -> {OUT_JSON}')

if __name__ == '__main__':
    {'detect': lambda: cmd_detect(int(sys.argv[2]), int(sys.argv[3])), 'link': cmd_link, 'events': cmd_events}[sys.argv[1]]()
