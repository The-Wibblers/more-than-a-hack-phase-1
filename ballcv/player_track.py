"""Metrica's players, kept on the picture: each of its 46 slots tracked on the player detector's players (player_det.py).

Metrica loses players exactly where they matter: one hidden behind another. Home #11 runs beside away #35 from
~11240, is lost behind him at ~11285, and is gone for ~120 frames while the two run down to the far-touchline duel;
Metrica then starts a new slot -- an away one, #33 -- on him. The detector sees him there, in the home kit.

One tracker, a frame at a time. Every slot is a track; a slot's place this frame is expected at
  Metrica's own position, when Metrica has him (live, not frozen), within POS_M (Metrica's error and the detector's)
    -- unless that is beyond running reach of where we had him a frame ago: Metrica has moved his number onto
    another player;
  otherwise where he was last seen moving on at his recent velocity, within POS_M + how far a player runs (RUN_MS)
  since then.
The detector's players (UNSURE and up, players/dets/) are matched one-to-one to the slots, nearest overall (Hungarian),
each only to a slot of his own kit: an outfield kit to that team's outfield slots, a keeper kit to either keeper slot
(Metrica's keeper tracks swap ends). A matched slot is where the picture has him. A slot with no match is at Metrica's
position when Metrica has one. Otherwise he is hidden: just lost where another player's body covers his in the
picture, he is behind that player and moves with him, at the same offset, until the picture sees him again (within
POS_M of there and that offset); lost in the open, he coasts on his velocity, at most COAST_S, then ends until Metrica
or a match picks him up again. A team has eleven players: when it comes out with more, its hidden and coasting
slots end, the one unseen longest first -- Metrica has found him again under another number.
A slot the picture does not see, within POS_M of a teammate it does, is that teammate under a second number, and ends:
Metrica sometimes carries one player on two numbers (away #35 and the #33 it starts beside him above). Frames with no detections run (not in players/dets/) keep Metrica's positions.

  python ballcv/player_track.py run A B     -> players/tracks.npz: P as tactical.npz (normalised, NaN where nobody),
                                               src (0 none, 1 Metrica, 2 picture, 3 coasting); ownball.py events reads it
  python ballcv/player_track.py show A B S  slot S's place and source every 5 frames
"""
import glob, os, sys
import numpy as np
from scipy.optimize import linear_sum_assignment

import ownball as ob
import kit_team as kt

ROOT = ob.S + 'players'
OUT = ob.PLAYER_TRACKS
POS_M = 1.0              # Metrica's position vs the picture's for the same player (head-only detections ~0.6 m off)
RUN_MS = 9.0             # the fastest a footballer runs, m/s
COAST_S = 2.0            # longest a player is carried on, unseen
VEL_FRAMES = 5           # his velocity: over this many frames
ELEVEN = 11
NONE, METRICA, PICTURE, COAST, HIDDEN = 0, 1, 2, 3, 4


def load_dets(a, b):
    """{frame: (G (n, 2) m, kit (n,))} for frames a..b with detections run, and the set of those frames."""
    out, done = {}, set()
    for fn in glob.glob(f'{ROOT}/dets/*.npz'):
        z = np.load(fn)
        fr = z['frames']; fr = fr[(fr >= a) & (fr <= b)]
        if not len(fr):
            continue
        done |= set(fr.tolist())
        keep = (z['f'] >= a) & (z['f'] <= b)
        f, G, k = z['f'][keep], z['G'][keep], z['kit'][keep]
        for g in np.unique(f):
            out[int(g)] = (G[f == g].astype(float), k[f == g].astype(int))
    return out, done


def allowed(n_slots, keepers):
    """(slots, kits) bool: which kit a slot may take -- kit_team.CLASSES order H, A, H keeper, A keeper."""
    ok = np.zeros((n_slots, len(kt.CLASSES)), bool)
    for s in range(n_slots):
        if s in keepers:
            ok[s, 2] = ok[s, 3] = True
        else:
            ok[s, 0 if ob.slot_teams()[s] == 'H' else 1] = True
    return ok


def box(f, G):
    """A player standing at G (m) in frame f's picture: (x0, y0, x1, y1), kit_team's body box."""
    feet, top = ob.column(kt.M[f], G); h = np.linalg.norm(top - feet); cx = (feet[0] + top[0]) / 2
    return cx - kt.BODY_W * h / 2, min(feet[1], top[1]), cx + kt.BODY_W * h / 2, max(feet[1], top[1])


def overlap(b1, b2):
    return b1[0] < b2[2] and b2[0] < b1[2] and b1[1] < b2[3] and b2[1] < b1[3]


def track(a, b):
    t = np.load(ob.S + 'tactical.npz'); Pn = t['P'][..., :2]
    stale = ob.stale_mask(Pn)
    Pm = Pn * [ob.L, ob.W]
    n = Pm.shape[1]
    dets, done = load_dets(a, b)
    ok = allowed(n, set(ob.keeper_slots()))
    P = Pn.copy(); src = np.where(np.isfinite(Pn[..., 0]), METRICA, NONE).astype(np.int8)
    last = np.full(n, -10 ** 9); hist = [[] for _ in range(n)]          # hist: [(frame, place)] seen, not coasting
    behind = np.full(n, -1); offset = np.zeros((n, 2))                  # hidden behind that slot, at that offset
    prev = np.full((n, 2), np.nan)                                       # every slot's place on the frame before
    for f in range(a, b + 1):
        if f not in done:
            prev[:] = np.nan; behind[:] = -1
            continue
        have = np.isfinite(Pm[f, :, 0]) & ~stale[f]
        # Metrica's slot beyond running reach of where we had him a frame ago is another player under his number
        jump = have & np.isfinite(prev[:, 0]) & (np.linalg.norm(np.nan_to_num(Pm[f] - prev), axis=1) > POS_M + RUN_MS / 25.0)
        have &= ~jump
        vel =np.array([(h[-1][1] - h[0][1]) / max(1, h[-1][0] - h[0][0]) if len(h) > 1 else np.zeros(2) for h in hist])
        seen = np.array([h[-1][1] if h else np.full(2, np.nan) for h in hist])      # where last seen (not coasting)
        hidden = ~have & (behind >= 0) & np.isfinite(prev[np.maximum(behind, 0), 0])
        alive = have | hidden | (np.isfinite(seen[:, 0]) & (f - last <= COAST_S * 25))
        pred = np.where(have[:, None], Pm[f], seen + vel * (f - last)[:, None])
        gate = np.where(have, POS_M, POS_M + RUN_MS * (f - last) / 25.0)
        pred[hidden] = prev[behind[hidden]] + offset[hidden]
        gate[hidden] = POS_M + np.linalg.norm(offset[hidden], axis=1)
        G, kit = dets.get(f, (np.zeros((0, 2)), np.zeros(0, int)))
        matched = {}
        if len(G) and alive.any():
            d = np.linalg.norm(pred[:, None] - G[None], axis=2)
            bad = ~alive[:, None] | ~ok[:, kit] | ~(d <= gate[:, None])
            cost = np.where(bad, 1e6, d)
            for s, j in zip(*linear_sum_assignment(cost)):
                if not bad[s, j]:
                    matched[s] = G[j]
        place = np.full((n, 2), np.nan)                                  # the slots seen this frame
        for s in range(n):
            if s in matched:
                place[s] = matched[s]; src[f, s] = PICTURE
            elif have[s]:
                place[s] = Pm[f, s]; src[f, s] = METRICA
        for s in range(n):
            if np.isfinite(place[s, 0]):
                behind[s] = -1; last[s] = f
                hist[s] = (hist[s] + [(f, place[s])])[-VEL_FRAMES:]
                continue
            if behind[s] < 0 and last[s] == f - 1:
                # just lost: hidden behind the player whose body covers his in the picture, the nearest such
                me = box(f, seen[s]); cx = (me[0] + me[2]) / 2
                cover = [(abs((bx[0] + bx[2]) / 2 - cx), o) for o in range(n) if o != s and np.isfinite(place[o, 0])
                         for bx in [box(f, place[o])] if overlap(me, bx)]
                if cover:
                    o = min(cover)[1]; behind[s] = o; offset[s] = seen[s] - place[o]
            if behind[s] >= 0 and np.isfinite(place[behind[s], 0]):
                place[s] = place[behind[s]] + offset[s]; src[f, s] = HIDDEN
            elif alive[s] and not hidden[s]:
                behind[s] = -1; place[s] = pred[s]; src[f, s] = COAST
            else:
                behind[s] = -1; src[f, s] = NONE; hist[s] = []
        # a slot the picture does not see, within POS_M of a teammate it does, is that teammate under a second number
        for team in (slice(0, 24), slice(24, n)):
            idx = np.arange(n)[team]
            seen_now = [o for o in idx if src[f, o] == PICTURE]
            for s in idx:
                if src[f, s] != PICTURE and np.isfinite(place[s, 0]) and any(
                        np.linalg.norm(place[s] - place[o]) <= POS_M for o in seen_now):
                    place[s] = np.nan; src[f, s] = NONE; behind[s] = -1; hist[s] = []
        # a team has eleven on the pitch: over that, its slots unseen longest end -- found again under another number
        for team in (slice(0, 24), slice(24, n)):
            idx = np.arange(n)[team]
            extra = np.isfinite(place[idx, 0]).sum() - ELEVEN
            unseen = sorted((last[s], s) for s in idx if src[f, s] in (HIDDEN, COAST))
            for _l, s in unseen[:max(0, extra)]:
                place[s] = np.nan; src[f, s] = NONE; behind[s] = -1; hist[s] = []
        P[f] = place / [ob.L, ob.W]; prev = place
    return P, src


def robust_sd(x):
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def noise(P, src, a, b):
    """The smoother's three noise levels, measured on this match: (picture sd, Metrica sd, acceleration sd), m and m/s2.
    The picture's: its positions' second differences on runs seen in the picture (white noise sd s gives 6 s^2 there).
    Metrica's: how far it sits from the picture where both have a player, less the picture's share. How sharply
    players change speed: Metrica's own second differences (its tracks are smooth)."""
    t = np.load(ob.S + 'tactical.npz')['P'][a:b + 1, :, :2] * [ob.L, ob.W]
    live = np.isfinite(t[..., 0]) & ~ob.stale_mask(np.load(ob.S + 'tactical.npz')['P'][..., :2])[a:b + 1]
    Pm, sw = P[a:b + 1] * [ob.L, ob.W], src[a:b + 1]
    pic = (sw[2:] == PICTURE) & (sw[1:-1] == PICTURE) & (sw[:-2] == PICTURE)
    d2 = Pm[2:] - 2 * Pm[1:-1] + Pm[:-2]
    s_pic = robust_sd(d2[pic].ravel()) / np.sqrt(6)
    both = (sw == PICTURE) & live
    s_met = np.sqrt(max(robust_sd((Pm - t)[both].ravel()) ** 2 - s_pic ** 2, 1e-4))
    m2 = live[2:] & live[1:-1] & live[:-2]
    s_acc = robust_sd(((t[2:] - 2 * t[1:-1] + t[:-2]) * 25 ** 2)[m2].ravel())
    return s_pic, s_met, s_acc


def smooth(P, src, a, b, group=6):
    """Every slot's path through a constant-velocity Kalman filter and its smoother (forwards and back): the picture,
    Metrica and a hidden player's place behind another are measurements of their own noise; a coasting slot has none.
    A slot's path starts again wherever it has had nobody. -> P smoothed (normalised)."""
    s_pic, s_met, s_acc = noise(P, src, a, b)
    print(f'noise measured on the match: picture {s_pic:.2f} m, Metrica {s_met:.2f} m, players accelerate {s_acc:.1f} m/s2 (sd)')
    dt = 1 / 25
    F = np.array([[1, dt], [0, 1]]); Q = s_acc ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
    var = {PICTURE: s_pic ** 2, METRICA: s_met ** 2, HIDDEN: POS_M ** 2, COAST: np.inf}
    Z = P[a:b + 1] * [ob.L, ob.W]; sw = src[a:b + 1]; T = len(Z)
    out = Z.copy()
    for g0 in range(0, Z.shape[1], group):
        z = Z[:, g0:g0 + group].reshape(T, -1)                                     # (T, series): each slot's x, then y
        sr = np.repeat(sw[:, g0:g0 + group], 2, axis=1)
        R = np.full(sr.shape, np.inf)
        for k, v in var.items():
            R[sr == k] = v
        valid = sr != NONE; K = z.shape[1]
        xp = np.zeros((T, K, 2)); Cp = np.zeros((T, K, 2, 2)); xf = np.zeros((T, K, 2)); Cf = np.zeros((T, K, 2, 2))
        x = np.zeros((K, 2)); C = np.zeros((K, 2, 2))
        for t in range(T):
            x = x @ F.T; C = F @ C @ F.T + Q
            new = valid[t] & (~valid[t - 1] if t else True)
            x[new] = np.c_[z[t, new], np.zeros(new.sum())]
            C[new] = np.array([[1.0, 0], [0, 0]]) * np.where(np.isfinite(R[t, new]), R[t, new], POS_M ** 2)[:, None, None] \
                + np.array([[0, 0], [0, RUN_MS ** 2]])
            xp[t], Cp[t] = x, C
            m = valid[t] & np.isfinite(R[t]) & ~new
            if m.any():
                S = C[m, 0, 0] + R[t, m]; Kg = C[m, :, 0] / S[:, None]
                x[m] = x[m] + Kg * (z[t, m] - x[m, 0])[:, None]
                C[m] = C[m] - Kg[:, :, None] * C[m, 0][:, None, :]
            xf[t], Cf[t] = x, C
        xs = xf.copy()
        for t in range(T - 2, -1, -1):
            m = valid[t] & valid[t + 1]
            if m.any():
                G = Cf[t, m] @ F.T @ np.linalg.inv(Cp[t + 1, m])
                xs[t, m] = xf[t, m] + np.einsum('kij,kj->ki', G, xs[t + 1, m] - xp[t + 1, m])
        sm = np.where(valid, xs[..., 0], np.nan)
        out[:, g0:g0 + group] = sm.reshape(T, -1, 2)
    Ps = P.copy(); Ps[a:b + 1] = out / [ob.L, ob.W]
    return Ps


def cmd_run(a, b):
    P, src = track(a, b)
    P = smooth(P, src, a, b)
    np.savez_compressed(OUT, P=P, src=src, frames=np.array([a, b]))
    w = src[a:b + 1]
    lost = (w != NONE) & ~np.isfinite(np.load(ob.S + 'tactical.npz')['P'][a:b + 1, :, 0])
    print(f'{a}-{b}: slot-frames from Metrica {np.mean(w == METRICA):.1%}, the picture {np.mean(w == PICTURE):.1%}, '
          f'coasting {np.mean(w == COAST):.1%}, hidden behind another {np.mean(w == HIDDEN):.1%}; {lost.sum()} slot-frames where Metrica has nobody -> {OUT}')


def cmd_show(a, b, slot):
    z = np.load(OUT); P, src = z['P'], z['src']
    Pm = np.load(ob.S + 'tactical.npz')['P'][..., :2]
    names = {NONE: '-', METRICA: 'Metrica', PICTURE: 'picture', COAST: 'coast', HIDDEN: 'hidden'}
    for f in range(a, b + 1, 5):
        q, m = P[f, slot - 1] * [ob.L, ob.W], Pm[f, slot - 1] * [ob.L, ob.W]
        print(f, 'ours', 'nobody' if not np.isfinite(q[0]) else q.round(1), names[int(src[f, slot - 1])],
              '  Metrica', 'nobody' if not np.isfinite(m[0]) else m.round(1))


if __name__ == '__main__':
    {'run': lambda: cmd_run(int(sys.argv[2]), int(sys.argv[3])),
     'show': lambda: cmd_show(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))}[sys.argv[1]]()
