"""The ball in 3D where the tactical camera alone cannot say: from the broadcast camera as well.

One camera puts the ball on a ray. The tactical camera looks down at the pitch, so a ball bouncing on the grass several
metres behind a player sits on his body in the picture, and a break in its motion straight up the picture's vertical is
the same whether it bounced there or he took it on his body. The broadcast camera stands about 14 m away and is zoomed
about three times further in: there the two are tens of pixels apart. So where the touch rule cannot tell (ownball.py
events: a break that looks like a bounce, a body within CONTACT_M in the picture), the ball's ray from each camera says
where it really was.

The broadcast ball is ours: heatmap.py's ball detector on the live broadcast picture, shrunk by the two cameras' focal
lengths so a ball is the size the detector learned on, its TOP candidates from CAND_LOGIT up (ownball's). The tactical
ray is ball3d.py's: the refined homography (refine_h.py) and the fixed camera centre (ball3d.CAMERA). The broadcast
camera moves, so its centre comes from its supplied homography, frame by frame (cameras.decompose). The candidate whose ray passes nearest our tactical
ball's ray is the ball -- if the two rays meet (within ball3d.ON_RAYS_M) at a place not under the grass (ball3d.BELOW_M):
two lines in space do not meet by chance. Otherwise the broadcast says nothing. Live broadcast frames only (the
alignment, alignment/align_broadcast.py: no replays, wipes or the half-time join).

  ball_at(f, uv)     the ball's place (x, y m) and height (m) at tactical frame f, its tactical pixel uv; or None
                     (cached by frame and pixel in work/align/broadcast_ball3d.json)
"""
import csv, json, os, subprocess
import numpy as np
import cv2

import ownball as ob
import ball3d
import cameras
import refine_h

L, W = ob.L, ob.W
REPO = __import__('paths').REPO
VIDEO = __import__('paths').BROADCAST_VIDEO
ALIGN = __import__('paths').ALIGNMENT
CACHE = ob.S + 'broadcast_ball3d.json'
CAMERA_FRAMES = 5                 # each camera's centre and focal length: the median over this many frames either side

_state = {}


def _load():
    if not _state:
        _state['Mt'] = np.load(refine_h.OUT)                         # the tactical camera: refined, as ball3d.py
        _state['Mb'], _state['vb'], _ = cameras.load_h('broadcast')
        _state['live'] = {int(float(r['tactical_frame'])): int(r['broadcast_frame']) for r in csv.DictReader(open(ALIGN))
                          if r['kind'] == 'live' and r['tactical_frame']}
        _state['cache'] = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    return _state


def _camera(M, f):
    """(centre (m, z negative = up), focal length px) around frame f."""
    fl, _fb, _R, _t, C = cameras.decompose(M[max(0, f - CAMERA_FRAMES):f + CAMERA_FRAMES + 1])
    return np.nanmedian(C, 0), float(np.nanmedian(fl))


def _ray(M, f, C, uv):
    q = np.linalg.inv(M[f]) @ [uv[0], uv[1], 1.0]
    d = np.r_[q[:2] / q[2] * [L, W], 0.0] - C
    return d / np.linalg.norm(d)


def _meet(C1, d1, C2, d2):
    """the midpoint of the two rays' closest points, and how far apart those are (m)."""
    w = C1 - C2; a, b, c, d, e = d1 @ d1, d1 @ d2, d2 @ d2, d1 @ w, d2 @ w
    den = a * c - b * b
    p1 = C1 + (b * e - c * d) / den * d1; p2 = C2 + (a * e - b * d) / den * d2
    return (p1 + p2) / 2, float(np.linalg.norm(p1 - p2))


def _decode(b):
    raw = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-ss', f'{b / 25:.2f}', '-i', VIDEO, '-frames:v', '1', '-f', 'rawvideo',
                          '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3)


def _measure(f, uv):
    s = _load(); b = s['live'].get(f)
    if b is None or not s['vb'][b]:
        return None
    import heatmap
    if 'net' not in s:
        s['net'] = heatmap.load()
    Ct = ball3d.CAMERA * [1, 1, -1]                                    # z negative = up here (cameras.decompose's frame)
    ft, (Cb, fb) = _camera(s['Mt'], f)[1], _camera(s['Mb'], b)
    k = ft / fb                                                        # broadcast px -> the tactical scale
    if not (np.isfinite(k) and np.isfinite(Ct).all() and np.isfinite(Cb).all()):
        return None                                                    # a homography with no camera behind it
    small = cv2.resize(_decode(b), (int(1920 * k), int(1080 * k)), interpolation=cv2.INTER_AREA)
    dt = _ray(s['Mt'], f, Ct, uv)
    best = None
    for u, v, _logit in heatmap.peaks_in(s['net'], small, (0, 0, small.shape[1], small.shape[0]), thresh=ob.CAND_LOGIT, top=ob.TOP):
        p, miss = _meet(Ct, dt, Cb, _ray(s['Mb'], b, Cb, np.array([u, v]) / k))
        if best is None or miss < best[1]:
            best = (p, miss)
    if best is None or best[1] > ball3d.ON_RAYS_M or -best[0][2] < -ball3d.BELOW_M:
        return None
    p, _ = best
    return [round(float(p[0]), 3), round(float(p[1]), 3), round(float(-p[2]), 3)]


def ball_at(f, uv):
    """[x, y, height] (m) of the ball at tactical frame f, its tactical pixel uv, from both cameras; or None."""
    s = _load(); key = f'{f}:{uv[0]:.1f},{uv[1]:.1f}'
    if key not in s['cache']:
        s['cache'][key] = _measure(f, uv)
        json.dump(s['cache'], open(CACHE, 'w'))
    return s['cache'][key]
