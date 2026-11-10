"""The tactical camera's homography, snapped to the painted lines.

The supplied homography is smooth and close (painted lines within ~0.1 m), but a ball's place near a line needs better.
In every STEP-th frame we look across each painted line where the homography puts it (touchlines, goal lines, halfway,
penalty boxes, goal areas, centre circle) for the paint itself (a bright, unsaturated peak, to the sub-pixel), and
across each edge of the mowing stripes (bands ~6 m wide across the pitch, at fixed x: STRIPE_EDGES -- a ruler along the
pitch where little is painted) for the step in brightness, and adjust the homography's 8 parameters so the lines meet it (least squares on the distance across each line: paint says
where a line is, not where along it a point falls). The camera moves smoothly, so the adjustment, as an image-to-image
homography, is smoothed over SMOOTH frames and interpolated to every frame.

  python refine_h.py fit         -> work/align/homography_refined.npy   (n, 3, 3) pitch (normalised) -> pixel
  python refine_h.py check [N]   painted-line error before / after, on N frames that were not fitted
"""
import sys, time
import numpy as np
import cv2

import ownball as ob
from ownball import L, W

OUT = ob.S + 'homography_refined.npy'
STEP = 5
SMOOTH = 25                 # frames the correction is smoothed over (the camera moves smoothly)
SEARCH_PX = 14              # paint is looked for this far either side of the projected line
MIN_CONTRAST = 25           # paint is this much brighter than the grass across it ...
MAX_SAT = 70                # ... and white (low saturation)
MIN_POINTS = 12
# the mowing stripes: bands ~6 m wide across the pitch, their edges at these x (m) in every frame (measured on the
# line-snapped homography over 120 frames of both halves) -- a ruler along the pitch where few lines are painted
STRIPE_EDGES = [5.65, 11.05, 16.75, 22.15, 28.2, 34.35, 40.45, 46.6, 52.9, 58.85, 64.85, 71.0, 77.0, 83.25, 89.05, 94.35]
EDGE_SEARCH_PX = 10
MIN_STEP = 1.5              # an edge: the brightness gradient across it at least this (grey levels / px) ...
STEP_OVER_MEDIAN = 3        # ... and this many times the gradient elsewhere across it


def _seg(a, b, n):
    return [((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t), (b[0] - a[0], b[1] - a[1])) for t in np.linspace(0.02, 0.98, n)]


_C, _R = (L / 2, W / 2), 9.15
LINES = (_seg((0, 0), (L, 0), 80) + _seg((0, W), (L, W), 80)                                      # touchlines
         + _seg((0, 0), (0, W), 40) + _seg((L, 0), (L, W), 40) + _seg((L / 2, 0), (L / 2, W), 40)  # goal lines, halfway
         + sum([_seg((x, W / 2 - 20.16), (x, W / 2 + 20.16), 24) for x in (16.5, L - 16.5)], [])   # penalty boxes
         + sum([_seg((x0, y), (x1, y), 12) for x0, x1 in ((0, 16.5), (L - 16.5, L)) for y in (W / 2 - 20.16, W / 2 + 20.16)], [])
         + sum([_seg((x, W / 2 - 9.16), (x, W / 2 + 9.16), 10) for x in (5.5, L - 5.5)], [])       # goal areas
         + [((_C[0] + _R * np.cos(a), _C[1] + _R * np.sin(a)), (-np.sin(a), np.cos(a)))           # centre circle
            for a in np.linspace(0, 2 * np.pi, 48, endpoint=False)])
_P = np.array([p for p, _ in LINES]); _D = np.array([d for _, d in LINES]); _D /= np.linalg.norm(_D, axis=1, keepdims=True)


def proj(H, P):
    q = (H @ np.c_[P[:, 0] / L, P[:, 1] / W, np.ones(len(P))].T).T
    return q[:, :2] / q[:, 2:]


def observe(img, H, P=_P, D=_D):
    """(projected points, normals, the paint's offset across each line in px -- nan where no paint is seen)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sat = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32)
    a, b, c = proj(H, P - 0.5 * D), proj(H, P + 0.5 * D), proj(H, P)
    t = b - a; nrm = np.c_[-t[:, 1], t[:, 0]] / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    off = np.arange(-SEARCH_PX, SEARCH_PX + 1, dtype=np.float32)
    xs = (c[:, :1] + off[None] * nrm[:, :1]).astype(np.float32); ys = (c[:, 1:] + off[None] * nrm[:, 1:]).astype(np.float32)
    inside = (xs > 1) & (xs < img.shape[1] - 2) & (ys > 1) & (ys < img.shape[0] - 2)
    prof = cv2.remap(gray, xs, ys, cv2.INTER_LINEAR); s = cv2.remap(sat, xs, ys, cv2.INTER_LINEAR)
    rows = np.arange(len(P)); k = np.argmax(np.where(inside, prof, -1), 1); peak = prof[rows, k]
    ok = (inside.all(1) & (peak - np.median(prof, 1) > MIN_CONTRAST) & (s[rows, k] < MAX_SAT)
          & (k > 0) & (k < len(off) - 1))
    y0, y2 = prof[rows, np.clip(k - 1, 0, None)], prof[rows, np.clip(k + 1, None, len(off) - 1)]
    den = y0 - 2 * peak + y2
    sub = np.clip(0.5 * (y0 - y2) / np.where(np.abs(den) > 1e-6, den, 1e-6), -0.5, 0.5)    # sub-pixel peak
    return c, nrm, np.where(ok, off[k] + sub, np.nan)


_SP = np.array([(x, y) for x in STRIPE_EDGES for y in np.linspace(4, W - 4, 16)]); _SD = np.tile([0.0, 1.0], (len(_SP), 1))


def observe_edges(img, H, P=_SP, D=_SD):
    """like observe(), for the stripes' edges: the brightness step across each (the strongest gradient), sub-pixel."""
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32), (5, 5), 0)
    a, b, c = proj(H, P - 0.5 * D), proj(H, P + 0.5 * D), proj(H, P)
    t = b - a; nrm = np.c_[-t[:, 1], t[:, 0]] / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    off = np.arange(-EDGE_SEARCH_PX, EDGE_SEARCH_PX + 1, dtype=np.float32)
    xs = (c[:, :1] + off[None] * nrm[:, :1]).astype(np.float32); ys = (c[:, 1:] + off[None] * nrm[:, 1:]).astype(np.float32)
    inside = (xs > 1) & (xs < img.shape[1] - 2) & (ys > 1) & (ys < img.shape[0] - 2)
    gr = np.abs(np.gradient(cv2.remap(gray, xs, ys, cv2.INTER_LINEAR), axis=1))
    rows = np.arange(len(P)); k = np.argmax(np.where(inside, gr, -1), 1); pk = gr[rows, k]
    ok = (inside.all(1) & (pk > MIN_STEP) & (pk > STEP_OVER_MEDIAN * np.median(gr, 1)) & (k > 0) & (k < len(off) - 1))
    y0, y2 = gr[rows, np.clip(k - 1, 0, None)], gr[rows, np.clip(k + 1, None, len(off) - 1)]
    den = y0 - 2 * pk + y2
    sub = np.clip(0.5 * (y0 - y2) / np.where(np.abs(den) > 1e-6, den, 1e-6), -0.5, 0.5)
    return c, nrm, np.where(ok, off[k] + sub, np.nan)


def snap_frame(img, H):
    """H snapped to the painted lines and the stripes' edges."""
    c1, n1, o1 = observe(img, H); c2, n2, o2 = observe_edges(img, H)
    return snap(H, np.r_[c1, c2], np.r_[n1, n2], np.r_[o1, o2], np.r_[_P, _SP])


def snap(H, c, nrm, o, P=_P, iters=6):
    """H adjusted so the projected lines meet the paint (robust least squares across the lines); H if too little."""
    m = np.isfinite(o)
    if m.sum() < MIN_POINTS:
        return H
    P, nrm, target = P[m], nrm[m], c[m] + o[m, None] * nrm[m]
    h = (H / H[2, 2]).ravel()
    res = lambda hv: np.einsum('ij,ij->i', proj(np.r_[hv[:8], 1].reshape(3, 3), P) - target, nrm)
    for _ in range(iters):
        r0 = res(h); J = np.empty((len(P), 8))
        for j in range(8):
            d = np.zeros(9); d[j] = 1e-6 * max(1, abs(h[j])); J[:, j] = (res(h + d) - r0) / d[j]
        w = 1 / np.maximum(1, np.abs(r0) / 2)                                 # points far off count less
        h = h + np.r_[np.linalg.lstsq(J * w[:, None], -r0 * w, rcond=None)[0], 0]
    return np.r_[h[:8], 1].reshape(3, 3)


def cmd_fit():
    from cameras import load_h
    from video import blocks, sampled
    M, _, _ = load_h('tactical'); n = len(M)
    corr = {}; t = time.time()
    for _half, h0, h1 in blocks():
        for f, img in sampled(h0, h1, STEP):
            Hs = snap_frame(img, M[f])
            corr[f] = Hs @ np.linalg.inv(M[f])                               # the adjustment, image -> image
            if len(corr) % 4000 == 0:
                print(f'  frame {f}: {len(corr)} fitted ({time.time() - t:.0f} s)', flush=True)
    out = M.copy()
    for _half, h0, h1 in blocks():
        fs = np.array(sorted(f for f in corr if h0 <= f <= h1))
        C = np.array([corr[f] / corr[f][2, 2] for f in fs]).reshape(len(fs), 9)
        k = max(1, SMOOTH // STEP)
        Cs = np.array([np.median(C[max(0, i - k):i + k + 1], 0) for i in range(len(fs))])      # smooth over SMOOTH frames
        frames = np.arange(h0, h1 + 1)
        Ci = np.stack([np.interp(frames, fs, Cs[:, j]) for j in range(9)], 1).reshape(-1, 3, 3)
        out[h0:h1 + 1] = Ci @ M[h0:h1 + 1]
    np.save(OUT, out)
    print(f'{len(corr)} frames fitted in {time.time() - t:.0f} s; refined homography for every frame of both halves -> {OUT}')


def cmd_check(n):
    """painted-line error, supplied vs refined, on frames that were not fitted (between the fitted ones)."""
    from cameras import load_h
    from video import read_frames, blocks
    M, _, _ = load_h('tactical'); R = np.load(OUT)
    rng = np.random.default_rng(7)
    frames = [f for f in sorted(int(rng.integers(a, b)) for _h, a, b in blocks() for _ in range(n // 2)) if f % STEP]
    err = {'supplied': [], 'refined': []}
    for f in frames:
        img = next(iter(read_frames(f, 1)))
        for name, H in (('supplied', M[f]), ('refined', R[f])):
            c, _, o = observe(img, H); m = np.isfinite(o)
            err[name] += [abs(v) / ob.ppm(H, p) for v, p in zip(o[m], c[m])]
    for name, e in err.items():
        print(f'{name:9s} painted lines off by: median {np.median(e):.3f} m, p90 {np.percentile(e, 90):.3f} m ({len(e)} points, {len(frames)} unfitted frames)')


if __name__ == '__main__':
    {'fit': cmd_fit, 'check': lambda: cmd_check(int(sys.argv[2]) if len(sys.argv) > 2 else 80)}[sys.argv[1]]()
