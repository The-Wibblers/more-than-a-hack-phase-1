"""The assistants' flags, by a detector that teaches itself: where in the assistant's picture is his flag?

Every crop is one assistant (assistant_self.py's model, both touchlines), his feet at FEET, CROP_H person heights
(ownball.BODY_H) tall, so a metre is the same number of pixels near the camera and far from it. Crops come in
CLIPS: CLIP consecutive frames of one assistant, so the flag can be followed from frame to frame. A heatmap network
(heatmap.py's, the ball's) is taught where the flag is -- hanging by his knee, held out, held up -- and from its place
on every frame, against his shoulders, whether and when it goes up.

Round 0 (seed): where the picture and the flag's own behaviour agree. A flag is red and yellow squares, which blur to
orange-red (FLAG_HSV: never the officials' lime, which starts at a higher hue); about FLAG_M across; in his hand.
The seed keeps to below his shoulders, within REACH_M of his body: above them, on the far side, are the boards and
the crowd behind him, below only the grass and his legs. A crop is labelled when exactly one flag-sized orange-red
patch is there (none larger touching it: a striped shirt going past), and it travels with him: on the frames around
it, NEAR frames either side, there is again exactly one such patch, no further from it than a hand moves (MOVE_M a
frame). Raised flags the seed leaves to the model: the network finds the same thing anywhere in the crop, and the
boards above his shoulders, in the seed's crops, are what it learns is not a flag.

From round 1: model K trains on every label so far (from scratch, EPOCHS passes, the epoch with the lowest validation
loss kept, at least MIN_STEPS steps an epoch), then looks at a fresh CLIPS_A_ROUND clips. A crop is labelled at the heatmap's peak when it is at least
conf_min(r) sure (selftrain.py's schedule), nothing else SEP_M or more away scores DETECTED, and the flag travels
with him the same way -- the peaks of the frames around it, DETECTED and up, within a hand's move.

Clips: centred on frames in a half (never the public five minutes), half of them where play has stopped (the
players still, selftrain.STILL), where signals happen, the rest anywhere. Every VAL_EVERY-th clip of round 0 is for
validation only, never trained on.

Checked every round, with no labels, on round 0's validation clips:
  found   the seed's flags with the model's peak within FOUND_M, DETECTED and up
  seen    crops where the model sees a flag at all
  steady  of frames where it sees one, the next frame's within a hand's move
  up      flags above his shoulders, where play is on (should be rare) and where it has stopped

  python ballcv/flag_self.py run          every round in turn (you run this); finished rounds are skipped
  python ballcv/flag_self.py round K      one round

Signals: the flag model (ballcv/models/flags.pt) on every DET_EVERY-th frame of A..B (the assistants found on each), then on
every frame around those where a flag is above his shoulders -- a raise is held at least HOLD frames, so it is never
missed between two samples; only its instant needs every frame -- his place between samples interpolated
-> flags/signals_A_B.npz (frame, line, x, the flag's probability, its place against his
feet in metres: out to the side, up). A RAISE is the flag above his shoulders, DETECTED and up, for at least HOLD
frames (gaps of at most GAP frames bridged); its INSTANT is where the rise began: from the first frame above the
shoulders, back while the flag was still climbing (at most BACK frames).
  python ballcv/flag_self.py signals A B  (you run this; the whole match is 11185 190197)
  python ballcv/flag_self.py raises A B   the raises, checked with no labels: how many follow Metrica's ball leaving
                                          the pitch (within EXIT_S) against how many would by chance, and how many
                                          fall in live play (LIVE_S from any exit, whistle or set-piece tag's span)
"""
import csv, json, os, shutil, sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

import ownball as ob
import selftrain as st
import heatmap
import assistant_net as an
import assistant_self as asl

ROOT = ob.S + 'flags'
RAISES = ROOT + '/raises.json'    # every raise: start (the instant), up (first frame above the shoulders), end, line, x
MODEL = __import__('paths').MODELS + 'flags.pt'
CLIP, NEAR = 9, 2                # frames a clip; neighbours either side a flag must travel through
CLIPS_A_ROUND = 600
STILL_SHARE = 0.5                # clips where play has stopped
ROUNDS, EPOCHS, BATCH = 8, 30, 64
MIN_STEPS = 150                  # steps an epoch at least: few labels are passed over more than once
CROP = 96                        # crop side (px)
CROP_H = 2.0                     # ... in person heights (ownball.BODY_H), the feet FEET_AT of the way down
FEET_AT = 0.8
M_PX = CROP / (CROP_H * ob.BODY_H)                   # px a metre, in a crop
FEET = (CROP / 2, FEET_AT * CROP)
SHOULDER_M = 1.45                # shoulders above the feet
REACH_M = 1.1                    # arm and flag: the flag is this close to his shoulder, or below it
FLAG_M = 0.4                     # a flag across; with blur, a patch at most 1.5 times that
MOVE_M = 0.2                     # a hand moves at most this much a frame (5 m/s: a flag raised fast)
SEP_M = 0.6                      # two peaks this far apart are two things
FOUND_M = 0.15
FLAG_HSV = (((0, 110, 120), (20, 255, 255)), ((170, 110, 120), (180, 255, 255)))   # orange-red
DETECTED = 0.5
VAL_EVERY = 5
DEV = heatmap.DEV
SH_Y = FEET[1] - SHOULDER_M * M_PX


def rdir(r):
    d = f'{ROOT}/r{r}'
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------- clips
def frames_of(r):
    """Round r's clip centres: one from each equal slice of the stopped frames (STILL_SHARE of them) and of the rest."""
    rng = np.random.default_rng(700 + r)
    f = np.concatenate([np.arange(a + CLIP, b - CLIP) for a, b in st.HALVES])
    f = f[~((f >= st.PUBLIC[0] - CLIP) & (f <= st.PUBLIC[1] + CLIP))]
    out = []
    for group, n in ((f[st.STILL[f]], int(CLIPS_A_ROUND * STILL_SHARE)), (f[~st.STILL[f]], CLIPS_A_ROUND - int(CLIPS_A_ROUND * STILL_SHARE))):
        edges = np.linspace(0, len(group), n + 1).astype(int)
        out += [int(group[rng.integers(a, b)]) for a, b in zip(edges, edges[1:]) if b > a]
    keep = []
    for c in sorted(out):
        if not keep or c - keep[-1] >= CLIP:
            keep.append(c)
    return keep


def crop_of(img, Mf, x, y, o):
    """The assistant standing at (x, just outside touchline y): CROP x CROP, feet at FEET, M_PX a metre; his height in
    the picture (px). None when his feet are off the picture."""
    feet, top = ob.column(Mf, np.array([x, y + o * an.OUT_M])); H = float(np.linalg.norm(top - feet))
    if not (0 <= feet[0] < img.shape[1] and 0 <= feet[1] < img.shape[0]):
        return None, H
    s = M_PX * ob.BODY_H / H
    A = np.array([[s, 0, FEET[0] - feet[0] * s], [0, s, FEET[1] - feet[1] * s]])
    return cv2.warpAffine(img, A, (CROP, CROP), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT), H


def cmd_clips(r):
    """Round r's clips: on each, the assistants found (assistant_self.py's model) on its first, middle and last frames,
    his place between them interpolated, a crop on every frame -> rK/clips.npz."""
    net = asl.load_net(asl.MODEL)
    centres = frames_of(r)
    clip_of = {f: i for i, c in enumerate(centres) for f in range(c - CLIP // 2, c + CLIP // 2 + 1)}
    lines = {n: (y, o) for n, y, o in an.LINES}
    X, FR, LN, CL, HP = [], [], [], [], []
    buf = {}
    for f, img in st.read(sorted(clip_of), f'round {r} clips'):
        i = clip_of[f]; buf[f] = img
        c = centres[i]
        if f != c + CLIP // 2:
            continue
        found = {}
        for g in (c - CLIP // 2, c, c + CLIP // 2):
            if g not in buf:
                continue
            for name, xs, wins in asl.lines_of(g, buf[g]):
                if len(wins):
                    s = asl.scores(net, wins); j = int(np.argmax(s))
                    if s[j] >= asl.DETECTED:
                        found.setdefault(name, {})[g] = float(xs[j])
        for name, d in found.items():
            if c not in d:
                continue                                                  # not seen in the middle of the clip
            ks = sorted(d)
            for g in sorted(buf):
                x = float(np.interp(g, ks, [d[k] for k in ks]))
                im, H = crop_of(buf[g], asl.M[g], x, *lines[name])
                if im is not None:
                    X.append(im); FR.append(g); LN.append([n for n, *_ in an.LINES].index(name)); CL.append(r * 10000 + i); HP.append(H)
        buf = {}
    np.savez_compressed(f'{rdir(r)}/clips.npz', X=np.stack(X), frame=np.array(FR), line=np.array(LN, np.int8),
                        clip=np.array(CL), H=np.array(HP, np.float32))
    tqdm.write(f'round {r}: {len(centres)} clips, {len(X)} assistant crops')


def load_clips(r):
    z = np.load(f'{rdir(r)}/clips.npz')
    return {k: z[k] for k in z.files}


def neighbours(d):
    """{(clip, line, frame): index} of a round's crops."""
    return {(int(c), int(l), int(f)): i for i, (c, l, f) in enumerate(zip(d['clip'], d['line'], d['frame']))}


def travels(i, d, at, idx, ok):
    """Does the flag at at[i] travel with him: on each of the NEAR frames either side the clip has, ok there and
    within a hand's move of it?"""
    c, l, f = int(d['clip'][i]), int(d['line'][i]), int(d['frame'][i])
    n = 0
    for k in range(-NEAR, NEAR + 1):
        j = idx.get((c, l, f + k))
        if k == 0 or j is None:
            continue
        if not ok[j] or np.hypot(*(at[j] - at[i])) > MOVE_M * M_PX * abs(k):
            return False
        n += 1
    return n >= NEAR


# ---------------------------------------------------------------- the seed's rule
REACH = np.zeros((CROP, CROP), bool)
REACH[int(SH_Y):int(FEET[1]) + 1, int(FEET[0] - REACH_M * M_PX):int(FEET[0] + REACH_M * M_PX) + 1] = True


def flag_patch(img):
    """The flag's place (crop px) when exactly one flag-sized orange-red patch is in his reach below his shoulders
    and nothing bigger of that colour touches it; else None."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in FLAG_HSV:
        m |= cv2.inRange(hsv, lo, hi)
    n, lab, stt, cen = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8))
    big = 1.5 * FLAG_M * M_PX
    inside = [k for k in range(1, n) if REACH[lab == k].any() and stt[k, 4] >= 3]
    if len(inside) != 1 or stt[inside[0], 2] > big or stt[inside[0], 3] > big:
        return None
    return cen[inside[0]]


# ---------------------------------------------------------------- the model
def load_net(r):
    net = heatmap.BallFCN()
    net.load_state_dict(torch.load(r if isinstance(r, str) else f'{rdir(r)}/model.pt', map_location='cpu'))
    return net.to(DEV).eval()


def to_t(X):
    return torch.as_tensor(X).permute(0, 3, 1, 2).float().div(255).to(DEV)


@torch.no_grad()
def peaks(net, X, batch=256):
    """Each crop's best heatmap peak: place (crop px), probability; and the best probability SEP_M or more from it."""
    net.eval(); at, p1, p2 = [], [], []
    sep = SEP_M * M_PX / heatmap.STRIDE
    for k in range(0, len(X), batch):
        hm = torch.sigmoid(net(to_t(X[k:k + batch]))).cpu().numpy()
        h, w = hm.shape[1:]; yy, xx = np.mgrid[0:h, 0:w]
        for m in hm:
            i = int(np.argmax(m)); y, x = divmod(i, w)
            at.append(((x + 0.5) * heatmap.STRIDE, (y + 0.5) * heatmap.STRIDE)); p1.append(m[y, x])
            p2.append(m[np.hypot(xx - x, yy - y) >= sep].max())
    return np.array(at).reshape(-1, 2), np.array(p1), np.array(p2)


# ---------------------------------------------------------------- labels
def label_round(r, net):
    """Round r's labels -> rK/labels.npz (X, P crop px, frame, line, clip)."""
    if not os.path.exists(f'{rdir(r)}/clips.npz'):
        cmd_clips(r)
    d = load_clips(r); idx = neighbours(d)
    bar = st.conf_min(r) if r > 0 else None
    if r == 0:
        got = [flag_patch(x) for x in tqdm(d['X'], desc='round 0 seed', leave=False)]
        ok = np.array([g is not None for g in got]); at = np.array([g if g is not None else (np.nan, np.nan) for g in got])
        sure = ok
    else:
        at, p1, p2 = peaks(net, d['X'])
        ok = (p1 >= DETECTED) & (p2 < DETECTED)
        sure = ok & (p1 >= bar)
    keep = np.array([bool(sure[i]) and travels(i, d, at, idx, ok) for i in range(len(at))])
    np.savez_compressed(f'{rdir(r)}/labels.npz', X=d['X'][keep], P=at[keep].astype(np.float32), frame=d['frame'][keep],
                        line=d['line'][keep], clip=d['clip'][keep])
    up = (at[keep, 1] < SH_Y)
    still = st.STILL[d['frame'][keep]]
    names = [n for n, *_ in an.LINES]
    per = ', '.join(f'{n} {np.sum(d["line"][keep] == k)} of {np.sum(d["line"] == k)}' for k, n in enumerate(names))
    tqdm.write(f'round {r}: {keep.sum()} flags labelled of {len(keep)} crops ({per}); {ok.sum()} with one flag, '
               f'{sure.sum()} sure; above the shoulders {np.mean(up[~still]) if (~still).any() else 0:.1%} where play is on, '
               f'{np.mean(up[still]) if still.any() else 0:.1%} where it has stopped' + (f'; bar {bar}' if bar else ''))
    if not os.path.exists(f'{ROOT}/rounds.csv'):
        open(f'{ROOT}/rounds.csv', 'w').write('round,crops,labels,far,near,up_live,up_stopped,bar\n')
    with open(f'{ROOT}/rounds.csv', 'a') as fh:
        fh.write(f'{r},{len(keep)},{keep.sum()},{np.sum(d["line"][keep] == 0)},{np.sum(d["line"][keep] == 1)},'
                 f'{np.mean(up[~still]) if (~still).any() else 0:.4f},{np.mean(up[still]) if still.any() else 0:.4f},{bar or ""}\n')
    return int(keep.sum())


# ---------------------------------------------------------------- training
def all_labels(upto):
    parts = [np.load(f'{rdir(q)}/labels.npz') for q in range(upto + 1) if os.path.exists(f'{rdir(q)}/labels.npz')]
    cat = lambda k: np.concatenate([p[k] for p in parts])
    return cat('X'), cat('P'), cat('clip')


def is_val(clip):
    return (clip < 10000) & (clip % VAL_EVERY == VAL_EVERY - 1)          # round 0's every VAL_EVERY-th clip


def targets(P):
    h = CROP // heatmap.STRIDE
    return torch.as_tensor(np.stack([heatmap.target(h, h, [p / heatmap.STRIDE]) for p in P.astype(np.float32)]), dtype=torch.float32)


def val_loss(net, X, T):
    net.eval(); tot = 0.0
    with torch.no_grad():
        for k in range(0, len(X), BATCH):
            tot += float(heatmap.focal(net(to_t(X[k:k + BATCH])), T[k:k + BATCH].to(DEV))) * len(X[k:k + BATCH])
    return tot / max(1, len(X))


def train(r):
    torch.manual_seed(r); np.random.seed(r)
    X, P, clip = all_labels(r - 1)
    v = is_val(clip); T = targets(P)
    Xt, Tt, Xv, Tv = X[~v], T[~v], X[v], T[v]
    net = heatmap.BallFCN().to(DEV)
    torch.nn.init.constant_(net.head.bias, -4.0)                         # start sure there is no flag (CenterNet)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    steps = max(MIN_STEPS, -(-len(Xt) // BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EPOCHS * steps)
    base = val_loss(net, Xv, Tv)
    tqdm.write(f'model {r}: {len(Xt)} training flags, {len(Xv)} validation; baseline to beat (untrained) val loss {base:.3f}')
    log = open(f'{rdir(r)}/log.csv', 'w'); w = csv.writer(log); w.writerow(['epoch', 'train_loss', 'val_loss', 'val_found', 'best'])
    best = np.inf
    for ep in range(1, EPOCHS + 1):
        net.train(); tot = 0.0
        order = np.concatenate([np.random.permutation(len(Xt)) for _ in range(-(-steps * BATCH // len(Xt)))])
        for k in tqdm(range(steps), desc=f'epoch {ep}', leave=False):
            b = order[k * BATCH:(k + 1) * BATCH]
            x, t = heatmap.augment(to_t(Xt[b]), Tt[b].to(DEV))
            loss = heatmap.focal(net(x), t)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += float(loss)
        vl = val_loss(net, Xv, Tv)
        at, p1, _ = peaks(net, Xv)
        found = np.mean((p1 >= DETECTED) & (np.hypot(*(at - P[v]).T) <= FOUND_M * M_PX)) if len(Xv) else 0
        mark = vl < best
        if mark:
            best = vl; torch.save(net.state_dict(), f'{rdir(r)}/model.pt')
        w.writerow([ep, round(tot / steps, 4), round(vl, 4), round(float(found), 4), int(mark)]); log.flush()
        tqdm.write(f'epoch {ep}: train loss {tot / steps:.3f}, val loss {vl:.3f} (baseline {base:.3f}), '
                   f'validation flags found {found:.1%}' + (' *' if mark else ''))
    tqdm.write('good = val loss falling well under the baseline, validation flags found near 100%')
    return load_net(r)


def evaluate(r, net):
    """found / seen / steady / up on round 0's validation clips (see the module docstring)."""
    d = load_clips(0); v = is_val(d['clip']); idx = neighbours(d)
    seed = np.load(f'{rdir(0)}/labels.npz'); sv = is_val(seed['clip'])
    at, p1, p2 = peaks(net, d['X'][v]); seen = p1 >= DETECTED
    full_at = np.full((len(d['X']), 2), np.nan); full_at[v] = at
    full_ok = np.zeros(len(d['X']), bool); full_ok[v] = seen
    pos = {k: i for k, i in idx.items() if v[i]}
    steady = [np.hypot(*(full_at[pos[(c, l, f + 1)]] - full_at[i])) <= MOVE_M * M_PX
              for (c, l, f), i in pos.items() if full_ok[i] and (c, l, f + 1) in pos and full_ok[pos[(c, l, f + 1)]]]
    sat, sp1, _ = peaks(net, seed['X'][sv])
    found = np.mean((sp1 >= DETECTED) & (np.hypot(*(sat - seed['P'][sv]).T) <= FOUND_M * M_PX)) if sv.any() else 0
    up = at[seen, 1] < SH_Y; still = st.STILL[d['frame'][v][seen]]
    upl, ups = (np.mean(up[~still]) if (~still).any() else 0), (np.mean(up[still]) if still.any() else 0)
    tqdm.write(f'model {r} on round 0\'s validation clips: seed flags found {found:.1%}; a flag seen in {seen.mean():.1%} of '
               f'{len(seen)} crops; steady frame to frame {np.mean(steady) if steady else 0:.1%}; above the shoulders '
               f'{upl:.1%} where play is on, {ups:.1%} where it has stopped')
    with open(f'{ROOT}/checks.csv', 'a') as fh:
        fh.write(f'{r},{found:.4f},{seen.mean():.4f},{np.mean(steady) if steady else 0:.4f},{upl:.4f},{ups:.4f}\n')


# ---------------------------------------------------------------- signals
DET_EVERY = 5
HOLD, GAP, BACK = 10, 2, 25      # frames: a raise held 0.4 s; gaps bridged; the rise looked back for, at most 1 s
EXIT_S, LIVE_S = 4, 10


def in_halves(f):
    return any(h0 <= f <= h1 for h0, h1 in st.HALVES)


def scan(runs, step, anet, fnet, desc):
    """The flag on frames s, s + step, .. e of each run (s, e) (runs starting and ending on multiples of DET_EVERY):
    the assistants looked for on every DET_EVERY-th frame, his place between interpolated -> {frame, line, x, p,
    side, up}."""
    frames = [f for s0, e in runs for f in range(s0, e + 1, step)]
    lines = {n: (y, o) for n, y, o in an.LINES}; names = [n for n, *_ in an.LINES]
    out = {k: [] for k in ('frame', 'line', 'x', 'p', 'side', 'up')}
    buf, crops, meta = {}, [], []
    prev_f, prev = None, {}                                                # the last frame the assistants were looked for

    def flush(force=False):
        if crops and (force or len(crops) >= 512):
            at, p1, _ = peaks(fnet, np.stack(crops))
            for (f, k, x), q, pr in zip(meta, at, p1):
                out['frame'].append(f); out['line'].append(k); out['x'].append(x); out['p'].append(pr)
                out['side'].append((q[0] - FEET[0]) / M_PX); out['up'].append((FEET[1] - q[1]) / M_PX)
            crops.clear(); meta.clear()

    for f, img in tqdm(st.stream(runs, step), total=len(frames), desc=desc, unit='frame', dynamic_ncols=True):
        buf[f] = img
        if f % DET_EVERY and f != frames[-1]:
            continue
        if prev_f is not None and f - prev_f > DET_EVERY:
            prev_f, prev = None, {}                                        # a gap between runs: start afresh
        now = {}
        for name, xs, wins in asl.lines_of(f, img):
            if len(wins):
                sc = asl.scores(anet, wins); j = int(np.argmax(sc))
                if sc[j] >= asl.DETECTED:
                    now[name] = float(xs[j])
        for g in sorted(buf):
            for name in names:
                ends = [(h, d[name]) for h, d in ((prev_f, prev), (f, now)) if h is not None and name in d]
                if not ends:
                    continue                                               # not seen on either side of this frame
                x = float(np.interp(g, [h for h, _ in ends], [v for _, v in ends]))
                im, _H = crop_of(buf[g], asl.M[g], x, *lines[name])
                if im is not None:
                    crops.append(im); meta.append((g, names.index(name), x))
            flush()
        buf = {}; prev_f, prev = f, now
    flush(True)
    return {k: np.array(v) for k, v in out.items()}


def cmd_signals(a, b):
    anet = asl.load_net(asl.MODEL); fnet = load_net(MODEL)
    halves = [(max(a, h0) + (-max(a, h0)) % DET_EVERY, min(b, h1)) for h0, h1 in st.HALVES if max(a, h0) < min(b, h1)]
    halves = [(s0, e - e % DET_EVERY) for s0, e in halves]
    coarse = [f for s0, e in halves for f in range(s0, e + 1, DET_EVERY)]
    c = scan(halves, DET_EVERY, anet, fnet, 'flag signals, every 5th frame')
    ups = np.unique(c['frame'][(c['p'] >= DETECTED) & (c['up'] > SHOULDER_M)])
    want = set()
    for u in ups:                                                          # the rise before it, the hold after it
        lo = u - BACK - DET_EVERY; hi = u + HOLD + DET_EVERY
        want.update(range(lo - lo % DET_EVERY, hi - hi % DET_EVERY + DET_EVERY + 1))
    dense = sorted(f for f in want if a <= f <= b and in_halves(f))
    d = scan(st.runs_of(dense), 1, anet, fnet, 'flag signals, every frame near a flag up') if dense else {k: np.zeros(0) for k in c}
    keep = ~np.isin(c['frame'], dense)
    out = {k: np.concatenate([c[k][keep], d[k]]) for k in c}
    fn = f'{ROOT}/signals_{a}_{b}.npz'
    np.savez_compressed(fn, **out)
    print(f'{len(coarse)} frames sampled, {len(ups)} with a flag up; {len(dense)} frames read in full around them; '
          f'{len(out["frame"])} assistant frames -> {fn}')


def raises(sig, k):
    """[(instant, first frame up, last frame up, x)] of line k's raises."""
    s = sig['line'] == k
    fr, up, p, x = sig['frame'][s], sig['up'][s], sig['p'][s], sig['x'][s]
    o = np.argsort(fr); fr, up, p, x = fr[o], up[o], p[o], x[o]
    h = dict(zip(fr.tolist(), np.where(p >= DETECTED, up, np.nan)))
    above = [f for f in fr.tolist() if h[f] > SHOULDER_M]
    runs, run = [], []
    for f in above + [None]:
        if run and f is not None and f - run[-1] <= GAP + 1:
            run.append(f); continue
        if len(run) >= 2 and run[-1] - run[0] + 1 >= HOLD:
            runs.append((run[0], run[-1]))
        run = [f] if f is not None else []
    out = []
    for f0, f1 in runs:
        i = f0
        while f0 - i < BACK and np.isfinite(h.get(i - 1, np.nan)) and h[i - 1] < h[i]:
            i -= 1
        out.append((i, f0, f1, float(x[fr == f0][0])))
    return out


def cmd_raises(a, b):
    import csv as _csv
    import stoppages
    sig = np.load(f'{ROOT}/signals_{a}_{b}.npz')
    B = asl.BALL
    on = np.isfinite(B[:, 0]) & (B[:, 0] >= 0) & (B[:, 0] <= ob.L) & (B[:, 1] >= 0) & (B[:, 1] <= ob.W)
    off = np.isfinite(B[:, 0]) & ~on
    exits = np.flatnonzero(off[1:] & on[:-1]) + 1
    exits = exits[(exits >= a) & (exits <= b)]
    marks = [(e, e) for e in exits] + [(t - stoppages.OUT_S * 25, t + stoppages.TAG_LEAD_S * 25) for t, _k, _ in stoppages.tagged_outs()] + \
        [(int(r['tactical_frame']),) * 2 for r in _csv.DictReader(open(__import__('paths').DATA + 'broadcast/BAR-ATH_broadcast_whistles.csv'))]
    seen = np.zeros(200000, bool); seen[np.unique(sig['frame'])] = True
    after_exit = np.zeros(200000, bool); busy = np.zeros(200000, bool)
    for e in exits:
        after_exit[e:e + EXIT_S * 25 + 1] = True
    for m0, m1 in marks:                                                 # a tag: its whole span (the out is somewhere in it)
        busy[max(0, m0 - LIVE_S * 25):m1 + LIVE_S * 25 + 1] = True
    frames = np.flatnonzero(seen); found = []
    print(f'frames {a}-{b}: {len(frames)} with an assistant; Metrica\'s ball leaves the pitch {len(exits)} times')
    for k, name in enumerate(n for n, *_ in an.LINES):
        R = raises(sig, k)
        if not R:
            print(f'  {name}: no raises'); continue
        inst = np.array([r[0] for r in R])
        near = after_exit[inst].mean(); chance = after_exit[frames].mean()
        live = ~busy[inst]; live_min = (~busy[frames]).sum() / 25 / 60 / 2
        delay = [min((i - e for e in exits if 0 <= i - e <= EXIT_S * 25), default=None) for i in inst]
        delay = [d / 25 for d in delay if d is not None]
        print(f'  {name}: {len(R)} raises, held {np.median([r[2] - r[1] + 1 for r in R]) / 25:.1f} s (median), rising '
              f'{np.median([r[1] - r[0] for r in R]) / 25:.2f} s; within {EXIT_S} s after the ball left the pitch {near:.0%} '
              f'(by chance {chance:.0%}), median {np.median(delay) if delay else float("nan"):.1f} s after it; in live play '
              f'{live.sum()} ({live.sum() / max(1e-9, live_min):.2f} a minute of it)')
        for r in R[:12]:
            print(f'      instant {r[0]}  up {r[1]}-{r[2]}  at x {r[3]:.0f} m' + ('  (live play)' if not busy[r[0]] else ''))
        found += [dict(start=int(r[0]), up=int(r[1]), end=int(r[2]), line=name, x=round(r[3], 1)) for r in R]
    json.dump(dict(frames=[a, b], raises=sorted(found, key=lambda r: r['start'])), open(RAISES, 'w'), indent=0)
    print(f'{len(found)} raises -> {RAISES}')


# ---------------------------------------------------------------- commands
def cmd_round(r):
    net = None
    if r > 0:
        net = load_net(r) if os.path.exists(f'{rdir(r)}/model.pt') and os.path.exists(f'{rdir(r)}/log.csv') else train(r)
        evaluate(r, net)
    return label_round(r, net)


def cmd_run():
    for r in range(ROUNDS):
        if os.path.exists(f'{rdir(r)}/labels.npz'):
            continue
        if cmd_round(r) == 0:
            tqdm.write(f'round {r}: no labels -- stopping'); break
    last = max((r for r in range(ROUNDS) if os.path.exists(f'{rdir(r)}/model.pt')), default=None)
    if last is not None:
        shutil.copy(f'{rdir(last)}/model.pt', MODEL)
        tqdm.write(f'model {last} -> {MODEL}')


if __name__ == '__main__':
    os.makedirs(ROOT, exist_ok=True)
    {'run': cmd_run, 'round': lambda: cmd_round(int(sys.argv[2])),
     'signals': lambda: cmd_signals(int(sys.argv[2]), int(sys.argv[3])),
     'raises': lambda: cmd_raises(int(sys.argv[2]), int(sys.argv[3]))}[sys.argv[1]]()
