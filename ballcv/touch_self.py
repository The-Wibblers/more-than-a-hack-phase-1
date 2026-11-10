"""Touches, by a model that teaches itself from the picture: at this frame, did someone touch the ball?

Two views of a moment that do not share their mistakes. The ball's PATH (our ball track, model 17, and the motion
model's contacts, motion_model.py, over it) knows when the ball's motion changed; the PICTURE knows whether a foot
met it. The seed takes moments the path makes obvious; the model learns from the picture only -- a clip of CLIP
frames around the ball, the pitch held still (each frame's box is the same patch of ground, through its own
homography), CROP_H person heights across at the ball. A foot meeting the ball looks the same whether the ball
then goes two metres or thirty, so a model taught on the obvious touches can find the light ones -- a dribble, a
nudge -- where the path barely changes and no rule can tell.

Round 0 (seed), all physical, cautious:
  touch      the motion model finds a contact at that frame, the ball ROLLING into it and out of it (the model's
             own choice of mode for both stretches: on the ground, so its place in the picture is its place on the
             pitch -- a ball in the air can sit on a player's body in the picture while flying past him); within
             CONTACT_M of exactly one player's body (ownball.bodies: the distance to his body column), nobody else
             within FREE_M; and the change in its motion pushing it away from him
             -- and the contact beyond doubt by the motion model's own numbers: 2 ln(Bayes factor) of a contact over
             one smooth path at least DECISIVE (Kass & Raftery's "very strong")
  free       nobody within FREE_M on any frame of the clip, the ball rolling all through it (in the air its place in
             the picture is not its place on the pitch: a header can look far from everyone) (FREE_EVERY-th frames;
             also every contact there: a bounce, nobody touched it)
Both need every player located (ALL_PLAYERS, our tracker or Metrica's live slots) on every frame of the clip: nobody
else near is only known when nobody is missing. Rather fewer labels than wrong ones: no near-miss rule (a light touch
on a fast ball barely bends its path, and that rule would call it no touch). Slow balls at a player's feet with no
clear change are left out: those are what the rounds are for.

From round 1: model K trains on every label so far (from scratch, EPOCHS passes of at least MIN_STEPS steps, the
epoch with the lowest validation loss kept), then looks at SEGMENTS fresh stretches of SEG frames. Every frame with
the ball within NEAR_M of a body is scored. A touch label: at least conf_min(r) sure, the surest frame within HALF
either side (one touch a moment), and not a fast ball that kept its path (the near-miss rule overrules the model).
A not-a-touch label: at most 1 - conf_min(r); the surest of them first, at most NEG_PER_POS a touch. Free
frames of the stretches are added as in the seed.

Frames: in a half, never the public five minutes. Every VAL_EVERY-th block of BLOCK frames is for validation only.
Checked on each round's labels: touches with a contact in the path within HALF (most, not all: light touches),
touches the near-miss rule overruled.

  python ballcv/touch_self.py run          (you run this) the contacts, then every round in turn
  python ballcv/touch_self.py round K      one round
"""
import csv, json, os, sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import ownball as ob
import selftrain as st
import motion_model as mm

L, W = ob.L, ob.W
ROOT = ob.S + 'touches'
TRACK = ob.S + 'fullmatch/track.json'
BREAKS = ROOT + '/breaks.json'
ROLLS = ROOT + '/rolls.json'     # the motion model's rolling stretches [f0, f1]
CLIP, HALF = 5, 2                # frames a clip, either side of its moment
CROP = 64                        # px
CROP_H = 1.5                     # person heights (ownball.BODY_H) across, at the ball
CONTACT_M = ob.CONTACT_M         # the ball this close to a body: at it
NEAR_M = ob.MAX_REACH_M          # ... this close: a touch is possible
FREE_M = 2.0                     # nobody this close: nobody touched it
RUN_MS = 9.0                     # faster than any footballer runs
QUIET = 4                        # frames either side with no contact in the path
FAR_M = 10.0                     # a body this far from the ball's ground point is far from the ball (a ball in the
                                 # air has its ground point beyond it, never this far)
FREE_EVERY = 25
DECISIVE = 10.0                  # 2 ln(Bayes factor): Kass & Raftery's "very strong" evidence
ALL_PLAYERS = 22                 # eleven a side
SEGMENTS, SEG = 60, 250
NEG_PER_POS = 3
ROUNDS, EPOCHS, BATCH, MIN_STEPS = 8, 20, 64, 150
VAL_EVERY, BLOCK = 5, 5000
DEV = st.DEV


def rdir(r):
    d = f'{ROOT}/r{r}'
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------- the match: ball, players, contacts
class Match:
    def __init__(self):
        from cameras import load_h
        self.M, _, _ = load_h('tactical')
        self.track = {int(k): np.array(v[:2]) for k, v in json.load(open(TRACK)).items()}
        P = np.load(ob.S + 'tactical.npz')['P'][..., :2].copy()
        ob.STALE = ob.stale_mask(P)
        if os.path.exists(ob.PLAYER_TRACKS):                                # our tracker where it has run
            z = np.load(ob.PLAYER_TRACKS); a, b = z['frames']
            P[a:b + 1] = z['P'][a:b + 1]; ob.STALE[a:b + 1] = False
        self.P = P
        self.located = (np.isfinite(P[..., 0]) & ~ob.STALE).sum(1)
        fr = np.array(sorted(self.track)); uv = np.array([self.track[f] for f in fr])
        q = np.einsum('nij,nj->ni', np.linalg.inv(self.M[fr]), np.c_[uv, np.ones(len(fr))])
        G = q[:, :2] / q[:, 2:] * [L, W]
        d = np.linalg.norm(P[fr] * [L, W] - G[:, None], axis=2)
        self.ground_min = dict(zip(fr.tolist(), np.nanmin(np.where(ob.STALE[fr], np.nan, d), axis=1)))
        self.breaks = {int(k): v for k, v in json.load(open(BREAKS)).items()}
        self.rolling = set(g for a, b in json.load(open(ROLLS)) for g in range(a, b + 1))

    def ground(self, f):
        q = np.linalg.inv(self.M[f]) @ np.r_[self.track[f], 1.0]
        return q[:2] / q[2] * [L, W]

    def bodies(self, f):
        """[(distance m, slot)] nearest first, of the ball at f; none when every body is FAR_M away."""
        if not self.ground_min.get(f, np.inf) < FAR_M:
            return []
        return [(d, s) for d, s, _h, _q in ob.bodies(self.M, self.P, f, self.track[f])]

    def everyone(self, f):
        """Every player located on each frame of the clip around f."""
        return all(self.located[g] >= ALL_PLAYERS for g in range(f - HALF, f + HALF + 1))

    def speed(self, f):
        """The ball's speed over f-HALF..f+HALF (m/s), None unless tracked on all of them."""
        if not all(g in self.track for g in range(f - HALF, f + HALF + 1)):
            return None
        G = np.array([self.ground(g) for g in range(f - HALF, f + HALF + 1)])
        return float(np.median(np.linalg.norm(np.diff(G, axis=0), axis=1)) * 25)

    def contact_near(self, f, k):
        return any(g in self.breaks for g in range(f - k, f + k + 1))


def cmd_breaks():
    """The motion model's contacts over the full-match track -> BREAKS ({frame: [velocity change m/s, its direction
    in the picture (unit x, y), the mode of the stretch before it, after it]})."""
    br, stretch = mm.run_model(track_json=TRACK)
    starts = np.array([s_['f0'] for s_ in stretch])
    mode = lambda i: stretch[i]['kind'] if 0 <= i < len(stretch) else 'none'

    from cameras import load_h
    M, _, _ = load_h('tactical')
    out = {}
    for b in br:
        d = np.subtract(b['vout'], b['vin']); n = float(np.linalg.norm(d))
        dv = n * 25 / ob.ppm(M[b['f']], np.array(b['pos']))
        i = int(np.searchsorted(starts, b['f'], side='right')) - 1           # the stretch the contact starts (or is in)
        before, after = (mode(i - 1), mode(i)) if starts[i] == b['f'] else (mode(i), mode(i + 1))
        out[str(b['f'])] = [round(float(dv), 2), *(np.round(d / max(n, 1e-9), 3).tolist()), before, after,
                            round(b['evidence'], 1)]
    os.makedirs(ROOT, exist_ok=True)
    json.dump(out, open(BREAKS, 'w'))
    json.dump([[s_['f0'], s_['f1']] for s_ in stretch if s_['kind'] == 'roll'], open(ROLLS, 'w'))
    print(f'{len(out)} contacts in the path -> {BREAKS}')


# ---------------------------------------------------------------- which frames
_OPEN = None


def open_frame(f):
    """May frame f be used for training: in a half, not the public five minutes."""
    global _OPEN
    if _OPEN is None:
        ok = np.zeros(200000, bool)
        for a, b in st.HALVES:
            ok[a:b + 1] = True
        ok[st.PUBLIC[0] - CLIP:st.PUBLIC[1] + CLIP + 1] = False
        _OPEN = ok
    return bool(_OPEN[f])


def is_val(f):
    return (np.asarray(f) // BLOCK) % VAL_EVERY == VAL_EVERY - 1


# ---------------------------------------------------------------- the seed's rules
def seed_call(m, f):
    """'touch', 'miss', 'free' or None at frame f, by the path and the bodies alone."""
    if f not in m.track:
        return None
    b = m.bodies(f)
    if not b or b[0][0] >= FREE_M:
        clear = all(g in m.track and (not m.bodies(g) or m.bodies(g)[0][0] >= FREE_M) for g in range(f - HALF, f + HALF + 1))
        rolling = all(g in m.rolling for g in range(f - HALF, f + HALF + 1))      # a ball in the air is not where it looks
        return 'free' if clear and rolling and m.everyone(f) else None
    if b[0][0] > CONTACT_M:
        return None
    if f in m.breaks:
        _d, _s, _h, q = ob.bodies(m.M, m.P, f, m.track[f])[0]
        dv, dx, dy, before, after, ev = m.breaks[f]
        away = float(np.dot([dx, dy], m.track[f] - q)) > 0
        rolling = before == 'roll' and after == 'roll'
        alone = len(b) < 2 or b[1][0] >= FREE_M
        return 'touch' if alone and rolling and away and ev >= DECISIVE and m.everyone(f) else None
    return None


def fast_kept_path(m, f):
    v = m.speed(f)
    return v is not None and v >= RUN_MS and not m.contact_near(f, QUIET)


# ---------------------------------------------------------------- clips
def clips(m, centres, desc):
    """{centre: (CLIP, CROP, CROP, 3) uint8}: the same patch of pitch around the ball at the centre frame, on each
    of the frames around it (through each frame's homography), CROP_H person heights across."""
    need = sorted({g for c in centres for g in range(c - HALF, c + HALF + 1)})
    want = set(centres); buf, out = {}, {}
    for f, img in st.read(need, desc):
        buf[f] = img
        c = f - HALF
        if c in want and all(g in buf for g in range(c - HALF, c + HALF + 1)) and c in m.track:
            G = m.ground(c)
            feet, top = ob.column(m.M[c], G); s = CROP / (CROP_H * float(np.linalg.norm(top - feet)))
            frames = []
            for g in range(c - HALF, c + HALF + 1):
                q = m.M[g] @ np.array([G[0] / L, G[1] / W, 1.0]); u = q[:2] / q[2]
                A = np.array([[s, 0, CROP / 2 - u[0] * s], [0, s, CROP / 2 - u[1] * s]])
                frames.append(cv2.warpAffine(buf[g], A, (CROP, CROP), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT))
            out[c] = np.stack(frames)
        for g in [g for g in buf if g < f - CLIP]:
            del buf[g]
    return out


# ---------------------------------------------------------------- the model
class ClipNet(nn.Module):
    """The clip's frames stacked as channels -> a touch logit."""

    def __init__(self):
        super().__init__()

        def cbr(i, o, s=1):
            return nn.Sequential(nn.Conv2d(i, o, 3, stride=s, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.body = nn.Sequential(cbr(3 * CLIP, 32), cbr(32, 32, 2), cbr(32, 64, 2), cbr(64, 96, 2), cbr(96, 128, 2),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        return self.head(self.body(x))[:, 0]


def to_t(X):
    """(n, CLIP, H, W, 3) uint8 -> (n, 3 * CLIP, H, W)."""
    x = torch.as_tensor(X).to(DEV).float().div(255)
    return x.permute(0, 1, 4, 2, 3).reshape(len(X), 3 * CLIP, CROP, CROP)


def augment(x):
    """Mirror; time reversed (a touch played backwards is still a touch); a little light."""
    n = len(x); v = x.view(n, CLIP, 3, CROP, CROP)
    flip = torch.rand(n, device=x.device) < 0.5
    v = torch.where(flip[:, None, None, None, None], v.flip(-1), v)
    rev = torch.rand(n, device=x.device) < 0.5
    v = torch.where(rev[:, None, None, None, None], v.flip(1), v)
    gain = 1 + 0.15 * (torch.rand(n, 1, 1, 1, 1, device=x.device) * 2 - 1)
    return (v * gain).clamp(0, 1).reshape(n, 3 * CLIP, CROP, CROP)


def load_net(r):
    net = ClipNet(); net.load_state_dict(torch.load(f'{rdir(r)}/model.pt', map_location='cpu'))
    return net.to(DEV).eval()


@torch.no_grad()
def probs(net, X, batch=256):
    net.eval()
    return np.concatenate([torch.sigmoid(net(to_t(X[k:k + batch]))).cpu().numpy() for k in range(0, len(X), batch)]) \
        if len(X) else np.zeros(0)


# ---------------------------------------------------------------- labels
def save_labels(r, X, y, frame, n):
    np.savez_compressed(f'{rdir(r)}/labels.npz', X=X, y=np.array(y, np.uint8), frame=np.array(frame))
    tqdm.write(f'round {r}: ' + ', '.join(f'{k} {v}' for k, v in n.items()))
    head = not os.path.exists(f'{ROOT}/rounds.csv')
    with open(f'{ROOT}/rounds.csv', 'a') as fh:
        w = csv.writer(fh)
        if head:
            w.writerow(['round'] + list(n))
        w.writerow([r] + list(n.values()))


def label_seed(m):
    rng = np.random.default_rng(0)
    calls = {}
    for f in tqdm(sorted(m.track), desc='round 0 seed', leave=False):
        if open_frame(f) and (f in m.breaks or f % FREE_EVERY == 0 or (m.bodies(f) or [(np.inf, None)])[0][0] <= CONTACT_M):
            c = seed_call(m, f)
            if c:
                calls[f] = c
    touch = [f for f, c in calls.items() if c == 'touch']
    each = NEG_PER_POS * len(touch) // 2                 # every touch kept; NEG_PER_POS negatives a touch, half of each kind
    pick = lambda fs: sorted(rng.choice(fs, min(len(fs), each), replace=False).tolist()) if fs else []
    free_b = pick([f for f, c in calls.items() if c == 'free' and f in m.breaks])
    free = pick([f for f, c in calls.items() if c == 'free' and f % FREE_EVERY == 0 and f not in m.breaks])
    want = {**{f: 1 for f in touch}, **{f: 0 for f in free_b + free}}
    got = clips(m, sorted(want), 'round 0 clips')
    fr = sorted(got)
    save_labels(0, np.stack([got[f] for f in fr]), [want[f] for f in fr], fr,
                dict(touch=sum(want[f] for f in fr),
                     bounce_free=sum(f in free_b for f in fr), free=sum(f in free for f in fr)))
    return len(fr)


def segments(r):
    rng = np.random.default_rng(500 + r)
    f = np.concatenate([np.arange(a + SEG, b - SEG) for a, b in st.HALVES])
    f = f[[open_frame(int(g)) and open_frame(int(g) + SEG) for g in f]]
    edges = np.linspace(0, len(f), SEGMENTS + 1).astype(int)
    return [int(f[rng.integers(a, b)]) for a, b in zip(edges, edges[1:]) if b > a]


def label_round(r, net, m):
    bar = st.conf_min(r); rng = np.random.default_rng(r)
    near, free = [], []
    for s in segments(r):
        for f in range(s, s + SEG):
            if f in m.track and open_frame(f):
                d = m.bodies(f)[0][0] if m.bodies(f) else np.inf
                if d <= NEAR_M:
                    near.append(f)
                elif d >= FREE_M and f % FREE_EVERY == 0:
                    free.append(f)
    got = clips(m, near + free, f'round {r} clips')
    near = [f for f in near if f in got]; free = [f for f in free if f in got]
    p = dict(zip(near, probs(net, np.stack([got[f] for f in near])) if near else []))
    touch, veto = [], 0
    for f in near:
        if p[f] >= bar and all(p[f] >= p.get(g, 0) for g in range(f - HALF, f + HALF + 1)):
            if fast_kept_path(m, f):
                veto += 1
            else:
                touch.append(f)
    no = [f for f in near if p[f] <= 1 - bar and not any(abs(f - t) <= HALF for t in touch)]
    no = sorted(no, key=lambda f: p[f])[:NEG_PER_POS * max(1, len(touch))]     # the surest first
    with_contact = np.mean([m.contact_near(f, HALF) for f in touch]) if touch else 0
    want = {**{f: 1 for f in touch}, **{f: 0 for f in no + free}}
    fr = sorted(want)
    save_labels(r, np.stack([got[f] for f in fr]), [want[f] for f in fr], fr,
                dict(scored=len(near), touch=len(touch), not_touch=len(no), free=len(free), overruled=veto,
                     touch_with_contact=round(float(with_contact), 4), bar=bar))
    return len(fr)


# ---------------------------------------------------------------- training
def all_labels(upto):
    parts = [np.load(f'{rdir(q)}/labels.npz') for q in range(upto + 1) if os.path.exists(f'{rdir(q)}/labels.npz')]
    return (np.concatenate([p['X'] for p in parts]), np.concatenate([p['y'] for p in parts]).astype(np.float32),
            np.concatenate([p['frame'] for p in parts]))


def auc(p, y):
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return 0.5
    r = np.argsort(np.argsort(np.r_[pos, neg])) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def train(r):
    torch.manual_seed(r); np.random.seed(r)
    X, y, fr = all_labels(r - 1); v = is_val(fr)
    Xt, yt, Xv, yv = X[~v], y[~v], X[v], y[v]
    pw = torch.tensor(float(max(1.0, (yt == 0).sum() / max(1, (yt == 1).sum()))), dtype=torch.float32, device=DEV)
    lossf = lambda lg, t: F.binary_cross_entropy_with_logits(lg, t, pos_weight=pw)
    net = ClipNet().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    steps = max(MIN_STEPS, -(-len(Xt) // BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EPOCHS * steps)

    def val():
        p = probs(net, Xv)
        lg = torch.logit(torch.as_tensor(p, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)).to(DEV)
        return float(lossf(lg, torch.as_tensor(yv, device=DEV))), auc(p, yv)
    base, _ = val()
    tqdm.write(f'model {r}: {len(Xt)} training clips ({int(yt.sum())} touches), {len(Xv)} validation ({int(yv.sum())}); '
               f'baseline to beat (untrained) val loss {base:.3f}, AUC 0.5')
    log = open(f'{rdir(r)}/log.csv', 'w'); w = csv.writer(log); w.writerow(['epoch', 'train_loss', 'val_loss', 'val_auc', 'best'])
    best = np.inf
    for ep in range(1, EPOCHS + 1):
        net.train(); tot = 0.0
        order = np.concatenate([np.random.permutation(len(Xt)) for _ in range(-(-steps * BATCH // len(Xt)))])
        for k in tqdm(range(steps), desc=f'epoch {ep}', leave=False):
            b = order[k * BATCH:(k + 1) * BATCH]
            loss = lossf(net(augment(to_t(Xt[b]))), torch.as_tensor(yt[b], device=DEV))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += float(loss)
        vl, va = val(); mark = vl < best
        if mark:
            best = vl; torch.save(net.state_dict(), f'{rdir(r)}/model.pt')
        w.writerow([ep, round(tot / steps, 4), round(vl, 4), round(va, 4), int(mark)]); log.flush()
        tqdm.write(f'epoch {ep}: train loss {tot / steps:.3f}, val loss {vl:.3f} (baseline {base:.3f}), val AUC {va:.3f}'
                   + (' *' if mark else ''))
    tqdm.write('good = val loss falling well under the baseline, val AUC near 1')
    return load_net(r)


# ---------------------------------------------------------------- commands
def cmd_round(r, m):
    if r == 0:
        if not os.path.exists(f'{rdir(0)}/labels.npz'):
            label_seed(m)
        return
    log = f'{rdir(r)}/log.csv'
    trained = os.path.exists(log) and sum(1 for _ in open(log)) > EPOCHS     # every epoch in it: a stopped run trains again
    net = load_net(r) if trained else train(r)
    return label_round(r, net, m)


def cmd_run():
    os.makedirs(ROOT, exist_ok=True)
    if not (os.path.exists(BREAKS) and os.path.exists(ROLLS)):
        cmd_breaks()
    m = Match()
    for r in range(ROUNDS):
        if os.path.exists(f'{rdir(r)}/labels.npz'):
            continue
        if cmd_round(r, m) == 0:
            tqdm.write(f'round {r}: no labels -- stopping'); break


if __name__ == '__main__':
    os.makedirs(ROOT, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == 'run':
        cmd_run()
    elif cmd == 'round':
        cmd_round(int(sys.argv[2]), Match())
