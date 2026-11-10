"""A ball detector that teaches itself: the ball's centre pixel in a frame, like heatmap.py, but with no hand labels.

  python ballcv/selftrain.py run         every round in turn, 0 to ROUNDS - 1 (you run this); a finished round (its
                                         labels.json written) is skipped, a trained model (trained.json) is not
                                         trained again, and labels are saved as they come (labels.partial.json),
                                         so a stopped run goes on where it was;
                                         it stops early when a round takes no new labels
  python ballcv/selftrain.py round 0     label what the seed can -> self/r0/labels.json
  python ballcv/selftrain.py round K     (K >= 1) train model K on every label so far, then label round K's frames
                                         with it -> self/rK/model.pt, log.csv, labels.json; a line a round in
                                         self/rounds.csv
  python ballcv/selftrain.py orbita      the match ball's colour, ORBITA_SAT, measured again from ORBITA_PNG: Puma's
                                         product photo of the Orbita LaLiga 1 24/25 (ORBITA_URL; saved into data/,
                                         which is not committed)

Round 0 labels the frames where Metrica's tactical ball and simple CV on the picture agree (seed): the ball is a blob
brighter than the grass around it (by NOISE times the frame's own spread), the size of a ball lying there (0.22 m
through the homography), round and whitish (the match ball, the Orbita, from afar: ORBITA_SAT); it must be the only such blob
within SEARCH_M of Metrica's ball, lie within NEAR_M of it, and have nothing else bright within CLEAR_M of it in the
picture. Metrica alone coasts and latches onto players; the picture alone finds socks and kit; both wrong at the same
spot is rare.

From round 1 the labeller is the model (trained EPOCHS passes over its labels, the epoch with the lowest validation loss kept):
its confidence in a frame is its strongest peak's probability, and every frame where that is at least conf_min(r)
becomes a label -- as many as the model is sure of, not a quota. Every label lies on the pitch (OFF_PITCH_M): left to
itself, model 1 took a white thing beside the benches for the ball now and then, and model 2, taught by those labels,
did so more and more. There is one ball, so in a labelled frame every other detection more than NEG_WIDTHS ball
widths from it is a "not ball".

Each round tries FRAMES_A_ROUND frames, one from each equal slice of the frames still open: never a still one
(STILL_M: the players standing about, as in a stoppage), never one already labelled, and none within MOMENT of a label
from the same block of BLOCK_ROUNDS rounds -- one picture of a moment, not twenty copies; a new block opens every moment
again for the better model. The frames are never in the public five minutes. The last model (round 17) is the ball detector in use:
ballcv/models/ball.pt (heatmap.load).
"""
import csv, json, os, shutil, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import torch
from tqdm import tqdm

import ownball as ob
import heatmap
from video import FW, FH, decode

L, W = ob.L, ob.W
ROOT = ob.S + 'self'
R = np.load(ob.S + 'homography_refined.npy')         # pitch (normalised) -> pixel, every frame
FRAMES_A_ROUND = 3000
SEED_FRAMES = 15000                                  # round 0 tries more: the seed is cheap and rarely fires
EPOCHS = 30                                          # passes over the crops a round: every label seen as often,
                                                     # however many there are
ROUNDS = 18
MOMENT = 50                                          # frames (2 s): labels at least this far apart -- one a moment
BLOCK_ROUNDS = 4                                     # ... within a block of this many rounds; a new block opens every
                                                     #     moment again for the better model to look at
OFF_PITCH_M = 1.0                                    # a label is on the pitch, at most this far past a line
STILL_M = 1.0                                        # a frame is still when the median player moved less than this
                                                     # over the last MOMENT (standing about; walking covers ~3 m)
PUBLIC = (11185, 18684)                              # the public ground truth's frames: never used
HALVES = ((11185, 90651), (110808, 190197))
CONF_START = 0.8                                     # round 1 labels where the model's top peak is this sure; the
CONF_STEP = 0.015                                    # bar comes down this much every CONF_EVERY rounds, as the
CONF_EVERY = 1                                       # models grow better (a schedule: no ground truth to set it by)
CONF_FLOOR = 0.6                                     # ... but never below this
NEG_WIDTHS = 3
TOP = 24                                             # detections kept per frame
BALL_M = 0.22
AIR_SCALE = 2.0
NOISE = 3.0                                          # standard deviations
CROP = heatmap.CROP
DEV = heatmap.DEV
P = np.load(ob.S + 'tactical.npz')['P'][..., :2]


def still_frames():
    """Frames where the players have barely moved over the last MOMENT: a stoppage, the same picture again and again.
    Never a frame here -- a ball lying still for two minutes would otherwise give dozens of copies of one label."""
    import warnings
    pm = P * [L, W]
    moved = np.full(len(pm), np.inf)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)              # frames with no tracked player
        moved[MOMENT:] = np.nanmedian(np.linalg.norm(pm[MOMENT:] - pm[:-MOMENT], axis=2), axis=1)
    return moved < STILL_M


STILL = still_frames()
MB = np.load(ob.S + 'tactical.npz')['B'][:, :2]     # Metrica's tactical ball, pitch fractions
SEARCH_M = 3.0                                       # round 0 looks this close to Metrica's ball for ball-like blobs
NEAR_M = 1.5                                         # ... and takes the one only this close (Metrica far off is off
                                                     #     by more; the decoys it finds lie further out)
CLEAR_M = 1.0                                        # ... with nothing else bright this close to it
ORBITA_SAT = 0.108                                   # the Orbita's mean colour: (max - min) / max (cmd_orbita)
ORBITA_URL = ('https://www.prodirectsport.us/products/puma-orbita-laliga-1-24-25-fifa-quality-pro-puma-white-multicolor-'
              'puma-white-multicolor-1008765')
ORBITA_PNG = __import__('paths').DATA + 'orbita-ball.png'   # the product photo from ORBITA_URL
SAT_K = 1.5                                          # ... a blob's may be up to this much more coloured (light, blur)


def conf_min(r):
    """The confidence round r labels at: CONF_START, CONF_STEP lower every CONF_EVERY rounds, at least CONF_FLOOR."""
    return max(CONF_FLOOR, round(CONF_START - CONF_STEP * ((r - 1) // CONF_EVERY), 4))


def rdir(r):
    d = f'{ROOT}/r{r}'
    os.makedirs(d, exist_ok=True)
    return d


def frames_of(r):
    """Round r's frames: round 0, the SEED_FRAMES of frames.json; a later round, FRAMES_A_ROUND drawn when it starts
    from the frames still open in its block (open_frames), one from each equal slice of them, and kept in
    rK/frames.json so a stopped round goes on with the same ones."""
    if r == 0:
        if not os.path.exists(ROOT + '/frames.json'):
            make_frames()
        return json.load(open(ROOT + '/frames.json'))['seed']
    path = f'{rdir(r)}/frames.json'
    if not os.path.exists(path):
        pool = open_frames(r)
        rng = np.random.default_rng(r)
        pick = pool if len(pool) <= FRAMES_A_ROUND else \
            sorted(int(rng.choice(sl)) for sl in np.array_split(pool, FRAMES_A_ROUND))
        json.dump([int(f) for f in pick], open(path, 'w'))
        tqdm.write(f'round {r}: {len(pick)} frames drawn from the {len(pool)} still open in its block '
                   f'(rounds {block(r) * BLOCK_ROUNDS}-{block(r) * BLOCK_ROUNDS + BLOCK_ROUNDS - 1})')
    return json.load(open(path))


def block(r):
    return r // BLOCK_ROUNDS


def taken(r):
    """The labels round r keeps its distance from: those of the earlier rounds in its block."""
    lab = {}
    for q in range(block(r) * BLOCK_ROUNDS, r):
        lab.update(json.load(open(f'{rdir(q)}/labels.json')))
    return np.array(sorted(int(f) for f in lab), int)


def allowed():
    """Every frame a round may try: in the halves, not the public five minutes, not still."""
    pool = np.r_[np.arange(*HALVES[0]), np.arange(*HALVES[1])]
    bad = np.zeros(pool.max() + 2, bool)
    bad[PUBLIC[0] - 50:PUBLIC[1] + 51] = True
    return pool[~bad[pool] & ~STILL[pool]]


def open_frames(r):
    """The allowed frames at least MOMENT from every label of round r's block so far, and not a label already."""
    pool = allowed()
    near = np.zeros(pool.max() + MOMENT + 1, bool)
    for t in taken(r):
        near[max(0, t - MOMENT + 1):t + MOMENT] = True
    for f in all_labels(r):
        near[f] = True
    return pool[~near[pool]]


VIDEO = __import__('paths').DATA + 'tactical/BAR-ATH_tactical-001.mp4'


def stream(runs, step=1):
    """(frame, RGB image) for frames s, s + step, .. e of each run (s, e): the video read in order, one ffmpeg per
    run (a seek per frame costs far more than decoding the frames between)."""
    import subprocess
    for s0, e in runs:
        n = (e - s0) // step + 1
        cmd = ['ffmpeg', '-nostdin', '-v', 'error', '-ss', f'{s0 / 25:.3f}', '-i', VIDEO]
        if step > 1:
            cmd += ['-vf', f'select=not(mod(n\\,{step}))', '-fps_mode', 'passthrough']
        cmd += ['-frames:v', str(n), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        size = 1080 * 1920 * 3
        for k in range(n):
            raw = proc.stdout.read(size)
            if len(raw) < size:
                break
            yield s0 + k * step, np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3)
        proc.stdout.close(); proc.wait()


def runs_of(frames):
    """Sorted frames -> [(first, last)] of their consecutive runs."""
    out = []
    for f in frames:
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return [tuple(r) for r in out]


def read(frames, desc, status=None, workers=6):
    """RGB frames, decoded in parallel, in order, under a progress bar; status(), if given, is shown on it as it
    goes. At most 2 * workers frames are decoded ahead (6 MB each): the decoders outrun the model, and frames
    piling up in memory slowed everything down."""
    from collections import deque
    with ThreadPoolExecutor(workers) as ex:
        todo, ahead = iter(frames), deque()
        for f in todo:
            ahead.append((f, ex.submit(decode, f)))
            if len(ahead) >= 2 * workers:
                break
        bar = tqdm(total=len(frames), desc=desc, unit='frame', dynamic_ncols=True)
        while ahead:
            f, job = ahead.popleft()
            nxt = next(todo, None)
            if nxt is not None:
                ahead.append((nxt, ex.submit(decode, nxt)))
            yield f, job.result()
            bar.update()
            if status:
                bar.set_postfix(status(), refresh=False)
        bar.close()


# ---------------------------------------------------------------- geometry
def px(Hf, xy):
    """pitch metres (n, 2) -> pixels (n, 2)."""
    q = np.c_[xy[:, 0] / L, xy[:, 1] / W, np.ones(len(xy))] @ Hf.T
    return q[:, :2] / q[:, 2:]


def ground(Hf, uv):
    """pixels (n, 2) -> pitch metres (n, 2), where each pixel's ray meets the grass."""
    q = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(Hf).T
    return q[:, :2] / q[:, 2:] * [L, W]


def ball_width(Hf, uv):
    """px a ball on the grass at each pixel is across."""
    g = ground(Hf, uv)
    return np.linalg.norm(px(Hf, g + [BALL_M, 0]) - px(Hf, g), axis=1)


# ---------------------------------------------------------------- round 0: simple CV
def robust_sd(x):
    return 1.4826 * float(np.median(np.abs(x - np.median(x)))) + 1e-6


def blobs(img, Hf):
    """Every bright blob on the pitch: dict of arrays u, v, z (how much brighter than the grass around it, in noise
    units), area, ball (the disc a ball lying there covers, px), aspect (of its box), sat (its mean colour's
    (max - min) / max)."""
    corners = px(Hf, np.array([[-3, -3], [L + 3, -3], [L + 3, W + 3], [-3, W + 3]]))
    x0, y0 = np.clip(np.floor(corners.min(0)).astype(int), 0, [FW, FH])
    x1, y1 = np.clip(np.ceil(corners.max(0)).astype(int), 0, [FW, FH])
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    pitch = np.zeros((FH, FW), np.uint8)
    cv2.fillConvexPoly(pitch, np.round(corners).astype(np.int32), 1)
    pitch = pitch[y0:y1, x0:x1].astype(bool)
    c = img[y0:y1, x0:x1]
    lum = c.max(-1).astype(np.float32)
    d_mid = max(2.0, float(ball_width(Hf, px(Hf, np.array([[L / 2, W / 2]])))[0]))
    win = int(2 * round(2.5 * AIR_SCALE * d_mid) + 1)               # wider than the biggest ball
    top = lum - cv2.blur(lum, (win, win), borderType=cv2.BORDER_REFLECT)
    sd = robust_sd(top[pitch][::7])
    mask = (top > NOISE * sd) & pitch
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    on = lab > 0
    li = lab[on] - 1
    peak = np.zeros(n - 1, np.float32); np.maximum.at(peak, li, top[on])
    area = st[1:, cv2.CC_STAT_AREA].astype(float)
    rgb = np.stack([np.bincount(li, c[..., k][on].astype(float), n - 1) for k in range(3)], 1) / area[:, None]
    bw, bh = st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]
    uv = cen[1:] + [x0, y0]
    return dict(u=uv[:, 0], v=uv[:, 1], z=peak / sd, area=area, ball=np.pi * (ball_width(Hf, uv) / 2) ** 2,
                aspect=np.maximum(bw, bh) / np.minimum(bw, bh), sat=(rgb.max(1) - rgb.min(1)) / np.maximum(rgb.max(1), 1))


def ball_like(b):
    """Blobs that look like the ball: about the size of a ball lying there (blur and a lifted ball make it bigger),
    round, and whitish as the Orbita is from afar (ORBITA_SAT)."""
    return ((b['area'] >= b['ball'] / 2) & (b['area'] <= 3 * b['ball']) & (b['aspect'] <= 1.5)
            & (b['sat'] <= SAT_K * ORBITA_SAT))


def seed(img, f):
    """Round 0: (the ball's pixel or None, [not-ball pixels]) of one frame. The ball is where Metrica's tactical ball
    and the picture agree: the only ball-like blob within SEARCH_M of Metrica's ball on the grass, itself within
    NEAR_M, with nothing else bright within CLEAR_M of it in the picture. A lifted ball's pixel is above its ground
    point, so it fails, and the frame goes unlabelled."""
    if not np.isfinite(MB[f, 0]):
        return None, []
    b = blobs(img, R[f])
    if b is None:
        return None, []
    to_metrica = np.hypot(*(ground(R[f], np.c_[b['u'], b['v']]) - MB[f] * [L, W]).T)
    near = to_metrica <= SEARCH_M
    ok = ball_like(b)
    if (near & ok).sum() != 1:
        return None, []
    i = int(np.flatnonzero(near & ok)[0])
    width = np.sqrt(b['ball'] * 4 / np.pi)
    beside = np.hypot(b['u'] - b['u'][i], b['v'] - b['v'][i]) * BALL_M / width[i]       # m, at the ball's scale
    if to_metrica[i] > NEAR_M or (near & (beside > 0) & (beside <= CLEAR_M)).any():
        return None, []
    far = ok & ~near & (beside > NEG_WIDTHS * BALL_M)
    neg = np.flatnonzero(far)[np.argsort(-b['z'][far])][:TOP - 1]
    return (float(b['u'][i]), float(b['v'][i])), [(float(b['u'][j]), float(b['v'][j])) for j in neg]


# ---------------------------------------------------------------- frames
def make_frames():
    os.makedirs(ROOT, exist_ok=True)
    pool = allowed()
    rng = np.random.default_rng(0)
    seed_ = sorted(int(rng.choice(sl)) for sl in np.array_split(pool, SEED_FRAMES))   # one a slice of time
    json.dump(dict(seed=seed_), open(ROOT + '/frames.json', 'w'))
    print(f'{SEED_FRAMES} seed frames from {len(pool)} allowed -> {ROOT}/frames.json')


# ---------------------------------------------------------------- the model
def load_net(r):
    net = heatmap.BallFCN()
    net.load_state_dict(torch.load(f'{rdir(r)}/model.pt', map_location='cpu'))
    return net.to(DEV).eval()


def peaks(net, img, f):
    """[(u, v, probability)] of the model's heatmap peaks in one frame, strongest first."""
    return [(u, v, float(1 / (1 + np.exp(-lg)))) for u, v, lg in
            heatmap.peaks_in(net, img, heatmap.pitch_box(R[f]), thresh=-4.0, top=TOP)]


# ---------------------------------------------------------------- labels
def all_labels(upto):
    """Every label of rounds 0 .. upto - 1: {frame: dict(ball, neg)}."""
    lab = {}
    for q in range(upto):
        lab.update({int(f): v for f, v in json.load(open(f'{rdir(q)}/labels.json')).items()})
    return lab


def on_pitch(f, uv):
    """Is pixel uv of frame f on the pitch (its ground point at most OFF_PITCH_M past a line)? A ball in play is."""
    x, y = ground(R[f], np.array([uv]))[0]
    return -OFF_PITCH_M <= x <= L + OFF_PITCH_M and -OFF_PITCH_M <= y <= W + OFF_PITCH_M


def label_round(r, net):
    """Label round r's frames, in order: with no model, where the seed finds the ball; else where the model's top peak
    is at least conf_min(r); each at least MOMENT from every label before it in its block. -> its labels."""
    near = taken(r)                                                  # this block's earlier labels
    partial = f'{rdir(r)}/labels.partial.json'                      # saved as it goes: a stopped round goes on
    lab, upto = {}, -1
    if os.path.exists(partial):
        z = json.load(open(partial))
        lab, upto = {int(f): v for f, v in z['labels'].items()}, z['upto']
        tqdm.write(f'round {r}: going on after frame {upto} with the {len(lab)} labels found before')
    frames = [f for f in frames_of(r) if f > upto]
    who = 'seed' if net is None else f'model {r} at {conf_min(r):.2f}'
    for k, (f, img) in enumerate(read(frames, f'round {r}: {who} on new frames', lambda: {'labels': len(lab)})):
        if k % 100 == 99:
            json.dump(dict(upto=f - 1, labels={str(g): v for g, v in lab.items()}), open(partial, 'w'))
        if STILL[f] or (len(near) and np.abs(near - f).min() < MOMENT) or (lab and f - max(lab) < MOMENT):
            continue                                                 # frames come in order: the last is the nearest
        if net is None:
            ball, neg = seed(img, f)
            conf = 1.0
        else:
            pk = peaks(net, img, f)
            if not pk or pk[0][2] < conf_min(r):
                continue
            (u, v, conf), w = pk[0], NEG_WIDTHS * max(2.0, float(ball_width(R[f], np.array([pk[0][:2]]))[0]))
            ball, neg = (u, v), [(a, b) for a, b, _ in pk[1:] if np.hypot(a - u, b - v) > w]
        if ball and on_pitch(f, ball):
            lab[f] = dict(ball=[round(ball[0], 1), round(ball[1], 1)], conf=round(conf, 4),
                          neg=[[round(a, 1), round(b, 1)] for a, b in neg])
    json.dump({str(f): v for f, v in lab.items()}, open(f'{rdir(r)}/labels.json', 'w'))
    if os.path.exists(partial):
        os.remove(partial)
    return lab


# ---------------------------------------------------------------- crops
def crops(lab, seed_):
    """Crops from labels: the ball off-centre twice, three of the frame's not-balls, one anywhere. Every fifth
    labelled frame is validation."""
    rng = np.random.default_rng(seed_)
    frames = sorted(lab)
    X, T, val = [], [], []
    for k, (f, img) in enumerate(read(frames, f'crops from {len(frames)} labelled frames')):
        (bu, bv), neg = lab[f]['ball'], lab[f]['neg']
        is_val = k % 5 == 4

        def add(cx, cy):
            x0 = int(np.clip(cx - CROP // 2, 0, FW - CROP)); y0 = int(np.clip(cy - CROP // 2, 0, FH - CROP))
            pts = [((bu - x0) / heatmap.STRIDE, (bv - y0) / heatmap.STRIDE)] if x0 <= bu < x0 + CROP and y0 <= bv < y0 + CROP else []
            X.append(img[y0:y0 + CROP, x0:x0 + CROP]); T.append(heatmap.target(CROP // heatmap.STRIDE, CROP // heatmap.STRIDE, pts))
            val.append(is_val)
        for _ in range(2):
            add(bu + rng.integers(-30, 31), bv + rng.integers(-30, 31))
        for nu, nv in [neg[i] for i in rng.permutation(len(neg))[:3]]:
            add(nu + rng.integers(-8, 9), nv + rng.integers(-8, 9))
        add(rng.integers(48, FW - 48), rng.integers(200, FH - 48))
    X = torch.tensor(np.array(X, np.uint8)).permute(0, 3, 1, 2).float().div(255)
    T = torch.tensor(np.array(T, np.float32))
    return X, T, torch.tensor(val)


# ---------------------------------------------------------------- training
def scores(net, X, T, batch=256):
    """loss; found: on a crop with the ball, the heatmap's top peak is on it (within a cell); alarm: on a crop
    without, the top peak is over a half."""
    found = alarm = npos = nneg = 0
    loss = 0.0
    net.eval()
    with torch.no_grad():
        for k in range(0, len(X), batch):
            x, t = X[k:k + batch].to(DEV), T[k:k + batch].to(DEV)
            lg = net(x)
            loss += float(heatmap.focal(lg, t)) * len(x)
            p, tt = torch.sigmoid(lg).flatten(1), t.flatten(1)
            pos = tt.max(1).values > 0.5
            pk, at = p.max(1)
            w, ball = t.shape[-1], tt.argmax(1)
            near = (torch.abs(at % w - ball % w) <= 1) & (torch.abs(at // w - ball // w) <= 1)
            found += int((pos & near).sum()); npos += int(pos.sum())
            alarm += int((~pos & (pk > 0.5)).sum()); nneg += int((~pos).sum())
    return loss / len(X), found / max(1, npos), alarm / max(1, nneg)


def train(r, X, T, val, batch=64, lr=2e-3):
    """Model r on the crops for EPOCHS passes, from scratch every round (a model fitted only to the labels as they now
    stand, not carrying the last one's habits); the epoch with the lowest validation loss is kept. A bar per epoch
    with the batch's loss on it, then a line per epoch."""
    Xt, Tt, Xv, Tv = X[~val], T[~val], X[val], T[val]
    epochs = EPOCHS
    net = heatmap.BallFCN()
    torch.nn.init.constant_(net.head.bias, -4.0)                     # start sure there is no ball (CenterNet)
    net = net.to(DEV)
    if r > 1:                                                        # the last model: the baseline, not the start
        l0, f0, a0 = scores(load_net(r - 1), Xv, Tv)
        tqdm.write(f'round {r}: baseline to beat, model {r - 1} on this validation: loss {l0:.3f}, found {f0:.1%}, '
                   f'alarm {a0:.1%}')
    tqdm.write(f'round {r}: training on {len(Xt)} crops ({int((Tt.flatten(1).max(1).values > 0.5).sum())} with the '
               f'ball), validating on {len(Xv)}; {epochs} epochs, {epochs * -(-len(Xt) // batch)} steps')
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(Xt) + batch - 1) // batch))
    log = open(f'{rdir(r)}/log.csv', 'w', newline='')
    w = csv.writer(log)
    w.writerow(['epoch', 'train_loss', 'val_loss', 'val_found', 'val_alarm'])
    best = np.inf
    for ep in range(1, epochs + 1):
        net.train()
        perm, tot, seen = torch.randperm(len(Xt)), 0.0, 0
        bar = tqdm(range(0, len(Xt), batch), desc=f'round {r} epoch {ep}/{epochs}', unit='batch', dynamic_ncols=True,
                   leave=False)
        for k in bar:
            idx = perm[k:k + batch]
            x, t = heatmap.augment(Xt[idx].to(DEV), Tt[idx].to(DEV))
            loss = heatmap.focal(net(x), t)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss.detach()) * len(idx); seen += len(idx)
            bar.set_postfix(loss=f'{float(loss.detach()):.3f}', mean=f'{tot / seen:.3f}')
        vl, found, alarm = scores(net, Xv, Tv)
        w.writerow([ep, round(tot / seen, 4), round(vl, 4), round(found, 4), round(alarm, 4)]); log.flush()
        mark = ''
        if vl < best:
            best = vl
            torch.save(net.state_dict(), f'{rdir(r)}/model.pt'); mark = '  saved'
        tqdm.write(f'round {r} epoch {ep:2d}  train {tot / seen:.3f}  val {vl:.3f}  found {found:.1%}  alarm {alarm:.1%}{mark}')
    json.dump(dict(epochs=epochs, best_val_loss=round(best, 4)), open(f'{rdir(r)}/trained.json', 'w'))
    return load_net(r)


# ---------------------------------------------------------------- a round
def spread(lab):
    """How the labels lie over the match: the 5-minute stretches of play holding any, and the most in one."""
    block = 25 * 300
    edges = [(a, b) for a, b in HALVES]
    n_blocks = sum((b - a + block - 1) // block for a, b in edges)
    per = {}
    for f in lab:
        for h, (a, b) in enumerate(edges):
            if a <= f <= b:
                per[(h, (f - a) // block)] = per.get((h, (f - a) // block), 0) + 1
    return f'in {len(per)} of {n_blocks} five-minute stretches, at most {max(per.values()) if per else 0} in one'


def cmd_run():
    for r in range(ROUNDS):
        if os.path.exists(f'{ROOT}/r{r}/labels.json'):
            tqdm.write(f'round {r}: done already, {len(all_labels(r + 1)) - len(all_labels(r))} labels')
            continue
        tqdm.write(f'\n================ round {r} of {ROUNDS - 1} ================')
        if not cmd_round(r):
            tqdm.write(f'round {r} took no new labels: stopping')
            break
    last = max((r for r in range(ROUNDS) if os.path.exists(f'{rdir(r)}/model.pt')), default=None)
    if last is not None:
        shutil.copy(f'{rdir(last)}/model.pt', heatmap.BALL_MODEL)
        tqdm.write(f'model {last} -> {heatmap.BALL_MODEL}')


def cmd_round(r):
    """One round; -> how many new labels it took."""
    if r == 0:
        lab = label_round(0, None)
        tqdm.write(f'round 0: the seed labelled {len(lab)} of {SEED_FRAMES} frames '
                   f'({sum(len(v["neg"]) for v in lab.values())} not-balls), {spread(lab)} -> {rdir(0)}/labels.json')
        log_round(0, len(lab), len(lab), None)
        return len(lab)
    so_far = all_labels(r)
    if os.path.exists(f'{rdir(r)}/trained.json'):                   # trained already: label with it
        tqdm.write(f'round {r}: model {r} is trained already ({rdir(r)}/model.pt), on {len(so_far)} labels')
        net = load_net(r)
    else:
        tqdm.write(f'round {r}: {len(so_far)} labels from rounds 0-{r - 1}')
        X, T, val = crops(so_far, r)
        net = train(r, X, T, val)
    lab = label_round(r, net)
    conf = [v['conf'] for v in lab.values()]
    tqdm.write(f'round {r}: took {len(lab)} new labels of {FRAMES_A_ROUND} frames, confidence '
               f'{min(conf) if conf else float("nan"):.3f} to {max(conf) if conf else float("nan"):.3f}; '
               f'{len(so_far) + len(lab)} labels in all, {spread({**so_far, **lab})}')
    log_round(r, len(lab), len(so_far) + len(lab), min(conf) if conf else None)
    return len(lab)


def log_round(r, new, total, conf_min):
    path = ROOT + '/rounds.csv'
    rows = list(csv.reader(open(path))) if os.path.exists(path) else []
    head = ['round', 'new_labels', 'total_labels', 'min_confidence']
    row = [r, new, total, '' if conf_min is None else round(conf_min, 4)]
    rows = sorted([x for x in rows[1:] if x[0] != str(r)] + [row], key=lambda x: int(x[0]))
    csv.writer(open(path, 'w', newline='')).writerows([head] + rows)


def cmd_orbita():
    """ORBITA_SAT from the product photo: the ball is the box of pixels differing from the photo's background (its
    corners) by more than 25 in summed RGB; its mean colour over the disc inside it (95% of the radius, the edge left
    out), and that colour's (max - min) / max."""
    rgb = cv2.cvtColor(cv2.imread(ORBITA_PNG), cv2.COLOR_BGR2RGB).astype(float)
    h, w = rgb.shape[:2]
    corner = np.median(np.r_[rgb[:5, :5].reshape(-1, 3), rgb[:5, -5:].reshape(-1, 3), rgb[-5:, :5].reshape(-1, 3)], 0)
    ys, xs = np.nonzero(np.abs(rgb - corner).sum(-1) > 25)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2; r = min(ys.max() - ys.min(), xs.max() - xs.min()) / 2
    yy, xx = np.mgrid[:h, :w]
    m = rgb[(yy - cy) ** 2 + (xx - cx) ** 2 <= (0.95 * r) ** 2].mean(0)
    print(f'the Orbita\'s mean colour {m.round(1)}: (max - min) / max = {(m.max() - m.min()) / m.max():.3f} (ORBITA_SAT {ORBITA_SAT})')


if __name__ == '__main__':
    cmd = sys.argv[1]
    r = int(sys.argv[2]) if len(sys.argv) > 2 else None
    {'run': cmd_run, 'round': lambda: cmd_round(r), 'orbita': cmd_orbita}[cmd]()
