"""Whose kit a tracked player is wearing, from the picture: a model that teaches itself, like selftrain.py's ball.

Metrica's tracking names every slot's team, and mostly rightly -- but not always where it matters: in a duel a new
track can start on the wrong team's player (slot 33 at 11405 was born an Athletic track on a Barcelona player, a second
before the touch credited to it), and the goalkeeper tracks swap. The touch rule credits the nearest slot's team, so
those mistakes become possession changes. The kit does not lie: four kits in this match, each team's outfield and
goalkeeper one (CLASSES).

Round 0 (seed): Metrica's team where it is sure -- a track at least MATURE frames into its life and MATURE from its end
(a new track can sit on the wrong player for its first second), not frozen (ownball.stale_mask), alone in the picture
(no other player's body box overlaps its own: only one kit in the crop), feet and head in the image. A goalkeeper slot
is a seed only inside a penalty area, as that box's team's keeper (keepers do not leave their box often, and their
tracks swap); an outfield slot in a box is still its slot's team.

From round 1: model K trains on every label so far (from scratch, EPOCHS passes, the epoch with the lowest validation
loss kept), then labels a fresh FRAMES_A_ROUND frames, only the crops the seed cannot take -- players overlapping
others, tracks young or ending, keepers out of their box -- where its top class is at least CONF sure. Those labels
are the model's, not Metrica's: where they disagree with the slot's team is exactly what it is for. Each round's
frames are new; frames are never in the public five minutes (PUBLIC), never still (selftrain.STILL) and always in a
half.

Nothing is scored against hand labels (there are none for kits). Printed every epoch: validation accuracy on held-out
labels (every VAL_EVERY-th frame), per class; every round: how many new labels, and how many disagree with Metrica.

  python ballcv/kit_team.py run        every round in turn (you run this); finished rounds are skipped; at the end the
                                       last model is copied to ballcv/models/kit.pt, which player_det.py's seed uses
  python ballcv/kit_team.py round K    one round
"""
import json, os, sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import ownball as ob
import selftrain as st
from cameras import load_h

L, W = ob.L, ob.W
ROOT = ob.S + 'kit'
MODEL = ob.KIT_MODEL
CLASSES = ('H', 'A', 'H keeper', 'A keeper')
TEAM_OF = ('H', 'A', 'H', 'A')
CROP_H, CROP_W = 48, 24                  # each crop, pixels
BOX = 1.6                                # crop height in body heights (the 2 m column), centred on the body
BODY_W = 0.5                             # a body's box is this wide, in body heights (for "alone": boxes overlapping)
MATURE = 50                              # frames (2 s) of a track's life at either end not trusted for its team
FRAMES_A_ROUND = 3000
SEED_FRAMES = 2000
ROUNDS = 4
EPOCHS = 20
CONF = 0.99                              # a model's label: its top class at least this sure
VAL_EVERY = 5                            # every 5th labelled frame is for validation
DEV = st.DEV

M, _, _ = load_h('tactical')
_t = np.load(ob.S + 'tactical.npz')
P = _t['P'][..., :2]
SLOT_TEAM = np.array([0] * 24 + [1] * 22)
KEEPERS = ob.keeper_slots()
STALE = ob.stale_mask(P)
SIDES = ob.sides_by_half(P)
LIVE = np.all(np.isfinite(P), axis=2) & ~STALE


def life():
    """(n, slots): frames since each slot's track began, and until it ends (a gap ends a track)."""
    n, k = LIVE.shape
    born, left = np.zeros((n, k), np.int32), np.zeros((n, k), np.int32)
    for f in range(1, n):
        born[f] = np.where(LIVE[f], np.where(LIVE[f - 1], born[f - 1] + 1, 0), 0)
    for f in range(n - 2, -1, -1):
        left[f] = np.where(LIVE[f], np.where(LIVE[f + 1], left[f + 1] + 1, 0), 0)
    return born, left


BORN, LEFT = life()


def rdir(r):
    d = f'{ROOT}/r{r}'
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------- frames, players, crops
def open_frames():
    """Frames a round may use: in a half, not the public five minutes, not still."""
    f = np.concatenate([np.arange(a, b + 1) for a, b in st.HALVES])
    return f[~((f >= st.PUBLIC[0]) & (f <= st.PUBLIC[1])) & ~st.STILL[f]]


def frames_of(r):
    """Round r's frames: one from each equal slice of the open frames, at an offset of the round's own (new every round)."""
    o = open_frames(); n = SEED_FRAMES if r == 0 else FRAMES_A_ROUND
    rng = np.random.default_rng(r)
    edges = np.linspace(0, len(o), n + 1).astype(int)
    return sorted(int(o[rng.integers(a, b)]) for a, b in zip(edges, edges[1:]) if b > a)


def players(f):
    """[(slot, feet, top, box)] of the live slots at f whose feet and head are in the image; box = (x0, y0, x1, y1)."""
    out = []
    for s in np.where(LIVE[f])[0]:
        feet, top = ob.column(M[f], P[f, s] * [L, W])
        h = np.linalg.norm(top - feet)
        if h < 4 or not all(0 <= p[0] < st.FW and 0 <= p[1] < st.FH for p in (feet, top)):
            continue
        cx = (feet[0] + top[0]) / 2
        out.append((int(s), feet, top, (cx - BODY_W * h / 2, min(feet[1], top[1]), cx + BODY_W * h / 2, max(feet[1], top[1]))))
    return out


def alone(i, people):
    x0, y0, x1, y1 = people[i][3]
    return not any(j != i and b[0] < x1 and x0 < b[2] and b[1] < y1 and y0 < b[3] for j, (*_, b) in enumerate(people))


def crop(img, feet, top):
    """the player's box (BOX body heights tall, centred on the body), resized to CROP_H x CROP_W."""
    h = np.linalg.norm(top - feet) * BOX; w = h * CROP_W / CROP_H
    cx, cy = (feet + top) / 2
    A = np.array([[CROP_W / w, 0, CROP_W / 2 - cx * CROP_W / w], [0, CROP_H / h, CROP_H / 2 - cy * CROP_H / h]])
    return cv2.warpAffine(img, A, (CROP_W, CROP_H), flags=cv2.INTER_AREA, borderValue=(20, 20, 20))


def seed_class(f, s, i, people):
    """Metrica's class for slot s at f where it is sure (see the module docstring), else None."""
    if BORN[f, s] < MATURE or LEFT[f, s] < MATURE or not alone(i, people):
        return None
    if s in KEEPERS:
        t = ob.box_team(P[f, s] * [L, W], SIDES(f))
        return None if t is None else 2 + 'HA'.index(t)
    return int(SLOT_TEAM[s])


# ---------------------------------------------------------------- the model
class KitNet(nn.Module):
    """Small CNN: three conv blocks, global average pool, one linear layer to the four kits."""

    def __init__(self):
        super().__init__()
        def block(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
                                 nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2))
        self.body = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.head = nn.Linear(128, len(CLASSES))

    def forward(self, x):
        return self.head(self.body(x).mean((2, 3)))


def to_input(X):
    return torch.as_tensor(X).permute(0, 3, 1, 2).float().div(255.0)


def augment(x):
    """x: (n,3,H,W) 0-1. Horizontal flip, brightness/contrast jitter (sun and shade), a shift of up to 2 px."""
    flip = torch.rand(len(x), device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    gain = torch.empty(len(x), 1, 1, 1, device=x.device).uniform_(0.75, 1.25)
    bias = torch.empty(len(x), 1, 1, 1, device=x.device).uniform_(-0.08, 0.08)
    x = (x * gain + bias).clamp(0, 1)
    dx, dy = np.random.randint(-2, 3, 2)
    return torch.roll(x, shifts=(int(dy), int(dx)), dims=(2, 3))


def load_net(path):
    net = KitNet(); net.load_state_dict(torch.load(path, map_location='cpu'))
    return net.to(DEV).eval()


@torch.no_grad()
def probs(net, X, batch=512):
    out = [F.softmax(net(to_input(X[k:k + batch]).to(DEV)), 1).cpu().numpy() for k in range(0, len(X), batch)]
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)))


# ---------------------------------------------------------------- a round's labels
def label_round(r, net):
    """Round r's crops and labels -> rK/labels.npz (X crops, y class, f, slot, metrica class, kind)."""
    X, y, fs, ss, mc, kind = [], [], [], [], [], []
    n_seen = 0
    status = lambda: dict(labels=len(y), disagree=int(sum(TEAM_OF[a] != TEAM_OF[b] for a, b in zip(y, mc))))
    for f, img in st.read(frames_of(r), f'round {r} labelling', status):
        people = players(f)
        cand = []
        for i, (s, feet, top, _b) in enumerate(people):
            c = seed_class(f, s, i, people)
            if r == 0 and c is not None:
                cand.append((s, feet, top, c, 'seed'))
            elif r > 0 and c is None:
                k = 'keeper out of box' if s in KEEPERS else 'young/ending' if min(BORN[f, s], LEFT[f, s]) < MATURE else 'overlap'
                cand.append((s, feet, top, None, k))
        if not cand:
            continue
        crops = np.stack([crop(img, feet, top) for s, feet, top, *_ in cand])
        if r > 0:
            pr = probs(net, crops)
        for j, (s, feet, top, c, k) in enumerate(cand):
            n_seen += 1
            if r > 0:
                if pr[j].max() < CONF:
                    continue
                c = int(pr[j].argmax())
            met = 2 + int(SLOT_TEAM[s]) if s in KEEPERS else int(SLOT_TEAM[s])
            X.append(crops[j]); y.append(c); fs.append(f); ss.append(s + 1); mc.append(met); kind.append(k)
    X = np.stack(X) if X else np.zeros((0, CROP_H, CROP_W, 3), np.uint8)
    np.savez_compressed(f'{rdir(r)}/labels.npz', X=X, y=np.array(y, np.int8), f=np.array(fs), slot=np.array(ss),
                        metrica=np.array(mc, np.int8), kind=np.array(kind))
    report(r, np.array(y), np.array(mc), np.array(kind), n_seen)
    return len(y)


def report(r, y, mc, kind, n_seen):
    dis = np.array([TEAM_OF[a] != TEAM_OF[b] for a, b in zip(y, mc)], bool)
    tqdm.write(f'round {r}: {len(y)} labels of {n_seen} crops tried; by kit ' +
               ', '.join(f'{CLASSES[c]} {int((y == c).sum())}' for c in range(len(CLASSES))) +
               f'; team differs from Metrica\'s slot on {int(dis.sum())}')
    for k in sorted(set(kind)):
        m = kind == k
        tqdm.write(f'    {k:<18} {int(m.sum()):6d} labels, {int(dis[m].sum()):5d} on the other team from Metrica')
    if not os.path.exists(f'{ROOT}/rounds.csv'):
        open(f'{ROOT}/rounds.csv', 'w').write('round,labels,tried,team_differs_from_metrica\n')
    with open(f'{ROOT}/rounds.csv', 'a') as fh:
        fh.write(f'{r},{len(y)},{n_seen},{int(dis.sum())}\n')


def all_labels(upto):
    parts = [np.load(f'{rdir(k)}/labels.npz') for k in range(upto + 1) if os.path.exists(f'{rdir(k)}/labels.npz')]
    return (np.concatenate([p['X'] for p in parts]), np.concatenate([p['y'] for p in parts]).astype(np.int64),
            np.concatenate([p['f'] for p in parts]))


# ---------------------------------------------------------------- training
def train(r, batch=128, lr=2e-3):
    """Model r on the labels of rounds 0..r-1, from scratch, EPOCHS passes; the lowest validation loss is kept."""
    X, y, f = all_labels(r - 1)
    frames = np.unique(f)
    val = np.isin(f, frames[::VAL_EVERY])
    Xt, yt, Xv, yv = X[~val], torch.as_tensor(y[~val]), X[val], y[val]
    counts = np.bincount(yt.numpy(), minlength=len(CLASSES)).astype(float)
    weight = torch.as_tensor(counts.sum() / np.maximum(counts, 1) / len(CLASSES), dtype=torch.float32, device=DEV)
    tqdm.write(f'round {r}: training on {len(Xt)} crops (' + ', '.join(f'{CLASSES[c]} {int(counts[c])}' for c in range(len(CLASSES))) +
               f'), validating on {len(Xv)}; {EPOCHS} epochs; classes weighted by how rare they are')
    net = KitNet().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = EPOCHS * -(-len(Xt) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    log = open(f'{rdir(r)}/log.csv', 'w')
    log.write('epoch,train_loss,val_loss,val_acc,' + ','.join(f'acc_{c.replace(" ", "_")}' for c in CLASSES) + '\n')
    best = np.inf
    for ep in range(1, EPOCHS + 1):
        net.train()
        perm, tot, seen = torch.randperm(len(Xt)), 0.0, 0
        bar = tqdm(range(0, len(Xt), batch), desc=f'round {r} epoch {ep}/{EPOCHS}', unit='batch', dynamic_ncols=True, leave=False)
        for k in bar:
            idx = perm[k:k + batch]
            x = augment(to_input(Xt[idx.numpy()]).to(DEV))
            loss = F.cross_entropy(net(x), yt[idx].to(DEV), weight=weight)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss.detach()) * len(idx); seen += len(idx)
            bar.set_postfix(loss=f'{float(loss.detach()):.3f}', mean=f'{tot / seen:.3f}')
        net.eval()
        pv = probs(net, Xv)
        vl = float(F.nll_loss(torch.log(torch.as_tensor(pv).clamp_min(1e-9)), torch.as_tensor(yv, dtype=torch.long),
                              weight=weight.cpu()))
        hit = pv.argmax(1) == yv
        per = [hit[yv == c].mean() if (yv == c).any() else np.nan for c in range(len(CLASSES))]
        log.write(f'{ep},{tot / seen:.4f},{vl:.4f},{hit.mean():.4f},' + ','.join(f'{p:.4f}' for p in per) + '\n'); log.flush()
        mark = ''
        if vl < best:
            best = vl; torch.save(net.state_dict(), f'{rdir(r)}/model.pt'); mark = '  saved'
        tqdm.write(f'round {r} epoch {ep:2d}  train {tot / seen:.3f}  val {vl:.3f}  right {hit.mean():.2%}  (' +
                   ', '.join(f'{CLASSES[c]} {per[c]:.1%}' for c in range(len(CLASSES))) + f'){mark}')
    json.dump(dict(epochs=EPOCHS, best_val_loss=round(best, 4)), open(f'{rdir(r)}/trained.json', 'w'))
    return load_net(f'{rdir(r)}/model.pt')


def cmd_round(r):
    net = None
    if r > 0:
        net = load_net(f'{rdir(r)}/model.pt') if os.path.exists(f'{rdir(r)}/trained.json') else train(r)
    return label_round(r, net)


def cmd_run():
    last = None
    for r in range(ROUNDS):
        if os.path.exists(f'{rdir(r)}/labels.npz'):
            last = r if os.path.exists(f'{rdir(r)}/model.pt') else last
            continue
        new = cmd_round(r)
        last = r if r > 0 else last
        if r > 0 and new == 0:
            tqdm.write(f'round {r}: no new labels -- stopping'); break
    if last is None:                                   # only the seed so far: train the first model on it
        train(1); last = 1
    import shutil
    shutil.copy(f'{rdir(last)}/model.pt', MODEL)
    tqdm.write(f'model {last} -> {MODEL}')


if __name__ == '__main__':
    os.makedirs(ROOT, exist_ok=True)
    cmd = sys.argv[1]
    {'run': cmd_run, 'round': lambda: cmd_round(int(sys.argv[2]))}[cmd]()
