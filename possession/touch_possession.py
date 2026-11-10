"""Which team has the ball in live play, from the touches alone: a touch by home or away at each 10 Hz step in, H or A
out. Dead ball is not this model's job (stoppages and the assistants' flags decide it); the gaps between possessions
are left out of the loss and the scores.

Learned from SkillCorner (20 games): its player possessions give the touches, exact to the frame (a player's first
touch and his last: the touches in between are his own team's, so they cannot change possession), and its phases of
play give the team in possession, the closest open labels to the challenge's (median 7.7 s; BAR-ATH 7.6 s). Nothing
from BAR-ATH is fitted or chosen.

Our touches will not be SkillCorner's: some are missed, and a bounce may be called one. Training degrades the touches
every time a chunk is drawn -- a share of them dropped, U(0, MISS_MAX), and false ones added at random times for a
random team, U(0, FALSE_MAX) a minute -- so the model never relies on every touch being there, nor on every touch
being real. `test` shows how much each kind of error costs, for the model and for two rules:
  last touch    the team of the latest touch
  control       ownball.control_roles, the touch pipeline's rule: the other team takes the ball when its touch is
                followed by another of its own (or by no touch); a lone touch between two of ours is a block

Split by game (split_of): every 5th game is test, the one before it val. Val picks the epoch; test is looked
at once.

  build                      -> data/possession/touch_games.npz                                     (~1 min)
  train [--epochs N]         -> ballcv/models/touch_possession.pt, per-epoch log data/possession/touch_possession_log.csv
  test                       the rules and the model on the test games, touches missed / false at several rates
"""
import argparse, csv, json, os, sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, 'ballcv')]

HZ = 10
H, A, UNK = 0, 1, -1
CHUNK = 600                  # training window, steps (60 s)
CONTEXT = 100                # inference: steps either side of a window's middle kept as context only
MISS_MAX = 0.3               # training: up to this share of the touches dropped ...
FALSE_MAX = 1.0              # ... and up to this many false touches a minute added
EPOCHS, STEPS, BATCH = 20, 200, 32
GRID = [(0.0, 0.0), (0.15, 0.0), (0.3, 0.0), (0.0, 0.5), (0.15, 0.5), (0.3, 1.0)]   # test: (missed share, false a minute)
VAL_AT = (0.15, 0.5)

GAMES_DIR = Path(ROOT) / 'data' / 'external' / 'games'      # python -m external.build skillcorner
OUT = Path(ROOT) / 'data' / 'possession'
CACHE, LOG = OUT / 'touch_games.npz', OUT / 'touch_possession_log.csv'
CKPT = Path(ROOT) / 'ballcv' / 'models' / 'touch_possession.pt'                 # the model in use: committed


# ---------------------------------------------------------------- games -> periods: touches (T, 2) and labels (T,)
def split_of(game):
    """Every 5th SkillCorner game (sorted) is test, the one before it val."""
    i = sorted(p.name for p in GAMES_DIR.glob('skillcorner-*')).index(game)
    return 'test' if i % 5 == 4 else 'val' if i % 5 == 3 else 'train'


def sc_game(d):
    m = json.load(open(d / 'meta.json'))
    ev = json.load(open(d / 'events.json'))
    n = m['n_frames']
    lab = np.full(n, UNK, np.int8)
    for e in ev:
        if e['type'] == 'phase' and e['f_end'] is not None and e['team'] in ('H', 'A'):
            lab[e['f']:e['f_end'] + 1] = H if e['team'] == 'H' else A
    touch = np.zeros((n, 2), np.int8)
    for e in ev:
        if e['type'] == 'player_possession' and e['team'] in ('H', 'A'):
            k = H if e['team'] == 'H' else A
            for f in (e['f'], e['f_end']):
                if f is not None:
                    touch[f, k] = 1
    return [dict(game=d.name, split=split_of(d.name), touch=touch[p['start']:p['end'] + 1],
                 lab=lab[p['start']:p['end'] + 1]) for p in m['periods']]


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    periods = [p for d in sorted(GAMES_DIR.glob('skillcorner-*')) for p in sc_game(d)]
    assert periods, f'no SkillCorner games in {GAMES_DIR}: python -m external.build skillcorner'
    meta = [{k: p[k] for k in ('game', 'split')} for p in periods]
    np.savez(CACHE, meta=json.dumps(meta), **{f'{i}_{k}': p[k] for i, p in enumerate(periods) for k in ('touch', 'lab')})
    for split in ('train', 'val', 'test'):
        sel = [p for p in periods if p['split'] == split]
        live = sum(int((p['lab'] != UNK).sum()) for p in sel)
        print(f'{split:5s}: {len({p["game"] for p in sel})} games, {live / HZ / 3600:.1f} h live, '
              f'{sum(int(p["touch"].sum()) for p in sel) / max(1, live / HZ / 60):.1f} touches a live minute')
    print(f'-> {CACHE.relative_to(ROOT)}')


def load(splits):
    z = np.load(CACHE)
    return [dict(m, touch=z[f'{i}_touch'], lab=z[f'{i}_lab']) for i, m in enumerate(json.loads(str(z['meta'])))
            if m['split'] in splits]


# ---------------------------------------------------------------- our touches' errors, put on theirs
def degrade(touch, miss, false_per_min, rng):
    t = touch * (rng.random(touch.shape) >= miss)
    n_false = rng.poisson(false_per_min * len(t) / HZ / 60)
    t[rng.integers(0, len(t), n_false), rng.integers(0, 2, n_false)] = 1
    return t.astype(np.int8)


# ---------------------------------------------------------------- rules
def last_touch(touch):
    team = np.where(touch[:, H] > 0, H, np.where(touch[:, A] > 0, A, UNK))
    out, cur = np.full(len(team), UNK, np.int8), UNK
    for i, k in enumerate(team):
        cur = k if k != UNK else cur
        out[i] = cur
    first = np.flatnonzero(team != UNK)
    if len(first):
        out[:first[0]] = team[first[0]]
    return out


def control(touch):
    from ownball import control_roles
    ev = [dict(f=int(f), team='HA'[k]) for f, k in zip(*np.nonzero(touch))]
    changes = control_roles(ev, [])
    out = np.full(len(touch), UNK, np.int8)
    for c, nxt in zip(changes, changes[1:] + [None]):
        out[c['f']:nxt['f'] if nxt else len(out)] = 'HA'.index(c['team'])
    if changes:
        out[:changes[0]['f']] = 'HA'.index(changes[0]['team'])
    return out


# ---------------------------------------------------------------- the model
def make_net():
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(2, 32, num_layers=2, batch_first=True, bidirectional=True)
            self.head = nn.Linear(64, 2)

        def forward(self, x):
            return self.head(self.gru(x)[0])

    return Net()


def run(net, touch):
    """H/A for every step of one period, as it was trained: CHUNK-step windows, each kept only in its middle (CONTEXT
    steps either side are context)."""
    import torch
    T, stride = len(touch), CHUNK - 2 * CONTEXT
    n = -(-T // stride)
    pad = np.zeros((n * stride + 2 * CONTEXT, 2), np.float32); pad[CONTEXT:CONTEXT + T] = touch
    wins = np.stack([pad[i * stride:i * stride + CHUNK] for i in range(n)])
    with torch.no_grad():
        o = net(torch.from_numpy(wins))[:, CONTEXT:CONTEXT + stride]
    return o.argmax(-1).reshape(-1)[:T].numpy().astype(np.int8)


def accuracy(periods, preds):
    """Share of live steps (labelled H or A) given the right team."""
    ok = n = 0
    for p, q in zip(periods, preds):
        live = p['lab'] != UNK
        ok += int((q[live] == p['lab'][live]).sum()); n += int(live.sum())
    return ok / max(1, n)


def degraded(periods, miss, false_per_min, seed):
    rng = np.random.default_rng(seed)
    return [degrade(p['touch'], miss, false_per_min, rng) for p in periods]


def train(epochs):
    import torch, torch.nn.functional as F
    torch.manual_seed(0); rng = np.random.default_rng(0)
    tr, va = load({'train'}), load({'val'})
    va_t = degraded(va, *VAL_AT, seed=1)
    base = accuracy(va, [control(t) for t in va_t])
    print(f'train {len({p["game"] for p in tr})} games, val {len({p["game"] for p in va})}; baseline to beat: the control '
          f'rule on val at {VAL_AT[0]:.0%} missed, {VAL_AT[1]} false a minute: {base:.3f}')
    net = make_net(); opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=epochs * STEPS)
    w = csv.writer(open(LOG, 'w')); w.writerow(['epoch', 'train_loss', 'val_acc', 'control_val_acc', 'best'])
    best = -1.0
    for ep in range(1, epochs + 1):
        net.train(); tot = 0.0
        for _ in range(STEPS):
            X, Y = [], []
            for _b in range(BATCH):
                p = tr[rng.integers(len(tr))]; s = rng.integers(0, max(1, len(p['lab']) - CHUNK))
                t = degrade(p['touch'][s:s + CHUNK], rng.uniform(0, MISS_MAX), rng.uniform(0, FALSE_MAX), rng)
                y = p['lab'][s:s + CHUNK].astype(np.int64)
                if rng.random() < 0.5:                                         # the teams swapped
                    t, y = t[:, ::-1], np.where(y == UNK, UNK, 1 - y)
                pad = CHUNK - len(t)
                X.append(np.pad(t, ((0, pad), (0, 0)))); Y.append(np.pad(y, (0, pad), constant_values=UNK))
            lg = net(torch.from_numpy(np.stack(X).astype(np.float32)))
            loss = F.cross_entropy(lg.reshape(-1, 2), torch.from_numpy(np.stack(Y)).reshape(-1), ignore_index=UNK)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += float(loss)
        net.eval()
        acc = accuracy(va, [run(net, t) for t in va_t]); mark = acc > best
        if mark:
            best = acc; torch.save(net.state_dict(), CKPT)
        w.writerow([ep, round(tot / STEPS, 4), round(acc, 4), round(base, 4), int(mark)])
        print(f'epoch {ep}: train loss {tot / STEPS:.3f}, val accuracy {acc:.3f} (control rule {base:.3f})' + (' *' if mark else ''),
              flush=True)


def load_net():
    import torch
    net = make_net(); net.load_state_dict(torch.load(CKPT)); net.eval()
    return net


def test():
    te = load({'test'})
    net = load_net() if CKPT.exists() else None
    print(f'test: {len({p["game"] for p in te})} games; accuracy on live steps (the team SkillCorner says has the ball)')
    print(f'{"missed":>7s} {"false/min":>9s} {"last touch":>10s} {"control":>8s} {"model":>6s}')
    for k, (miss, fpm) in enumerate(GRID):
        ts = degraded(te, miss, fpm, seed=100 + k)
        row = [accuracy(te, [last_touch(t) for t in ts]), accuracy(te, [control(t) for t in ts])]
        row.append(accuracy(te, [run(net, t) for t in ts]) if net else float('nan'))
        print(f'{miss:7.0%} {fpm:9.1f} ' + ' '.join(f'{v:{w}.3f}' for v, w in zip(row, (10, 8, 6))))


def longest_gap():
    """The longest gap (s) between two touches within one possession in the training games: longer silences the model
    never learned."""
    out = 0.0
    for p in load({'train'}):
        t = np.flatnonzero(p['touch'].any(1))
        for x, y in zip(t, t[1:]):
            if p['lab'][x] != UNK and (p['lab'][x:y + 1] == p['lab'][x]).all():
                out = max(out, (y - x) / HZ)
    return out


def public():
    """BAR-ATH where our touches run: dead ball as predict.py decides it (stoppages), the model's team on every other
    frame. Our touches and the restarts (the team taking a set piece touches the ball) are the model's input, at 10 Hz.
    -> evaluation/results/possession_touch_model.csv"""
    import predict as pr
    labels, _ = pr.predict()
    data = json.load(open(pr.TOUCHES_JSON.format(game=pr.GAME)))
    f0, f1 = data['frames']
    touches = [(e['f'], 'HA'.index(e['team'])) for e in data['events'] if e['kind'] == 'touch' and e.get('team') in ('H', 'A')]
    touches += [(f, H if k == pr.HOME else A) for f, k in pr.stoppage_restarts(pr.STOPPAGES_JSON.format(game=pr.GAME))
                if f0 <= f <= f1]
    n10 = int((f1 - f0) * HZ / 25) + 1
    t = np.zeros((n10, 2), np.int8)
    for f, k in touches:
        t[int(round((f - f0) * HZ / 25)), k] = 1
    team = run(load_net(), t)
    longest = longest_gap()
    at = sorted(f for f, _ in touches); kept = 0
    for a, b in zip(at, at[1:]):                      # between touches; before the first and after the last, nothing to go on
        if (b - a) / 25 > longest:
            continue                                  # a longer silence than it ever learned: predict.py's last touch stands
        for f in range(a, b):
            if labels[f] in ('H', 'A'):
                labels[f] = 'HA'[team[min(n10 - 1, int(round((f - f0) * HZ / 25)))]]; kept += 1
    print(f'{len(touches)} touches and restarts over frames {f0}-{f1}; the model decides H/A on {kept / 25 / 60:.1f} min of '
          f'live play (touch gaps up to {longest:.1f} s, the longest within a SkillCorner possession), dead ball as predict.py')
    print('-> ' + os.path.relpath(pr.write_submission(os.path.join(pr.RESULTS_DIR, 'possession_touch_model.csv'), labels), ROOT))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'train', 'test', 'public'])
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    a = ap.parse_args()
    {'build': build, 'train': lambda: train(a.epochs), 'test': test, 'public': public}[a.cmd]()
