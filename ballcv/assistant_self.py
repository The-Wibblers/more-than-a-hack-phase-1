"""The assistant referees, by a detector that teaches itself: where along each touchline is the assistant?

An assistant is always on a touchline, so we only ask along the lines: assistant_net.py's windows, a person-sized box
every WIN_STEP_M of each line, resampled to 32 x 64. A small classifier (assistant_net.py's) scores every window; on
each line the best window is the assistant, as sure as its score.

Round 0 (seed): where two independent sources agree. The colour rules (assistant_rule.py: a lime shirt and lime
socks, glowing brighter than the grass) find an official on the line; AND he is where an assistant stands: level
with the offside line of the half he is in (OFFSIDE_M), from Metrica's tracking alone -- the second-last player of
the team defending that end, or the ball when it is nearer the goal line, and never past halfway. A yellow board, a
steward's bib, the fourth official at the bench do not follow that line. And he is in his own half: the two
assistants patrol opposite halves, each his own all through a half -- the half where most of the seed's officials on
his line stand -- so he is level with that end's line, not the other's. And he is the only official the rules see on
that line (none other NEG_M or more away; the fourth official's spot at halfway on the near line, which the rules
already set aside, apart): two yellow figures on one line are not labelled at all.

From round 1: model K trains on every label so far (from scratch, EPOCHS passes, the epoch with the lowest validation
loss kept), then looks at a fresh FRAMES_A_ROUND frames. On each line, its best window at least conf_min(r) sure (the
schedule of selftrain.py: no ground truth to set it by) is a label -- by the model alone, wherever on the line --
when every other window NEG_M or more away is under DETECTED: the model sees nobody else there it would call an
assistant (the referee by the line, the fourth official); and the window is in the half that line's assistant patrols
(the seed's, above) -- a steward or ball boy by the other corner is not him, however sure. Seed or model, a line
with two candidates is skipped.
Every labelled line also gives negatives: windows on the same line in the same frame at least NEG_M away (one
assistant to a line), NEG_RANDOM anywhere and NEG_HARD where the model (round 0: the colour rules) scores highest --
the fourth official, the benches, boards, players by the line. Windows nearer than NEG_M are left out.

Frames: kit_team's -- in a half, never the public five minutes, never still. Every VAL_EVERY-th labelled frame is for
validation only.

Checked every round, with no labels, on the held-out seed frames (EVAL: round 0's validation frames, both lines):
  found      the seed's assistants with the model's best window on that line within NEG_M, DETECTED and up
  seen       lines where the model finds an assistant at all (DETECTED and up)
  offside    of those, how many stand within OFFSIDE_M of the offside line of the end that line's assistant patrols
             -- an independent source the model never sees after round 0

  python ballcv/assistant_self.py run          every round in turn (you run this); finished rounds are skipped; the
                                               last model is copied to ballcv/models/assistants.pt
  python ballcv/assistant_self.py round K      one round
"""
import json, os, shutil, sys
import numpy as np
import torch
from tqdm import tqdm

import ownball as ob
import selftrain as st
import kit_team as kt
import assistant_net as an
import assistant_rule as ar

L, W = ob.L, ob.W
ROOT = ob.S + 'assistants'
MODEL = __import__('paths').MODELS + 'assistants.pt'
SEED_FRAMES = 2000
FRAMES_A_ROUND = 2000
ROUNDS = 10
EPOCHS = 20
OFFSIDE_M = 3.0                  # an assistant keeps level with the offside line: within this, he is
NEG_M = an.NEG_M
NEG_RANDOM, NEG_HARD = 6, 6      # negatives per labelled line
DETECTED = 0.5
VAL_EVERY = 5
DEV = st.DEV
M = kt.M
_t = np.load(ob.S + 'tactical.npz')
PM = _t['P'][..., :2] * [L, W]
BALL = _t['B'][:, :2] * [L, W]
STALE = ob.stale_mask(_t['P'][..., :2])
SIDES = ob.sides_by_half(_t['P'][..., :2])


def rdir(r):
    d = f'{ROOT}/r{r}'
    os.makedirs(d, exist_ok=True)
    return d


def frames_of(r):
    """Round r's frames: one from each equal slice of kit_team's open frames, at an offset of the round's own."""
    o = kt.open_frames(); n = SEED_FRAMES if r == 0 else FRAMES_A_ROUND
    rng = np.random.default_rng(900 + r)
    edges = np.linspace(0, len(o), n + 1).astype(int)
    return sorted(int(o[rng.integers(a, b)]) for a, b in zip(edges, edges[1:]) if b > a)


# ---------------------------------------------------------------- where an assistant stands
def offside_line(f, end):
    """The x the assistant watching end ('left', 'right') keeps level with on frame f, or None: the second-last player
    of the team defending that end, or the ball when it is nearer the goal line; never past halfway."""
    team = next((t for t, sd in SIDES(f).items() if sd == end), None)
    slots = range(0, 24) if team == 'H' else range(24, PM.shape[1])
    xs = [PM[f, s, 0] for s in slots if np.isfinite(PM[f, s, 0]) and not STALE[f, s]]
    if len(xs) < 2:
        return None
    if end == 'left':
        line = sorted(xs)[1]
        if np.isfinite(BALL[f, 0]):
            line = min(line, BALL[f, 0])
        return min(line, L / 2)
    line = sorted(xs)[-2]
    if np.isfinite(BALL[f, 0]):
        line = max(line, BALL[f, 0])
    return max(line, L / 2)


def level(f, x, ends=('left', 'right')):
    """Is an assistant at x level with the offside line of one of ends? Each line stays in its own half, so with both
    ends only the half he is in can be near him (at halfway, either)."""
    return any(o is not None and abs(x - o) <= OFFSIDE_M for o in (offside_line(f, e) for e in ends))


def half_of(f):
    return next(i for i, (a, b) in enumerate(st.HALVES) if a <= f <= b)


def patrols(labels):
    """{(half, line): end}: the half each line's assistant patrols in each half of the match -- where most of
    labels ({frame: {line: x}}) on that line stand."""
    xs = {}
    for f, v in labels.items():
        for name, x in v.items():
            xs.setdefault((half_of(int(f)), name), []).append(x)
    return {k: 'left' if np.median(v) < L / 2 else 'right' for k, v in xs.items()}


_PATROL = {}


def patrol(f, name):
    """The end line name's assistant patrols on frame f: the seed's (round 0's labels)."""
    if not _PATROL:
        _PATROL.update(patrols(json.load(open(f'{rdir(0)}/labels.json'))))
    return _PATROL[half_of(f), name]


def in_half(x, end):
    """Is x in end's half (or within OFFSIDE_M of halfway, where he waits while play is in the other)?"""
    return x <= L / 2 + OFFSIDE_M if end == 'left' else x >= L / 2 - OFFSIDE_M


# ---------------------------------------------------------------- the model
def load_net(r):
    """Model of round r, or the model at path r."""
    net = an.build_net(torch)
    net.load_state_dict(torch.load(r if isinstance(r, str) else f'{rdir(r)}/model.pt', map_location='cpu'))
    return net.to(DEV).eval()


@torch.no_grad()
def scores(net, wins):
    if not len(wins):
        return np.zeros(0)
    return torch.sigmoid(net(an.to_tensor(torch, np.stack(wins)).to(DEV))[:, 0]).cpu().numpy()


def lines_of(f, img):
    """[(name, xs, windows)] for both touchlines of frame f."""
    return [(name, *ar.windows_on_line(img, M[f], y, o)) for name, y, o in an.LINES]


# ---------------------------------------------------------------- labels
def negatives(xs, score, x, rng):
    """Window indices on a line to teach as not an assistant: NEG_M or more from x, NEG_RANDOM anywhere, NEG_HARD
    where score is highest."""
    far = np.where(np.abs(xs - x) >= NEG_M)[0]
    if not len(far):
        return []
    hard = far[np.argsort(-score[far])[:NEG_HARD]]
    rest = np.setdiff1d(far, hard)
    return list(hard) + list(rng.choice(rest, min(NEG_RANDOM, len(rest)), replace=False))


def label_round(r, net):
    """Round r's labels -> rK/labels.json ({frame: {line: x}}), rK/windows.npz (X, Y, frame)."""
    bar = st.conf_min(r) if r > 0 else None
    frames = frames_of(r)
    rng = np.random.default_rng(2000 + r)
    found = []                                                          # (frame, line, x, window, negatives)
    n = dict(labels=0, far=0, near=0, two=0, half=0)
    for f, img in st.read(frames, f'round {r} labelling', lambda: n):
        for name, xs, wins in lines_of(f, img):
            if not len(wins):
                continue
            if r == 0:
                kits = [ar.kit(w) for w in wins]
                a = ar.assistant(xs, kits, name == 'near')
                if a is None or not level(f, a[0]):
                    continue
                x, score = a[0], np.array(kits)
                fourth = (name == 'near') & (np.abs(xs - L / 2) < ar.FOURTH_M)
                if any(ar.official(k) for k, fx in zip(kits, ~fourth & (np.abs(xs - x) >= NEG_M)) if fx):
                    n['two'] += 1; continue                                    # another official on the line
            else:
                score = scores(net, wins)
                i = int(np.argmax(score))
                if score[i] < bar:
                    continue
                x = float(xs[i])
                if np.any(score[np.abs(xs - x) >= NEG_M] >= DETECTED):
                    n['two'] += 1; continue                                    # the model sees another on the line
            i = int(np.argmin(np.abs(xs - x)))
            found.append((f, name, float(x), wins[i], [wins[j] for j in negatives(xs, score, x, rng)]))
    # in the half his line's assistant patrols: the seed's own officials say which (round 0: level with that end)
    if r == 0:
        seed = {}
        for f, name, x, _w, _n in found:
            seed.setdefault(str(f), {})[name] = x
        side = patrols(seed)
        ok = lambda f, name, x: level(f, x, (side[half_of(f), name],))
    else:
        ok = lambda f, name, x: in_half(x, patrol(f, name))
    out, X, Y, FR = {}, [], [], []
    for f, name, x, win, negs in found:
        if not ok(f, name, x):
            n['half'] += 1; continue
        X.append(win); Y.append(1); FR.append(f)
        for w in negs:
            X.append(w); Y.append(0); FR.append(f)
        out.setdefault(str(f), {})[name] = round(x, 2)
        n['labels'] += 1; n[name] += 1
    json.dump(out, open(f'{rdir(r)}/labels.json', 'w'))
    if not X:
        tqdm.write(f'round {r}: no labels'); return 0
    np.savez_compressed(f'{rdir(r)}/windows.npz', X=np.stack(X), Y=np.array(Y, np.uint8), frame=np.array(FR))
    agree = sum(level(int(f), x, (patrol(int(f), name),)) for f, v in out.items() for name, x in v.items())
    tqdm.write(f'round {r}: {n["labels"]} assistants labelled on {len(out)} of {len(frames)} frames (far {n["far"]}, '
               f'near {n["near"]}; {n["two"]} lines skipped, two candidates on them; {n["half"]} not in his half)' + ('' if r == 0 else f'; {agree / max(1, n["labels"]):.1%} level with the offside '
                                                          f'line (not used to label); bar {bar}'))
    if not os.path.exists(f'{ROOT}/rounds.csv'):
        open(f'{ROOT}/rounds.csv', 'w').write('round,labels,far,near,offside_level,bar\n')
    with open(f'{ROOT}/rounds.csv', 'a') as fh:
        fh.write(f'{r},{n["labels"]},{n["far"]},{n["near"]},{agree / max(1, n["labels"]):.4f},{bar or ""}\n')
    return n['labels']


# ---------------------------------------------------------------- training
def all_windows(upto):
    parts = [np.load(f'{rdir(q)}/windows.npz') for q in range(upto + 1) if os.path.exists(f'{rdir(q)}/windows.npz')]
    return (np.concatenate([p['X'] for p in parts]), np.concatenate([p['Y'] for p in parts]).astype(np.float32),
            np.concatenate([p['frame'] for p in parts]))


def augment(x):
    """Mirror (an assistant faces either way along his line) and a little light."""
    flip = torch.rand(len(x), device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    gain = 1 + 0.15 * (torch.rand(len(x), 1, 1, 1, device=x.device) * 2 - 1)
    bias = 0.05 * (torch.rand(len(x), 1, 1, 1, device=x.device) * 2 - 1)
    return (x * gain + bias).clamp(0, 1)


def loss_on(net, X, Y, lossf, batch=512):
    net.eval(); tot = 0.0
    with torch.no_grad():
        for k in range(0, len(X), batch):
            x = an.to_tensor(torch, X[k:k + batch]).to(DEV); y = torch.as_tensor(Y[k:k + batch]).to(DEV)
            tot += float(lossf(net(x)[:, 0], y)) * len(x)
    return tot / max(1, len(X))


def train(r, batch=128, lr=3e-3):
    X, Y, f = all_windows(r - 1)
    frames = np.unique(f)
    val = np.isin(f, frames[::VAL_EVERY])
    pos_w = torch.tensor(float((Y[~val] == 0).sum() / max(1, (Y[~val] == 1).sum())), device=DEV)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    tqdm.write(f'round {r}: training on {int((~val).sum())} windows ({int(Y[~val].sum())} assistants) from '
               f'{len(frames) - len(frames[::VAL_EVERY])} frames, validating on {int(val.sum())}; {EPOCHS} epochs, from scratch')
    if r > 1:
        tqdm.write(f'round {r}: baseline to beat, model {r - 1} on this validation: loss '
                   f'{loss_on(load_net(r - 1), X[val], Y[val], lossf):.3f}')
    Xt, Yt = X[~val], Y[~val]
    net = an.build_net(torch).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=EPOCHS * -(-len(Xt) // batch))
    log = open(f'{rdir(r)}/log.csv', 'w'); log.write('epoch,train_loss,val_loss\n')
    best = np.inf
    for ep in range(1, EPOCHS + 1):
        net.train()
        perm, tot, seen = np.random.permutation(len(Xt)), 0.0, 0
        bar = tqdm(range(0, len(Xt), batch), desc=f'round {r} epoch {ep}/{EPOCHS}', unit='batch', dynamic_ncols=True, leave=False)
        for k in bar:
            idx = perm[k:k + batch]
            x = augment(an.to_tensor(torch, Xt[idx]).to(DEV)); y = torch.as_tensor(Yt[idx]).to(DEV)
            loss = lossf(net(x)[:, 0], y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss.detach()) * len(idx); seen += len(idx)
            bar.set_postfix(loss=f'{float(loss.detach()):.3f}', mean=f'{tot / seen:.3f}')
        vl = loss_on(net, X[val], Y[val], lossf)
        log.write(f'{ep},{tot / seen:.4f},{vl:.4f}\n'); log.flush()
        mark = ''
        if vl < best:
            best = vl; torch.save(net.state_dict(), f'{rdir(r)}/model.pt'); mark = '  saved'
        tqdm.write(f'round {r} epoch {ep:2d}  train {tot / seen:.3f}  val {vl:.3f}{mark}')
    json.dump(dict(epochs=EPOCHS, best_val_loss=round(best, 4)), open(f'{rdir(r)}/trained.json', 'w'))
    return load_net(r)


# ---------------------------------------------------------------- label-free checks
def evaluate(r, net):
    """found / seen / offside on round 0's validation frames (see the module docstring)."""
    lab = json.load(open(f'{rdir(0)}/labels.json'))
    frames = sorted(int(f) for f in lab)[::VAL_EVERY]                    # never trained on: validation frames
    found = total = seen = lines = level_n = 0
    for f, img in st.read(frames, f'model {r} check'):
        for name, xs, wins in lines_of(f, img):
            if not len(wins):
                continue
            s = scores(net, wins); i = int(np.argmax(s)); lines += 1
            if s[i] >= DETECTED:
                seen += 1; level_n += level(f, float(xs[i]), (patrol(f, name),))
            if name in lab[str(f)]:
                total += 1
                found += s[i] >= DETECTED and abs(xs[i] - lab[str(f)][name]) < NEG_M
    tqdm.write(f'model {r} on {len(frames)} held-out seed frames: found {found}/{total} seed assistants '
               f'({found / max(1, total):.1%}); an assistant seen on {seen}/{lines} lines ({seen / max(1, lines):.1%}), '
               f'{level_n / max(1, seen):.1%} of them level with the offside line')
    with open(f'{ROOT}/checks.csv', 'a') as fh:
        fh.write(f'{r},{found / max(1, total):.4f},{seen / max(1, lines):.4f},{level_n / max(1, seen):.4f}\n')


# ---------------------------------------------------------------- commands
def cmd_round(r):
    net = None
    if r > 0:
        net = load_net(r) if os.path.exists(f'{rdir(r)}/trained.json') else train(r)
        evaluate(r, net)
    return label_round(r, net)


def cmd_run():
    for r in range(ROUNDS):
        if os.path.exists(f'{rdir(r)}/windows.npz'):
            continue
        if cmd_round(r) == 0:
            tqdm.write(f'round {r}: no labels -- stopping'); break
    last = max(r for r in range(ROUNDS) if os.path.exists(f'{rdir(r)}/model.pt'))
    shutil.copy(f'{rdir(last)}/model.pt', MODEL)
    tqdm.write(f'model {last} -> {MODEL}')


if __name__ == '__main__':
    os.makedirs(ROOT, exist_ok=True)
    {'run': cmd_run, 'round': lambda: cmd_round(int(sys.argv[2]))}[sys.argv[1]]()
