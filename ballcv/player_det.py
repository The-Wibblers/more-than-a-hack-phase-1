"""A player detector that teaches itself: every player's feet in a frame, and whose kit he wears -- no hand labels.

Metrica's tracking misses players, and mostly where it matters: all eleven of a team are tracked on only ~60% of
live frames, and the ones missing are the hidden ones -- a Barcelona player marked from the camera side at the far
touchline (11340-11430) has no track at all, so every touch there goes to the Athletic player in front of him. This
finds players in the picture itself: heatmaps of feet and of heads, a pair per kit (kit_team.CLASSES: each team's
outfield and goalkeeper kit), like heatmap.py's ball heatmap.

Feet stand on the grass, so a feet peak is a ground point through the homography, exactly. But a player hidden behind
another shows his head and shoulders, not his feet. So a head peak with no feet peak under it is a player too: his
ground point is where the ray through his head is HEAD_M up (the camera's centre known, ownball.CT) -- rougher (a
player 0.15 m taller or shorter moves it ~0.6 m at the far touchline), enough to know he is in a duel. Every label
teaches both points where they truly are, hidden or not: a player found by his head teaches where his feet were.

Round 0 (seed): where two independent sources agree -- Metrica's player alone in the picture (kit_team: no other
body's box overlaps his), on a mature track, on the pitch, AND the kit model (kit_team.py, from the picture) reading
the kit his slot says (a keeper slot: as kit_team's seed, inside a penalty area). Every other body Metrica has in the
frame is "unknown" (IGNORE: not taught as a player, not as grass either).

From round 1: model K trains on every label so far (from scratch, EPOCHS passes, the epoch with the lowest validation
loss kept), then looks at a fresh FRAMES_A_ROUND frames over the whole pitch. A peak at least conf_min(r) sure (the
schedule of selftrain.py: no ground truth to set it by) is a label, wherever it is -- with or without a Metrica track
there: the players Metrica lost are what this is for. In crowds the model is rarely that sure, and crowds are where it
is needed, so a Metrica body is a label too where three sources agree as the seed's two do: Metrica has him there, the
model has a peak (UNSURE and up) on his feet, and the kit model reads his slot's kit (a keeper slot: inside a penalty
area). And close pairs, where the model often has one peak for two players and the kit crop holds both: a crowded
Metrica body is a label where this round's own model saw him alone in his team's kit within PAIR_S before and within
PAIR_S after (look_around: it looks at the PAIR_LOOK nearest frames on each side where he stands alone), and Metrica's
track of him runs on in between without a jump (pair_sure). Nothing comes from earlier models' files: from round 0
on, the rounds are the code alone. Any other weaker peak and Metrica body with no label on it is unknown. So a player the model does not yet see
is never taught as grass.

Training: per labelled frame CROPS_A_FRAME crops of CROP pixels: on a crowd (players whose boxes overlap) when there is
one, around labels (jittered), and anywhere on the pitch;
Gaussian targets at each label's feet and head in its kit's channels, the size of the body there (SIGMA_BODY); focal
loss (heatmap.focal), unknowns left out, each kit weighted back to the seed's mix of kits (kit_weights). Frames: never the public five minutes, never still, always in a half; every
VAL_EVERY-th labelled frame is for validation only.

Checked every round with no labels, on EVAL_FRAMES held-out seed frames (full frames):
  found      the seed players (Metrica + kit agree, held out) with a detection on their feet (HIT_BODY of a body) in
             the same team
  extra      detections nowhere near a Metrica body (MISS_BODY): players Metrica lost -- or false alarms
  over 11    frames where a team comes out with more than eleven (or a keeper kit more than once): false alarms for sure

  python ballcv/player_det.py run        every round in turn (you run this); finished rounds are skipped; the last
                                         model is copied to ballcv/models/players.pt
  python ballcv/player_det.py round K    one round
  python ballcv/player_det.py detect A B ballcv/models/players.pt over frames A..B -> players/dets/ (player_track.py)
"""
import json, os, shutil, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import heatmap
import ownball as ob
import selftrain as st
import kit_team as kt
import player_track as pt

L, W = ob.L, ob.W
ROOT = ob.S + 'players'
CLASSES, TEAM_OF = kt.CLASSES, kt.TEAM_OF
CROP = 96                                # training crop, pixels (a multiple of 16)
STRIDE = 2                               # heatmap pixels per image pixel
CROPS_A_FRAME = 3                        # two around labels, one anywhere on the pitch
SIGMA_BODY = 0.06                        # target Gaussian, in body heights (a far player ~1 heatmap pixel)
SEED_FRAMES = 2000
FRAMES_A_ROUND = 2000
ROUNDS = 16
EPOCHS = 20
UNSURE = 0.3                             # a peak this strong may be a body: unknown, not grass
HIT_BODY = 0.25                          # a detection this close to a player's feet (in his body heights) is on him
MISS_BODY = 0.6                          # ... and one farther than this from every Metrica body is "extra"
OFF_PITCH_M = 1.0                        # labels stay on the pitch (subs, coaches, ball boys are not players)
VAL_EVERY = 5
EVAL_FRAMES = 200
DETECTED = 0.5                           # a detection: a peak this sure (the checks)
HEAD_M = 1.7                             # the middle of a player's head above the grass
DEV = st.DEV
M = kt.M
MODEL = ob.PLAYER_MODEL                  # the last round's model, copied here by run


def conf_min(r):
    return st.conf_min(r)


def rdir(r):
    d = f'{ROOT}/r{r}'
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------- geometry
def body_at(f, G):
    """(feet, top) pixels of a player standing at ground point G (m) in frame f."""
    return ob.column(M[f], np.asarray(G, float))


def point_at(f, G, h):
    """pixel of the point h m above ground point G (m), as ownball.column."""
    Q = ob.CT[:2] + (np.asarray(G, float) - ob.CT[:2]) * ob.CT[2] / (ob.CT[2] - h)
    q = M[f] @ [Q[0] / L, Q[1] / W, 1.0]
    return q[:2] / q[2]


def ground_under_head(f, uv):
    """ground point (m) of a player whose head (HEAD_M up) is at pixel uv: point_at inverted."""
    Q = ground(f, uv)
    return ob.CT[:2] + (Q - ob.CT[:2]) * (ob.CT[2] - HEAD_M) / ob.CT[2]


def ground(f, uv):
    q = np.linalg.inv(M[f]) @ [uv[0], uv[1], 1.0]
    return q[:2] / q[2] * [L, W]


def on_pitch(G):
    return -OFF_PITCH_M <= G[0] <= L + OFF_PITCH_M and -OFF_PITCH_M <= G[1] <= W + OFF_PITCH_M


def box_of(feet, top):
    h = np.linalg.norm(top - feet); cx = (feet[0] + top[0]) / 2
    return (cx - kt.BODY_W * h / 2, min(feet[1], top[1]), cx + kt.BODY_W * h / 2, max(feet[1], top[1]) + 0.1 * h)


# ---------------------------------------------------------------- the model
class PlayerFCN(nn.Module):
    """heatmap.BallFCN's encoder-decoder one stage deeper (a body, not a ball, has to fit in what a pixel sees); heatmaps
    at 1/STRIDE resolution: feet, one channel per kit, then heads, one channel per kit."""

    def __init__(self):
        super().__init__()

        def cbr(i, o, s=1):
            return nn.Sequential(nn.Conv2d(i, o, 3, stride=s, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.e1 = nn.Sequential(cbr(3, 16), cbr(16, 16))                 # 1
        self.e2 = nn.Sequential(cbr(16, 32, 2), cbr(32, 32))             # 1/2
        self.e3 = nn.Sequential(cbr(32, 64, 2), cbr(64, 64))             # 1/4
        self.e4 = nn.Sequential(cbr(64, 96, 2), cbr(96, 96))             # 1/8
        self.e5 = nn.Sequential(cbr(96, 128, 2), cbr(128, 128))          # 1/16
        self.d4 = cbr(128 + 96, 96)
        self.d3 = cbr(96 + 64, 64)
        self.d2 = cbr(64 + 32, 32)
        self.head = nn.Conv2d(32, 2 * len(CLASSES), 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3); e5 = self.e5(e4)
        d4 = self.d4(torch.cat([F.interpolate(e5, size=e4.shape[-2:]), e4], 1))
        d3 = self.d3(torch.cat([F.interpolate(d4, size=e3.shape[-2:]), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, size=e2.shape[-2:]), e2], 1))
        return self.head(d2)                                            # (n, 2 * kits, H/2, W/2) logits


def load_net(r):
    """Model of round r, or the model at path r."""
    path = r if isinstance(r, str) else f'{rdir(r)}/model.pt'
    net = PlayerFCN(); net.load_state_dict(torch.load(path, map_location='cpu'))
    return net.to(DEV).eval()


def part_peaks(p, x0, y0):
    """[(u, v, kit, p)] local maxima (3x3) of one part's heatmaps (kits, H, W), the strongest kit at each pixel."""
    best, kit = p.max(0)
    keep = (best == F.max_pool2d(best[None, None], 3, stride=1, padding=1)[0, 0]) & (best >= UNSURE)
    best, kit, keep = best.cpu().numpy(), kit.cpu().numpy(), keep.cpu().numpy()
    return [(x0 + (xx + 0.5) * STRIDE, y0 + (yy + 0.5) * STRIDE, int(kit[yy, xx]), float(best[yy, xx]))
            for yy, xx in zip(*np.nonzero(keep))]


@torch.no_grad()
def detect(net, img, f):
    """[(u, v, kit, p)] players over the pitch area, strongest first; (u, v) his feet. A feet peak and the head peak
    where his head would be (within HIT_BODY of a body) are one player, as sure as the surer of the two, in its kit;
    a head peak with no feet under it is a player whose feet are hidden (ground_under_head)."""
    x0, y0, x1, y1 = (int(a) for a in heatmap.pitch_box(M[f]))
    crop = img[y0:y1, x0:x1]
    h, w = crop.shape[:2]; ph, pw = (-h) % 16, (-w) % 16
    if ph or pw:
        crop = np.pad(crop, ((0, ph), (0, pw), (0, 0)), mode='edge')
    p = torch.sigmoid(net(torch.tensor(crop).permute(2, 0, 1)[None].float().div(255).to(DEV))[0])
    k = len(CLASSES)
    feet, heads = part_peaks(p[:k], x0, y0), part_peaks(p[k:], x0, y0)
    used, out = set(), []
    for u, v, c, pf in sorted(feet, key=lambda d: -d[3]):
        G = ground(f, (u, v)); ft, top = body_at(f, G); hd = point_at(f, G, HEAD_M)
        near = [(np.hypot(hu - hd[0], hv - hd[1]), j) for j, (hu, hv, _c, _p) in enumerate(heads) if j not in used]
        near = [j for d, j in sorted(near) if d <= HIT_BODY * np.linalg.norm(top - ft)][:1]
        if near:
            j = near[0]; used.add(j)
            if heads[j][3] > pf:
                c, pf = heads[j][2], heads[j][3]
        out.append((u, v, c, pf))
    for j, (u, v, c, ph_) in enumerate(heads):
        if j not in used:
            ft, _top = body_at(f, ground_under_head(f, (u, v)))
            out.append((float(ft[0]), float(ft[1]), c, ph_))
    return sorted(out, key=lambda d: -d[3])


# ---------------------------------------------------------------- labels
def metrica_bodies(f):
    """[(slot, feet, top)] of Metrica's live players at f with their feet in the image."""
    return [(s, feet, top) for s, feet, top, _b in kt.players(f)]


def seed_frame(f, img, kitnet):
    """Round 0 on one frame -> labels [(u, v, kit)], unknown boxes [(x0, y0, x1, y1)]."""
    people = kt.players(f)
    labels, unknown, cand = [], [], []
    for i, (s, feet, top, _b) in enumerate(people):
        c = kt.seed_class(f, s, i, people)
        if c is not None and on_pitch(P_m(f, s)):
            cand.append((s, feet, top, c))
        else:
            unknown.append(box_of(feet, top))
    if cand:
        pr = kt.probs(kitnet, np.stack([kt.crop(img, feet, top) for _s, feet, top, _c in cand]))
        for (s, feet, top, c), p in zip(cand, pr):
            if int(p.argmax()) == c:                                    # Metrica and the picture agree
                labels.append((float(feet[0]), float(feet[1]), c))
            else:
                unknown.append(box_of(feet, top))
    return labels, unknown


def P_m(f, s):
    return kt.P[f, s] * [L, W]


def slot_class(f, s):
    """The kit Metrica's slot s says at f: its team's; a keeper slot only inside a penalty area (that box's keeper)."""
    if s in kt.KEEPERS:
        t = ob.box_team(P_m(f, s), kt.SIDES(f))
        return None if t is None else 2 + 'HA'.index(t)
    return int(kt.SLOT_TEAM[s])


PAIR_S = 2.0                             # a crowded player is sure if seen alone this close before and after (player_track.COAST_S)
PAIR_LOOK = 2                            # nearest frames on each side where he stands alone, that the model looks at
_SEEN = {}                               # this round's model's players on the frames it looked at: {frame: (G (n, 2), team)}
_ALONE = {}


def alone_slots(g):
    """Metrica's slots standing alone in frame g's picture (no other body's box overlaps theirs)."""
    if g not in _ALONE:
        people = kt.players(g)
        _ALONE[g] = {sl for i, (sl, *_r) in enumerate(people) if kt.alone(i, people)}
    return _ALONE[g]


def track_runs_on(f, s):
    """Metrica's track of slot s is live from PAIR_S before f to PAIR_S after, never stepping further than a player
    runs (player_track's rule): the same man throughout."""
    k = int(PAIR_S * 25)
    if f - k < 0 or f + k >= len(kt.LIVE) or not kt.LIVE[f - k:f + k + 1, s].all():
        return False
    step = np.linalg.norm(np.diff(kt.P[f - k:f + k + 1, s] * [L, W], axis=0), axis=1)
    return not np.any(step > pt.POS_M + pt.RUN_MS / 25)


def look_frames(f, s):
    """The PAIR_LOOK nearest frames before f, and after, within PAIR_S, where slot s stands alone."""
    k = int(PAIR_S * 25)
    before = [g for g in range(f - 1, f - k - 1, -1) if s in alone_slots(g)][:PAIR_LOOK]
    after = [g for g in range(f + 1, f + k + 1) if s in alone_slots(g)][:PAIR_LOOK]
    return before, after


def crowded(f):
    """[(slot, kit)] of the Metrica players in a crowd at f who could be pair labels: on the pitch, a kit from their
    slot, their track running on."""
    people = kt.players(f)
    return [(s, slot_class(f, s)) for i, (s, *_r) in enumerate(people)
            if not kt.alone(i, people) and slot_class(f, s) is not None and on_pitch(P_m(f, s)) and track_runs_on(f, s)]


def look_around(frames, net):
    """This round's model on the frames where the crowded players of its frames stand alone (look_frames) -> _SEEN."""
    _SEEN.clear(); _ALONE.clear()
    need = set()
    for f in frames:
        for s, _c in crowded(f):
            b, a = look_frames(f, s)
            if b and a:
                need.update(b + a)
    for g, img in st.read(sorted(need), 'looking around the crowds'):
        det = [d for d in detect(net, img, g) if d[3] >= DETECTED]
        G = np.array([ground(g, d[:2]) for d in det]).reshape(-1, 2)
        _SEEN[g] = (G, np.array([d[2] % 2 for d in det], int))          # 0 home, 1 away (as kt.SLOT_TEAM)
    return len(need)


def seen_alone(g, s):
    """At frame g, where slot s stands alone, this round's model has a player of his team where Metrica has him."""
    G, team = _SEEN.get(g, (np.zeros((0, 2)), np.zeros(0)))
    return bool(len(G)) and bool(np.any((np.linalg.norm(G - P_m(g, s), axis=1) <= pt.POS_M) & (team == kt.SLOT_TEAM[s])))


def pair_sure(f, s):
    """A player in a crowd at f is sure where Metrica has him when two sources agree: this round's model saw him alone,
    in his team's kit, within PAIR_S before and within PAIR_S after; and Metrica's track of him runs on in between."""
    if not track_runs_on(f, s):
        return False
    b, a = look_frames(f, s)
    return any(seen_alone(g, s) for g in b) and any(seen_alone(g, s) for g in a)


def model_frame(f, net, img, bar, kitnet):
    """Round r >= 1 on one frame -> labels, unknown boxes, how many labels the three sources gave. Labels: peaks at
    least bar sure on the pitch; and Metrica bodies with a weaker peak (UNSURE and up) on their feet whose crop the kit
    model reads as the slot's kit. Unknown: every other peak, and every Metrica body with no label on it."""
    labels, unknown, weak = [], [], []
    for u, v, c, p in detect(net, img, f):
        G = ground(f, (u, v))
        feet, top = body_at(f, G)
        if p >= bar and on_pitch(G):
            labels.append((u, v, c))
        else:
            unknown.append(box_of(feet, top))
            if on_pitch(G):
                weak.append((u, v, len(unknown) - 1))
    agreed, dropped, pairs = [], set(), 0
    for s, feet, top in metrica_bodies(f):
        h = np.linalg.norm(top - feet)
        if any(np.hypot(u - feet[0], v - feet[1]) <= HIT_BODY * h for u, v, _c in labels):
            continue
        c = slot_class(f, s)
        on = [i for u, v, i in weak if np.hypot(u - feet[0], v - feet[1]) <= HIT_BODY * h]
        if c is not None and on and on_pitch(P_m(f, s)):
            agreed.append((feet, top, c, on))
        elif c is not None and on_pitch(P_m(f, s)) and pair_sure(f, s):        # seen alone before and after
            labels.append((float(feet[0]), float(feet[1]), c)); dropped.update(on); pairs += 1
        else:
            unknown.append(box_of(feet, top))
    n = 0
    if agreed:
        pr = kt.probs(kitnet, np.stack([kt.crop(img, feet, top) for feet, top, _c, _on in agreed]))
        for (feet, top, c, on), p in zip(agreed, pr):
            if int(p.argmax()) == c:                                    # Metrica, the model and the kit agree
                labels.append((float(feet[0]), float(feet[1]), c)); dropped.update(on); n += 1
            else:
                unknown.append(box_of(feet, top))
    return labels, [bx for i, bx in enumerate(unknown) if i not in dropped], n, pairs


def crowd_feet(f):
    """Feet of Metrica's bodies whose boxes overlap another's: where the crowds are."""
    people = kt.players(f)
    return [(float(feet[0]), float(feet[1])) for i, (_s, feet, _t, _b) in enumerate(people) if not kt.alone(i, people)]


def crops_of(f, img, labels, unknown, rng, crowd=()):
    """CROPS_A_FRAME (crop, target, unknown mask): on a crowd if there is one, around labels, anywhere on the pitch."""
    x0, y0, x1, y1 = heatmap.pitch_box(M[f])
    centres = [crowd[rng.integers(len(crowd))]] if crowd else []
    take = min(CROPS_A_FRAME - 1 - len(centres), len(labels))
    centres += [labels[i][:2] for i in rng.choice(len(labels), take, replace=False)] if take > 0 else []
    centres = [(u + rng.uniform(-CROP / 3, CROP / 3), v + rng.uniform(-CROP / 3, CROP / 3)) for u, v in centres]
    while len(centres) < CROPS_A_FRAME:
        centres.append((rng.uniform(x0, x1), rng.uniform(y0, y1)))
    hs = CROP // STRIDE
    yy, xx = np.mgrid[0:hs, 0:hs]
    out = []
    for cu, cv in centres:
        a = int(np.clip(cu - CROP / 2, 0, st.FW - CROP)); b = int(np.clip(cv - CROP / 2, 0, st.FH - CROP))
        X = img[b:b + CROP, a:a + CROP]
        T = np.zeros((2 * len(CLASSES), hs, hs), np.float32)
        for u, v, c in labels:
            G = ground(f, (u, v)); feet, top = body_at(f, G)
            sig = max(1.0, SIGMA_BODY * np.linalg.norm(top - feet) / STRIDE)
            for ch, (pu, pv) in ((c, (u, v)), (len(CLASSES) + c, point_at(f, G, HEAD_M))):   # his feet, his head
                gu, gv = (pu - a) / STRIDE, (pv - b) / STRIDE
                if not (-4 <= gu < hs + 4 and -4 <= gv < hs + 4):
                    continue
                cx, cy = int(np.floor(gu)), int(np.floor(gv))
                T[ch] = np.maximum(T[ch], np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig ** 2)))
        IGN = np.zeros((hs, hs), bool)
        for bx0, by0, bx1, by1 in unknown:
            IGN[max(0, int((by0 - b) / STRIDE)):max(0, int(np.ceil((by1 - b) / STRIDE))),
                max(0, int((bx0 - a) / STRIDE)):max(0, int(np.ceil((bx1 - a) / STRIDE)))] = True
        IGN &= T.max(0) < 0.05                                          # a label's own peak is always taught
        out.append((X, (T * 255).astype(np.uint8), IGN))
    return out


def label_round(r, net):
    """Round r's labels and crops -> rK/labels.json (per frame: labels, unknown boxes), rK/crops.npz."""
    kitnet = kt.load_net(kt.MODEL)
    bar = conf_min(r) if r > 0 else None
    frames = frames_of(r)
    rng = np.random.default_rng(1000 + r)
    out, X, T, IGN, FR = {}, [], [], [], []
    n_lab = [0, 0, 0, 0]                 # labels; with no Metrica body there; from the three sources; seen alone around a crowd
    status = lambda: dict(labels=n_lab[0], lost=n_lab[1], crowd=n_lab[2], pairs=n_lab[3])
    if r > 0:
        n_look = look_around(frames, net)
        tqdm.write(f'round {r}: the model looked at {n_look} frames where crowded players stand alone')
    for f, img in st.read(frames, f'round {r} labelling', status):
        labels, unknown, agreed, pairs = (*seed_frame(f, img, kitnet), 0, 0) if r == 0 else model_frame(f, net, img, bar, kitnet)
        bodies = metrica_bodies(f)
        lost = sum(not any(np.hypot(u - feet[0], v - feet[1]) <= MISS_BODY * np.linalg.norm(top - feet) for _s, feet, top in bodies)
                   for u, v, _c in labels)
        n_lab[0] += len(labels); n_lab[1] += lost; n_lab[2] += agreed; n_lab[3] += pairs
        out[str(f)] = dict(labels=[(round(u, 1), round(v, 1), c) for u, v, c in labels],
                           unknown=[tuple(round(float(z), 1) for z in bx) for bx in unknown])
        if labels:
            for x, t, ig in crops_of(f, img, labels, unknown, rng, crowd_feet(f) if r > 0 else ()):
                X.append(x); T.append(t); IGN.append(ig); FR.append(f)
    json.dump(out, open(f'{rdir(r)}/labels.json', 'w'))
    if not X:
        tqdm.write(f'round {r}: no labels'); return 0
    np.savez_compressed(f'{rdir(r)}/crops.npz', X=np.stack(X), T=np.stack(T), IGN=np.stack(IGN), f=np.array(FR))
    kits = np.bincount([c for v in out.values() for *_x, c in v['labels']], minlength=len(CLASSES))
    tqdm.write(f'round {r}: {n_lab[0]} players labelled on {sum(bool(v["labels"]) for v in out.values())} of '
               f'{len(frames)} frames (' + ', '.join(f'{CLASSES[c]} {kits[c]}' for c in range(len(CLASSES))) +
               f'); {n_lab[1]} where Metrica has no body' + ('' if r == 0 else f'; {n_lab[2]} from Metrica + model + kit '
                                                                  f'agreeing; {n_lab[3]} in a crowd, seen alone before and after; bar {bar}'))
    if not os.path.exists(f'{ROOT}/rounds.csv'):
        open(f'{ROOT}/rounds.csv', 'w').write('round,labels,no_metrica_body,bar\n')
    with open(f'{ROOT}/rounds.csv', 'a') as fh:
        fh.write(f'{r},{n_lab[0]},{n_lab[1]},{bar or ""}\n')
    return n_lab[0]


def frames_of(r):
    """Round r's frames: kit_team's (one per equal slice of the open frames, new every round), its own seed."""
    o = kt.open_frames(); n = SEED_FRAMES if r == 0 else FRAMES_A_ROUND
    rng = np.random.default_rng(500 + r)
    edges = np.linspace(0, len(o), n + 1).astype(int)
    return sorted(int(o[rng.integers(a, b)]) for a, b in zip(edges, edges[1:]) if b > a)


# ---------------------------------------------------------------- training
def masked_focal(logits, t, ign, weight, alpha=2, beta=4):
    """heatmap.focal over every channel, unknown pixels left out, each channel's loss times its kit's weight
    (kit_weights)."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    keep = ~ign[:, None].expand_as(t)
    pos = (t >= 0.99) & keep
    neg = (t < 0.99) & keep
    w = weight.repeat(2)[None, :, None, None].expand_as(t)             # feet and head channels: the same kit weight
    pos_loss = -(w * (1 - p) ** alpha * torch.log(p))[pos].sum()
    neg_loss = -(w * ((1 - t) ** beta) * (p ** alpha) * torch.log(1 - p))[neg].sum()
    return (pos_loss + neg_loss) / max(1, int(pos.sum()))


def kit_counts(k):
    n = np.zeros(len(CLASSES))
    for v in json.load(open(f'{rdir(k)}/labels.json')).values():
        for *_x, c in v['labels']:
            n[c] += 1
    return n


def kit_weights(upto):
    """Each kit's weight in the loss: its share of the seed's labels / its share of all labels of rounds 0..upto. The
    seed is the fair sample (Metrica and the kit model agreeing, both teams alike); a model surer of one kit labels
    more of it, and unweighted, the next model would lean further that way (round 2: 74% Barcelona from a 50/50
    seed). With no lean every weight is 1; keepers stay as rare as they are in the seed (one a team)."""
    seed = kit_counts(0)
    n = sum(kit_counts(k) for k in range(upto + 1))
    w = (seed / seed.sum()) / np.maximum(n / n.sum(), 1e-9)
    return n, torch.as_tensor(w, dtype=torch.float32, device=DEV)


def augment(x, t, ign):
    flip = torch.rand(len(x), device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    t = torch.where(flip[:, None, None, None], t.flip(-1), t)
    ign = torch.where(flip[:, None, None], ign.flip(-1), ign)
    gain = torch.empty(len(x), 1, 1, 1, device=x.device).uniform_(0.75, 1.25)
    return (x * gain).clamp(0, 1), t, ign


def all_crops(upto):
    parts = [np.load(f'{rdir(k)}/crops.npz') for k in range(upto + 1)]
    return tuple(np.concatenate([p[k] for p in parts]) for k in ('X', 'T', 'IGN', 'f'))


def loss_on(net, X, T, IGN, weight, batch=128):
    net.eval(); tot, n = 0.0, 0
    with torch.no_grad():
        for k in range(0, len(X), batch):
            x = torch.as_tensor(X[k:k + batch]).permute(0, 3, 1, 2).float().div(255).to(DEV)
            t = torch.as_tensor(T[k:k + batch]).float().div(255).to(DEV); ig = torch.as_tensor(IGN[k:k + batch]).to(DEV)
            tot += float(masked_focal(net(x), t, ig, weight)) * len(x); n += len(x)
    return tot / max(1, n)


def train(r, batch=32, lr=2e-3):
    X, T, IGN, f = all_crops(r - 1)
    frames = np.unique(f)
    val = np.isin(f, frames[::VAL_EVERY])
    counts, weight = kit_weights(r - 1)
    tqdm.write(f'round {r}: kit weights ' + ', '.join(f'{CLASSES[c]} {int(counts[c])} labels x{float(weight[c]):.2f}'
                                                       for c in range(len(CLASSES))))
    tqdm.write(f'round {r}: training on {int((~val).sum())} crops from {len(frames) - len(frames[::VAL_EVERY])} frames, '
               f'validating on {int(val.sum())}; {EPOCHS} epochs, from scratch')
    if r > 1:
        tqdm.write(f'round {r}: baseline to beat, model {r - 1} on this validation: loss '
                   f'{loss_on(load_net(r - 1), X[val], T[val], IGN[val], weight):.3f}')
    Xt, Tt, It = X[~val], T[~val], IGN[~val]
    net = PlayerFCN()
    torch.nn.init.constant_(net.head.bias, -4.0)                     # start sure there is nobody (CenterNet)
    net = net.to(DEV)
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
            x = torch.as_tensor(Xt[idx]).permute(0, 3, 1, 2).float().div(255).to(DEV)
            t = torch.as_tensor(Tt[idx]).float().div(255).to(DEV); ig = torch.as_tensor(It[idx]).to(DEV)
            x, t, ig = augment(x, t, ig)
            loss = masked_focal(net(x), t, ig, weight)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss.detach()) * len(idx); seen += len(idx)
            bar.set_postfix(loss=f'{float(loss.detach()):.3f}', mean=f'{tot / seen:.3f}')
        vl = loss_on(net, X[val], T[val], IGN[val], weight)
        log.write(f'{ep},{tot / seen:.4f},{vl:.4f}\n'); log.flush()
        mark = ''
        if vl < best:
            best = vl; torch.save(net.state_dict(), f'{rdir(r)}/model.pt'); mark = '  saved'
        tqdm.write(f'round {r} epoch {ep:2d}  train {tot / seen:.3f}  val {vl:.3f}{mark}')
    json.dump(dict(epochs=EPOCHS, best_val_loss=round(best, 4)), open(f'{rdir(r)}/trained.json', 'w'))
    return load_net(r)


# ---------------------------------------------------------------- label-free checks
def evaluate(r, net):
    """found / extra / over 11 on EVAL_FRAMES held-out seed frames (see the module docstring)."""
    lab = json.load(open(f'{rdir(0)}/labels.json'))
    frames = sorted(int(f) for f in lab)[::VAL_EVERY][:EVAL_FRAMES]          # never trained on: validation frames
    found, total, extra, over, n = 0, 0, 0, 0, 0
    for f, img in st.read(frames, f'model {r} check'):
        det = [d for d in detect(net, img, f) if d[3] >= DETECTED and on_pitch(ground(f, d[:2]))]
        for u, v, c in lab[str(f)]['labels']:
            G = ground(f, (u, v)); feet, top = body_at(f, G); h = np.linalg.norm(top - feet)
            total += 1
            found += any(np.hypot(du - u, dv - v) <= HIT_BODY * h and TEAM_OF[dc] == TEAM_OF[c] for du, dv, dc, _p in det)
        bodies = metrica_bodies(f)
        extra += sum(not any(np.hypot(du - feet[0], dv - feet[1]) <= MISS_BODY * np.linalg.norm(top - feet) for _s, feet, top in bodies)
                     for du, dv, _c, _p in det)
        k = np.bincount([d[2] for d in det], minlength=len(CLASSES))
        over += (k[0] + k[2] > 11) or (k[1] + k[3] > 11) or k[2] > 1 or k[3] > 1
        n += 1
    tqdm.write(f'model {r} on {n} held-out frames (detections >= {DETECTED}): found {found}/{total} seed players '
               f'({found / max(1, total):.1%}) in the right team; {extra / max(1, n):.2f} extra a frame (no Metrica body '
               f'there); a team over eleven on {over} frames ({over / max(1, n):.1%})')
    with open(f'{ROOT}/checks.csv', 'a') as fh:
        fh.write(f'{r},{found / max(1, total):.4f},{extra / max(1, n):.3f},{over / max(1, n):.4f}\n')


# ---------------------------------------------------------------- rounds
def cmd_round(r):
    net = None
    if r > 0:
        net = load_net(r) if os.path.exists(f'{rdir(r)}/trained.json') else train(r)
        evaluate(r, net)
    return label_round(r, net)


def cmd_run():
    for r in range(ROUNDS):
        if os.path.exists(f'{rdir(r)}/crops.npz'):
            continue
        if cmd_round(r) == 0:
            tqdm.write(f'round {r}: no labels -- stopping'); break
    last = max(r for r in range(ROUNDS) if os.path.exists(f'{rdir(r)}/model.pt'))
    shutil.copy(f'{rdir(last)}/model.pt', MODEL)
    tqdm.write(f'model {last} -> {MODEL}')


def cmd_detect(a, b, step=4000):
    """MODEL over frames a..b (in the halves) -> players/dets/{a}_{b}.npz: every player at UNSURE and up on the pitch,
    f, ground point (m), kit, p -- what player_track.py tracks. Written in chunks of step frames; done chunks skipped."""
    sys.path.insert(0, __import__('paths').REPO)
    from games import GAME, half_video_blocks
    os.makedirs(f'{ROOT}/dets', exist_ok=True)
    net = load_net(MODEL)
    for c0 in range(a, b + 1, step):
        c1 = min(b, c0 + step - 1); out = f'{ROOT}/dets/{c0}_{c1}.npz'
        frames = [f for f in range(c0, c1 + 1) if any(h0 <= f <= h1 for _h, h0, h1 in half_video_blocks(GAME))]
        if os.path.exists(out) or not frames:
            continue
        rows = []
        for f, img in tqdm(st.stream(st.runs_of(frames)), total=len(frames), desc=f'players {c0}-{c1}', unit='frame'):
            for u, v, c, p in detect(net, img, f):
                G = ground(f, (u, v))
                if p >= UNSURE and on_pitch(G):
                    rows.append((f, G[0], G[1], c, p))
        r = np.array(rows, dtype=float).reshape(-1, 5)
        np.savez_compressed(out, f=r[:, 0].astype(int), G=r[:, 1:3].astype(np.float32), kit=r[:, 3].astype(np.int8),
                            p=r[:, 4].astype(np.float32), frames=np.array(frames))


if __name__ == '__main__':
    os.makedirs(ROOT, exist_ok=True)
    {'run': cmd_run, 'round': lambda: cmd_round(int(sys.argv[2])), 'detect': lambda: cmd_detect(int(sys.argv[2]), int(sys.argv[3]))}[sys.argv[1]]()
