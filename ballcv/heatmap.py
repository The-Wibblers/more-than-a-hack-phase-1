"""The heatmap network our detectors share: a small fully-convolutional encoder-decoder whose output is a "here"
heatmap at 1/STRIDE resolution, its Gaussian targets, augmentation and focal loss, and peak finding over the pitch.

The ball detector is BallFCN trained by selftrain.py: models/ball.pt, the last of its 18 rounds.
player_det.py's PlayerFCN is the same network one stage deeper.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CROP = 96
STRIDE = 2
L, W = 105.3, 68.0
DEV = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


class BallFCN(nn.Module):
    """Encoder-decoder; heatmap logits at 1/STRIDE resolution."""

    def __init__(self):
        super().__init__()

        def cbr(i, o, s=1):
            return nn.Sequential(nn.Conv2d(i, o, 3, stride=s, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.e1 = nn.Sequential(cbr(3, 16), cbr(16, 16))                 # 1
        self.e2 = nn.Sequential(cbr(16, 32, 2), cbr(32, 32))             # 1/2
        self.e3 = nn.Sequential(cbr(32, 64, 2), cbr(64, 64))             # 1/4
        self.e4 = nn.Sequential(cbr(64, 96, 2), cbr(96, 96))             # 1/8
        self.d3 = cbr(96 + 64, 64)
        self.d2 = cbr(64 + 32, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3)
        d3 = self.d3(torch.cat([F.interpolate(e4, size=e3.shape[-2:]), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, size=e2.shape[-2:]), e2], 1))
        return self.head(d2)[:, 0]                                      # (n, H/2, W/2) logits


def target(h, w, pts, sigma=1.0):
    """Gaussian peaks at the ball positions inside the crop, centred on the NEAREST heatmap pixel so every
    ball has an exact 1.0 peak (otherwise most balls sit between pixels and the loss sees no positives)."""
    yy, xx = np.mgrid[0:h, 0:w]
    t = np.zeros((h, w), np.float32)
    for (x, y) in pts:
        cx, cy = int(np.clip(np.floor(x), 0, w - 1)), int(np.clip(np.floor(y), 0, h - 1))
        t = np.maximum(t, np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
    return t



# ---------------------------------------------------------------- training
def augment(x, t):
    """x: (n,3,H,W) 0-1, t: (n,h,w). Horizontal flip, scale jitter (+-25%, centre crop/pad), colour jitter."""
    flip = torch.rand(len(x), device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x); t = torch.where(flip[:, None, None], t.flip(-1), t)
    sc = float(np.random.uniform(0.8, 1.25))                     # one scale per batch: far/near balls
    if abs(sc - 1) > 0.02:
        H = x.shape[-1]; n = int(round(H * sc / 16)) * 16
        xs = F.interpolate(x, size=(n, n), mode='bilinear', align_corners=False)
        ts = F.max_pool2d(F.interpolate(t[:, None], size=(n // STRIDE, n // STRIDE), mode='bilinear', align_corners=False), 1)[:, 0]
        if n >= H:                                              # crop the centre back to H
            o = (n - H) // 2; x = xs[..., o:o + H, o:o + H]; ot = o // STRIDE; t = ts[..., ot:ot + H // STRIDE, ot:ot + H // STRIDE]
        else:                                                   # pad back to H
            p = (H - n) // 2; x = F.pad(xs, (p, H - n - p, p, H - n - p), mode='replicate'); pt = p // STRIDE
            t = F.pad(ts, (pt, H // STRIDE - n // STRIDE - pt, pt, H // STRIDE - n // STRIDE - pt))
        # keep an exact 1.0 at each resampled peak
        mx = F.max_pool2d(t[:, None], 3, stride=1, padding=1)[:, 0]
        t = torch.where((t == mx) & (t > 0.5), torch.ones_like(t), t)
    gain = 1 + 0.2 * (torch.rand(len(x), 1, 1, 1, device=x.device) * 2 - 1)
    bias = 0.06 * (torch.rand(len(x), 1, 1, 1, device=x.device) * 2 - 1)
    return (x * gain + bias).clamp(0, 1), t


def focal(logits, t, alpha=2, beta=4):
    """CenterNet-style focal loss on Gaussian targets."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = t >= 0.99
    pos_loss = -((1 - p) ** alpha * torch.log(p))[pos].sum()
    neg_loss = -(((1 - t) ** beta) * (p ** alpha) * torch.log(1 - p))[~pos].sum()
    return (pos_loss + neg_loss) / max(1, int(pos.sum()))


# ---------------------------------------------------------------- inference
BALL_MODEL = os.environ.get('BALL_MODEL') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'ball.pt')


def load(path=None):
    """the ball detector: BALL_MODEL (selftrain.py's last round unless set), or the model at path."""
    path = path or BALL_MODEL
    net = BallFCN(); net.load_state_dict(torch.load(path, map_location='cpu')); return net.to(DEV).eval()


SEARCH_MARGIN_M = 5.0     # full-frame searches reach this far past the lines (balls going out of play)


def pitch_box(Mf):
    """Image rows/cols covering the pitch (+SEARCH_MARGIN_M), padded to a multiple of 16."""
    pts = []
    m = SEARCH_MARGIN_M
    for x, y in ((-m, -m), (L + m, -m), (L + m, W + m), (-m, W + m)):
        q = Mf @ np.array([x / L, y / W, 1.0]); pts.append(q[:2] / q[2])
    pts = np.array(pts)
    x0, y0 = np.clip(np.floor(pts.min(0)).astype(int), 0, [1920, 1080]); x1, y1 = np.clip(np.ceil(pts.max(0)).astype(int), 0, [1920, 1080])
    x0, y0 = x0 - x0 % 16, y0 - y0 % 16
    x1, y1 = min(1920, x0 + ((x1 - x0 + 15) // 16) * 16), min(1080, y0 + ((y1 - y0 + 15) // 16) * 16)
    return x0, y0, x1, y1


@torch.no_grad()
def peaks(net, img, Mf, thresh=-2.0, top=50):
    """[(u, v, logit)] local maxima of the heatmap over the pitch area, strongest first."""
    return peaks_in(net, img, pitch_box(Mf), thresh, top)


@torch.no_grad()
def peaks_in(net, img, box, thresh=-2.0, top=50):
    """[(u, v, logit)] local maxima of the heatmap inside an image box (x0, y0, x1, y1), strongest first."""
    x0, y0, x1, y1 = box
    x0, y0 = max(0, int(x0)), max(0, int(y0)); x1, y1 = min(1920, int(x1)), min(1080, int(y1))
    x0, y0 = x0 - x0 % STRIDE, y0 - y0 % STRIDE
    if x1 - x0 < 16 or y1 - y0 < 16:
        return []
    crop = img[y0:y1, x0:x1]
    h, w = crop.shape[:2]; ph, pw = (-h) % 16, (-w) % 16
    if ph or pw:
        crop = np.pad(crop, ((0, ph), (0, pw), (0, 0)), mode='edge')
    x = torch.tensor(crop).permute(2, 0, 1)[None].float().div(255).to(DEV)
    hm = net(x)[0]
    mx = F.max_pool2d(hm[None, None], 5, stride=1, padding=2)[0, 0]
    keep = (hm == mx) & (hm > thresh)
    ys, xs = torch.nonzero(keep, as_tuple=True)
    vals = hm[ys, xs]
    order = torch.argsort(vals, descending=True)[:top]
    ys, xs, vals = ys[order].cpu().numpy(), xs[order].cpu().numpy(), vals[order].cpu().numpy()
    h = hm.cpu().numpy()

    def sub(a, b, c):
        # sub-cell peak position from a parabola through three neighbouring heatmap values (cuts the 2-px grid's
        # rounding: error to clicked ball centres RMS 1.50 -> 1.16 px)
        den = a - 2 * b + c
        return 0.0 if den >= 0 else float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    out = []
    for yy, xx, v in zip(ys, xs, vals):
        dx = sub(h[yy, xx - 1], h[yy, xx], h[yy, xx + 1]) if 0 < xx < h.shape[1] - 1 else 0.0
        dy = sub(h[yy - 1, xx], h[yy, xx], h[yy + 1, xx]) if 0 < yy < h.shape[0] - 1 else 0.0
        out.append((float(x0 + (xx + 0.5 + dx) * STRIDE), float(y0 + (yy + 0.5 + dy) * STRIDE), float(v)))
    return out
