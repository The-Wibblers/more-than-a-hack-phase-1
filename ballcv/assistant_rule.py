"""The assistant referees by colour rules, no learned model: is an assistant standing at this point of the touchline?

Same windows as assistant_net.py: a person-sized box every WIN_STEP_M along each touchline, resampled to 32 x 64. A
window holds an official when, down its body (the most lime band of BODY_W columns), there is a lime shirt and, below
it, lime socks -- each across at least KIT_MIN of the body's width. Lime is the kit's yellow-green glowing brighter
than the sunlit grass in the same window (GLOW), so grass in sun beside shade is not kit.

On each line the window with the most kit is the assistant. On the near line the fourth official stands at the
technical area, within FOURTH_M of halfway (where the officials the kit test finds stand on that line when the
assistant is elsewhere: 90% of them within 2.5 m); a figure there is not taken for the assistant.

assistant_self.py's seed: these rules, where Metrica's offside line agrees.
"""
import numpy as np
import cv2

from ownball import L
from assistant_net import window, WIN_STEP_M

LIME = ((24, 80, 110), (42, 255, 255))      # the kit's yellow-green, in sun and shade (with GLOW: brighter than grass)
GLOW = 1.15                                  # a kit pixel is at least this much brighter than the grass around it
CLIPPED = 250                                # as bright as the camera records: the shirt in sun (grass never gets there)
BODY_W = 10                     # the body's width in a window (columns of 32)
SHIRT_ROWS = (8, 48)            # the shirt's peak is searched in these rows (of 64)
SOCKS_BELOW = (8, 26)           # the socks' peak, this many rows below the shirt's
KIT_MIN = 1 / 3                 # shirt and socks: lime across at least this much of the body's width
FOURTH_M = 3.0                  # the fourth official: within this of halfway on the near line
GRASS_HUE = (28, 60)


def kit(win):
    """The lime across the body at the shirt's and the socks' peaks, the lesser of the two, as a share of the body's
    width. win: RGB uint8, 64 x 32."""
    hsv = cv2.cvtColor(win, cv2.COLOR_RGB2HSV); V = hsv[..., 2].astype(np.float32)
    grassy = (hsv[..., 0] >= GRASS_HUE[0]) & (hsv[..., 0] <= GRASS_HUE[1])
    grass = float(np.percentile(V[grassy], 75)) if grassy.sum() > 20 else float(np.median(V))
    lime = (cv2.inRange(hsv, *LIME) > 0) & (V >= min(GLOW * grass, CLIPPED))
    c0 = int(np.argmax(np.convolve(lime.mean(0), np.ones(BODY_W), 'valid')))
    p = np.convolve(lime[:, c0:c0 + BODY_W].mean(1), np.ones(3) / 3, 'same')
    top = SHIRT_ROWS[0] + int(np.argmax(p[SHIRT_ROWS[0]:SHIRT_ROWS[1]]))
    lo, hi = min(63, top + SOCKS_BELOW[0]), min(64, top + SOCKS_BELOW[1])
    low = lo + int(np.argmax(p[lo:hi])) if hi > lo else top
    return float(min(p[top], p[low]))


def official(k):
    return k >= KIT_MIN


def assistant(xs, kits, near):
    """(ground x, kit) of the assistant on a line from its windows' kits, or None."""
    ok = [i for i, k in enumerate(kits) if official(k) and not (near and abs(xs[i] - L / 2) < FOURTH_M)]
    if not ok:
        return None
    i = max(ok, key=lambda i: kits[i])
    return float(xs[i]), kits[i]


def windows_on_line(img, Mf, line, out):
    xs, wins = [], []
    for x in np.arange(-2, L + 2, WIN_STEP_M):
        win = window(img, Mf, x, line, out)
        if win is not None:
            xs.append(x); wins.append(win)
    return np.array(xs), wins
