"""Where an assistant referee may stand, and the small classifier that scores it (trained by assistant_self.py).

An assistant is always on a touchline, so we only ask along the lines: a window every WIN_STEP_M metres of each line,
sized by the camera to a person standing there (PERSON_W x PERSON_H person heights, with room above the head for a
raised flag), resampled to a fixed WW x WH.
"""
import numpy as np
import cv2

import ownball as ob
from ownball import W

WW, WH = 32, 64                    # window size fed to the model (px)
PERSON_W, PERSON_H = 1.0, 2.0      # window, in person heights: the body centred, feet at FEET_AT of the height
FEET_AT = 0.8                      # (room above the head for a raised flag)
OUT_M = 0.75                       # windows sit this far outside the line (where assistants stand)
WIN_STEP_M = 0.5
NEG_M = 3.0                        # one assistant to a line: windows this far from him are not him
LINES = (('far', 0.0, -1), ('near', W, 1))      # name, touchline y, outwards


def window(img, Mf, x, line, out):
    """The window for a person standing at (x, just outside line): WH x WW pixels, or None if his feet are off the
    image (whoever is in the window then stands somewhere else -- the referee, when the near line is below the picture)."""
    feet, top = ob.column(Mf, np.array([x, line + out * OUT_M]))
    H = np.linalg.norm(top - feet)
    if not (8 < H < 400) or not (0 <= feet[0] < img.shape[1] and 0 <= feet[1] < img.shape[0]):
        return None
    w, h = PERSON_W * H, PERSON_H * H
    x0, y0 = feet[0] - w / 2, feet[1] - FEET_AT * h
    A = np.array([[WW / w, 0, -x0 * WW / w], [0, WH / h, -y0 * WH / h]])
    return cv2.warpAffine(img, A, (WW, WH), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)


def build_net(torch):
    nn = torch.nn
    block = lambda a, b: nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.Conv2d(b, b, 3, padding=1),
                                       nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2))
    return nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Linear(64, 1))


def to_tensor(torch, X):
    return torch.from_numpy(X).permute(0, 3, 1, 2).contiguous().float() / 255
