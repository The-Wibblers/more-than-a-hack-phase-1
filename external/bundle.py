"""Writing and indexing the per-game bundles. See __init__.py for the format."""

import json
import os
from pathlib import Path

import numpy as np

from .catalogue import DATASETS

DATA_DIR = Path(os.environ.get("MTAH_DATA_DIR") or Path(__file__).resolve().parent.parent / "data")
RAW_DIR = DATA_DIR / "external" / "raw"
GAMES_DIR = DATA_DIR / "external" / "games"

# state.u8 bits
ALIVE = 1
ALIVE_KNOWN = 2
OWNER_H = 4
OWNER_A = 8

# The clock each period starts at, so the viewer can show a match clock.
PERIOD_CLOCK_S = {1: 0, 2: 45 * 60, 3: 90 * 60, 4: 105 * 60, 5: 120 * 60}


def write_bundle(meta: dict, xyz: np.ndarray, state: np.ndarray, events: list[dict]) -> Path:
    """Write one game. xyz is [n_frames, n_objects, 3] in metres, object 0 the ball."""
    n, k, _ = xyz.shape
    assert k == len(meta["objects"]), "one tracking column per object"
    assert state.shape == (n,)
    meta = {**meta, "n_frames": n, "dataset_info": DATASETS[meta["dataset"]]}
    out = GAMES_DIR / meta["id"]
    out.mkdir(parents=True, exist_ok=True)
    xyz.astype("<f4").tofile(out / "tracking.f32")
    state.astype(np.uint8).tofile(out / "state.u8")
    events = sorted(events, key=lambda e: e["f"])
    (out / "events.json").write_text(json.dumps(events, separators=(",", ":")))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return out


def write_index() -> list[dict]:
    """index.json: one summary line per built game, for the viewer's game picker."""
    games = []
    for meta_path in sorted(GAMES_DIR.glob("*/meta.json")):
        m = json.loads(meta_path.read_text())
        games.append({k: m.get(k) for k in ("id", "dataset", "title", "competition", "date", "score", "fps", "n_frames", "has")})
    (GAMES_DIR / "index.json").write_text(json.dumps({"datasets": DATASETS, "games": games}, indent=1))
    return games


def periods_from(period_ids: np.ndarray) -> list[dict]:
    """Contiguous frame ranges of each period, in frame order."""
    out = []
    for pid in dict.fromkeys(period_ids.tolist()):
        idx = np.flatnonzero(period_ids == pid)
        assert idx[-1] - idx[0] + 1 == len(idx), f"period {pid} is not contiguous"
        out.append({"id": int(pid), "start": int(idx[0]), "end": int(idx[-1]), "clock_start_s": PERIOD_CLOCK_S.get(int(pid), 0)})
    return out


def num(v) -> float | None:
    """A finite float, rounded to the centimetre, or None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, 2) if np.isfinite(f) else None
