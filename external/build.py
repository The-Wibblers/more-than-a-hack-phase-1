"""Normalise each downloaded dataset into one bundle per game (see __init__.py).

    python -m external.build skillcorner                  # every game whose raw files are present
    python -m external.build skillcorner --games 1886347  # some games

SkillCorner goes through kloppy, which already turns it into its own normalised coordinates (0-1,
top-left origin, y down); multiplying by the pitch gives our metres. It keeps the real orientation,
so the second half is played the other way round, as it was.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .bundle import ALIVE, ALIVE_KNOWN, OWNER_A, OWNER_H, RAW_DIR, num, periods_from, write_bundle, write_index


# ---------------------------------------------------------------------------------------------
# kloppy tracking -> arrays
# ---------------------------------------------------------------------------------------------

def from_kloppy_tracking(ds) -> dict:
    """Objects, xyz, state, period ids and per-period seconds of a kloppy TrackingDataset."""
    from kloppy.domain import BallState, Ground

    md = ds.metadata
    L, W = float(md.pitch_dimensions.pitch_length), float(md.pitch_dimensions.pitch_width)
    side = {t.team_id: ("H" if t.ground == Ground.HOME else "A") for t in md.teams}

    # Only players who are ever tracked get a column: the squad lists unused substitutes too. Taken
    # from the frames rather than the squads, because kloppy adds tracks it cannot identify (SkillCorner
    # 2020's "away_anon_25") as players who are in no squad.
    seen = {}
    for fr in ds.frames:
        for p in fr.players_data:
            seen.setdefault(p.player_id, p)
    order = {t.team_id: i for i, t in enumerate(md.teams)}
    players = sorted(seen.values(), key=lambda p: (order.get(p.team.team_id, 9), p.jersey_no is None, p.jersey_no or 0, str(p.player_id)))
    col = {p.player_id: j + 1 for j, p in enumerate(players)}
    objects = [{"key": "ball", "team": "ball", "jersey": None, "name": "Ball", "position": None}] + [
        {
            "key": str(p.player_id),
            "team": side[p.team.team_id],
            "jersey": p.jersey_no,
            "name": p.name,
            "position": getattr(p.starting_position, "code", None) if p.starting_position else None,
        }
        for p in players
    ]

    n = len(ds.frames)
    xyz = np.full((n, len(objects), 3), np.nan, dtype=np.float32)
    state = np.zeros(n, dtype=np.uint8)
    period_ids = np.zeros(n, dtype=np.int16)
    seconds = np.zeros(n, dtype=np.float64)
    frame_ids = np.zeros(n, dtype=np.int64)
    for i, fr in enumerate(ds.frames):
        period_ids[i] = fr.period.id
        seconds[i] = fr.timestamp.total_seconds()
        frame_ids[i] = fr.frame_id
        b = fr.ball_coordinates
        if b is not None and b.x is not None:
            xyz[i, 0] = (b.x * L, b.y * W, b.z if getattr(b, "z", None) is not None else np.nan)
        for p, pd in fr.players_data.items():
            c = pd.coordinates
            if c is not None:
                xyz[i, col[p.player_id], :2] = (c.x * L, c.y * W)
        if fr.ball_state is not None:
            state[i] |= ALIVE_KNOWN | (ALIVE if fr.ball_state == BallState.ALIVE else 0)
        if fr.ball_owning_team is not None:
            state[i] |= OWNER_H if side.get(fr.ball_owning_team.team_id) == "H" else OWNER_A

    return {
        "objects": objects,
        "col": {str(k): v for k, v in col.items()},
        "xyz": xyz,
        "state": state,
        "period_ids": period_ids,
        "seconds": seconds,
        "frame_ids": frame_ids,
        "pitch": [L, W],
        "teams": {side[t.team_id]: t.name for t in md.teams},
        "team_ids": {str(t.team_id): side[t.team_id] for t in md.teams},
    }


# ---------------------------------------------------------------------------------------------
# SkillCorner: broadcast tracking JSONL + dynamic events + phases of play, ten matches
# ---------------------------------------------------------------------------------------------

SKILLCORNER_DROP = {"passing_option"}  # ~half of all rows: options, not things that happened


def skillcorner_games(raw: Path) -> list[str]:
    d = raw / "data" / "matches"
    return sorted(p.name for p in d.iterdir() if (p / f"{p.name}_tracking_extrapolated.jsonl").stat().st_size > 1000) if d.exists() else []


def build_skillcorner(match: str, raw: Path) -> Path:
    from kloppy import skillcorner

    d = raw / "data" / "matches" / match
    stem = d / match
    print(f"  {match}: tracking...", flush=True)
    t = from_kloppy_tracking(
        skillcorner.load(meta_data=f"{stem}_match.json", raw_data=f"{stem}_tracking_extrapolated.jsonl", include_empty_frames=True, only_alive=False)
    )
    L, W = t["pitch"]
    index_of = {int(fid): i for i, fid in enumerate(t["frame_ids"])}
    m = json.loads(Path(f"{stem}_match.json").read_text())

    def frame(v) -> int | None:
        return index_of.get(int(v)) if v not in ("", None) else None

    def pitch_xy(x, y, attacking_side):
        # Events are in the attacking team's frame: centre origin, y up, rotated 180 degrees when
        # it plays right to left. Checked against the tracked possessor at frame_start.
        x, y = num(x), num(y)
        if x is None or y is None:
            return None, None
        if attacking_side == "right_to_left":
            return num(L / 2 - x), num(W / 2 + y)
        return num(L / 2 + x), num(W / 2 - y)

    events = []
    for r in csv.DictReader(open(f"{stem}_dynamic_events.csv", newline="")):
        if r["event_type"] in SKILLCORNER_DROP:
            continue
        f = frame(r["frame_start"])
        if f is None:
            continue
        x, y = pitch_xy(r["x_start"], r["y_start"], r["attacking_side"])
        ex, ey = pitch_xy(r["x_end"], r["y_end"], r["attacking_side"])
        sub = r["event_subtype"] or None
        if r["event_type"] == "player_possession":
            sub = r["end_type"] + (f" ({r['pass_outcome']})" if r["pass_outcome"] else "")
        events.append({
            "f": f,
            "f_end": frame(r["frame_end"]),
            "period": int(r["period"]),
            "type": r["event_type"],
            "sub": sub,
            "team": t["team_ids"].get(r["team_id"]),
            "obj": t["col"].get(r["player_id"]),
            "player": r["player_name"] or None,
            "x": x, "y": y, "ex": ex, "ey": ey,
        })
    for r in csv.DictReader(open(f"{stem}_phases_of_play.csv", newline="")):
        f = frame(r["frame_start"])
        if f is None:
            continue
        events.append({
            "f": f,
            "f_end": frame(r["frame_end"]),
            "period": int(r["period"]),
            "type": "phase",
            "sub": r["team_in_possession_phase_type"],
            "team": t["team_ids"].get(r["team_in_possession_id"]),
            "obj": None, "player": None, "x": None, "y": None, "ex": None, "ey": None,
        })

    comp = m.get("competition_edition") or {}
    meta = {
        "id": f"skillcorner-{match}",
        "dataset": "skillcorner",
        "source_id": match,
        "title": f"{m['home_team']['short_name']} v {m['away_team']['short_name']}",
        "competition": comp.get("name") if isinstance(comp, dict) else None,
        "date": (m.get("date_time") or "")[:10] or None,
        "score": f"{m['home_team_score']}:{m['away_team_score']}",
        "home": m["home_team"]["name"],
        "away": m["away_team"]["name"],
        "fps": 10,
        "pitch": [L, W],
        "periods": periods_from(t["period_ids"]),
        "objects": t["objects"],
        "has": {"tracking": True, "ball": True, "ball_z": True, "alive": False, "possession": True, "events": True, "video": False},
        "notes": "Tracking from the broadcast feed: off-screen players are extrapolated, so positions far from the ball are estimates.",
        "video": None,
    }
    return write_bundle(meta, t["xyz"], t["state"], events)


# ---------------------------------------------------------------------------------------------

BUILDERS = {
    "skillcorner": (skillcorner_games, build_skillcorner),
}


def main(argv=None) -> None:
    builders = BUILDERS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", help=f"any of {', '.join(builders)}; default: all with raw files")
    ap.add_argument("--games", nargs="*", help="only these source match ids")
    args = ap.parse_args(argv)
    unknown = set(args.datasets) - set(builders)
    if unknown:
        ap.error(f"unknown dataset(s): {', '.join(sorted(unknown))}")

    for name in args.datasets or list(builders):
        list_games, build = builders[name]
        raw = RAW_DIR / name
        found = list_games(raw) if raw.exists() else []
        todo = [g for g in found if not args.games or g in args.games]
        print(f"{name}: {len(todo)} game(s) to build" + ("" if found else f" (nothing in {raw}; run python -m external.download {name})"))
        for g in todo:
            try:
                print(f"    -> {build(g, raw)}", flush=True)
            except Exception as error:  # one broken game must not stop the rest
                print(f"    !! {g}: {error!r}", file=sys.stderr, flush=True)
    games = write_index()
    print(f"index: {len(games)} game(s)")


if __name__ == "__main__":
    main()
