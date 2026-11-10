"""Fetch SkillCorner's open data into data/external/raw/skillcorner/.

    python -m external.download skillcorner       # no account needed (~1.8 GB)

Files already present at the expected size are skipped, so an interrupted run can be re-run.
Everything lands in data/, which is gitignored.
"""

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

from .bundle import RAW_DIR

SKILLCORNER_RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/master"
SKILLCORNER_LFS = "https://media.githubusercontent.com/media/SkillCorner/opendata/master"


def fetch(url: str, dest: Path, size: int | None = None) -> None:
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    print(f"  {dest.name}", flush=True)
    with urllib.request.urlopen(url) as r, open(part, "wb") as out:
        shutil.copyfileobj(r, out, length=1 << 20)
    part.replace(dest)


def skillcorner(out: Path, games: list[str] | None) -> None:
    fetch(f"{SKILLCORNER_RAW}/LICENSE", out / "LICENSE")
    fetch(f"{SKILLCORNER_RAW}/data/matches.json", out / "data" / "matches.json")
    ids = [str(m["id"]) for m in json.loads((out / "data" / "matches.json").read_text())]
    for g in ids:
        if games and g not in games:
            continue
        d = f"data/matches/{g}/{g}"
        for suffix in ("_match.json", "_dynamic_events.csv", "_phases_of_play.csv"):
            fetch(f"{SKILLCORNER_RAW}/{d}{suffix}", out / f"{d}{suffix}")
        # Tracking is in Git LFS: raw.githubusercontent would return the pointer, not the file.
        fetch(f"{SKILLCORNER_LFS}/{d}_tracking_extrapolated.jsonl", out / f"{d}_tracking_extrapolated.jsonl")


SOURCES = {"skillcorner": skillcorner}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+", help=f"any of {', '.join(SOURCES)}")
    ap.add_argument("--games", nargs="*", help="only these source match ids")
    args = ap.parse_args(argv)
    unknown = set(args.datasets) - set(SOURCES)
    if unknown:
        ap.error(f"unknown dataset(s): {', '.join(sorted(unknown))}")
    for name in args.datasets:
        print(name)
        SOURCES[name](RAW_DIR / name, args.games)


if __name__ == "__main__":
    main()
