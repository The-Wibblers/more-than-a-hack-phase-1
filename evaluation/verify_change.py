"""Verify what a change actually did to a possession file, beyond the 5-minute score.

The public ground truth covers only frames 11185-18684 (5 minutes of 199,319). A change can move hundreds
of frames out in the 90% of the match that has no ground truth, where run_evaluation.py is blind. This tool
makes those changes reviewable instead of a bare frame count: it groups them into possession sequences you
can open in the viewer, and flags whether any land in the scored window (if none do, the 5-min score cannot
judge the change at all).

    # before a change:
    cp evaluation/results/possession_touches.csv /tmp/before.csv
    # ... make the change, regenerate ...
    python evaluation/verify_change.py /tmp/before.csv evaluation/results/possession_touches.csv

It prints, in order:
  1. the possession SEQUENCES that changed  (frame range, duration, old -> new team) -- each one to eyeball
  2. how many changed frames fall in the SCORED window 11185-18684 (0 => the score says nothing about this)
  3. possession count and mean duration, before and after, against the ground truth's ~14.2 s reference
"""
import csv
import sys

SCORED = (11185, 18684)          # the only frames with public ground truth
GT_MEAN_SPELL_S = 14.2           # mean possession length in the public window (EVALUATION.md)
N_FRAMES = 199319


def load(path):
    return {int(r["frame"]): (r.get("label") or "").strip().upper() for r in csv.DictReader(open(path))}


def possession_runs(labels):
    """[(start, end, team)] maximal runs of H or of A (dead ball and X are not possessions)."""
    runs, start, cur = [], None, None
    for f in range(N_FRAMES):
        lab = labels.get(f)
        if lab != cur:
            if cur in ("H", "A"):
                runs.append((start, f - 1, cur))
            cur, start = lab, f
    if cur in ("H", "A"):
        runs.append((start, N_FRAMES - 1, cur))
    return runs


def main(before_path, after_path):
    before, after = load(before_path), load(after_path)

    changed = [f for f in range(N_FRAMES) if before.get(f) != after.get(f)]
    seqs = []                                                 # contiguous changed stretches
    for f in changed:
        if seqs and f == seqs[-1][1] + 1:
            seqs[-1][1] = f
        else:
            seqs.append([f, f])

    print(f"{len(changed)} frames changed, across {len(seqs)} sequences\n")
    print("  frames              dur     before -> after")
    for a, b in seqs:
        print(f"  {a:6d}-{b:<6d}  {(b - a + 1) / 25:5.1f}s   {before.get(a) or '-'} -> {after.get(a) or '-'}")

    in_window = [f for f in changed if SCORED[0] <= f <= SCORED[1]]
    print(f"\nchanged frames in the SCORED window {SCORED[0]}-{SCORED[1]}: {len(in_window)}", end="")
    print("   (0 => the 5-minute score cannot judge this change)" if not in_window else "")

    for name, labels in (("before", before), ("after", after)):
        runs = possession_runs(labels)
        durs = [(b - a + 1) / 25 for a, b, _ in runs]
        mean = sum(durs) / len(durs) if durs else 0.0
        print(f"\n{name}: {len(runs)} possessions, mean {mean:.1f}s  "
              f"(ground-truth reference ~{GT_MEAN_SPELL_S}s)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python evaluation/verify_change.py <before.csv> <after.csv>")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
