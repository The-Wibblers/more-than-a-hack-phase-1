"""Verify what a change actually did to a possession file, beyond the 5-minute score.

WHY THIS EXISTS
    The public ground truth covers only frames 11185-18684 -- 5 minutes of 199,319 (2.5%). A rule or
    detection change can move hundreds of frames out in the other 97.5% of the match, where there is no
    ground truth and run_evaluation.py is blind. The 5-minute score staying flat does NOT mean a change was
    safe: it usually means the change did not touch those 7,500 frames at all. (A real example: a touch rule
    moved 1,340 frames across 22 possessions, none of them in the scored window -- the score said 'unchanged'
    while saying nothing about the 22 sequences that actually changed.)

    This tool turns 'N frames changed' into something you can actually review: the possession SEQUENCES that
    changed, each one openable in the viewer, plus whether any land in the scored window, plus whether the
    possession structure stayed realistic.

SETUP
    Nothing to install -- standard library only (csv). Run it with the repo's .venv python so paths resolve:
        .venv/bin/python evaluation/verify_change.py <before.csv> <after.csv>

HOW TO USE IT (the loop for every change)
    1. Snapshot the current possession file BEFORE you change anything:
         cp evaluation/results/possession_touches.csv /tmp/before.csv
    2. Make your code change and regenerate the possession file, e.g.:
         python ballcv/ownball.py events && python possession/stoppages.py && python possession/predict.py
       (regenerate only the stages downstream of what you changed; a predict.py-only change needs just
       predict.py.)
    3. Compare:
         .venv/bin/python evaluation/verify_change.py /tmp/before.csv evaluation/results/possession_touches.csv
    4. Also run the 5-minute score as before:
         python evaluation/run_evaluation.py

WHAT IT PRINTS
  1. the possession SEQUENCES that changed  (frame range, duration, old -> new team) -- open each in the
     viewer at /frame/<start> to confirm it is a fix and not a regression
  2. how many changed frames fall in the SCORED window 11185-18684
       -> 0 means the 5-minute score CANNOT judge this change; it must be reviewed by eye
  3. possession count and mean duration, before and after, against the ground truth's ~14.2 s reference
       -> a change that explodes the count, or drifts mean duration far from ~14 s, is a red flag

HOW TO READ IT
  - Few sequences, each a clear fix in the viewer, structure steady  -> good change, keep it.
  - Many sequences you cannot explain, or structure drifting         -> likely regression, revert or narrow.
  - 0 changed frames in the scored window                            -> the score is no evidence either way;
                                                                        the eyeball review IS the evidence.

It is a review aid, not a pass/fail gate: outside the 5 minutes there is no ground truth, so judgement
(does each changed sequence look right in the viewer?) is the final check.
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
