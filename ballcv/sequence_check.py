"""Check the sequence of play: where our stoppages and possession tell a story the match's own evidence rules out.

Each finding is a place where the order of events cannot be right -- not a score, a list to look at:

  tag without a stoppage   a tagged set piece (a throw-in, a corner, ...) is taken from a dead ball: one of our
                           stoppages must end at it (from TAG_LEAD_S before its restart to TAG_LEAD_S after)
  tag disagrees            ... and wait for that set piece, taken by the team the tag names
  whistle in live play     the referee's stoppage whistle stops play: we have it live then
  restart whistle in play  the referee's restart whistle restarts a dead ball: we have the ball live, no restart near
  replays over live play   the broadcast away on replays for longer than OUT_S: nobody misses that much live play
  live with nobody running a live spell in which the fastest players' pace (over DROP_S) stays under a jog (RUN_MS)
                           for longer than OUT_S: play has someone running
  goal not dead            a tagged goal with no goal stoppage to its kick-off

  python sequence_check.py      # reads data/ballcv/BAR-ATH_stoppages.json and evaluation/results/possession_touches.csv
"""
import csv, json, os, sys
import numpy as np

import ownball as ob
import stoppages as S

sys.path[:0] = [os.path.join(__import__('paths').REPO, 'possession'), __import__('paths').REPO]
import predict

LABELS = os.path.join(__import__('paths').REPO, 'evaluation', 'results', 'possession_touches.csv')


def main():
    st = json.load(open(S.OUT_JSON))['stoppages']
    rows = list(csv.reader(open(LABELS)))[1:]
    label = np.array([r[-1] for r in rows])
    halves = predict.half_video_blocks(predict.GAME)
    half_of = lambda f: next(((a, b) for _h, a, b in halves if a <= f <= b), None)
    dead = label == 'D'
    lead = S.TAG_LEAD_S * 25
    found = []

    for t, kind, team in S.tagged_outs():
        if not half_of(t):
            continue
        near = [s for s in st if s['restart'] is not None and s['start'] - lead <= t <= s['restart'] + lead]
        if not near:
            found.append((t, 'tag without a stoppage', f'{kind} ({team or "no team"}) tagged; '
                          f'live as {label[t]}'))
            continue
        s = near[0]
        if s['waits_for'] != kind or (team and s['restart_team'] != team):
            found.append((t, 'tag disagrees', f'tag: {kind} {team or ""}; ours: {s["waits_for"]} '
                          f'{s["restart_team"]} ({s["start"]}-{s["end"]})'))

    whistles = list(csv.DictReader(open(predict.WHISTLES_CSV.format(game=predict.GAME))))
    for w in whistles:
        f = int(w['tactical_frame'])
        if not half_of(f) or w['role'] not in ('stoppage', 'restart'):
            continue
        around = dead[max(0, f - 50):f + 75]
        if w['role'] == 'stoppage' and not around.any():
            found.append((f, 'whistle in live play', f'{w["score"]} dB, {w["kind"]}; live as {label[f]}'))
        if w['role'] == 'restart' and not around.any():
            found.append((f, 'restart whistle in play', f'{w["score"]} dB, {w["kind"]}; live as {label[f]}'))

    alignment = list(csv.DictReader(open(__import__('paths').ALIGNMENT)))
    i = 0
    while i < len(alignment):
        if alignment[i]['kind'] in ('replay', 'wipe'):
            j = i
            while j + 1 < len(alignment) and alignment[j + 1]['kind'] in ('replay', 'wipe'):
                j += 1
            t0 = next((int(float(alignment[k]['tactical_frame'])) for k in range(i - 1, -1, -1)
                       if alignment[k]['tactical_frame']), None)
            t1 = next((int(float(alignment[k]['tactical_frame'])) for k in range(j + 1, len(alignment))
                       if alignment[k]['tactical_frame']), None)
            if (t0 is not None and t1 is not None and half_of(t0) and j - i + 1 > S.OUT_S * 25
                    and not dead[t0:t1 + 1].any()):
                found.append((t0, 'replays over live play', f'{(j - i + 1) / 25:.0f} s of replays to {t1}, all live'))
            i = j + 1
        else:
            i += 1

    spd = np.load(ob.S + 'tactical.npz')['P'][..., 2]
    fastest = np.nanmean(-np.sort(-np.nan_to_num(spd, nan=-1), 1)[:, :S.FASTEST], 1)
    kd = S.DROP_S * 25
    pace = np.convolve(np.nan_to_num(fastest), np.ones(kd) / kd)[kd - 1:]       # pace[f]: over f..f+kd-1
    for _h, a, b in halves:
        f = a
        while f <= b:
            if dead[f] or pace[f] >= S.RUN_MS:
                f += 1
                continue
            g = f
            while g <= b and not dead[g] and pace[g] < S.RUN_MS:
                g += 1
            if g - f > S.OUT_S * 25:
                found.append((f, 'live with nobody running', f'{(g - f) / 25:.0f} s to {g - 1}, '
                              f'pace at most {pace[f:g].max():.1f} m/s; live as {label[f]}'))
            f = g

    goals = [s['start'] for s in st if s['cause'] == 'goal']
    for g0, g1, scorer in S.goal_spans():
        if not any(g0 <= g <= g1 for g in goals):
            found.append((g0, 'goal not dead', f'{scorer} goal tagged {g0}-{g1}, no goal stoppage'))

    found.sort()
    for f, what, why in found:
        print(f'{f:>7}  {what:<25} {why}')
    counts = {}
    for _f, what, _w in found:
        counts[what] = counts.get(what, 0) + 1
    print(f'\n{len(found)} findings: {counts}')


if __name__ == '__main__':
    main()
