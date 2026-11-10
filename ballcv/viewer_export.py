"""Our models' outputs in the files the viewer reads (viewer/src/lib/server/data.ts):

  assistants   the assistant model's referees at each flag sample (flag_self.py signals)  -> data/ballcv/BAR-ATH_assistants.json
  flags        where each one's flag was, and the flag model's raises (flag_self.py raises) -> data/ballcv/BAR-ATH_flags.json
  players      our player tracks (player_track.py), float32 [frame, slot, x | y], 0-1     -> data/ballcv/BAR-ATH_players_tracked.f32
               (+ .json: the frames and slots it covers)

  python ballcv/viewer_export.py
"""
import glob, json

import numpy as np

import ownball as ob
import flag_self

DATA = __import__('paths').DATA + 'ballcv/'
LINES = ('far', 'near')


def assistants_and_flags():
    sig = np.load(sorted(glob.glob(flag_self.ROOT + '/signals_*.npz'))[-1])
    raised = json.load(open(flag_self.RAISES))['raises']
    up = {line: np.zeros(200000, bool) for line in LINES}
    for r in raised:
        up[r['line']][r['start']:r['end'] + 1] = True
    det, samples = {}, {}
    for f, k, x, p, side, height in zip(sig['frame'].tolist(), sig['line'].tolist(), sig['x'].tolist(), sig['p'].tolist(),
                                        sig['side'].tolist(), sig['up'].tolist()):
        line = LINES[k]
        det.setdefault(str(f), {})[line] = [round(x, 1), 1.0]
        samples.setdefault(str(f), {})[line] = ([round(height, 2), round(side, 2), int(up[line][f])]
                                                if p >= flag_self.DETECTED else None)
    json.dump(dict(step=flag_self.DET_EVERY, threshold=0.5, frames=det), open(DATA + 'BAR-ATH_assistants.json', 'w'))
    json.dump(dict(step=flag_self.DET_EVERY, samples=samples,
                   signals=[dict(start=r['start'], end=r['end'], line=r['line'], x=r['x']) for r in raised]),
              open(DATA + 'BAR-ATH_flags.json', 'w'))
    print(f'assistants on {len(det)} samples, {len(raised)} raises -> {DATA}BAR-ATH_assistants.json, BAR-ATH_flags.json')


def players():
    z = np.load(ob.PLAYER_TRACKS); a, b = (int(v) for v in z['frames'])
    P = z['P'][a:b + 1].astype('<f4')
    P.tofile(DATA + 'BAR-ATH_players_tracked.f32')
    json.dump(dict(frames=[a, b], slots=int(P.shape[1])), open(DATA + 'BAR-ATH_players_tracked.json', 'w'))
    print(f'player tracks {a}-{b}, {P.shape[1]} slots -> {DATA}BAR-ATH_players_tracked.f32')


if __name__ == '__main__':
    assistants_and_flags()
    players()
