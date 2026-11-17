"""Everything next, in order, each step skipped when it is already done (so it can be stopped and run again):

  1. the flags: the flag model on the whole match, then its raises, checked     (flag_self.py signals, raises)
  2. the players over the whole match: detected, then tracked                   (player_det.py detect, player_track.py run)
  3. the touch detector: its seed (benchmarked first), then its rounds           (touch_self.py run)

  python ballcv/run_next.py
"""
import os, subprocess, sys
import numpy as np

import ownball as ob

HERE = os.path.dirname(os.path.abspath(__file__))
A, B = 11185, 190197
DONE_DETS = 18685                # the public five minutes were detected before, in their own chunks


def run(*args):
    print(f'\n=== python ballcv/{" ".join(map(str, args))}', flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, args[0]), *map(str, args[1:])], check=True, cwd=HERE)


if __name__ == '__main__':
    if not os.path.exists(f'{ob.S}flags/signals_{A}_{B}.npz'):
        run('flag_self.py', 'signals', A, B)
    run('flag_self.py', 'raises', A, B)
    run('player_det.py', 'detect', DONE_DETS, B)                             # finished chunks are skipped
    tracks = ob.PLAYER_TRACKS
    if not (os.path.exists(tracks) and tuple(np.load(tracks)['frames']) == (A, B)):
        run('player_track.py', 'run', A, B)
    run('touch_self.py', 'run')                                              # finished rounds are skipped
