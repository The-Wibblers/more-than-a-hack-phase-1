"""The whole match, from our models to possession, in order (a step whose output is already there for the full match,
and newer than what it is built on, is skipped; the cheap ones after it always run again, since they read what came before):

  1. the flag raises                     flag_self.py raises        -> work/align/flags/raises.json
  2. the touches                         ownball.py events          -> data/ballcv/BAR-ATH_touches.json
  3. the ball in 3D                      ball3d.py fit              -> work/align/ball3d.npz
  4. the stoppages                       stoppages.py               -> data/ballcv/BAR-ATH_stoppages.json
  5. possession, the rule                possession/predict.py      -> evaluation/results/possession_touches.csv
  6. possession, the touch model         touch_possession.py public -> evaluation/results/possession_touch_model.csv
  7. for the viewer                      viewer_export.py           -> our assistants, flags and player tracks

  python ballcv/run_match.py
"""
import json, os, shutil, subprocess, sys

import ownball as ob
import flag_self

HERE = os.path.dirname(os.path.abspath(__file__))
POSS = os.path.join(os.path.dirname(HERE), 'possession')
A, B = 11185, 190197


def run(path, *args):
    print(f'\n=== python {os.path.relpath(path, os.path.dirname(HERE))} {" ".join(map(str, args))}', flush=True)
    subprocess.run([sys.executable, path, *map(str, args)], check=True, cwd=os.path.dirname(path))


if __name__ == '__main__':
    if not os.path.exists(flag_self.RAISES):
        run(os.path.join(HERE, 'flag_self.py'), 'raises', A, B)
    track = json.load(open(ob.WORK + '/track.json'))
    full = [min(map(int, track)), max(map(int, track))]
    have = json.load(open(ob.OUT_JSON))['frames'] if os.path.exists(ob.OUT_JSON) else None
    if have != full or os.path.getmtime(ob.PLAYER_TRACKS) > os.path.getmtime(ob.OUT_JSON):    # or built on older players
        if have and not os.path.exists(ob.OUT_JSON.replace('.json', '.public.json')):
            shutil.copy(ob.OUT_JSON, ob.OUT_JSON.replace('.json', '.public.json'))     # the five-minute touches, kept
        run(os.path.join(HERE, 'ownball.py'), 'events')
    run(os.path.join(HERE, 'ball3d.py'), 'fit')
    run(os.path.join(HERE, 'stoppages.py'))
    run(os.path.join(POSS, 'predict.py'))
    run(os.path.join(POSS, 'touch_possession.py'), 'public')
    run(os.path.join(HERE, 'viewer_export.py'))
