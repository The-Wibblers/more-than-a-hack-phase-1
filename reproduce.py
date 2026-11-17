"""The submission from the match data, in one command: every step in order, each skipped when its output is already
there, so a stopped run carries on where it left off.

  python reproduce.py            the committed models (ballcv/models/) on the match -> evaluation/results/possession_touches.csv,
                                 and its copy for the submission form, submission/The Wibblers.csv
  python reproduce.py --train    train every model again from nothing first, then detect with the new models

  1. caches            the tracking as arrays                  ballcv/cache_tracking.py      -> data/ballcv/work/align/tactical.npz
  2. alignment         broadcast frame -> tactical frame       alignment/align_broadcast.py  -> data/broadcast/..._to_tactical.csv
  3. whistles          the referee's, from the kick-off tags   alignment/detect_whistles.py  -> data/broadcast/..._whistles.csv
  4. homography        snapped to the painted lines            ballcv/refine_h.py fit        -> work/align/homography_refined.npy
  5. ball              candidates, then their best path        ballcv/ownball.py detect/link -> work/align/fullmatch/
  6. players           detected, then tracked                  ballcv/player_det.py, player_track.py -> work/align/players/
  7. flags             the assistants' flags                   ballcv/flag_self.py signals   -> work/align/flags/
  8. possession        touches, stoppages, our rules, viewer   ballcv/run_match.py           -> evaluation/results/possession_touches.csv
  9. submission        named after the team, validated                                       -> submission/The Wibblers.csv

With --train, before step 5: the ball, kit, player, assistant and flag detectors (selftrain.py, kit_team.py,
player_det.py, assistant_self.py, flag_self.py run), each copying its last model into ballcv/models/; then the
detections of steps 5-7 are cleared, so they are made again by the new models.
"""
import glob, os, shutil, subprocess, sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, 'ballcv'))
import numpy as np  # noqa: E402

import paths  # noqa: E402

S = paths.S
A, B = 11185, 190197                    # the match's two halves in the video
N_FRAMES = 199319
CHUNK = 10000                           # ball candidates are written a chunk at a time
SIGNALS = f'{S}flags/signals_{A}_{B}.npz'
RESULT = os.path.join(REPO, 'evaluation', 'results', 'possession_touches.csv')
SUBMISSION = os.path.join(REPO, 'submission', 'The Wibblers.csv')   # SUBMISSION.md: the file is named after the team
TRACKS = f'{S}players/tracks.npz'


def run(path, *args):
    print(f'\n=== python {path} {" ".join(map(str, args))}', flush=True)
    subprocess.run([sys.executable, os.path.join(REPO, path), *map(str, args)], check=True,
                   cwd=os.path.dirname(os.path.join(REPO, path)))


def step(done, path, *args):
    if done:
        print(f'--- {path} {" ".join(map(str, args))}: done', flush=True)
    else:
        run(path, *args)


def train():
    for script in ('selftrain.py', 'kit_team.py', 'player_det.py', 'assistant_self.py', 'flag_self.py'):
        run(f'ballcv/{script}', 'run')                                  # finished rounds are skipped
    for old in glob.glob(f'{S}fullmatch/*') + glob.glob(f'{S}players/dets/*') + [TRACKS, SIGNALS]:
        if os.path.isdir(old):
            shutil.rmtree(old)
        elif os.path.exists(old):
            os.remove(old)


if __name__ == '__main__':
    step(os.path.exists(S + 'tactical.npz') and os.path.exists(S + 'broadcast.npz'), 'ballcv/cache_tracking.py')
    step(os.path.exists(paths.ALIGNMENT), 'alignment/align_broadcast.py')
    step(os.path.exists(paths.WHISTLES), 'alignment/detect_whistles.py')
    step(os.path.exists(S + 'homography_refined.npy'), 'ballcv/refine_h.py', 'fit')
    if '--train' in sys.argv[1:]:
        train()
    for s in range(0, N_FRAMES, CHUNK):
        step(os.path.exists(f'{S}fullmatch/cands_{s}.npz'), 'ballcv/ownball.py', 'detect', s, min(s + CHUNK, N_FRAMES))
    step(os.path.exists(f'{S}fullmatch/track.json'), 'ballcv/ownball.py', 'link')
    tracked = os.path.exists(TRACKS) and tuple(np.load(TRACKS)['frames']) == (A, B)
    step(tracked, 'ballcv/player_det.py', 'detect', A, B)                # a stopped run: finished chunks are skipped
    step(tracked, 'ballcv/player_track.py', 'run', A, B)
    step(os.path.exists(SIGNALS), 'ballcv/flag_self.py', 'signals', A, B)
    run('ballcv/run_match.py')
    os.makedirs(os.path.dirname(SUBMISSION), exist_ok=True)
    shutil.copyfile(RESULT, SUBMISSION)
    run('evaluation/validate_submission.py', SUBMISSION)
