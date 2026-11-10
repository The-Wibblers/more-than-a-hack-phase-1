"""The tactical video's frames, as RGB arrays (ffmpeg decodes)."""
import subprocess, sys
import numpy as np

VIDEO = __import__('paths').DATA + 'tactical/BAR-ATH_tactical-001.mp4'
FW, FH = 1920, 1080


def blocks():
    """[(half, first frame, last frame)] of the match's two halves in the video."""
    sys.path.insert(0, __import__('paths').REPO)
    from games import GAME, half_video_blocks
    return half_video_blocks(GAME)


def decode(frame):
    """One frame."""
    raw = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-ss', f'{frame / 25:.2f}', '-i', VIDEO, '-frames:v', '1',
                          '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(FH, FW, 3)


def read_frames(start, n):
    """n consecutive frames from start."""
    proc = subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-ss', f'{start / 25:.2f}', '-i', VIDEO, '-frames:v', str(n),
                             '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
    size = FW * FH * 3
    for _ in range(n):
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, np.uint8).reshape(FH, FW, 3)
    proc.wait()


def sampled(t0, t1, step):
    """(frame, image) for each frame of t0..t1 that is a multiple of step: one ffmpeg decodes (it already uses every
    core) and passes on only those frames, while we work on the last one."""
    a = -(-t0 // step) * step
    proc = subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-ss', f'{a / 25:.2f}', '-i', VIDEO, '-frames:v', str((t1 - a) // step + 1),
                             '-vf', f'select=not(mod(n\\,{step}))', '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                            stdout=subprocess.PIPE, bufsize=FW * FH * 3 * 4)
    for f in range(a, t1 + 1, step):
        buf = proc.stdout.read(FW * FH * 3)
        if len(buf) < FW * FH * 3:
            break
        yield f, np.frombuffer(buf, np.uint8).reshape(FH, FW, 3)
    proc.kill()
