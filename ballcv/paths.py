"""Where the ball pipeline reads and writes. Everything derived from the match lives under data/ (gitignored):
data/ballcv/work/align/ holds the cached tracking arrays, models and intermediate files. Nothing here is committed."""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, 'data') + '/'
S = os.path.join(DATA, 'ballcv', 'work', 'align') + '/'      # caches: tactical.npz, broadcast.npz, ...
MODELS = os.path.join(REPO, 'ballcv', 'models') + '/'          # the trained models in use: committed
os.makedirs(S, exist_ok=True)
