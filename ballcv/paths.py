"""Where the ball pipeline reads and writes. The match's own files are found by games.py (data/tactical/,
data/broadcast/; MTAH_DATA_DIR moves the whole data folder). Everything we derive from them lives under data/ballcv/:
our outputs at its top, the caches and intermediate files in data/ballcv/work/align/. Nothing under data/ is committed;
the trained models are, in ballcv/models/."""
import os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import games  # noqa: E402

DATA = games.DATA_DIR.rstrip('/') + '/'
TACTICAL_VIDEO = games.video_path()
BROADCAST_VIDEO = games.video_path(source='broadcast')
TAGS = games.asset_paths()['smart_tagging']                    # Metrica's smart tags
ATD_XML = games.asset_paths()['atd_meta']
ALIGNMENT = DATA + 'broadcast/BAR-ATH_broadcast_to_tactical.csv'   # alignment/align_broadcast.py
WHISTLES = DATA + 'broadcast/BAR-ATH_broadcast_whistles.csv'       # alignment/detect_whistles.py
S = os.path.join(DATA, 'ballcv', 'work', 'align') + '/'      # caches: tactical.npz, broadcast.npz, ...
MODELS = os.path.join(REPO, 'ballcv', 'models') + '/'          # the trained models in use: committed
os.makedirs(S, exist_ok=True)
