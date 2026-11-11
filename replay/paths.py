"""Paths of the dev chain replay.

The code in this directory is tracked. Its inputs (member posteriors of the six dev bases, core1 features, the
identity model; about 60 MB) are not: they live in runs/replay_data/ (gitignored). replay/data.sha256 lists them; a
copy is on Galvani under runs/diag/replay_data/. Set NI26_REPLAY_DATA to use another location."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("NI26_REPLAY_DATA", REPO + "/runs/replay_data")
WINDOWS = REPO + "/data/proteus_windows.npz"
for _p in (REPO + "/experiments", os.path.dirname(os.path.abspath(__file__))):  # replay first, then experiments
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
