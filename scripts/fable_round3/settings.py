"""Fixed Round 3 order blocks and the verified Round 2 fold loader."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/fable_round2"))
from folds import DATA_SHA, D, FOLDS, LEAD, LEAD_SHA, ORDERS, pinned_pair, streams
from live import digest, fold_model
from run import timed_load
from verify import FOLDS as FOLD_ASSETS

FIT = tuple(range(71300, 71308))
HELD = tuple(range(71308, 71316))
WEIGHTS = tuple(i / 1000 for i in range(200, 401, 25))
DEVICE = "cuda:0"
