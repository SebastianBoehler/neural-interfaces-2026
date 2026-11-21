"""Frozen arms and order blocks for the Fable R8 scheme audit."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/bottleneck_21nov"), str(ROOT / "submissions/final_stack")]
from capture import DATA_SHA, V
from tapes import PEOPLE

FOLDS = tuple(f"{cohort}_{person}" for cohort, people in PEOPLE.items() for person in people)
ORDERS = {
    "within-fit": tuple(range(72000, 72004)),
    "within-held": tuple(range(72004, 72008)),
    "interface": (74000, 74001),
    "full": tuple(range(71808, 71812)),
}
CHUNKS = tuple(orders[i:i + 2] for orders in ORDERS.values() for i in range(0, len(orders), 2))
ARMS = ("hss", "nine25_fc", "ba_six_fp16", "em_half", "quarter")
