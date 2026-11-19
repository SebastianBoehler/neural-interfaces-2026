"""Run Lane C's D1 trainer with a CBraMod member on arm C only; fail loudly if any patch is inactive."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "laneC")]

import data  # noqa: E402

data.FULL_ARMS = "C"  # Scherer-included pooled arm only; must precede the import of train/full_collect
import cbmod  # noqa: E402
import train as lane_c  # noqa: E402

lane_c.Member = cbmod.Member
lane_c.split_optimizer = cbmod.split_optimizer
_save = lane_c.save_json


def save_json(path, record):
    _save(path, {**record, "backbone": "cbramod", "checkpoint_sha": cbmod.BLOB})


lane_c.save_json = save_json

if __name__ == "__main__":
    assert lane_c.FULL_ARMS == "C" and lane_c.Member is cbmod.Member and lane_c.split_optimizer is cbmod.split_optimizer
    assert lane_c.save_json is save_json and lane_c.FULL_SEEDS == 3
    lane_c.main()
