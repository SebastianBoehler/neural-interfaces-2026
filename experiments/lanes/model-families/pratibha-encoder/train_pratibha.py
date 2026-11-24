"""Run Lane C's D1 trainer with a Pratibha-2 member on arm C only; fail loudly if any patch is inactive."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "laneC")]

import data  # noqa: E402

data.FULL_ARMS = "C"  # must precede the import of train/full_collect
import pratibha  # noqa: E402
import train as lane_c  # noqa: E402

lane_c.Member = pratibha.Member
lane_c.split_optimizer = pratibha.split_optimizer  # Lane C's augment (channel subsetting with positions) is used unchanged
_save = lane_c.save_json


def save_json(path, record):
    _save(path, {**record, "backbone": "pratibha2", "revision": pratibha.REVISION,
                 "checkpoint_sha": pratibha.HASHES["model.safetensors"]})


lane_c.save_json = save_json
lane_c.PARITY_ATOL = 5e-4  # batched vs singleton, fp32 kernel noise (C2 value)

if __name__ == "__main__":
    assert lane_c.FULL_ARMS == "C" and lane_c.Member is pratibha.Member and lane_c.split_optimizer is pratibha.split_optimizer
    assert lane_c.save_json is save_json and lane_c.PARITY_ATOL == 5e-4 and lane_c.FULL_SEEDS == 3
    lane_c.main()
