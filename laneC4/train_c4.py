"""Run Lane C's D1 trainer with a frozen-encoder member (BrainCodes or NeuroRVQ) on arm C only; fail loudly if a patch is inactive.

Extra arguments (removed before Lane C's parser): --model braincodes|neurorvq, --init pretrained|random.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "laneC")]


def pop(flag, choices):
    require(flag in sys.argv, f"{flag} is required")
    i = sys.argv.index(flag)
    value = sys.argv[i + 1]
    require(value in choices, f"{flag} must be one of {choices}")
    del sys.argv[i:i + 2]
    return value


import data  # noqa: E402
from oracle import require  # noqa: E402

data.FULL_ARMS = "C"  # must precede the import of train/full_collect
import braincodes_member  # noqa: E402
import common  # noqa: E402
import neurorvq_member  # noqa: E402
import train as lane_c  # noqa: E402

MODEL = pop("--model", ("braincodes", "neurorvq"))
INIT = pop("--init", ("pretrained", "random"))
module = {"braincodes": braincodes_member, "neurorvq": neurorvq_member}[MODEL]
PINS = {"braincodes": "braincodes/braincodes.pt", "neurorvq": "neurorvq/model.safetensors"}
CHECKPOINT_SHA = {"braincodes": braincodes_member.HASHES[PINS["braincodes"]], "neurorvq": neurorvq_member.HASHES["model.safetensors"]}[MODEL]


class Member(module.Member):
    def __init__(self, checkpoint):
        super().__init__(checkpoint, INIT)


lane_c.Member = Member
lane_c.split_optimizer = common.split_optimizer  # augment (channel subsetting with positions) is Lane C's, unchanged
_save = lane_c.save_json


def save_json(path, record):
    _save(path, {**record, "backbone": f"{MODEL}-{INIT}", "checkpoint_sha": CHECKPOINT_SHA, "frozen_body": True})


lane_c.save_json = save_json
lane_c.PARITY_ATOL = 5e-4  # batched vs singleton, fp32 kernel noise (C2/C3 value)

if __name__ == "__main__":
    assert lane_c.FULL_ARMS == "C" and lane_c.Member is Member and lane_c.split_optimizer is common.split_optimizer
    assert lane_c.save_json is save_json and lane_c.PARITY_ATOL == 5e-4 and lane_c.FULL_SEEDS == 3
    lane_c.main()
