"""Run Lane C's D1 trainer with a frozen-encoder member (BrainCodes or NeuroRVQ) on arm C only; fail loudly if a patch is inactive.

Extra arguments (removed before Lane C's parser): --model braincodes|neurorvq, --init pretrained|random, --input aligned|raw
(raw = no covariance alignment, window demeaned only: EA whitening decorrelates the electrodes a frozen encoder reads by position).
"""
import sys
from pathlib import Path

import numpy as np

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
INPUT = pop("--input", ("aligned", "raw"))
module = {"braincodes": braincodes_member, "neurorvq": neurorvq_member}[MODEL]
PINS = {"braincodes": "braincodes/braincodes.pt", "neurorvq": "neurorvq/model.safetensors"}
CHECKPOINT_SHA = {"braincodes": braincodes_member.HASHES[PINS["braincodes"]], "neurorvq": neurorvq_member.HASHES["model.safetensors"]}[MODEL]


class Member(module.Member):
    def __init__(self, checkpoint):
        super().__init__(checkpoint, INIT)


if INPUT == "raw":
    def raw_align(x, stream, order, prior=None):  # same call signature and shape as the causal alignment; nothing crosses windows
        return (x - x.mean(-1, keepdims=True)).astype(np.float32)

    data.align = raw_align
lane_c.Member = Member
lane_c.split_optimizer = common.split_optimizer  # augment (channel subsetting with positions) is Lane C's, unchanged
_save = lane_c.save_json


def save_json(path, record):
    _save(path, {**record, "backbone": f"{MODEL}-{INIT}", "checkpoint_sha": CHECKPOINT_SHA, "frozen_body": True, "head_form": 3, "input": INPUT})


lane_c.save_json = save_json
_step, _ema = lane_c.step, [None, 0]


def step(*a, **k):  # running mean of the training loss (the trainer prints one minibatch only)
    loss = _step(*a, **k)
    _ema[0] = loss if _ema[0] is None else 0.99 * _ema[0] + 0.01 * loss
    _ema[1] += 1
    if _ema[1] % 725 == 0:
        print("train loss ema", round(_ema[0], 4), "steps", _ema[1], flush=True)
    return loss


lane_c.step = step
# batched vs singleton: 5e-4 (C2/C3 value); BrainCodes 3e-3 because a discrete codec code can flip at a near-tie (form 1 saw 1.2e-3 in 3 of 36 fits)
lane_c.PARITY_ATOL = 3e-3 if MODEL == "braincodes" else 5e-4

if __name__ == "__main__":
    assert lane_c.FULL_ARMS == "C" and lane_c.Member is Member and lane_c.split_optimizer is common.split_optimizer
    assert lane_c.step is step and (INPUT == "aligned" or data.align is raw_align) and lane_c.save_json is save_json and lane_c.PARITY_ATOL == (3e-3 if MODEL == "braincodes" else 5e-4) and lane_c.FULL_SEEDS == 3
    lane_c.main()
