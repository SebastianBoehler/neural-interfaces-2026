"""Check the signed routing patch on 379 windows from two excluded-person folds."""
import argparse
from functools import partial
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from capture import DATA_SHA, DEVICE, V, fixture_assets, open_fold
from grow_replay import load
from replay import taped
from returning_em import ARMS, FP16_SHA, patch
from build_cb_speed import verify_pair
from patch_ba_six_fp16 import patch as fp16_patch

STREAMS = {"new_P01": "P01|1", "new_P02": "P02|1"}
ROSTER = "ba_six"


def stage(path, parent, source):
    folder = path / ROSTER
    folder.mkdir(parents=True)
    for file in parent.iterdir():
        if file.name != "submission.py":
            (folder / file.name).symlink_to(file.resolve())
    (folder / "submission.py").write_text(source)


def run(a):
    if V.digest(a.data) != DATA_SHA or V.digest(a.candidate) != a.sha:
        raise ValueError("unexpected data or candidate digest")
    if torch.cuda.get_device_capability(DEVICE) != (7, 5):
        raise ValueError("qualification must exercise the active Turing FP16 path")
    change = partial(patch, weight=ARMS[a.arm])
    verify_pair(a.parent, a.candidate, expected_sha=FP16_SHA, patch_source=change)
    data = V.D.load(a.data)
    x = torch.from_numpy(data["X"])
    windows, drift = 0, 0.0
    for fold, stream in STREAMS.items():
        pkg, receipts = open_fold(a.run, fold, "ba-confirm")
        receipt = receipts[ROSTER]
        parent = pkg / ROSTER
        fixture_assets(parent, receipt)
        parent_source = fp16_patch((parent / "submission.py").read_text())
        rows = V.D.plans(data, 71800, [stream])[stream]
        taps = []
        with TemporaryDirectory() as tmp:
            for name, source in (("parent", parent_source), ("candidate", change(parent_source))):
                root = Path(tmp) / name
                stage(root, parent, source)
                model, _ = load(data, root, ROSTER, receipt)
                delivery = taped(model, x, rows)
                if any(item.recognised for item in delivery):
                    raise ValueError(f"{fold} {name} was recognised")
                taps.append(delivery)
                del model
        p, q = taps
        np.testing.assert_array_equal([i.label for i in p], [i.label for i in q])
        for stage_name in ("route_out", "terminal"):
            before, after = (np.stack([getattr(i, stage_name) for i in values]) for values in (p, q))
            np.testing.assert_allclose(before, after, rtol=0, atol=2e-5)
            drift = max(drift, float(np.max(np.abs(before - after))))
        windows += len(rows)
        print(f"PASS {fold}: 0/{len(rows)} label changes and no recognition", flush=True)
    if windows != 379:
        raise ValueError(f"expected 379 windows, got {windows}")
    a.out.write_text(json.dumps(dict(sha256=a.sha, order=71800, windows=windows, label_changes=0,
                                     max_score_delta=drift, gpu=torch.cuda.get_device_name()), indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "candidate", "run", "data", "out"):
        p.add_argument(name, type=Path)
    p.add_argument("arm", choices=("half", "bypass"))
    p.add_argument("sha")
    run(p.parse_args())


if __name__ == "__main__":
    main()
