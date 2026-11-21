"""Replay the fixed hard, nine15 and nine25 arms on the real fold packages."""
import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "laneTB"))
import verify as V
from run import check_zero_mixer

ORDERS = tuple(range(71400, 71432))
PIN = V.W / "laneTB-verify-21nov/out/hard"
DATA_SHA = "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327"


def sessions(data, fold):
    mode, person = fold.split("_")
    visits = ("2", "3") if mode == "ret" else sorted(set(data["session"][data["person"] == person]))
    return [f"{person}|{visit}" for visit in visits]


def live_check(model, windows):
    if model.routed != 1 or model.cf.dead or model.cp.dead or model.cf.log["err"] or model.cp.log["err"]:
        raise ValueError(f"inactive or failed terminal stages: {model.log6}, {model.cf.log}, {model.cp.log}")
    if model.team_b is not None and len(model.team_b.windows) != windows:
        raise ValueError("Team B must consume each window exactly once")


def load_arms(data, pkgs):
    for arm, pkg in pkgs.items():
        model, _ = V.timed_load(pkg, data, V.DEVICE)
        V.check_class(model, "hard")
        weight, members = V.B.ARMS[arm]
        assert model.TEAM_B_MIX == weight and (model.team_b is None) == (arm == "base")
        if weight:
            assert len(model.team_b.nets) == 3 * members
        else:
            check_zero_mixer(model)
        yield arm, pkg, model


def smoke(data, signals, fold, pkgs):
    order = 71124
    stream = sessions(data, fold)[0]
    rows = V.D.plans(data, order, [stream])[stream]
    visit = stream.split("|")[1]
    mixes = {}
    for arm, _, model in load_arms(data, pkgs):
        path = PIN / fold / arm / f"pred_{fold}_{order}.npz"
        with np.load(path, allow_pickle=False) as pin:
            np.testing.assert_array_equal(pin[f"rows_{order}_{visit}"], rows)
            expected = pin[f"labels_{order}_{visit}"]
        labels, trace, meta = V.stream_session(model, signals, rows, V.DEVICE, capture=True, check=live_check)
        np.testing.assert_array_equal(labels, expected)
        mixes[arm] = trace[:, :3]
        if model.team_b is not None and model.team_b.windows:
            raise ValueError("a fresh session mutated the Team B template")
        print(f"{fold} {arm}: {len(rows)} pinned labels exact", meta, flush=True)
    for arm in ("nine15", "nine25"):
        if np.array_equal(mixes[arm], mixes["base"]):
            raise ValueError(f"{arm} did not reach the live mixer")
    print("PASS pinned delivery, same-call mixer zero and live stage checks", flush=True)


def replay(data, signals, fold, models, orders, out, *, check=live_check):
    for arm, model, receipt in models:
        payload = dict(receipt=np.asarray(json.dumps(receipt, sort_keys=True)))
        for order in orders:
            for stream, rows in V.D.plans(data, order, sessions(data, fold)).items():
                labels, _, meta = V.stream_session(model, signals, rows, V.DEVICE, check=check)
                suffix = f"{order}_{stream.split('|')[1]}"
                payload.update({f"rows_{suffix}": rows, f"labels_{suffix}": labels})
                print(f"{fold} {arm} {suffix} {len(rows)}", meta, flush=True)
        target = out / fold / arm
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"pred_{fold}_{orders[0]}.npz"
        with TemporaryDirectory(dir=target) as tmp:
            stage = Path(tmp) / path.name
            np.savez_compressed(stage, **payload)
            stage.replace(path)


def dose_models(data, pkgs, receipt):
    for arm, pkg, model in load_arms(data, pkgs):
        yield arm, model, {**receipt, "source_sha": V.digest(pkg / "submission.py"),
                          "bank_sha": V.digest(pkg / "team_b.pt")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--fold", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--first", type=int, choices=ORDERS[::2])
    a = p.parse_args()
    assert torch.cuda.is_available()
    data_sha = V.digest(a.data)
    if data_sha != DATA_SHA:
        raise ValueError("PROTEUS data digest differs")
    data = V.D.load(a.data)
    signals = torch.from_numpy(data["X"])
    weights = V.WEIGHTS / a.fold / "weights.pt"
    receipt = dict(V.audit(data, a.fold, weights), data_sha=data_sha, hard_sha=V.HARD_SHA)
    with TemporaryDirectory() as tmp:
        pkgs = V.stage("hard", a.fold, weights, Path(tmp))
        if a.first is None:
            smoke(data, signals, a.fold, pkgs)
        else:
            replay(data, signals, a.fold, dose_models(data, pkgs, receipt), (a.first, a.first + 1), a.out)


if __name__ == "__main__":
    main()
