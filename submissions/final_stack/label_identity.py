"""Exact-label (and score-drift) comparison of two ZIPs through the official Solver on the same streams.

python label_identity.py ZIPS_DIR DATA OUT.json REF[=SHA] NEW[=SHA] --device cuda|cpu --stream-set shuffle71000|zero379
shuffle71000: all 14 public sessions in shuffle 71000; zero379: P01|1 and P04|2 of shuffle 71120 (379 windows, bitwise compare).
A changed label, or on cpu / zero379 any differing tap bytes, fails the run."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

import check_hard_team_b as C
from build_hard_team_b import verify_parent

D = C.D


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("zips", type=Path); p.add_argument("data", type=Path); p.add_argument("out", type=Path)
    p.add_argument("ref"); p.add_argument("new")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--stream-set", choices=("shuffle71000", "zero379"), default="shuffle71000")
    p.add_argument("--part", type=int, default=0); p.add_argument("--count", type=int, default=1)
    a = p.parse_args()
    verify_parent(a.data, C.DATA_SHA)
    data = D.load(a.data)
    dev = "cuda:0" if a.device == "cuda" else "cpu"
    plan = (C.plans(data, "71000") if a.stream_set == "shuffle71000" else D.plans(data, 71120, C.ZERO_STREAMS))
    plan = list(plan.items())[a.part::a.count]
    models = {}
    with TemporaryDirectory() as scratch:
        for role, spec in (("ref", a.ref), ("new", a.new)):
            name, _, sha = spec.partition("=")
            if sha:
                verify_parent(a.zips / f"{name}.zip", sha)
            directory = Path(scratch) / role
            models[role] = C.load(C.unpack(a.zips / f"{name}.zip", directory), directory, data, dev)
        rep, total, changed = {}, 0, 0
        for session, rows in plan:
            w = torch.from_numpy(data["X"][rows])
            w = w.to(dev) if dev != "cpu" else w
            lab = C.stream(models["ref"], w, capture=True)
            lab_n = C.stream(models["new"], w, capture=True)
            diff = np.flatnonzero(lab[0] != lab_n[0]).tolist()
            drift = float(np.abs(lab[1] - lab_n[1]).max())
            bit = lab[1].tobytes() == lab_n[1].tobytes()
            rep[session] = dict(windows=len(rows), changed=diff, max_tap_abs_diff=drift, taps_bitwise_equal=bit)
            total += len(rows); changed += len(diff)
            print(session, rep[session], flush=True)
            if diff or (a.device == "cpu" and not bit):
                a.out.write_text(json.dumps(dict(failed=session, sessions=rep), indent=1) + "\n")
                raise RuntimeError(f"{session}: labels changed {diff} / bitwise {bit}")
    res = dict(device=torch.cuda.get_device_name() if dev != "cpu" else "cpu", stream_set=a.stream_set, windows=total, changed=changed, sessions=rep,
               max_tap_abs_diff=max(r["max_tap_abs_diff"] for r in rep.values()))
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print("PASS", total, "windows, changed", changed, "max tap diff", res["max_tap_abs_diff"])


if __name__ == "__main__":
    main()
