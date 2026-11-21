"""GPU cost of one Team B batch-64 feature forward (9 nets) per net kind under cudnn settings; diff to the default settings.

python bench_team_b_conv.py ZIPS_DIR DATA OUT.json ZIP_NAME"""
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import check_hard_team_b as C


def clock(fn, reps=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def main(zips, data_path, out, name):
    data = C.D.load(Path(data_path))
    with TemporaryDirectory() as scratch:
        directory = Path(scratch) / name
        module = C.unpack(Path(zips) / f"{name}.zip", directory)
        model = C.load(module, directory, data, "cuda:0")
    team = model.team_b
    x = torch.from_numpy(data["X"][:64]).cuda()
    x = (x - x.mean(-1, keepdim=True)).float()
    res = dict(gpu=torch.cuda.get_device_name())
    base = None
    with torch.no_grad():
        for label, bench, cl in (("default", False, False), ("benchmark", True, False), ("channels_last", False, True), ("benchmark+cl", True, True)):
            torch.backends.cudnn.benchmark = bench
            per = {}
            outs = []
            for kind in ("shallow", "slow", "fbc"):
                nets = [n for n, k in zip(team.nets, [k for k in ("shallow",) * 3 + ("slow",) * 3 + ("fbc",) * 3]) if k == kind]
                if cl:
                    for n in nets:
                        n.to(memory_format=torch.channels_last)
                per[kind] = clock(lambda: [n.feats(x) for n in nets])
                outs += [n.feats(x) for n in nets]
            res[label] = dict(ms=per, total=sum(per.values()))
            if base is None:
                base = outs
            res[label]["max_abs_diff_vs_default"] = max(float((a - b).abs().max()) for a, b in zip(outs, base))
            print(label, res[label], flush=True)
    Path(out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
