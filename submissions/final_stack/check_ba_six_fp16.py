"""Compare the exact fused archives through the active MANAS and Team B graphs."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

import check_ba_terminal as G
from patch_ba_six_fp16 import PARENT_SHA

C = G.C


def observe_precision(model, seen):
    forward = model.team_a.net.features

    def features(wave):
        mode = str(torch.get_autocast_dtype("cuda")) if torch.is_autocast_enabled("cuda") else "disabled"
        seen.add(mode)
        return forward(wave)

    model.team_a.net.features = features


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("parent", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("sha")
    p.add_argument("data", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--part", type=int)
    a = p.parse_args()
    C.verify_parent(a.data, C.DATA_SHA)
    data = C.D.load(a.data)
    device = "cuda:0"
    plan = list(C.D.plans(data, 71120, C.ZERO_STREAMS).items())
    if a.part is not None:
        plan = list(C.plans(data, "71000").items())[a.part::7]
    rows = {}
    modes = [set(), set()]
    with TemporaryDirectory() as tmp:
        scratch = Path(tmp)
        models = []
        for seen, (z, sha) in zip(modes, ((a.parent, PARENT_SHA), (a.candidate, a.sha)), strict=True):
            module, folder = G.unpack(z.parent, z.name, sha, scratch)
            model = C.load(module, folder, data, device)
            G.clean(model)
            observe_precision(model, seen)
            models.append(model)
        for i, (name, indices) in enumerate(plan):
            x = torch.from_numpy(data["X"][indices]).to(device)
            order = (0, 1) if i % 2 == 0 else (1, 0)
            results = {k: G.play(models[k], x, capture=True) for k in order}
            before, after = results[0], results[1]
            changed = int(np.count_nonzero(before[0] != after[0]))
            rows[name] = dict(windows=len(x), changed=changed, work_identical=before[2] == after[2],
                              max_abs=float(np.max(np.abs(before[1] - after[1]))))
            print(name, rows[name], flush=True)
    n = sum(r["windows"] for r in rows.values())
    if a.part is None and n != 379:
        raise ValueError(f"wrong smoke coverage: {n}")
    capability = torch.cuda.get_device_capability()
    expected = {"torch.float16"} if capability == (7, 5) else modes[0]
    passed = bool(modes[0]) and modes[1] == expected and all(r["changed"] == 0 and r["work_identical"] for r in rows.values())
    a.out.write_text(json.dumps(dict(sha=a.sha, windows=n, passed=passed,
                                     capability=capability, precision=[sorted(m) for m in modes], sessions=rows),
                                indent=1) + "\n")
    if not passed:
        raise SystemExit("The precision change alters active labels or work")


if __name__ == "__main__":
    main()
