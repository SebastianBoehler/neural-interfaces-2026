"""End-to-end Solver time per window on the GPU for several exact ZIPs on the same streams, plus load time.

python e2e_timing.py ZIPS_DIR DATA OUT.json NAME[=SHA] ...   (first NAME is the reference; sessions of shuffle 71000, every second one)
Each arm streams through the official fresh_model path (windows moved to the device as in compare()), arms alternate by
session; the first session is a discarded warm-up. Wall time includes everything per window: mixers, refits, graph capture
and actor restore at session starts. A degraded session raises.
"""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

import check_hard_team_b as C
from build_hard_team_b import verify_parent

D = C.D
INSIDE = {}  # arm -> {"team_b": seconds, "hard": seconds} spent inside the wrapped calls (CUDA-synchronised)


def graph_report(template):
    team = getattr(template, "team_b", None)
    return None if team is None else {size: (g.graph is not None, g.failed, g.replays) for size, g in team.feature_graphs.items()}


def wrap(cls, name, arm, key):
    """Time cls.name for one arm (every ZIP has its own module, hence its own classes)."""
    original = getattr(cls, name)

    def timed(self, *args, **kwargs):
        start = perf_counter()
        out = original(self, *args, **kwargs)
        torch.cuda.synchronize()
        INSIDE.setdefault(arm, {}).setdefault(key, 0.0)
        INSIDE[arm][key] += perf_counter() - start
        return out

    setattr(cls, name, timed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("zips", type=Path)
    p.add_argument("data", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("arms", nargs="+")
    p.add_argument("--windows", choices=("gpu", "cpu"), default="gpu", help="device of the delivered window tensor")
    a = p.parse_args()
    verify_parent(a.data, C.DATA_SHA)
    data = D.load(a.data)
    plan = list(C.plans(data, "71000").items())[::2]
    dev = "cuda:0"
    models, load_s, kinds = {}, {}, {}
    with TemporaryDirectory() as scratch:
        for spec in a.arms:
            name, _, sha = spec.partition("=")
            archive = a.zips / f"{name}.zip"
            if sha:
                verify_parent(archive, sha)
            directory = Path(scratch) / name
            module = C.unpack(archive, directory)
            start = perf_counter()
            models[name] = C.load(module, directory, data, dev)
            torch.cuda.synchronize()
            load_s[name] = perf_counter() - start
            kinds[name] = type(models[name]).__name__
            print(name, kinds[name], f"load {load_s[name]:.2f} s", flush=True)
        res = {n: dict(wall_s=[], windows=[]) for n in models}
        for n, m in models.items():  # profile: time inside Team B and the hard actors during the same streams
            if getattr(m, "team_b", None) is not None:
                wrap(type(m.team_b), "posterior", n, "team_b")
            if hasattr(type(m), "hard_probability"):
                wrap(type(m), "hard_probability", n, "hard")
        for k, (session, rows) in enumerate(plan):
            windows = torch.from_numpy(data["X"][rows])
            if a.windows == "gpu":
                windows = windows.to(dev)
            if k == 1:
                INSIDE.clear()  # session 0 was the warm-up
            order = list(models) if k % 2 == 0 else list(models)[::-1]
            for n in order:
                start = perf_counter()
                _, _, meta = C.stream(models[n], windows)
                torch.cuda.synchronize()
                wall = perf_counter() - start
                if k:  # session 0 is the warm-up
                    res[n]["wall_s"].append(wall)
                    res[n]["windows"].append(len(rows))
            print(session, {n: round(1e3 * res[n]["wall_s"][-1] / res[n]["windows"][-1], 2) for n in models if res[n]["wall_s"]}, flush=True)
    ref = next(iter(models))
    out = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, sessions=[s for s, _ in plan[1:]], load_s=load_s, kinds=kinds)
    for n, r in res.items():
        ms = 1e3 * sum(r["wall_s"]) / sum(r["windows"])
        inside = {key: 1e3 * v / sum(r["windows"]) for key, v in INSIDE.get(n, {}).items()}
        out[n] = dict(ms_per_window=ms, inside_ms_per_window=inside, windows=sum(r["windows"]), per_session_ms=[1e3 * w / c for w, c in zip(r["wall_s"], r["windows"], strict=True)])
    for n in models:
        out[n]["added_vs_ref_ms"] = out[n]["ms_per_window"] - out[ref]["ms_per_window"]
        out[n]["projected_s"] = 2554.9166 + 20.083 * out[n]["added_vs_ref_ms"]
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({n: (round(out[n]["ms_per_window"], 2), round(out[n]["added_vs_ref_ms"], 2), round(out[n]["projected_s"])) for n in models}))


if __name__ == "__main__":
    main()
