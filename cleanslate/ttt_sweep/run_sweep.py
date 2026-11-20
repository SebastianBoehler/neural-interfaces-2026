"""One sweep task: stream one held person's sessions through TTTModel(cfg) for each config of a chunk.

python run_sweep.py --pkg DIR_OF_HELD_PKG --held P01 --orders 71000 --cfgs stage.json --chunk 0/40 --out task.npz
Delivery/scoring path = harness (D.plans, official fresh_model, one (1,41,480) window per predict); labels never enter here.
Output keys: <cid>|rows_<order>_<ses>, <cid>|labels_..., <cid>|secs_... (seconds per session; windows = len(rows)).
Failure is loud: a config that raises aborts the task (no silent skip).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "harness"))
import data as D  # noqa: E402
import ttt_model as M  # noqa: E402
from cfgs import cid  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--held", required=True)
    ap.add_argument("--orders", default="71000")
    ap.add_argument("--cfgs", required=True)
    ap.add_argument("--chunk", default="0/1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = False
    cfgs = json.load(open(a.cfgs))
    i, n = map(int, a.chunk.split("/"))
    names = sorted(cfgs)[i::n]
    d = D.load(a.data)
    tj = json.load(open(Path(a.pkg) / "train.json"))
    assert a.held not in tj["people"], "held person in the package's training people"
    x = torch.from_numpy(d["X"])
    streams = [f"{a.held}|{s}" for s in sorted(set(d["session"][d["person"] == a.held]))]
    _, streaming = D.official()
    res, meta = {}, []
    for name in names:
        cfg = cfgs[name]
        h = cid(cfg)
        model = M.load_package(a.pkg, dev, cfg)  # unknown cfg keys raise KeyError
        for seed in map(int, a.orders.split(",")):
            for s, rows in D.plans(d, seed, streams).items():
                ses = s.split("|")[1]
                m = streaming.fresh_model(model)
                lab = []
                t0 = time.perf_counter()
                for r in rows:
                    y = m.predict(x[int(r):int(r) + 1].to(dev))
                    assert tuple(y.shape) == (1,) and y.dtype == torch.int64
                    lab.append(int(y[0]))
                if dev == "cuda":
                    torch.cuda.synchronize()
                secs = time.perf_counter() - t0
                res[f"{h}|rows_{seed}_{ses}"], res[f"{h}|labels_{seed}_{ses}"] = rows.astype(np.int64), np.array(lab, np.int8)
                res[f"{h}|secs_{seed}_{ses}"] = np.array(secs)
                del m
        meta.append(dict(name=name, cid=h, cfg=cfg))
        print("PASS", a.held, name, h, flush=True)
    np.savez(a.out, **res)
    json.dump(dict(held=a.held, orders=a.orders, pkg=str(a.pkg), cfgs=meta,
                   gpu=torch.cuda.get_device_name() if dev == "cuda" else "cpu",
                   code=M.__file__), open(a.out + ".json", "w"))


if __name__ == "__main__":
    main()
