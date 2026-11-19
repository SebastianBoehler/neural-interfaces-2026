"""Stream held sessions through a package's real Solver exactly like the official evaluator.

python stream.py --pkg DIR --held P01 [--sessions all|2,3] --orders 71000,71001 --out pred.npz
Per (order, session): official fresh_model (copy.deepcopy + reset_state), one (1,41,480) window per predict call in
full-session-shuffled order, int64 label (1,) checked. Labels are never passed to the model or read here.
Output keys: rows_<order>_<session> (delivery order), labels_<order>_<session>, secs_<order>_<session>.
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D  # noqa: E402


def load_solver(pkg, ch_names, device):
    spec = importlib.util.spec_from_file_location("cs_submission", Path(pkg) / "submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    meta = dict(submission_dir=Path(pkg), device=device, ch_names=list(ch_names), n_chans=len(ch_names), n_times=480,
                n_classes=3, sfreq=120.0, chs_info=[{"ch_name": c} for c in ch_names])
    return module.Solver().load_model(meta)


def stream_sessions(model, x, plan, device):
    _, streaming = D.official()
    out = {}
    for s, rows in plan.items():
        t0 = time.perf_counter()
        m = streaming.fresh_model(model)
        labels = []
        for r in rows:
            y = m.predict(x[int(r):int(r) + 1].to(device))
            y = y.detach().cpu() if torch.is_tensor(y) else torch.as_tensor(np.asarray(y))
            if y.dtype != torch.int64 or tuple(y.shape) != (1,) or not 0 <= int(y[0]) <= 2:
                raise ValueError(f"predict must return one int64 label in 0..2 per call, got {y.dtype} {tuple(y.shape)}")
            labels.append(int(y[0]))
        out[s] = (rows.astype(np.int64), np.array(labels, np.int8), time.perf_counter() - t0)
        del m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--held", required=True)
    ap.add_argument("--sessions", default="all")
    ap.add_argument("--orders", default=",".join(map(str, D.DEFAULT_ORDERS)))
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    d = D.load(a.data)
    x = torch.from_numpy(d["X"])
    sess = sorted(set(d["session"][d["person"] == a.held]))
    if a.sessions != "all":
        sess = [s for s in sess if s in a.sessions.split(",")]
    assert sess, f"no sessions for {a.held}"
    streams = [f"{a.held}|{s}" for s in sess]
    model = load_solver(a.pkg, d["ch_names"], a.device)
    res = {}
    for seed in map(int, a.orders.split(",")):
        plan = D.plans(d, seed, streams)
        for s, (rows, labels, secs) in stream_sessions(model, x, plan, a.device).items():
            ses = s.split("|")[1]
            res[f"rows_{seed}_{ses}"], res[f"labels_{seed}_{ses}"], res[f"secs_{seed}_{ses}"] = rows, labels, secs
            print("PASS", a.held, seed, ses, len(rows), f"{secs:.1f}s", flush=True)
    np.savez(a.out, **res)


if __name__ == "__main__":
    main()
