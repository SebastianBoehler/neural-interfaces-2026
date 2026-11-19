"""Team B dev evaluator: one fold (train unless the package exists), stream through the real Solver, keep every arm.

python dev.py --mode new|ret --held P03 --orders 71000,71001 --run DIR [--variants "base|kappa=4|kappa=16,alpha=0.5"]
Same protocol as harness/stream.py (official fresh_model = deep copy + reset_state, one window per predict, full-session
shuffle from harness/data.py). Variants change only label-free session settings of the loaded model (no retraining).
Output DIR/dev_<mode>_<held>.npz: rows_<order>_<ses>, <arm>@<variant>_<order>_<ses>, secs_...; for the first order also
the end-of-session posteriors and projections (non-causal diagnostics, never fed back).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))
import data as D  # noqa: E402
import stream as S  # noqa: E402
import train as TR  # noqa: E402


def parse_variant(v):
    return {} if v == "base" else {k: float(x) if k != "final" else x for k, x in (kv.split("=") for kv in v.replace("+", ",").split(","))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("new", "ret"), required=True)
    ap.add_argument("--held", required=True)
    ap.add_argument("--orders", default="71000,71001")
    ap.add_argument("--run", required=True)
    ap.add_argument("--variants", default="base")
    a = ap.parse_args()
    run, dev = Path(a.run), "cuda:0" if torch.cuda.is_available() else "cpu"
    pkg = run / f"{a.mode}_{a.held}" / "pkg"
    if not (pkg / ".trained").exists():
        people = D.PEOPLE if a.mode == "ret" else [p for p in D.PEOPLE if p != a.held]
        TR.fit(people, [a.held] if a.mode == "ret" else [], pkg)
        (pkg / ".trained").touch()
    d = D.load()
    x = torch.from_numpy(d["X"])
    sess = sorted(set(d["session"][d["person"] == a.held]))
    sess = [s for s in sess if a.mode == "new" or s != "1"]
    streams = [f"{a.held}|{s}" for s in sess]
    model = S.load_solver(pkg, d["ch_names"], dev)
    _, streaming = D.official()
    res, base_cfg = {}, dict(model.cfg)
    for vi, variant in enumerate(a.variants.split("|")):
        model.cfg = dict(base_cfg, **parse_variant(variant))
        for oi, seed in enumerate(map(int, a.orders.split(","))):
            for s, rows in D.plans(d, seed, streams).items():
                ses, t0 = s.split("|")[1], time.perf_counter()
                m = streaming.fresh_model(model)
                assert not m.X and m.S is None, "state crossed the session boundary"
                final = [int(m.predict(x[int(r):int(r) + 1].to(dev))[0]) for r in rows]
                secs = time.perf_counter() - t0
                assert final == m.arms[m.cfg["final"]]
                res[f"rows_{seed}_{ses}"] = rows.astype(np.int64)
                for arm, lab in m.arms.items():
                    res[f"{arm}@{variant}_{seed}_{ses}"] = np.array(lab, np.int8)
                if vi == 0:
                    res[f"secs_{seed}_{ses}"] = secs
                    for n, who, z, scores in m.gate_log:
                        res[f"gate_{seed}_{ses}_{n}"] = np.array([float(who[1:]), z] + scores)
                if vi == 0 and oi == 0:
                    P, U, r = m.last
                    res[f"Pend_{seed}_{ses}"] = P.cpu().numpy().astype(np.float32)
                    res[f"Rend_{seed}_{ses}"] = r.cpu().numpy().astype(np.float32)
                    res[f"Uend_{seed}_{ses}"] = U.cpu().numpy().astype(np.float16)
                print("PASS", a.mode, a.held, variant, seed, ses, len(rows), f"{secs:.1f}s", flush=True)
    np.savez(run / f"dev_{a.mode}_{a.held}.npz", **res)


if __name__ == "__main__":
    main()
