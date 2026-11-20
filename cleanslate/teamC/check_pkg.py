"""Package checks for a team C submission dir: official fresh_model path, no cross-session state, timing.

python cleanslate/teamC/check_pkg.py --pkg DIR [--n 80] [--time-windows 400]
1. Equality (CPU, deterministic): session A streamed on fresh_model(master) == A streamed after B on another fresh copy
   == A streamed on a copy that first saw B and then got reset_state(). The master's weights and the submission
   module's globals are unchanged by streaming. Fails loudly on any difference.
2. Timing (cuda if present): warm ms per window over a real session prefix, projected over 20,083 windows.
"""
import argparse
import copy
import hashlib
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "2026-competition/tracks/bci_decoding"))
from benchmark_utils.streaming import fresh_model  # noqa: E402


def load(pkg, dev):
    spec = importlib.util.spec_from_file_location("submission", Path(pkg) / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    solver = object.__new__(mod.Solver)
    return mod, solver.load_model(dict(submission_dir=str(pkg), device=dev))


def run(m, X):
    return [int(m.predict(torch.as_tensor(x[None]))[0]) for x in X]


def snap(mod, model):
    g = {k: repr(v) for k, v in vars(mod).items() if not k.startswith("__") and not callable(v) and not isinstance(v, type(sys))}
    w = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for n in model.base for p in n.state_dict().values())).hexdigest()
    return g, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--time-windows", type=int, default=190)
    a = ap.parse_args()
    d = np.load(ROOT / "data/proteus_windows.npz")
    st = np.char.add(d["subject"], d["session"])
    s1, s2 = list(dict.fromkeys(st))[:2]
    XA, XB = d["X"][st == s1][: a.n], d["X"][st == s2][: a.n]
    torch.use_deterministic_algorithms(True)
    mod, master = load(a.pkg, "cpu")
    g0, w0 = snap(mod, master)
    l1 = run(fresh_model(master), XA)
    run(fresh_model(master), XB)
    l2 = run(fresh_model(master), XA)
    m = fresh_model(master)
    run(m, XB)
    m.reset_state()
    l3 = run(m, XA)
    g1, w1 = snap(mod, master)
    assert l1 == l2 == l3, ("cross-session state", l1, l2, l3)
    assert w0 == w1, "master weights changed"
    assert g0 == g1, ("module globals changed", {k for k in g0 if g0[k] != g1.get(k)})
    print(f"RESET_OK {a.n} windows x 3 paths identical; master weights and module globals unchanged", flush=True)
    torch.use_deterministic_algorithms(False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, master = load(a.pkg, dev)
    X = d["X"][st == s1]
    X = np.concatenate([X] * (a.time_windows // len(X) + 1))[: a.time_windows]
    m = fresh_model(master)
    run(m, X[:20])  # warm-up
    m = fresh_model(master)
    sec = []
    for x in X:
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        m.predict(torch.as_tensor(x[None]))
        if dev == "cuda":
            torch.cuda.synchronize()
        sec.append(time.perf_counter() - t0)
    ms = 1e3 * np.mean(sec)
    gpu = torch.cuda.get_device_name() if dev == "cuda" else "cpu"
    print(f"TIMING {gpu}: {ms:.1f} ms/window (p95 {1e3 * np.percentile(sec, 95):.1f}) -> {ms * 20083 / 1e3:.0f} s for 20,083 windows", flush=True)


if __name__ == "__main__":
    main()
