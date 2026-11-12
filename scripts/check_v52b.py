"""v5.2b (runtime-only build) against v5.2: identical outputs through the official streamed evaluation (ledger s112).

Usage: python scripts/check_v52b.py PKG_V52B PKG_V52 DEVICE [--split test|all] [--tol 0] [--out JSON]
Both packages stream the chosen windows through benchmark_utils.streaming.predict_streams (a deepcopy + reset_state per
session, one window per predict, batches moved to DEVICE by the official loader). Recorded per window: the routed-tail
posterior (Route3.step's output) and, for v5.2b, whether the personal posteriors came from the prefetch. Fails unless
the labels are identical on every window, the posteriors differ by at most --tol (default 0: the prefetch runs the same
arithmetic at the same batch size), and the prefetch served most routed windows (otherwise the build is a no-op).
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts"), str(ROOT / "2026-competition")]
import protocol  # noqa: E402
from add_rebalance import load_solver  # noqa: E402
from benchmark_utils.data import make_loader  # noqa: E402
from benchmark_utils.streaming import predict_streams  # noqa: E402

arg = lambda k, dflt: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else dflt


def stream(pkg, tag, d, rows, dev):
    m = load_solver(pkg, tag, dev); R = type(m.route); step0 = R.step; out, hits = [], [0, 0]

    def step(self, *a):
        o = step0(self, *a); out.append(np.array(o)); return o
    R.step = step
    P = type(m.pnets) if getattr(m, "pnets", None) is not None else None
    if P is not None and hasattr(P, "prefetch"):
        pr0 = P.proba

        def proba(self, j, t):
            hits[1] += 1; hits[0] += int(self.cache is not None and self.cache[0] == (j, t)); return pr0(self, j, t)
        P.proba = proba
    st = np.unique(d["stream"][rows], return_inverse=True)[1]
    ld = make_loader(torch.from_numpy(np.ascontiguousarray(d["X"][rows], np.float32)), torch.from_numpy(d["y"][rows]), batch_size=32,
                     record_id=st * 100 + d["run"][rows].astype(int), onset=np.arange(len(rows), dtype=np.float64), stream_id=st, device=dev)
    t0 = time.perf_counter(); y, p, _ = predict_streams(m, ld); sec = time.perf_counter() - t0
    R.step = step0
    return np.asarray(p), np.asarray(out), hits, round(sec, 1)


def main():
    new, ref, dev = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]; tol = float(arg("--tol", 0.0))
    d = protocol.load(); rows = np.flatnonzero(d["split"] == "test") if arg("--split", "all") == "test" else np.arange(len(d["y"]))
    pn, on, hits, sn = stream(new, "v52b", d, rows, dev)
    pr, orf, _, sr = stream(ref, "v52", d, rows, dev)
    diff = float(np.abs(on - orf).max())
    res = dict(device=dev, windows=len(rows), labels_identical=bool(np.array_equal(pn, pr)), n_label_diff=int((pn != pr).sum()),
               max_posterior_diff=diff, tol=tol, prefetch_hits=hits[0], personal_calls=hits[1], seconds_v52b=sn, seconds_v52=sr,
               acc_v52b=float(np.mean(pn == d["y"][rows])), acc_v52=float(np.mean(pr == d["y"][rows])))
    print(json.dumps(res, indent=1))
    if "--out" in sys.argv:
        Path(arg("--out", "")).write_text(json.dumps(res, indent=1))
    assert res["labels_identical"] and diff <= tol and hits[0] >= 0.9 * hits[1] > 0, "v5.2b check FAILED"
    print("V52B CHECK OK")


if __name__ == "__main__":
    main()
