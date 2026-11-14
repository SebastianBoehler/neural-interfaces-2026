"""Fit the personal TSEM template prior "T" (ledger PV2 / PV4, coordinator 14 Nov; route4's `pri`): per public person q,
the mean over q's public sessions of the class means (labels, offline) of the session-standardised TSEM coordinates
(rlib3.stdz: whole-session mean, SD floor 1e-6) of every window. The coordinates are the package's own: core1 streamed
causally over the session in recording order, u = (z - view_m) @ view_V[0] (the first n_tsem = 128 view columns, TSEM's
input). Rows in route_names order -> OUT.npz key tpri (P, 3, 128); build6.py stores it as v6_tpri.
Dev (pview / route4) built pri from P04's and P10's session 1 only (the seen_session protocol); here every public session
of all ten people is used.
Usage: python lean/tools/fit_tpri.py PKG [OUT.npz] [--sessions-of P04:1,P10:1]  (--sessions-of: only these sessions per person)"""
import copy, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(Path(__file__).resolve().parent)]
import protocol  # noqa: E402
from record import load_model  # noqa: E402


def stdz(X):  # replay/rlib3.stdz
    return (X - X.mean(0)) / np.maximum(X.std(0), 1e-6)


def main():
    pkg = Path(sys.argv[1]); out = Path(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else ROOT / "runs/tpri/tpri.npz"
    only = dict(s.split(":") for s in sys.argv[sys.argv.index("--sessions-of") + 1].split(",")) if "--sessions-of" in sys.argv else {}
    d = protocol.load(); m = load_model(pkg, None, "tpri"); z = np.load(pkg / "weights.npz"); names = [str(n) for n in z["route_names"]]
    V0, vm = np.asarray(m.view_V[0]), np.asarray(m.view_m); per = {q: [] for q in names}
    for s in dict.fromkeys(d["stream"]):
        q, sn = s.split("/")[-1].split("|")
        if q not in per or (q in only and only[q] != sn):
            continue
        rows = np.flatnonzero(d["stream"] == s); c = copy.deepcopy(m); c.reset_state(); U = []
        for x in np.nan_to_num(np.asarray(d["X"][rows], np.float32)):
            _, zz = c.core1.step(x); U.append((zz - vm) @ V0)
        U, y = stdz(np.asarray(U)), d["y"][rows].astype(int)
        per[q].append(np.stack([U[y == k].mean(0) for k in range(3)]))
        print(f"{s}: {len(rows)} windows, class counts {np.bincount(y, minlength=3).tolist()}", flush=True)
    assert all(per[q] for q in names), {q: len(v) for q, v in per.items()}
    tpri = np.stack([np.mean(per[q], 0) for q in names])
    out.parent.mkdir(parents=True, exist_ok=True); np.savez(out, tpri=tpri, names=np.array(names), sessions=np.array([len(per[q]) for q in names]))
    print(f"wrote {out}: tpri {tpri.shape}, sessions per person {[len(per[q]) for q in names]}, |pri| per person {np.round(np.abs(tpri).mean((1, 2)), 3).tolist()}")


if __name__ == "__main__":
    main()
