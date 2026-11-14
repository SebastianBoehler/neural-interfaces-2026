"""RUNS-0 DIAGNOSIS (ledger s226, step 5): why the TRUE-run count filter costs v6f on runs 0 (s215: V6Fr -0.33, P0r +0.07).
Per order and arm: per-person change, per-run net corrected windows (filter on - off, mean over bases) with run type
and delivery position, the capacity check (true counts vs cap) and the calibration of the chain's outputs (mean max
posterior vs accuracy, ECE in 10 bins, NLL), offsets off. Usage: python diag_runs0.py order [order ...]"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import jointcount as J  # noqa: E402


def calib(P):
    conf, pred = P.max(1), P.argmax(1); ok = pred == C.Y; b = np.minimum((conf * 10).astype(int), 9)
    ece = sum(abs(ok[b == i].mean() - conf[b == i].mean()) * (b == i).mean() for i in range(10) if (b == i).any())
    return conf.mean(), ok.mean(), ece, -np.log(np.clip(P[np.arange(len(P)), C.Y], 1e-9, 1)).mean()


over = {u: np.bincount(C.Y[C.RUN == u], minlength=3).max() - J.TRUE_CAP[u] for u in np.unique(C.RUN)}
print("capacity check: runs whose largest class count differs from the cap:", {u: v for u, v in over.items() if v != 0} or "none above", flush=True)
for order in sys.argv[1:]:
    sc, sd = order.rstrip("0123456789"), int(order[len(order.rstrip("0123456789")):]); g = C.PERMS[f"{sc}|{sd}|{C.p0}"]
    rk = np.empty(C.N, int); rk[g] = np.arange(C.N); z = np.load(f"{C.REPO}/runs/chainout/s215_{order}_off.npz"); bases = sorted({k.split("|")[0] for k in z.files})
    for arm in ("P0r", "V6Fr"):
        Ps = [z[f"{bn}|{arm}"].astype(np.float64) for bn in bases]; Fs = [J.filt_groups(P, rk, C.RUNI, lambda r: J.TRUE_CAP[C.RUN[r[0]]]) for P in Ps]
        d = np.mean([J.SC.person(F, C.p0) - J.SC.person(P, C.p0) for P, F in zip(Ps, Fs)], 0)
        cal = np.mean([calib(P) for P in Ps], 0); calf = np.mean([calib(F) for F in Fs], 0)
        print(f"{order} {arm}: change {d.mean():+.2f} | per person " + " ".join(f"{p} {v:+.1f}" for p, v in zip(C.PEOPLE, d)))
        print(f"   calibration off: conf {cal[0]:.3f} acc {cal[1]:.3f} ECE {cal[2]:.3f} NLL {cal[3]:.3f} | on: conf {calf[0]:.3f} acc {calf[1]:.3f} ECE {calf[2]:.3f}")
        net = {u: np.mean([((F[C.RUN == u].argmax(1) == C.Y[C.RUN == u]).sum() - (P[C.RUN == u].argmax(1) == C.Y[C.RUN == u]).sum()) for P, F in zip(Ps, Fs)]) for u in np.unique(C.RUN)}
        worst = sorted(net.items(), key=lambda kv: kv[1])[:6]
        pos = {u: int(np.argsort(np.argsort([rk[np.flatnonzero(C.RUN == v)].min() for v in np.unique(C.RUN[C.SES == C.SES[C.RUN == u][0]])]))[list(np.unique(C.RUN[C.SES == C.SES[C.RUN == u][0]])).index(u)]) for u, _ in worst}
        print("   worst runs (net corrected windows, delivery position): " + "; ".join(f"{C.PER[C.RUN == u][0]} {u.split('|', 1)[1]} {v:+.1f} pos {pos[u]}" for u, v in worst))
        by = {}
        for u, v in net.items():
            by.setdefault(u.split("|")[1], []).append(v)
        print("   net by run type: " + " ".join(f"{k} {np.sum(v):+.1f}" for k, v in sorted(by.items())), flush=True)
