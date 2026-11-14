"""DESIGNDEC (2) labels (ledger s239): r15's per-refit labels (runs/labels/relab_v51b_<order>.npz, relab[:, k] = the v5.1b
label chain's output under its state at refit k, windows < tgrid[k]) smoothed by the run urn over that prefix (J-wprior,
smoother.py), from the detector latch on (refits before the latch keep r15's labels). Causal: refit k sees windows < t only.
Writes runs/labels/relab_dd239_<order>.npz (same keys; only 'relab' changes) for ftrel_dd.py and prints the label accuracy
(balanced, mean over refits, per person) of emitted / r15 / smoothed labels.
Usage: python studies/14nov/designdec/labels.py within0 within1 runs0 runs1"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage as ST  # noqa: E402
import smoother as SM  # noqa: E402
import proxy as PX  # noqa: E402
C = ST.C


def main():
    for order in sys.argv[1:]:
        z = dict(np.load(f"{C.REPO}/runs/labels/relab_v51b_{order}.npz")); R, E, TG = z["relab"].copy(), z["all_loso"], z["tgrid"]; acc = {}
        for s in C.SESSIONS:
            rows = C.order_rows(order, s); y = C.Y[rows]; w0 = ST.latch(rows); lp, reset, cap = ST.hyps(rows); q = C.PER[rows[0]]
            for k, t in enumerate(TG):
                if t >= len(rows):  # ftrel refits at t < n only
                    break
                assert np.isfinite(R[rows[:t], k]).all() and np.isnan(R[rows[t:], k]).all(), (order, s, k)
                old = R[rows[:t], k].copy()
                if w0 is not None and t > w0:
                    u = SM.PrefixUrn(lp, reset[:t], cap[:t])
                    for i in range(t):
                        u.step(i, old[i])
                    R[rows[:t], k] = u.smooth(t)
                for f, lab in (("emit", E[rows[:t]]), ("r15", old), ("dd", R[rows[:t], k])):
                    acc.setdefault((q, f), []).append(PX.ba(y[:t], lab.argmax(1)))
        z["relab"] = R; np.savez(f"{C.REPO}/runs/labels/relab_dd239_{order}.npz", **z)
        people = list(dict.fromkeys(C.PER)); r = {f: np.array([np.mean(acc[p, f]) for p in people]) for f in ("emit", "r15", "dd")}
        print(f"{order}: " + " | ".join(f"{f} {v.mean():.2f}" for f, v in r.items()) + f" || dd - r15 {(r['dd'] - r['r15']).mean():+.2f} [min {(r['dd'] - r['r15']).min():+.1f}]", flush=True)


if __name__ == "__main__":
    main()
