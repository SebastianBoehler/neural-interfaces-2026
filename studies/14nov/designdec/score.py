"""DESIGNDEC scorer (ledger s239, package form): for each order, the unmodified chain (runs/chainout/<order>.npz) and each
variant's per-window outputs (runs/designdec/sc_<order>_<fb>_<bnd>.npz, wrap.py), arms P0r / V6Fr, bases in both files.
Per variant: change against off with the same output filter (none, gated J-wprior/urn, gated true runs), and the decision
number of s239: variant + gated J filter minus off + gated J filter. Mean over orders per scheme, 80 % person bootstrap.
A variant's off/J file, where present, must equal the chainout file (no-op check).
Usage: python studies/14nov/designdec/score.py within0 within1 runs0 runs1"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalx as EX  # noqa: E402
REPO = EX.C.REPO; ARMS = ("P0r", "V6Fr"); VARS = ("rep_J", "ext_J", "fwd_J", "rep_true", "ext_true", "rep_J_ts", "ext_J_ts", "ext_true_ts")


def main():
    rng = np.random.default_rng(0); res = {}
    for order in sys.argv[1:]:
        ref = np.load(f"{REPO}/runs/chainout/{order}.npz"); f0 = f"{REPO}/runs/designdec/sc_{order}_off_J.npz"
        if os.path.exists(f0):
            z0 = np.load(f0); d = max(np.abs(z0[k] - ref[k]).max() for k in z0.files); assert d == 0, f"{order}: off/J differs from chainout ({d})"
            print(f"{order}: no-op check OK ({len(z0.files)} keys equal to chainout)")
        have = {v: np.load(f) for v in VARS if os.path.exists(f := f"{REPO}/runs/designdec/sc_{order}_{v}.npz")}
        if not have:
            continue
        for arm in ARMS:
            bases = sorted({k.split("|")[0] for z in have.values() for k in z.files if k.endswith("|" + arm)})
            for bn in bases:
                outs = {"off": ref[f"{bn}|{arm}"]} | {v: z[f"{bn}|{arm}"] for v, z in have.items() if f"{bn}|{arm}" in z}
                r = EX.person_scores(order, outs); EX.report(r, f"{order} {arm} {bn}")
                for (v, f), s in r.items():
                    res.setdefault((order, arm, v, f), []).append(s)
    print("SUMMARY: mean over orders and bases; change against off with the same filter [80 % person bootstrap]; decision = gJ")
    for sc in ("within", "runs", "full"):
        os_ = [o for o in sys.argv[1:] if o.rstrip("0123456789") == sc]
        for arm in ARMS:
            for v in VARS:
                if not all((o, arm, v, "none") in res for o in os_) or not os_:
                    continue
                cells = []
                for f in EX.FILTERS:
                    d = np.mean([np.mean(res[o, arm, v, f], 0) - np.mean(res[o, arm, "off", f], 0) for o in os_], 0)
                    bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])
                    cells.append(f"{f} {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] min {d.min():+.1f}")
                print(f"  {sc:6s} {arm:4s} {v:8s} ({len(os_)} orders) " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
