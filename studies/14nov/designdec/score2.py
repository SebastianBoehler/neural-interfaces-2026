"""DESIGNDEC (2) scorer (ledger s239): fine-tuning on run-smoothed labels (ftrel_dd.py, tag ttt18<order>LPD, arm r15) against
s229's r15 (tag ttt18<order>LP), both through shufcombo (wrap.py off/J, DD_SFX _LPDr15 / _LPr15 -> runs/designdec/
sc_<order>_off_J_<sfx>.npz). Arms V6Fr (the FT arm) and P0r (no FT: must be identical, a check), each alone and with the
gated J / true filters (evalx). Mean over orders per scheme, paired per base, 80 % person bootstrap.
Usage: python studies/14nov/designdec/score2.py within0 within1 runs0 runs1"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalx as EX  # noqa: E402
REPO = EX.C.REPO


def main():
    rng = np.random.default_rng(0); res = {}
    for order in sys.argv[1:]:
        a, b = (np.load(f"{REPO}/runs/designdec/sc_{order}_off_J_{s}.npz") for s in ("LPr15", "LPDr15"))
        for k in a.files:
            if k.endswith("|P0r"):
                assert np.mean(a[k].argmax(1) == b[k].argmax(1)) > 0.998, f"{order} {k}: P0r differs between the FT tags beyond GPU float noise"  # each FT run re-writes its base outputs: up to 0.05 on single-seed bases, <= 1 flip in 2655
        for k in [k for k in a.files if k.endswith("|V6Fr")]:
            r = EX.person_scores(order, {"off": a[k], "dd": b[k]})
            for f in EX.FILTERS:
                res.setdefault((order, f), []).append(r["dd", f] - r["off", f])
        print(f"{order}: " + " | ".join(f"{f} {np.mean(res[order, f]):+.2f}" for f in EX.FILTERS), flush=True)
    for sc in ("within", "runs", "full"):
        os_ = [o for o in sys.argv[1:] if o.rstrip("0123456789") == sc]
        if os_:
            cells = []
            for f in EX.FILTERS:
                d = np.mean([np.mean(res[o, f], 0) for o in os_], 0); bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])
                cells.append(f"{f} {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] min {d.min():+.1f}")
            print(f"  {sc:6s} V6Fr smoothed-label r15 - r15 ({len(os_)} orders): " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
