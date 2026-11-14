"""DESIGNDEC (2) proxy (ledger s239): are prefix-smoothed run posteriors better fine-tuning labels? At each s15 refit time t
(runs/labels/relab_v51b_<order>.npz tgrid), the labels of windows < t under three forms, scored with the true classes
(balanced accuracy over the prefix, mean over refits, then per person over sessions):
  emit   as emitted by the label chain (v5.1b, chain_v51b_<order>.npz = relab file 'all_loso'; what s15 trains on)
  relab  r15: the label chain's output under its state at t (relab[:, k])
  smooth the run urn (J-wprior or true runs) smoothed over the prefix, on the emitted outputs, from the detector latch on
Usage: [DD_B=J|true] python studies/14nov/designdec/proxy.py within0 within1 runs0 runs1"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage as ST  # noqa: E402
import smoother as SM  # noqa: E402
C = ST.C


def ba(y, p):
    return 100 * np.mean([np.mean(p[y == k] == k) for k in np.unique(y)])


def main():
    print(f"PROXY (s239 (2)): FT label accuracy at s15 refits, boundaries {ST.CFG['bnd']}, new people; per person mean [min]")
    allr = {}
    for order in sys.argv[1:]:
        z = np.load(f"{C.REPO}/runs/labels/relab_v51b_{order}.npz"); E, R, TG = z["all_loso"], z["relab"], z["tgrid"]; acc = {}
        for s in C.SESSIONS:
            rows = C.order_rows(order, s); y = C.Y[rows]; w0 = ST.latch(rows); u = SM.PrefixUrn(*ST.hyps(rows)); P = E[rows]
            for t in range(len(rows)):
                u.step(t, P[t])
                if t + 1 in TG:
                    k = list(TG).index(t + 1); n = t + 1
                    sm = u.smooth(n) if (w0 is not None and n > w0) else P[:n]
                    for f, lab in (("emit", P[:n]), ("relab", R[rows[:n], k]), ("smooth", sm)):
                        acc.setdefault((C.PER[rows[0]], f), []).append(ba(y[:n], lab.argmax(1)))
        people = list(dict.fromkeys(C.PER)); r = {f: np.array([np.mean(acc[q, f]) for q in people]) for f in ("emit", "relab", "smooth")}
        allr[order] = r
        print(f"  {order:8s} " + " | ".join(f"{f} {v.mean():.2f}" for f, v in r.items()) + f" || smooth - emit {(r['smooth'] - r['emit']).mean():+.2f} "
              f"[min {(r['smooth'] - r['emit']).min():+.1f}], smooth - relab {(r['smooth'] - r['relab']).mean():+.2f}", flush=True)
    for sc in ("within", "runs"):
        os_ = [o for o in allr if o.rstrip("0123456789") == sc]
        if os_:
            d = {f: np.mean([allr[o][f] for o in os_], 0) for f in ("emit", "relab", "smooth")}
            print(f"  {sc}: smooth - emit {(d['smooth'] - d['emit']).mean():+.2f}, smooth - relab {(d['smooth'] - d['relab']).mean():+.2f}, relab - emit {(d['relab'] - d['emit']).mean():+.2f}")


if __name__ == "__main__":
    main()
