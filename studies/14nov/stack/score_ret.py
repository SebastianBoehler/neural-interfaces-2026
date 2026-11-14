"""STACK scoring for returning people (ledger s245). Per order and base, the arms of SRC (pview chain SAVE_OUT, keys
<order>|<base>|<arm>, seen_session rows in recording layout) raw and through the detector-gated count filter
(rundetect/package.py with PROTO=seen_session: latch + J-wprior urn on the emitted output only). Every cell is the per-person
change against ER raw (P04, P10), mean over bases; per scheme the mean over its orders, plus the sealed weighting
0.2 P04 + 0.8 P10. Two people only: no bootstrap; per-order spread shown as min / max of the sealed-weighted value.
Usage: [SRC=runs/chainout/s245_ret.npz] [ARMS=T] python studies/14nov/stack/score_ret.py order [order ...]"""
import os, sys
import numpy as np
os.environ["PROTO"] = "seen_session"
REPO = os.path.abspath(__file__ + "/../../../.."); sys.argv, ORD = sys.argv[:1], sys.argv[1:]
sys.path.insert(0, REPO + "/studies/14nov/rundetect")
import package as K  # noqa: E402
assert K.SEEN, "package.py did not take PROTO=seen_session"
SRC = os.environ.get("SRC", "runs/chainout/s245_ret.npz"); ARMS = os.environ.get("ARMS", "T").split(",")
WHO = sorted(dict.fromkeys(K.C.A.L.Z["person|seen_session"])); assert WHO == ["P04", "P10"], WHO
z = np.load(REPO + "/" + SRC)


def outs(order, arm, hc, lat):
    bases = sorted({k.split("|")[1] for k in z.files if k.startswith(order + "|") and k.endswith("|" + arm)}); raw, fil = [], []
    for bn in bases:
        P = np.full((K.C.N, 3), 1 / 3); P[K.ROWS] = z[f"{order}|{bn}|{arm}"].astype(np.float64)
        raw.append(K.C.A.SC.person(P[K.ROWS], K.PROTO)); fil.append(K.C.A.SC.person(K.gate(P, K.J.filt_joint(P, hc, "urn"), order, lat)[K.ROWS], K.PROTO))
    return bases, np.mean(raw, 0), np.mean(fil, 0)


R = {}
for order in ORD:
    lat = K.latch(order); hc = K.J.hyp_cache(order, "wprior", K.X, K.LAM, sessions=K.SESS, proto=K.PROTO)
    be, e0, e1 = outs(order, "ER", hc, lat); R[order, "ER+filter"] = e1 - e0
    for arm in ARMS:
        ba, a0, a1 = outs(order, arm, hc, lat); assert ba == be, (order, arm, ba, be)
        R[order, arm] = a0 - e0; R[order, arm + "+filter"] = a1 - e0
    print(f"  {order:8s} latched {sum(w is not None for w in lat.values())}/{len(lat)} | " + " ".join(f"{k} {v[0]:+.2f}/{v[1]:+.2f}" for (o, k), v in R.items() if o == order), flush=True)

print("\nchange against ER raw, returning people: per scheme P04 / P10 / sealed (0.2 P04 + 0.8 P10); mean of schemes; sealed range over orders")
for k in dict.fromkeys(k for _, k in R):
    cells, sw = [], []
    for sc in ("within", "full", "runs"):
        os_ = [o for o in ORD if o.rstrip("0123456789") == sc]
        if os_:
            d = np.mean([R[o, k] for o in os_], 0); s = 0.2 * d[0] + 0.8 * d[1]; sw.append(s); cells.append(f"{sc} {d[0]:+.2f}/{d[1]:+.2f}/{s:+.2f}")
    allsw = [0.2 * R[o, k][0] + 0.8 * R[o, k][1] for o in ORD]
    print(f"  {k:10s} " + " | ".join(cells) + f" | mean sealed {np.mean(sw):+.2f} | range [{min(allsw):+.2f}, {max(allsw):+.2f}]")
np.savez(f"{REPO}/runs/s245_ret_score.npz", **{f"{o}|{k}": v for (o, k), v in R.items()})
