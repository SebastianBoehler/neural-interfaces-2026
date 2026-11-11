"""CSCH (ledger s21): core1's head fitted with Scherer 2015 rows as weighted extra training data.

Usage: python experiments/core_scherer.py OUT.npz [weights, default 0,0.1,0.25]
"""
import os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT + "/experiments"]
import protocol  # noqa: E402
import ts_core2 as T  # noqa: E402

OUT = sys.argv[1]; WS = [float(w) for w in (sys.argv[2] if len(sys.argv) > 2 else "0,0.1,0.25").split(",")]
d = protocol.load(external=("scherer2015~n",)); fe = T.Feats(d, T.parse("car_ov17_fs2_t0.1_le_k2")); y = d["y"]
ext = d["split"] == "external"; st = d["stream"]; bands = T.bandset("ov17")
low = np.zeros(fe.F.shape[1], bool)
for b, sl in enumerate(fe.slices[:-1]):
    if bands[b // 3][0] < 4:
        low[sl] = True
low[fe.slices[-1]] = True
print("features", fe.F.shape, "external rows", int(ext.sum()), "low-frequency columns", int(low.sum()), flush=True)
res = {}
for proto in ("all_loso", "seen_session"):
    for name, tr, te in protocol.folds(d, proto):
        pro, rows = np.flatnonzero(tr & ~ext), np.flatnonzero(te); ex = np.flatnonzero(ext)
        pri = [fe.prior(b, pro) for b in range(len(fe.C))]; d2p = [q[pro].mean() for q in fe.d2]
        sc = StandardScaler().fit(fe.F[pro]); Zp = sc.transform(fe.F[pro]); Ze = sc.transform(fe.F[ex]); Ze[:, low] = 0.0
        Fc = np.zeros((len(rows), fe.F.shape[1]))
        for s in dict.fromkeys(st[rows]):
            m = st[rows] == s; Fc[m] = fe.causal(rows[m], pri, d2p, None)
        Zc = sc.transform(Fc)
        for w in WS:
            if w == 0:
                lr = LogisticRegression(C=10.0, max_iter=2000).fit(Zp, y[pro])
            else:
                lr = LogisticRegression(C=10.0, max_iter=2000).fit(np.concatenate([Zp, Ze]), np.concatenate([y[pro], y[ex]]),
                                                                   sample_weight=np.r_[np.ones(len(pro)), np.full(len(ex), w)])
            res[f"w{w:g}|{proto}-{name}|proba"] = lr.predict_proba(Zc).astype(np.float32); res[f"w{w:g}|{proto}-{name}|rows"] = rows
        print(proto, name, "done", flush=True); np.savez(OUT, **res)
