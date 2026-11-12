"""SELMETA (ledger s118, explore): a learned, label-free selector for per-session member weightings. For each new
session and each of FAMSEL's 17 weightings, features from the first 60 windows only (mean top posterior, agreement with
the shipped output and with the majority of the 17, class-share entropy, agreement of the variant with itself at
lower confidence); a ridge model trained on the OTHER people's sessions predicts each variant's gain on windows 60+;
the session switches to the predicted best variant from window 60 if its predicted gain is > 0. Oracle on the same
windows for comparison. Usage: python studies/12nov/selmeta.py [smoke]"""
import os, sys
import numpy as np
from sklearn.linear_model import Ridge
REPO = os.path.abspath(__file__ + "/../../..")
F = REPO + "/studies/12nov/famsel.py"; src = open(F).read(); SMOKE = "smoke" in sys.argv
exec(compile(src.split("\nfor base in (")[0], F, "exec"))  # noqa: S102  (G, Z, sess, y, cell, chain, W, env)
per = Z[f"person|{p}"]; T0 = 60


def ba(P, r):
    return np.mean([np.mean(P[r][(cell[r] == c) & (y[r] == k)].argmax(1) == k) for c in np.unique(cell[r]) for k in range(3) if ((cell[r] == c) & (y[r] == k)).any()])


def feats(O, names, r0, maj):
    out = []
    for n in names:
        P = O[n][r0]; lab = P.argmax(1); sh = np.bincount(lab, minlength=3) / len(lab)
        out.append([P.max(1).mean(), np.mean(lab == O["shipped"][r0].argmax(1)), np.mean(lab == maj[r0]), -(sh[sh > 0] * np.log(sh[sh > 0])).sum(), np.mean(P.max(1) > 0.7)])
    return np.array(out)


for base in ("B2_pool",) if SMOKE else env.BASES:
    O = {n: chain(base, sc) for n, sc in W}; names = list(O); maj = np.stack([v.argmax(1) for v in O.values()]); maj = np.array([np.bincount(c, minlength=3).argmax() for c in maj.T])
    rows = []
    for s in np.unique(sess):
        r = np.flatnonzero(sess == s)
        if len(r) <= T0 + 30:
            continue
        r0, r1 = r[:T0], r[T0:]; Fv = feats(O, names, r0, maj); g = np.array([ba(O[n], r1) - ba(O["shipped"], r1) for n in names])
        rows.append((s, per[r[0]], r, Fv - Fv[0], g))
    out_l, out_o = O["shipped"].copy(), O["shipped"].copy()
    for s, q, r, Fv, g in rows:
        Xtr = np.concatenate([x[3] for x in rows if x[1] != q]); ytr = np.concatenate([x[4] for x in rows if x[1] != q])
        pred = Ridge(alpha=1.0).fit(Xtr, ytr).predict(Fv); k = int(np.argmax(pred)) if pred.max() > 0 else 0
        out_l[r[T0:]] = O[names[k]][r[T0:]]; out_o[r[T0:]] = O[names[int(np.argmax(g))]][r[T0:]]
    ref = G.scores(O["shipped"], p); dl, do = G.scores(out_l, p) - ref, G.scores(out_o, p) - ref
    bt = lambda d: np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])  # noqa: E731
    print(f"{base}: learned selector from window {T0}: {dl.mean():+.2f} {bt(dl).round(2)} | oracle from window {T0}: {do.mean():+.2f} | sessions {len(rows)}", flush=True)
