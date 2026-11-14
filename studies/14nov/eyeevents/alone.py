"""s240 gate 2: the eye-event view alone on new people. Columns causally z-scored per session in the order's delivery order
(running mean / sd of delivered windows, current included); LOSO logistic regression (3-class, and SUB-vs-WORD on those
windows only). Reports cell BA (person mean of interface cells), SUB-vs-WORD BA and within-session AUC, per-column
SUB-minus-WORD sign by interface x run type, and the margin / error correlation with the member mix (A.mix B2_pool) within
class x run, with the eye net (runs/ocunet/probs oc_nomix all_loso) as the reference.
Usage: python studies/14nov/eyeevents/alone.py [name=ev8]"""
import glob, os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
REPO = os.path.abspath(__file__ + "/../../../..")
sys.path[:0] = [REPO + "/replay", REPO + "/studies/14nov/extview"]
import env  # noqa: E402
A = env.A; p0 = "all_loso"; Z = A.L.Z
y = Z[f"y|{p0}"].astype(int); per = Z[f"person|{p0}"]; sess = Z[f"sess|{p0}"]
W = np.load(REPO + "/data/proteus_windows.npz"); assert (W["y"] == y).all() and (A.pl.B[f"idx|{p0}"] == np.arange(len(y))).all()
run = np.char.add(sess.astype(str), np.char.add("|", np.char.add(W["task"].astype(str), np.char.add("|", W["run"].astype(str)))))
cell = np.char.add(W["interface"].astype(str), np.char.add("|", np.where(np.char.startswith(W["task"].astype(str), "Online"), "online", "acq")))
icell = np.char.add(sess.astype(str), np.char.add("|", W["interface"].astype(str)))
LM = np.log(np.clip(A.mix("B2_pool", p0, A.FAMS)[0], 1e-6, 1))
OCU = np.log(np.clip(np.mean([np.load(f)["proba"] for f in sorted(glob.glob(f"{REPO}/runs/ocunet/probs/oc_nomix-{p0}-s*.npz"))], 0), 1e-3, 1))
pz = np.load(REPO + "/runs/shuffle_perms.npz")


def causal_z(V, order):
    out = np.empty_like(V); acc = {}
    for i in order:
        n, s1, s2 = acc.get(sess[i], (0, 0.0, 0.0)); n, s1, s2 = n + 1, s1 + V[i], s2 + V[i] ** 2; acc[sess[i]] = (n, s1, s2)
        m = s1 / n; out[i] = (V[i] - m) / np.sqrt(np.maximum(s2 / n - m ** 2, 0) + 1e-6) if n > 1 else 0.0
    return np.clip(out, -6, 6)


def loso(F, mask):
    P = np.full((len(y), 3), np.nan)
    for q in np.unique(per):
        tr, te = mask & (per != q), mask & (per == q)
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(F[tr], y[tr])
        P[np.ix_(te, m.classes_)] = m.predict_log_proba(F[te])
    return P


def cellba(pred, m):
    def ba(c):
        return np.mean([np.mean(pred[(icell == c) & (y == k) & m] == k) for k in np.unique(y[m & (icell == c)])])
    return 100 * np.mean([np.mean([ba(c) for c in np.unique(icell[(per == q) & m])]) for q in np.unique(per)])


def auc(s, yy):
    r = np.argsort(np.argsort(s)) + 1; n1 = (yy == 1).sum(); n0 = len(yy) - n1
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)


def dcorr(a, b, g):
    a, b = a.astype(float).copy(), b.astype(float).copy()
    for u in np.unique(g):
        k = g == u; a[k] -= a[k].mean(); b[k] -= b[k].mean()
    return np.corrcoef(a, b)[0, 1]


def main(name):
    z = np.load(f"{REPO}/runs/views_eye/{name}.npz"); cols = list(z["cols"]) if "cols" in z.files else [str(i) for i in range(z["rec"].shape[1])]
    R = z["rec"].astype(np.float64)
    print(f"{name}: SUB-minus-WORD mean per column (raw, recording order) by interface x run type, sign consistency:")
    for j, c in enumerate(cols):
        d = [R[(cell == u) & (y == 1), j].mean() - R[(cell == u) & (y == 2), j].mean() for u in np.unique(cell)]
        print(f"  {c:10s} " + " ".join(f"{u}:{x:+.3g}" for u, x in zip(np.unique(cell), d)) + f"  same sign {max(sum(x > 0 for x in d), sum(x < 0 for x in d))}/4", flush=True)
    sw = (y == 1) | (y == 2); out = []
    for k in ("rec", "within0", "full0", "runs0"):
        o = np.arange(len(y)) if k == "rec" else pz[f"{k.rstrip('0123456789')}|{k[-1]}|all_loso"].astype(int)
        F = causal_z(z[k].astype(np.float64), o); P3 = loso(F, np.ones(len(y), bool)); P2 = loso(F, sw)
        ba3 = cellba(P3.argmax(1), np.ones(len(y), bool)); lg = P2[:, 2] - P2[:, 1]
        ba2 = cellba(np.where(lg > 0, 2, 1), sw); au = np.mean([auc(lg[sw & (sess == u)], (y[sw & (sess == u)] == 2).astype(int)) for u in np.unique(sess)])
        tx = np.where(y == 2, lg, -lg)[sw]; tm = np.where(y == 2, LM[:, 2] - LM[:, 1], LM[:, 1] - LM[:, 2])[sw]
        to = np.where(y == 2, OCU[:, 2] - OCU[:, 1], OCU[:, 1] - OCU[:, 2])[sw]; g = np.char.add(run, y.astype(str))[sw]
        t3 = P3[np.arange(len(y)), y]; tm3 = LM[np.arange(len(y)), y]; g3 = np.char.add(run, y.astype(str))
        ba2o = cellba(np.where(OCU[:, 2] > OCU[:, 1], 2, 1), sw); ba2m = cellba(np.where(LM[:, 2] > LM[:, 1], 2, 1), sw)
        print(f" {k:8s} 3-class cell BA {ba3:.1f} | SUB-vs-WORD BA {ba2:.1f} (eye net {ba2o:.1f}, mix {ba2m:.1f}), within-session AUC {au:.3f} | "
              f"SW margin corr with mix {dcorr(tx, tm, g):+.3f}, errors {dcorr(tx < 0, tm < 0, g):+.3f}; eye net with mix {dcorr(to, tm, g):+.3f} "
              f"({dcorr(to < 0, tm < 0, g):+.3f}); view with eye {dcorr(tx, to, g):+.3f} | 3-class true-logp corr with mix {dcorr(t3, tm3, g3):+.3f}", flush=True)
        out.append((ba2, au))
    ba2, au = np.mean(out, 0); print(f"{name}: mean SUB-vs-WORD BA {ba2:.1f}, AUC {au:.3f} -> gate 2 {'STOP (chance)' if ba2 < 53 and au < 0.53 else 'PASS'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ev8")
