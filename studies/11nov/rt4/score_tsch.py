"""Score TSCH (ledger s18): teacher member rebuilt from head log-probs (m8s = 7 corpora + Scherer, m7c = control,
m7r = shipped recipe's checkpoints), swapped for alib member 10 at weight 5.  Usage: python score.py HEADS.npz"""
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
REPO = __import__("os").path.abspath(__file__ + "/../../../..")
sys.path[:0] = [REPO + "/replay", REPO + "/experiments"]
import env
import protocol
A = env.A; SC = A.SC
H = np.load(sys.argv[1], allow_pickle=True); d = protocol.load(); y = d["y"]; subj = d["subject"]
IDX = {p: A.pl.B[f"idx|{p}"].astype(int) for p in A.PROTOS}
print("keys", [k for k in H.files][:6], "...", len(H.files))


def teacher(tag, seeds=range(6)):
    F = np.concatenate([np.mean([H[f"{a}_ea-{tag}-s{s}"] for s in seeds], 0) for a in ("eegnex", "ifnet")], 1).astype(np.float64)
    fit = lambda tr, te: LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced").fit(F[tr], y[tr]).predict_proba(F[te])  # noqa: E731
    out = {}; T = np.zeros((len(y), 3))
    for q in sorted(set(subj)):
        te = subj == q; T[te] = fit(~te, te)
    out["all_loso"] = T[IDX["all_loso"]]; T2 = np.zeros((len(y), 3))
    for _, tr, te in protocol.folds(d, "seen_session"):
        T2[te] = fit(tr, te)
    out["seen_session"] = T2[IDX["seen_session"]]; return out, F.shape[1]


G1 = [SC.PEOPLE.index(p) for p in SC.G1]; G2 = [SC.PEOPLE.index(p) for p in SC.G2]; fams = [f for f in A.FAMS if f != "teacher"]; res = {}
for bn in env.EXPL:
    s = A.summary(A.system(bn, A.FAMS)); res[("shipped", bn)] = (s["dn"], s["ds"])
print("shipped chain J", [round(0.5 * res[("shipped", b)][0].mean() + 0.5 * res[("shipped", b)][1].mean(), 4) for b in env.EXPL])
for tag in ("m7r", "m7c", "m8s"):
    try:
        P, nc = teacher(tag)
    except KeyError as e:
        print(tag, "missing", e); continue
    sn, ss = SC.person(P["all_loso"], "all_loso"), SC.person(P["seen_session"], "seen_session"); A.EXTRA["t_" + tag] = P
    wr = np.mean(P["all_loso"].argmax(1)[SC.B["y|all_loso"] == 2] == 2)
    for bn in env.EXPL:
        s = A.summary(A.system(bn, fams, extra=[("t_" + tag, 5.0)])); res[(tag, bn)] = (s["dn"], s["ds"])
    J = [0.5 * res[(tag, b)][0].mean() + 0.5 * res[(tag, b)][1].mean() for b in env.EXPL]
    print(f"{tag}: {nc} head columns | teacher alone new {sn.mean():.2f} seen {ss.mean():.2f} WORD recall {100 * wr:.1f} | chain J {J[0]:.4f} {J[1]:.4f}", flush=True)
if ("m8s", env.EXPL[0]) in res and ("m7c", env.EXPL[0]) in res:
    dn = np.mean([res[("m8s", b)][0] - res[("m7c", b)][0] for b in env.EXPL], 0); ds = np.mean([res[("m8s", b)][1] - res[("m7c", b)][1] for b in env.EXPL], 0)
    print(f"m8s - m7c: dJ {0.5 * dn.mean() + 0.5 * ds.mean():+.2f} new {dn.mean():+.2f} (G1 {dn[G1].mean():+.2f}, G2 {dn[G2].mean():+.2f}; {int((dn >= 0).sum())}/10) seen {ds.mean():+.2f} | per person new {np.round(dn, 1).tolist()}")
