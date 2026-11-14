"""TEACHGAIN (ledger s207, edit of the teacher, slot 10): per-input gains on the teacher's map. The teacher is a logistic
regression (C 0.01, balanced, no scaler) on the seed-mean head log-probs of eegnex_ea-m7r and ifnet_ea-m7r (38 features: each
family head has 19 outputs), refitted here exactly as teachshuf.py / eval_external_teacher.fit_predict do, per held-out person. Its logit
splits into per-feature contributions z_j = coef[:, j] x_j; the edit is z = sum_j g_j z_j + b (= a gain on each input
column of the map). For each held-out person the gains (one per feature) are fitted on the other nine people's leave-one-person-out
contributions in recording order (log loss + REG ||g - 1||^2). Writes teacher files for shufcombo's TS_FILE: rec|all_loso =
the unedited refit in recording order, <order>|all_loso = the edited (or, tag base, unedited) refit in delivery order.
Usage: [REG=0.1] python studies/14nov/teachgain.py runs/teachfeat_<orders>.npz ORDER [ORDER ...]"""
import os, sys
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
REPO = os.path.abspath(__file__ + "/../../.."); sys.path[:0] = [REPO + "/replay", REPO + "/experiments"]
import env  # noqa: E402
import protocol  # noqa: E402
A = env.A; p0 = "all_loso"; d = protocol.load(); y, subj = d["y"].astype(int), d["subject"]; IDX = A.L.Z[f"idx|{p0}"].astype(int)
FF, ORDERS = np.load(sys.argv[1]), sys.argv[2:]; REG = float(os.environ.get("REG", 0.1)); Frec = FF[f"rec|{p0}|F"].astype(np.float64)
PEOPLE = sorted(set(subj[IDX])); NF = Frec.shape[1]  # 38 (two families x 19 head outputs)


def fit(tr):
    return LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced").fit(Frec[tr], y[tr])


def contrib(m, F, rows):
    return F[rows][:, :, None] * m.coef_.T[None], np.broadcast_to(m.intercept_, (len(rows), 3))  # (n, NF, 3), (n, 3)


def logp(g, zj, b):
    z = np.einsum("njc,j->nc", zj, g) + b; z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))


M = {p: fit(subj != p) for p in PEOPLE}  # the package's map for each held-out person (fitted without that person)
CR = {p: contrib(M[p], Frec, np.flatnonzero(subj == p)) for p in PEOPLE}  # each person's own leave-one-out contributions
Trec = np.zeros((len(y), 3))
for p in PEOPLE:
    Trec[subj == p] = np.exp(logp(np.ones(NF), *CR[p]))
ref = A.L.Z[f"B2_pool|Ps|{p0}"][10]; err = np.abs(Trec[IDX] - ref).max(); print(f"refit teacher against the stored slot 10: max |diff| {err:.2e}", flush=True)
assert err < 1e-2, "the refitted teacher is not the stored slot 10"
G, acc = {}, []
for q in PEOPLE:
    zj = np.concatenate([CR[p][0] for p in PEOPLE if p != q]); b = np.concatenate([CR[p][1] for p in PEOPLE if p != q])
    yy = np.concatenate([y[subj == p] for p in PEOPLE if p != q])
    f = lambda g: -logp(g, zj, b)[np.arange(len(yy)), yy].mean() + REG * ((g - 1) ** 2).sum()
    G[q] = minimize(f, np.ones(NF), method="L-BFGS-B").x; zq, bq = CR[q]; yq = y[subj == q]
    acc.append((np.mean(logp(np.ones(NF), zq, bq).argmax(1) == yq), np.mean(logp(G[q], zq, bq).argmax(1) == yq)))
    print(f"  {q}: gains " + " ".join(f"{v:.2f}" for v in G[q]) + f" | held-out teacher accuracy {100 * acc[-1][0]:.2f} -> {100 * acc[-1][1]:.2f}", flush=True)
a = np.array(acc); print(f"held-out teacher accuracy change {100 * (a[:, 1] - a[:, 0]).mean():+.2f} points")
for tag, gains in ((f"acc{REG:g}", G), ("base", {q: np.ones(NF) for q in PEOPLE})):
    OUT = {f"rec|{p0}": Trec[IDX]}
    for o in ORDERS:
        Fo = FF[f"{o}|{p0}|F"].astype(np.float64); T = np.zeros((len(y), 3))
        for q in PEOPLE:
            r = np.flatnonzero(subj == q); T[r] = np.exp(logp(gains[q], *contrib(M[q], Fo, r)))
        OUT[f"{o}|{p0}"] = T[IDX]
    os.makedirs(f"{REPO}/runs/wsp", exist_ok=True); np.savez(f"{REPO}/runs/wsp/teachgain_{tag}.npz", **OUT); print(f"written runs/wsp/teachgain_{tag}.npz")
