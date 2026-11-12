"""S2 (ledger s97): a nonlinear multi-view model with its own session stage, scored as a whole system (new people, LOSO).
Usage: python studies/12nov/sys2.py   (caches features in runs/sys2_feats.npz)"""
import os, sys
import numpy as np
import torch
from sklearn.decomposition import PCA
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/replay", REPO + "/studies/12nov"]
import env  # noqa: E402
import grid as G  # noqa: E402
import ownspace as O  # noqa: E402
A = env.A; Z = A.L.Z; p = "all_loso"; sess = Z[f"sess|{p}"]; y = O.y.astype(int); N = len(y); PEOPLE = sorted(set(O.PER)); torch.set_num_threads(6)
assert (Z[f"idx|{p}"] == np.arange(N)).all()
S_IDS = np.unique(sess); ROWS = {s: np.flatnonzero(sess == s) for s in S_IDS}


def cz(X):
    out = np.zeros_like(X, dtype=np.float64)
    for r in ROWS.values():
        out[r] = O.causal_z(X[r].astype(np.float64))
    return out


F = REPO + "/runs/sys2_feats.npz"
if not os.path.exists(F):
    TAN = {}
    for q in PEOPLE:  # causal recentring with the other people's mean as prior (fold-specific), all rows
        means = [O.recentred(r)[1] for s, r in ROWS.items() if O.PER[r[0]] != q]; prior = [np.mean([m[b] for m in means], 0) for b in range(len(O.BANDS))]
        T = np.zeros((N, 9 * 861), np.float32)
        for r in ROWS.values():
            T[r] = O.tang(O.recentred(r, prior)[0])
        TAN[q] = T; print("tangent fold", q, flush=True)
    BP = cz(np.log(np.stack([np.einsum("ncc->nc", C) for C in O.CV], 2)).reshape(N, -1)); SL = cz(O.SL); GE = cz(Z[f"GE|{p}"])
    np.savez(F, BP=BP, SL=SL, GE=GE, **{f"T_{q}": TAN[q] for q in PEOPLE})
FF = np.load(F)


class Net(torch.nn.Module):
    def __init__(s, d):
        super().__init__(); s.f = torch.nn.Sequential(torch.nn.Linear(d, 512), torch.nn.GELU(), torch.nn.Dropout(0.5), torch.nn.Linear(512, 128), torch.nn.GELU()); s.h = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(128, 3))

    def forward(s, x):
        e = s.f(x); return s.h(e), e


def fold(q, seed):
    te = O.PER == q; tr = ~te; T = FF[f"T_{q}"].astype(np.float64); T = (T - T[tr].mean(0)) / (T[tr].std(0) + 1e-6)
    pt = PCA(256, random_state=0).fit(T[tr]); ps = PCA(64, random_state=0).fit(FF["SL"][tr])
    X = np.c_[pt.transform(T), ps.transform(FF["SL"]), FF["BP"], FF["GE"]]; X = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6); X = np.clip(X, -6, 6)
    torch.manual_seed(seed); rng = np.random.default_rng(seed); net = Net(X.shape[1]); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    Xt, yt, it = torch.tensor(X[tr], dtype=torch.float32), torch.tensor(y[tr]), np.flatnonzero(tr); pers = O.PER[it]
    for ep in range(60):
        perm = rng.permutation(len(it))
        for b in range(0, len(perm), 128):
            j = perm[b:b + 128]; k = rng.permutation(len(j)); lam = float(rng.beta(0.4, 0.4)); lam = max(lam, 1 - lam)
            xb = lam * Xt[j] + (1 - lam) * Xt[j][k]; lo, _ = net(xb)
            loss = lam * torch.nn.functional.cross_entropy(lo, yt[j], label_smoothing=0.1) + (1 - lam) * torch.nn.functional.cross_entropy(lo, yt[j][k], label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        lo, e = net(torch.tensor(X, dtype=torch.float32))
    return torch.softmax(lo, 1).numpy(), e.numpy()


P = np.zeros((N, 3)); EMB = np.zeros((N, 128)); SEEDS = 5
for q in PEOPLE:
    te = O.PER == q; es = []
    for seed in range(SEEDS):
        pr, e = fold(q, seed); P[te] += pr[te] / SEEDS; es.append(e)
    E = np.concatenate(es, 1); pe = PCA(128, random_state=0).fit(E[~te]); EMB[te] = pe.transform(E)[te]; print(f"fold {q}: S2 alone {G.scores(P, p)[PEOPLE.index(q)]:.1f}", flush=True)
np.savez(REPO + "/runs/sys2_out.npz", P=P, EMB=EMB)
rng = np.random.default_rng(0)
boot = lambda d: f"{d.mean():+.2f} [{np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], 10):+.2f}, {np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], 90):+.2f}]"  # noqa: E731
a = G.scores(P, p); Pb = G.rebal(P, sess, 1.15); b = G.scores(Pb, p); L, tt = G.tsem_lik(Pb, sess, EMB)
print(f"S2 alone {a.mean():.2f} | + rebalancing {b.mean():.2f} | " + " ".join(f"{x:.0f}" for x in a))
for lam in (0.15, 0.3, 0.5):
    c = G.scores(G.apply_lam(Pb, L, tt, lam), p); print(f"S2 + rebalancing + TSEM in its embedding, lambda {lam}: {c.mean():.2f} | " + " ".join(f"{x:.0f}" for x in c))
S2s = G.apply_lam(Pb, L, tt, 0.3)
for bn in env.BASES:
    C = A.system(bn, A.FAMS)["out"][p]; ref = G.scores(C, p)
    for w, src in ((0.5, S2s), (0.25, S2s), (0.5, P)):
        Lx = np.log(np.clip(C, 1e-9, 1)) + w * np.log(np.clip(src, 1e-3, 1)); e = np.exp(Lx - Lx.max(1, keepdims=True)); d = G.scores(e / e.sum(1, keepdims=True), p) - ref
        print(f"  {bn}: chain {ref.mean():.2f} | product with S2{' staged' if src is S2s else ' raw'} ^{w}: {boot(d)} people >= 0 {(d >= 0).sum()}/10", flush=True)
