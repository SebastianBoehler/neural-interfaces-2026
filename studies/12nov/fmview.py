"""FMVIEW (ledger s71): frozen foundation-model embeddings as a view of the session stage (new people, grid chain without
routing). Usage: python studies/12nov/fmview.py runs/fmemb/<model>.npz [key=emb] [k=64]"""
import os, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay")
import grid as G  # noqa: E402
import score as SC  # noqa: E402
kw = dict(a.split("=") for a in sys.argv[2:]); KEY = kw.get("key", "emb"); K = int(kw.get("k", 64)); p = "all_loso"; Z = G.Z; rng = np.random.default_rng(0)
F = np.load(sys.argv[1]); E = F[KEY].astype(np.float64); sess, per, y = Z[f"sess|{p}"], Z[f"person|{p}"], Z[f"y|{p}"].astype(int)
assert E.shape[0] == len(y) and np.isfinite(E).all() and (F["y"].astype(int) == y).all(), "embedding rows are not the replay rows"
assert np.unique(np.round(E, 6), axis=0).shape[0] > 0.99 * len(E), "embeddings repeat across windows"


def causal_z(X, m0=10.0):
    n = np.arange(len(X))[:, None]; cs = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, 0)[:-1]]); c2 = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X ** 2, 0)[:-1]])
    mu = cs / (n + m0); var = (c2 + m0) / (n + m0) - mu ** 2; return (X - mu) / np.sqrt(np.maximum(var, 1e-4))


def boot(d):
    b = [d[rng.integers(0, 10, 10)].mean() for _ in range(4000)]; return f"{d.mean():+.2f} [{np.percentile(b, 10):+.2f}, {np.percentile(b, 90):+.2f}]"


V = np.zeros((len(y), K)); probe = np.zeros((len(y), 3))  # components fitted on the other nine people; probe on causally standardised components
for q in G.PEOPLE:
    te = per == q; mu, sd = E[~te].mean(0), np.maximum(E[~te].std(0), 1e-9); pca = PCA(K, random_state=0).fit((E[~te] - mu) / sd)
    C = pca.transform((E - mu) / sd); C = C / C[~te].std(0); V[te] = C[te]; Cz = np.zeros_like(C)
    for s in np.unique(sess):
        r = np.flatnonzero(sess == s); Cz[r] = causal_z(C[r])
    probe[te] = LogisticRegression(C=0.1, max_iter=2000).fit(Cz[~te], y[~te]).predict_proba(Cz[te])
pr = G.scores(probe, p); print(f"{os.path.basename(sys.argv[1])} [{KEY}] d {E.shape[1]} -> {K} components | linear probe, leave one person out: {pr.mean():.1f} | " + " ".join(f"{v:.0f}" for v in pr), flush=True)
for base in ("B2_pool", "B2_s345"):
    pre = G.pre_tsem(base, p, *G.SHIP); L, tt = G.tsem_lik(pre, sess, Z[f"pc128|{p}"]); chain = G.apply_lam(pre, L, tt, 0.15); ref = G.scores(chain, p)
    print(f"{base}: chain {ref.mean():.2f} (P04 {ref[3]:.1f})", flush=True)
    L2, t2 = G.tsem_lik(chain, sess, V)
    for lam in (0.05, 0.10, 0.15):
        d = G.scores(G.apply_lam(chain, L2, t2, lam), p) - ref; print(f"  second TSEM through the embedding view, lambda {lam}: new {boot(d)} | P04 {d[3]:+.1f} | people >= 0: {(d >= 0).sum()}/10")
    L3, t3 = G.tsem_lik(pre, sess, np.concatenate([Z[f"pc128|{p}"], V[:, :32]], 1)); d = G.scores(G.apply_lam(pre, L3, t3, 0.15), p) - ref
    print(f"  first TSEM with 32 embedding components appended: new {boot(d)} | P04 {d[3]:+.1f}")
    Lp, tp = G.tsem_lik(pre, sess, V); d = G.scores(G.apply_lam(pre, Lp, tp, 0.15), p) - ref; print(f"  TSEM through the embedding view only: new {boot(d)} | P04 {d[3]:+.1f}", flush=True)
    d = G.scores(np.exp(np.log(np.clip(chain, 1e-9, 1)) + 0.3 * np.log(np.clip(probe, 1e-3, 1))), p) - ref; print(f"  probe as a final member (exponent 0.3): new {boot(d)} | P04 {d[3]:+.1f}", flush=True)
