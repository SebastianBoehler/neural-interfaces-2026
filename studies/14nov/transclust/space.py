"""BALANCED TRANSDUCTIVE CLUSTERING (ledger s259), order-free stage. Per held-out person: features = filter-bank tangent
space of the EA-aligned band covariances (classic_feat.npz; same six bands as classic.py a1), each session recentred at its
own whole-session Riemannian mean (label-free); projection = standardise -> PCA(PCA_DIM) -> shrinkage LDA (2 directions)
fitted on the nine OTHER people's labelled, recentred rows. Per session of the held-out person: 3 clusters in that space,
each run's class balance as a constraint (balanced k-means: per-run exact balanced assignment by Hungarian on replicated
columns; centroids shared over the session). RESTARTS k-means++ starts, picked by within-cluster loss, never by labels.
Soft posterior: Sinkhorn plan (uniform marginals per run) at the chosen centroids, row-normalised.
Balance (amendment, 14 Nov 18:19): PRIMARY arm 'sess' = uniform class marginals over the session's windows (no run ids needed);
oracle arm 'run' = per-run marginals with TRUE run ids. Output runs/transclust/clusters.npz: 'Z' (N,2) projected rows,
'hard_<arm>' (N,) cluster ids, 'soft_<arm>' (N,3) cluster posteriors, 'loss|<arm>|<session>'.
Usage: python studies/14nov/transclust/space.py [PERSON ...]   (no argument: all ten)"""
import sys, time
from itertools import permutations
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "personal_ceiling"))
import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from pyriemann.tangentspace import TangentSpace  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
import classic as CL  # noqa: E402
import splits as S  # noqa: E402

OUT = S.REPO / "runs/transclust"; PCA_DIM = 200; RESTARTS = 20; ITERS = 30; EPS = 0.1; K = 3; ARMS = ("sess", "run")


def tangent(m, F):
    """-> (N, 6 * 861) tangent features, reference = each session's Riemannian mean per band"""
    Z = []
    for lo, hi in CL.TSB:
        C = F[f"cov{lo}_{hi}"]; z = None
        for s in S.sessions(m):
            r = np.flatnonzero(m["sess"] == s); t = TangentSpace("riemann").fit(C[r]).transform(C[r])
            if z is None:
                z = np.zeros((len(C), t.shape[1]))
            z[r] = t
        Z.append(z)
    return np.concatenate(Z, 1)


def balanced_assign(D, rows):
    """exact balanced hard assignment of each run's windows to K clusters (sizes differ by at most one)"""
    lab = np.zeros(len(D), int)
    for r in np.unique(rows):
        q = np.flatnonzero(rows == r); cap = int(np.ceil(len(q) / K)); cost = np.repeat(D[q], cap, axis=1)
        a, b = linear_sum_assignment(cost); lab[q[a]] = b // cap
    return lab


def sinkhorn(D, rows):
    """per-run plan with uniform marginals; epsilon = EPS x median cost; returns row-normalised plan"""
    P = np.zeros_like(D); sc = EPS * np.median(D)
    for r in np.unique(rows):
        q = np.flatnonzero(rows == r); Kq = np.exp(-(D[q] - D[q].min(1, keepdims=True)) / sc) + 1e-300; u = np.ones(len(q)); v = np.ones(K)
        for _ in range(200):
            u = (1 / len(q)) / (Kq @ v); v = (1 / K) / (Kq.T @ u)
        T = u[:, None] * Kq * v[None]; P[q] = T / T.sum(1, keepdims=True)
    return P


def cluster(Z, rows, rng):
    """balanced k-means, best of RESTARTS by within-cluster loss -> (hard labels, centroids, loss)"""
    best = None
    for _ in range(RESTARTS):
        c = [Z[rng.integers(len(Z))]]
        for _ in range(K - 1):
            d = np.min([((Z - x) ** 2).sum(1) for x in c], 0); c.append(Z[rng.choice(len(Z), p=d / d.sum())])
        mu = np.array(c); lab = None
        for _ in range(ITERS):
            D = ((Z[:, None] - mu[None]) ** 2).sum(-1); new = balanced_assign(D, rows)
            if lab is not None and (new == lab).all():
                break
            lab = new; mu = np.array([Z[lab == k].mean(0) for k in range(K)])
        loss = ((Z - mu[lab]) ** 2).sum()
        if best is None or loss < best[2]:
            best = (lab, mu, loss)
    return best


def main():
    m = S.meta(); F = CL.features(m); people = sys.argv[1:] or S.PEOPLE; OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    f = OUT / "clusters.npz"; res = {k: v for k, v in np.load(f).items()} if f.exists() and len(sys.argv) > 1 else {}
    N = len(m["y"]); Zp = tangent(m, F); print(f"tangent {Zp.shape} {time.time() - t0:.0f}s", flush=True)
    res.setdefault("Z", np.zeros((N, 2)))
    for a in ARMS:
        res.setdefault(f"hard_{a}", np.full(N, -1)); res.setdefault(f"soft_{a}", np.zeros((N, K)))
    for p in people:
        tr = np.flatnonzero(m["person"] != p); sc = StandardScaler().fit(Zp[tr]); pca = PCA(PCA_DIM, random_state=0).fit(sc.transform(Zp[tr]))
        lda = LDA(solver="eigen", shrinkage="auto").fit(pca.transform(sc.transform(Zp[tr])), m["y"][tr])
        for s in [x for x in S.sessions(m) if x.startswith(p)]:
            r = np.flatnonzero(m["sess"] == s); Z = lda.transform(pca.transform(sc.transform(Zp[r])))[:, :2]; res["Z"][r] = Z
            for a in ARMS:
                rows = np.zeros(len(r), int) if a == "sess" else m["run"][r]
                lab, mu, loss = cluster(Z, rows, np.random.default_rng(0)); D = ((Z[:, None] - mu[None]) ** 2).sum(-1)
                res[f"hard_{a}"][r] = lab; res[f"soft_{a}"][r] = sinkhorn(D, rows); res[f"loss|{a}|{s}"] = np.array(loss)
                pur = max(np.mean(np.array(pm)[lab] == m["y"][r]) for pm in permutations(range(K)))
                print(f"{s} {a} n {len(r)} loss {loss:.1f} best-permutation purity (diagnostic only) {pur:.3f} {time.time() - t0:.0f}s", flush=True)
    np.savez(f, **res); print("saved", f)


if __name__ == "__main__":
    main()
