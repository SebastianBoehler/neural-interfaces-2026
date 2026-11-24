"""Lane Y #5 label-free residual rotation: per cell and band, align the eigenbasis of the within-cell covariance-fluctuation matrix
M = mean_n (T_n - I)^2 (T = EA-whitened window covariance) to the source pool's (mean of the other people's cells), rotation Q = polar(V_src V_cell^T)
shrunk toward identity by Cayley-log scaling alpha (alpha 0 = identity: must reproduce the base member exactly). No labels at test time.
python labelfree_resrot.py OUT.json"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "laneR"))
from oracle import covs  # noqa: E402
from oracle_resrot import white, tangent_np, ba  # noqa: E402
from scipy.linalg import logm, expm  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
ROOT = Path(__file__).resolve().parents[1]; ALPHAS = (0.0, 0.5, 1.0)


def fluct(T):  # (n,41,41) -> M
    D = T - np.eye(41); return np.einsum("nij,njk->ik", D, D) / len(T)


def basis(M):
    w, V = np.linalg.eigh(M); return V[:, ::-1]  # descending eigenvalues


def rot_to(Mc, Ms, alpha):
    Vc, Vs = basis(Mc), basis(Ms); U, _, Vt = np.linalg.svd(Vs @ Vc.T); Q = U @ Vt
    if alpha == 1.0: return Q
    L = np.real(logm(Q)); return expm(alpha * (L - L.T) / 2)  # skew part, scaled


def apply(T, Qs):
    return np.stack([np.einsum("ij,njk,lk->nil", Qs[b], T[:, b], Qs[b]) for b in range(T.shape[1])], 1)


def main(dest):
    z = np.load(ROOT / "data/proteus_windows.npz", allow_pickle=True)
    X, y, subj, sess, itf = z["X"], z["y"], z["subject"], z["session"], z["interface"]
    cells = sorted({(a, b, c) for a, b, c in zip(subj, sess, itf)}); Wt, Y, Mc = {}, {}, {}
    for c in cells:
        r = np.flatnonzero((subj == c[0]) & (sess == c[1]) & (itf == c[2])); Wt[c] = white(covs(X[r])); Y[c] = y[r]
        Mc[c] = [fluct(Wt[c][:, b]) for b in range(Wt[c].shape[1])]
    res = []
    for p in sorted(set(subj)):
        tr = [c for c in cells if c[0] != p]
        Ms = [np.mean([Mc[c][b] for c in tr], 0) for b in range(3)]
        Ftr = np.concatenate([tangent_np(Wt[c]) for c in tr]); ytr = np.concatenate([Y[c] for c in tr]); mu, sd = Ftr.mean(0), np.maximum(Ftr.std(0), 1e-6)
        m = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit((Ftr - mu) / sd, ytr)
        for c in [c for c in cells if c[0] == p]:
            row = dict(cell="|".join(map(str, c)))
            for a in ALPHAS:
                Qs = [rot_to(Mc[c][b], Ms[b], a) for b in range(3)]; T = apply(Wt[c], Qs)
                row[f"a{a}"] = ba(Y[c], m.predict((tangent_np(T) - mu) / sd))
            res.append(row); print(row, flush=True)
    s = {k: float(np.mean([r[k] for r in res])) for k in res[0] if k != "cell"}; print("MEAN", s); json.dump(dict(mean=s, cells=res), open(dest, "w"))


if __name__ == "__main__":
    main(sys.argv[1])
