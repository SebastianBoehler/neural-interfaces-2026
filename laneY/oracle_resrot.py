"""Lane Y idea #5 oracle: label-informed near-identity Cayley rotation of the EA-whitened covariances per cell, frozen LOPO logistic member.
python laneY/oracle_resrot.py OUT.json  (CPU torch; a Slurm/capped job). Cross-fit: rotation fit on half a cell (labels), tested on the other half."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "laneR"))
from oracle import covs, isqrt  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IU = np.triu_indices(41)
LAMS = (0.3, 3.0)


def white(Cs):  # EA with the cell's own mean covariance -> (n, nb, 41, 41)
    ref = Cs.mean(0)
    return np.stack([np.einsum("ij,njk,kl->nil", isqrt(ref[b]), Cs[:, b], isqrt(ref[b])) for b in range(Cs.shape[1])], 1)


def tangent_np(T):
    out = []
    for b in range(T.shape[1]):
        w, V = np.linalg.eigh(T[:, b]); L = np.einsum("nij,nj,nkj->nik", V, np.log(np.maximum(w, 1e-9)), V); out.append(L[:, IU[0], IU[1]])
    return np.concatenate(out, 1)


def tangent_t(T):
    out = []
    for b in range(T.shape[1]):
        w, V = torch.linalg.eigh(T[:, b]); L = V @ torch.diag_embed(torch.log(w.clamp_min(1e-9))) @ V.transpose(-1, -2)
        out.append(L[:, IU[0], IU[1]])
    return torch.cat(out, 1)


def rotate(T, params):  # params (nb, 41*40/2); Cayley per band
    out = []
    for b in range(T.shape[1]):
        A = torch.zeros(41, 41, dtype=T.dtype); iu = torch.triu_indices(41, 41, 1); A[iu[0], iu[1]] = params[b]; A = A - A.T
        I = torch.eye(41, dtype=T.dtype); R = torch.linalg.solve(I - A, I + A)
        out.append(R @ T[:, b] @ R.T)
    return torch.stack(out, 1)


def ba(y, g): return 100 * np.mean([(g[y == c] == c).mean() for c in range(3) if (y == c).any()])


def fit_rot(T, y, mu, sd, W, bias, lam, steps=60):
    p = torch.zeros(T.shape[1], 820, dtype=T.dtype, requires_grad=True)
    opt = torch.optim.LBFGS([p], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")
    yt = torch.as_tensor(y); wcls = torch.as_tensor(1.0 / np.maximum(np.bincount(y, minlength=3), 1), dtype=T.dtype)[yt]
    def closure():
        opt.zero_grad(); f = (tangent_t(rotate(T, p)) - mu) / sd; lg = f @ W.T + bias
        l = (torch.nn.functional.cross_entropy(lg, yt, reduction="none") * wcls).sum() / wcls.sum() + lam * (p ** 2).sum(); l.backward(); return l
    opt.step(closure); return p.detach()


def main(dest):
    torch.set_num_threads(3)
    z = np.load(ROOT / "data/proteus_windows.npz", allow_pickle=True)
    X, y, subj, sess, itf = z["X"], z["y"], z["subject"], z["session"], z["interface"]
    cells = sorted({(a, b, c) for a, b, c in zip(subj, sess, itf)})
    Wt, F, Y = {}, {}, {}
    for c in cells:
        r = np.flatnonzero((subj == c[0]) & (sess == c[1]) & (itf == c[2])); Wt[c] = white(covs(X[r])); F[c] = tangent_np(Wt[c]); Y[c] = y[r]
    rng = np.random.default_rng(0); res = []
    for p in sorted(set(subj)):
        tr = [c for c in cells if c[0] != p]
        Ftr = np.concatenate([F[c] for c in tr]); ytr = np.concatenate([Y[c] for c in tr])
        mu, sd = Ftr.mean(0), np.maximum(Ftr.std(0), 1e-6)
        m = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit((Ftr - mu) / sd, ytr)
        W, b = torch.as_tensor(m.coef_), torch.as_tensor(m.intercept_); mu_t, sd_t = torch.as_tensor(mu), torch.as_tensor(sd)
        for c in [c for c in cells if c[0] == p]:
            n = len(Y[c]); perm = rng.permutation(n); halves = (perm[: n // 2], perm[n // 2:]); T = torch.as_tensor(Wt[c]); row = dict(cell="|".join(map(str, c)))
            g0 = m.predict((F[c] - mu) / sd); row["base"] = ba(Y[c], g0)
            for lam in LAMS:
                g = np.zeros(n, int)
                for a_, b_ in (halves, halves[::-1]):
                    pr = fit_rot(T[a_], Y[c][a_], mu_t, sd_t, W, b, lam)
                    f = (tangent_t(rotate(T[b_], pr)) - mu_t) / sd_t; g[b_] = (f @ W.T + b).argmax(1).numpy()
                row[f"rot_cf_lam{lam}"] = ba(Y[c], g)
            res.append(row); print(row, flush=True)
    summ = {k: float(np.mean([r[k] for r in res])) for k in res[0] if k != "cell"}
    print("MEAN", summ); json.dump(dict(mean=summ, cells=res), open(dest, "w"))


if __name__ == "__main__":
    main(sys.argv[1])
