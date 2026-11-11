"""WSP E9 (exploratory, added after E1 failed): block gains LEARNED on inner held-out people instead of set by a ratio.

For each fold, every training person is predicted by the inner head that never saw them; the logit is split into its
52 block contributions; gains g = c (1 + u) are fitted by cross-entropy on those rows with a ridge penalty lam |u|^2.
Usage: python experiments/wsp_gains.py CACHE_DIR
"""
import sys
import numpy as np
from scipy.optimize import minimize
import wsp_core as W

LAMS = (0.3, 1.0, 3.0)


def block_logits(f):
    Ls, ys = [], []
    for q, m in f["inner"].items():
        te, _ = W.inner_rows(f, q); Z = (W.F[te] - m["mean"]) / m["scale"]; C = m["coef"] - m["coef"].mean(0)
        Ls.append(np.stack([Z[:, a:b] @ C[:, a:b].T for a, b in W.SL], 1)); ys.append(W.Y[te])
    return np.concatenate(Ls), np.concatenate(ys)


def fit(L, y, lam):
    n, nb, _ = L.shape; Yh = np.eye(3)[y]

    def obj(th):
        c, u = th[0], th[1:]; g = c * (1 + u); z = np.einsum("nbk,b->nk", L, g); z -= z.max(1, keepdims=True)
        P = np.exp(z); P /= P.sum(1, keepdims=True); ce = -np.log(P[np.arange(n), y] + 1e-12).mean()
        dg = np.einsum("nk,nbk->b", P - Yh, L) / n
        return ce + lam * (u ** 2).mean(), np.concatenate([[dg @ (1 + u)], dg * c + 2 * lam * u / nb])
    r = minimize(obj, np.concatenate([[0.3], np.zeros(nb)]), jac=True, method="L-BFGS-B", bounds=[(1e-3, None)] + [(-1, None)] * nb)
    u = r.x[1:]; return (1 + u) / (1 + u).mean(), r.fun


if __name__ == "__main__":
    out = {k: {} for k in ["base"] + [f"E9_lam{l}" for l in LAMS]}
    for nm in W.FOLDS:
        f = W.load(nm); L, y = block_logits(f); out["base"][nm] = (f["rows"], f["proba"]); msg = []
        for lam in LAMS:
            g, ce = fit(L, y, lam); out[f"E9_lam{lam}"][nm] = (f["rows"], W.apply(f, W.e_scale(g[W.BLOCK])))
            msg.append(f"lam {lam}: CE {ce:.3f} gains min {g.min():.2f} max {g.max():.2f} slow {g[-1]:.2f}")
        print(nm, "|", " | ".join(msg), flush=True)
    np.savez(f"{sys.argv[1]}/edits_e9.npz", **{f"{k}|{nm}|{t}": v[i] for k, d in out.items() for nm, v in d.items() for i, t in enumerate(("rows", "proba"))})
