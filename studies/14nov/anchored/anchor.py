"""POPULATION-ANCHORED PERSONAL CLASSIC MODELS (ledger s254): features, population fits and the anchored members.
Space: s246's recentred features (classic_feat.npz, whole-session EA, non-causal) in the tangent space at the identity
(log-Euclidean; a common space across people after EA). Per held-out person one scaler per feature set, fit on the other
nine people's rows (all sessions, true labels: public data).
  L   multinomial LR on the six-band tangent features, loss weighted, penalty ||W - W_pop||^2 / 2C (intercept free);
      grid CS from 'pop' (W_pop, b_pop) through C to 'pers' (anchor 0, C 0.1 = s246's a1 form in this space)
  D1  LDA on the 8-30 Hz tangent features, D2 LDA on the slow waveform: class means, covariance and priors mixed with the
      population's by pseudo-count k (KS: inf = pop ... 0 = pers = classic.wlda)"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO / "studies/14nov/personal_ceiling"), str(REPO / "studies/14nov/causalpers")]
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from sklearn.covariance import ledoit_wolf  # noqa: E402
import classic as CL  # noqa: E402
import splits as S  # noqa: E402
from fast import logvec  # noqa: E402

OUT = REPO / "runs/anchored"
CS = ["pop", 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, "pers"]
AL = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]  # amendment: anchor a * W_pop at the personal C 0.1
KS = [np.inf, 1000.0, 300.0, 100.0, 30.0, 10.0, 3.0, 0.0]
SETS = ("L", "D1", "D2")


def feats(m):
    F = CL.features(m)
    return {"L": np.concatenate([logvec(F[f"cov{lo}_{hi}"]) for lo, hi in CL.TSB], 1), "D1": logvec(F["cov8_30"]), "D2": F["slow"]}


def lr_fit(X, y, s, C, anchor=None, x0=None):
    """min sum_i s_i CE_i + ||W - anchor||^2 / 2C over W (d, 3), b (3) (b unpenalised); -> W, b"""
    n, d = X.shape; Y = np.eye(3)[y]; A = np.zeros((d, 3)) if anchor is None else anchor

    def f(th):
        W = th[:3 * d].reshape(d, 3); Z = X @ W + th[3 * d:]; Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); se = E.sum(1)
        P = E / se[:, None]; D = W - A; G = (P - Y) * s[:, None]
        return float(np.sum(s * (np.log(se) - (Z * Y).sum(1))) + (D * D).sum() / (2 * C)), np.concatenate([(X.T @ G + D / C).ravel(), G.sum(0)])
    x0 = np.concatenate([A.ravel(), np.zeros(3)]) if x0 is None else x0
    r = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 20000, "gtol": 1e-7, "ftol": 1e-13})
    return r.x[:3 * d].reshape(d, 3), r.x[3 * d:]


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)


def population(Fs, m, p, W=None):
    """population fits for held-out person p (true labels of the other nine people): scalers, LDA moments, LR (fit
    unless W = cached (W, b) is given)"""
    r = m["person"] != p; y = m["y"][r]; pop = {}
    for k in SETS:
        mu, sd = Fs[k][r].mean(0), Fs[k][r].std(0) + 1e-12; pop[k + "_sc"] = (mu, sd); X = (Fs[k][r] - mu) / sd
        if k == "L":
            pop["W"], pop["b"] = lr_fit(X, y, np.ones(len(y)), 0.1) if W is None else W
        else:
            M = np.stack([X[y == c].mean(0) for c in range(3)]); pop[k + "_M"] = M
            pop[k + "_S"] = ledoit_wolf(X - M[y], assume_centered=True)[0]; pop[k + "_pri"] = np.bincount(y, minlength=3) / len(y)
    return pop


def lr_grid(Xtr, y, s, Xte, pop):
    """-> (len(CS), len(te), 3), warm-started from the pop end toward weaker anchoring"""
    out = []; x0 = None
    for C in CS:
        if C == "pop":
            W, b = pop["W"], pop["b"]
        elif C == "pers":
            W, b = lr_fit(Xtr, y, s, 0.1)
        else:
            W, b = lr_fit(Xtr, y, s, C, pop["W"], x0); x0 = np.concatenate([W.ravel(), b])
        out.append(softmax(Xte @ W + b))
    return np.stack(out)


def lra_grid(X, y, s, te, pop):
    """amendment (La): (len(AL), len(te), 3), penalty ||W - a W_pop||^2 / 2(0.1); X / te scaled"""
    out = []; x0 = None
    for a in AL:
        W, b = lr_fit(X, y, s, 0.1, a * pop["W"], x0); x0 = np.concatenate([W.ravel(), b]); out.append(softmax(te @ W + b))
    return np.stack(out)


def lda_grid(Xtr, y, s, Xte, M0, S0, p0):
    """-> (len(KS), len(te), 3): means (n_c mu_c + k M0) / (n_c + k), cov (N S_LW + k S0) / (N + k), priors alike"""
    w = s / s.mean(); N = w.sum(); nc = np.array([w[y == c].sum() for c in range(3)])
    mu = np.stack([np.average(Xtr[y == c], 0, w[y == c]) if nc[c] > 0 else np.zeros(Xtr.shape[1]) for c in range(3)])
    Sg = ledoit_wolf((Xtr - mu[y]) * np.sqrt(w)[:, None], assume_centered=True)[0]; out = []
    for k in KS:
        if np.isinf(k):
            M, Sig, pri = M0, S0, p0
        else:
            M = (nc[:, None] * mu + k * M0) / (nc + k + 1e-300)[:, None]; Sig = (N * Sg + k * S0) / (N + k); pri = (nc + k * p0) / (N + k)
        B = np.linalg.solve(Sig, M.T); out.append(softmax(Xte @ B - 0.5 * np.sum(M.T * B, 0) + np.log(np.maximum(pri, 1e-12))))
    return np.stack(out)


def grids(Fs, pop, y, s, tr, te, alpha=False):
    """-> {'L': (10, n, 3), 'D1': (8, n, 3), 'D2': (8, n, 3)} or, alpha, {'La': (8, n, 3)}; y, s labels / weights on tr only"""
    sc = {k: (lambda X, k=k: (X - pop[k + "_sc"][0]) / pop[k + "_sc"][1]) for k in SETS}; yt = y[tr]
    out = {"L": lr_grid(sc["L"](Fs["L"][tr]), yt, s, sc["L"](Fs["L"][te]), pop)}
    if alpha:
        return {"La": lra_grid(sc["L"](Fs["L"][tr]), yt, s, sc["L"](Fs["L"][te]), pop)}
    for k in ("D1", "D2"):
        out[k] = lda_grid(sc[k](Fs[k][tr]), yt, s, sc[k](Fs[k][te]), pop[k + "_M"], pop[k + "_S"], pop[k + "_pri"])
    for k, v in out.items():
        assert np.isfinite(v).all() and np.allclose(v.sum(-1), 1), k
    return out
