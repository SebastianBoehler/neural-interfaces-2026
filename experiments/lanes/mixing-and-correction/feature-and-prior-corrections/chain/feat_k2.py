"""Lane K2 features and source-only readouts (ideas 4 and 6 of research2_codex.md). Registered form: evidence/wave_20nov/laneK2_ideas456.md.

usage: feat_k2.py FEAT_NPZ OUT_NPZ     (X (N,41,480) at 120 Hz; labels only enter the leave-one-person-out source fits)
Writes ll4c / ll4m / ll4p{0,1,2} (N,3) class log-likelihoods of the cross-view models, x6g / x6a (N,3) log-probabilities of the gen / abs readouts, diagnostics."""
import json
import sys
from time import perf_counter

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression

FS = 120
BANDS = dict(theta=(4, 8), alpha=(8, 12), mu=(8, 13), beta=(13, 30), broad=(4, 40))
ROI = dict(central="Cz C1 C2 C3 C4 CPz CP1 CP2", frontal="Fz F1 F2 F3 F4 AF3 AF4 FCz", lateral="C5 C6 CP5 CP6 TP7 TP8 P5 P6")
LAM, SHR, RIDGE_EPS, EPS_D = 10.0, 0.5, 1e-3, 0.001


def bandpass(X, lo, hi):
    return sosfiltfilt(butter(4, (lo, hi), btype="band", fs=FS, output="sos"), X, axis=-1)


def roi_index(names):
    names = [str(n) for n in names]
    return [[names.index(c) for c in v.split()] for v in ROI.values()]


def view_vectors(X, names):
    """(N,18): log band power (theta, alpha, beta) x three ROIs of the first 240 and last 240 samples -> v1 (N,9), v2 (N,9)."""
    idx = roi_index(names)
    out = []
    for h in (slice(0, 240), slice(240, 480)):
        cols = []
        for b in ("theta", "alpha", "beta"):
            P = (bandpass(X, *BANDS[b])[..., h] ** 2).mean(-1)
            cols += [np.log(P[:, ix].mean(1) + 1e-12) for ix in idx]
        out.append(np.stack(cols, 1))
    return out


def covs(Xf):
    Xf = Xf - Xf.mean(-1, keepdims=True)
    return Xf @ Xf.transpose(0, 2, 1) / Xf.shape[-1]


def gen_spectra(N, D, kind):
    """sorted log eigenvalues, 4 low + 4 high, of the whitened (N + 0.001 D) (kind gen, invariant) or of N + 0.001 D (kind abs)."""
    out = np.empty((len(N), 8))
    for i in range(len(N)):
        M = N[i] + EPS_D * D[i]
        if kind == "gen":
            d, U = np.linalg.eigh(D[i]); keep = d > 1e-8 * d.max(); W = U[:, keep] / np.sqrt(d[keep])
            M = W.T @ M @ W
        e = np.log(np.linalg.eigvalsh(M)); out[i] = np.concatenate([e[:4], e[-4:]])
    return out


def spectra_from_cov(Nmu, Nbeta, D):
    return {k: np.concatenate([gen_spectra(Nmu, D, k), gen_spectra(Nbeta, D, k)], 1) for k in ("gen", "abs")}


def invariance_gate(X, rng):
    """gen unchanged, abs changed, under a fixed invertible sensor mixing applied to real windows (covariance level)."""
    ix = rng.choice(len(X), 200, replace=False)
    Xf = {b: bandpass(X[ix], *BANDS[b]) for b in ("mu", "beta", "broad")}
    base = spectra_from_cov(covs(Xf["mu"]), covs(Xf["beta"]), covs(Xf["broad"]))
    while True:
        M = rng.normal(size=(41, 41)) + 3 * np.eye(41)
        if np.linalg.cond(M) < 20:
            break
    mixed = spectra_from_cov(*(covs(np.einsum("ij,njt->nit", M, Xf[b])) for b in ("mu", "beta", "broad")))
    g = float(np.max(np.abs(mixed["gen"] - base["gen"]) / (1 + np.abs(base["gen"]))))
    a = float(np.max(np.abs(mixed["abs"] - base["abs"]) / (1 + np.abs(base["abs"]))))
    assert g < 1e-6 and a > 1e-3, (g, a)
    return dict(gen_max_rel_diff=g, abs_max_rel_diff=a, cond=float(np.linalg.cond(M)))


def gauss_ll(Y, mean, S):
    L = np.linalg.cholesky(S); z = np.linalg.solve(L, (Y - mean).T)
    return -0.5 * (z ** 2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * Y.shape[1] * np.log(2 * np.pi)


def ridge(A, B, prior):
    return np.linalg.solve(A.T @ A + LAM * np.eye(A.shape[1]), A.T @ (B - A @ prior)) + prior


def cross_view_models(V1, V2, y, groups, tr, conditional=True, permute=None):
    """fit on rows tr; returns a function (v1, v2) -> (n,3) class log-likelihoods. conditional: v2 | v1 with class maps; else class marginal of v2."""
    m1, s1, m2, s2 = V1[tr].mean(0), V1[tr].std(0) + 1e-9, V2[tr].mean(0), V2[tr].std(0) + 1e-9
    A, B, yt = (V1[tr] - m1) / s1, (V2[tr] - m2) / s2, y[tr]
    if permute is not None:  # v2 swapped within (person, session, interface) and class
        rng = np.random.default_rng(permute); B = B.copy(); g = groups[tr]
        for key in np.unique(g):
            for k in range(3):
                r = np.flatnonzero((g == key) & (yt == k)); B[r] = B[rng.permutation(r)]
    A1 = np.concatenate([A, np.ones((len(A), 1))], 1)
    if conditional:
        B0 = ridge(A1, B, np.zeros((A1.shape[1], B.shape[1])))
        maps = [ridge(A1[yt == k], B[yt == k], B0) for k in range(3)]
        res = [B[yt == k] - A1[yt == k] @ maps[k] for k in range(3)]
    else:
        maps = [np.zeros((A1.shape[1], B.shape[1])) for _ in range(3)]
        for k in range(3):
            maps[k][-1] = B[yt == k].mean(0)
        res = [B[yt == k] - B[yt == k].mean(0) for k in range(3)]
    pooled = np.cov(np.concatenate(res).T)
    S = [SHR * np.cov(r.T) + (1 - SHR) * pooled + RIDGE_EPS * np.eye(B.shape[1]) for r in res]

    def ll(v1, v2):
        a = np.concatenate([(v1 - m1) / s1, np.ones((len(v1), 1))], 1); b = (v2 - m2) / s2
        return np.stack([gauss_ll(b, a @ maps[k], S[k]) for k in range(3)], 1)
    return ll


def readout(F, y, tr):
    m, s = F[tr].mean(0), F[tr].std(0) + 1e-9
    clf = LogisticRegression(C=1.0, max_iter=1000).fit((F[tr] - m) / s, y[tr])
    return lambda G: clf.predict_log_proba((G - m) / s)


def main(feat, out):
    z = np.load(feat, allow_pickle=False)
    X, y, names = z["X"].astype(np.float64), z["y"], z["ch_names"]
    person = np.asarray([p.rsplit("/", 1)[-1] for p in z["subject"]])
    groups = np.asarray([f"{p}|{s}|{i}" for p, s, i in zip(person, z["session"], z["interface"])])
    t0 = perf_counter(); V1, V2 = view_vectors(X, names); t_view = perf_counter() - t0
    gate = invariance_gate(X, np.random.default_rng(11)); print("invariance gate", gate, flush=True)
    t0 = perf_counter()
    Xf = {b: bandpass(X, *BANDS[b]) for b in ("mu", "beta", "broad")}
    S = spectra_from_cov(covs(Xf["mu"]), covs(Xf["beta"]), covs(Xf["broad"])); t_spec = perf_counter() - t0
    arrays = {k: np.zeros((len(y), 3)) for k in ("ll4c", "ll4m", "ll4p0", "ll4p1", "ll4p2", "x6g", "x6a")}
    diag = dict(gate=gate, ms_per_window_views=1e3 * t_view / len(y), ms_per_window_spectra=1e3 * t_spec / len(y), per_person={})
    t_ll = 0.0
    for p in sorted(set(person)):
        te, tr = person == p, person != p
        t0 = perf_counter(); f = cross_view_models(V1, V2, y, groups, tr)
        arrays["ll4c"][te] = f(V1[te], V2[te]); t_ll += perf_counter() - t0
        arrays["ll4m"][te] = cross_view_models(V1, V2, y, groups, tr, conditional=False)(V1[te], V2[te])
        for s in range(3):
            arrays[f"ll4p{s}"][te] = cross_view_models(V1, V2, y, groups, tr, permute=s)(V1[te], V2[te])
        arrays["x6g"][te] = readout(S["gen"], y, tr)(S["gen"][te]); arrays["x6a"][te] = readout(S["abs"], y, tr)(S["abs"][te])
        # diagnostics: 3-class accuracy of each readout, and 2-way accuracy of cond / marg / perm among the true class and a random wrong class
        rng = np.random.default_rng(5); wrong = (y[te] + rng.integers(1, 3, te.sum())) % 3; idx = np.arange(te.sum())
        d = {"acc_x6g": float((arrays["x6g"][te].argmax(1) == y[te]).mean()), "acc_x6a": float((arrays["x6a"][te].argmax(1) == y[te]).mean())}
        for k in ("ll4c", "ll4m", "ll4p0"):
            L = arrays[k][te]; d["pair_acc_" + k] = float((L[idx, y[te]] > L[idx, wrong]).mean())
        diag["per_person"][p] = d
        print(p, d, flush=True)
    diag["ms_per_window_ll4"] = 1e3 * t_ll / len(y)  # includes the per-person fit
    for k in ("acc_x6g", "acc_x6a", "pair_acc_ll4c", "pair_acc_ll4m", "pair_acc_ll4p0"):
        diag["mean_" + k] = float(np.mean([v[k] for v in diag["per_person"].values()]))
    assert all(np.isfinite(a).all() for a in arrays.values())
    np.savez(out, y=y, person=person, **arrays)
    json.dump(diag, open(out.replace(".npz", ".json"), "w"), indent=1)
    print({k: v for k, v in diag.items() if k != "per_person"})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
