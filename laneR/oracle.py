"""Lane R step 1 (c): oracle within-person models trained on the enrolment visit (registered in evidence/wave_20nov/laneR_onesession.md).
O1 person-only; O2 = enrolment visit (weight 3) + all other people's visits (each EA-referenced to its own mean covariance).
python laneR/oracle.py OUT.json"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.linalg import eigh, fractional_matrix_power
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
BANDS = [(8, 13), (13, 20), (20, 30)]
SOS = [butter(4, b, "bandpass", fs=120, output="sos") for b in BANDS]
ba = lambda y, p: 100 * float(np.mean([(p[y == c] == c).mean() for c in range(3)]))


def covs(X):  # (n, nb, 41, 41)
    X = np.nan_to_num(np.asarray(X, np.float64))
    out = []
    for s in SOS:
        F = sosfiltfilt(s, X, axis=-1)
        C = np.einsum("nct,ndt->ncd", F, F) / F.shape[-1]
        out.append(C + 1e-3 * np.trace(C, axis1=1, axis2=2)[:, None, None] / 41 * np.eye(41))
    return np.stack(out, 1)


def isqrt(C):
    w, V = np.linalg.eigh(C); return (V / np.sqrt(w)) @ V.T


def feats(Cs, ref):  # EA with the reference mean covariance, tangent at identity
    iu = np.triu_indices(41)
    out = []
    for b in range(Cs.shape[1]):
        W = isqrt(ref[b])
        T = np.einsum("ij,njk,kl->nil", W, Cs[:, b], W)
        w, V = np.linalg.eigh(T)
        L = np.einsum("nij,nj,nkj->nik", V, np.log(w), V)
        out.append(L[:, iu[0], iu[1]])
    return np.concatenate(out, 1)


def block(X):  # features of one (person, visit) block, referenced to its own mean covariance
    C = covs(X); return feats(C, C.mean(0))


def fit_predict(Xtr, ytr, wtr, Xte):
    mu, sd = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-6)
    m = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit((Xtr - mu) / sd, ytr, sample_weight=wtr)
    return m.predict((Xte - mu) / sd)


def run(d, people, enrol_vis, test_vis, subj, sess, y, X):
    cache = {}
    for p in people:
        for v in sorted(set(sess[subj == p])):
            r = np.flatnonzero((subj == p) & (sess == v)); cache[(p, v)] = (block(X[r]), y[r], r)
    res = {}
    for p in people:
        Fe, ye, _ = cache[(p, enrol_vis)]
        for v in test_vis:
            if (p, v) not in cache: continue
            _, yt, rt = cache[(p, v)]
            C = covs(X[rt]); Ft = feats(C, covs(X[cache[(p, enrol_vis)][2]]).mean(0))  # test windows referenced to the enrolment mean covariance
            o1 = fit_predict(Fe, ye, np.ones(len(ye)), Ft)
            others = [(q, u) for (q, u) in cache if q != p]
            Fo = np.concatenate([cache[k][0] for k in others] + [Fe]); yo = np.concatenate([cache[k][1] for k in others] + [ye])
            wo = np.concatenate([np.ones(sum(len(cache[k][1]) for k in others)), 3 * np.ones(len(ye))])
            o2 = fit_predict(Fo, yo, wo, Ft)
            res[f"{p}|{v}"] = dict(O1=ba(yt, o1), O2=ba(yt, o2), n=len(yt))
            print(p, v, {k: round(x, 1) for k, x in res[f"{p}|{v}"].items() if k != "n"}, flush=True)
    return res


def main(dest):
    out = {}
    z = np.load(ROOT / "data/proteus_windows.npz", allow_pickle=True)
    X, y, subj, sess = z["X"], z["y"], z["subject"], z["session"].astype(str)
    out["proteus"] = run(None, ["Dreyer2026Proteus/P04", "Dreyer2026Proteus/P10"], "1", ["2", "3"], subj, sess, y, X)
    s = np.load(ROOT / "data/scherer2015/windows.npz", allow_pickle=True)
    subj, sess = s["subject"], s["session"].astype(str)
    out["scherer"] = run(None, sorted(set(subj)), "1", ["2"], subj, sess, s["y"], s["X"])
    json.dump(out, open(dest, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
