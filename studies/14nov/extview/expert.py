"""External pairwise experts as SessionEM views (ledger s233). Trained on public Shin2017 people only (no PROTEUS label
is read here), applied to every PROTEUS window, written as runs/views_ext/<name>.npz for shufcombo's EXTRA_VIEW hook.

Montage: the Shin exports sit on the 41 PROTEUS channels but 35 of them are spherical-spline interpolations of Shin's
30-channel cap (6 exact: Pz Cz P3 P4 F3 F4), so the exported data live in a 29-dim subspace. Both corpora are CAR'd and
projected onto the top R (= 20) eigenvectors of the pooled Shin covariance (the part of the field Shin's cap resolves).
Features: bands BANDS, shrunk covariance, log-Euclidean tangent vectors; each session's mean is subtracted (Shin:
full-session mean over the pseudo-session; PROTEUS: causal running mean of the session's delivered windows, current
included, in each delivery order); z-scored with Shin's pooled sd; L2 logistic regression, C chosen by 5-fold
grouped-by-person CV on Shin.
Experts:  mivma  Shin2017A motor imagery (left + right) vs Shin2017B mental arithmetic   -> PROTEUS MI vs SUB
          mavr   Shin2017B mental arithmetic vs rest                                     -> PROTEUS SUB vs (MI, WORD)
Shin's A and B sessions alternate on one day (A 0/2/4, B 1/3/5); mivma's pseudo-session is A_k + B_k (k = pair).
View file keys: 'rec' (recording order) and '<scheme><seed>' for ORDERS, rows in replay row order (all_loso idx =
window index), values = log posteriors (n, 2) computed with the causal recentring of that order.
Usage: python studies/14nov/extview/expert.py [mivma mavr]  -> runs/views_ext/{mivma,mavr,shin2}.npz + cv report."""
import hashlib, json, os, sys
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

REPO = os.path.abspath(__file__ + "/../../../..")
OUT = REPO + "/runs/views_ext"
R, SHRINK = 20, 0.05
BANDS = ((1, 4), (4, 8), (8, 13), (13, 20), (20, 30))
CS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
ORDERS = [f"{s}{k}" for s in ("within", "full", "runs") for k in range(2)]
SET = dict(R=R, SHRINK=SHRINK, BANDS=BANDS, files=["data/shin2017a/windows_1-29.npz", "data/shin2017b/windows_1-29.npz",
                                                   "data/proteus_windows.npz"])
KEY = hashlib.sha256(json.dumps(SET, sort_keys=True).encode()).hexdigest()[:16]


def tangent(X, U):
    X = X - X.mean(1, keepdims=True); X = np.einsum("cr,nct->nrt", U, X); iu = np.triu_indices(R)
    wgt = np.where(iu[0] == iu[1], 1.0, np.sqrt(2)); out = []
    for lo, hi in BANDS:
        sos = butter(4, [lo, hi], "bandpass", fs=120, output="sos"); vb = []
        for a in range(0, len(X), 1000):
            Xb = sosfiltfilt(sos, X[a:a + 1000], axis=-1); C = np.einsum("nct,ndt->ncd", Xb, Xb) / Xb.shape[-1]
            tr = np.trace(C, axis1=1, axis2=2)[:, None, None] / R; C = (1 - SHRINK) * C + SHRINK * tr * np.eye(R)
            w, V = np.linalg.eigh(C); vb.append(np.einsum("ncd,nd,ned->nce", V, np.log(w), V)[:, iu[0], iu[1]] * wgt)
        out.append(np.concatenate(vb))
    return np.concatenate(out, 1)


def features():
    """cached: Shin A, Shin B, PROTEUS tangent vectors (uncentred) + metadata; the cache key holds every setting."""
    f = f"{OUT}/feat_{KEY}.npz"
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True); assert str(z["key"]) == KEY and json.loads(str(z["set"])) == json.loads(json.dumps(SET)); return z
    os.makedirs(OUT, exist_ok=True); A, B, P = (np.load(REPO + "/" + p) for p in SET["files"])
    for z in (A, B):
        assert list(z["ch_names"]) == list(P["ch_names"]), "channel order differs from PROTEUS"
    Xs = np.concatenate([A["X"][::4], B["X"][::4]]).astype(np.float64); Xs -= Xs.mean(1, keepdims=True)
    C = np.einsum("nct,ndt->cd", Xs, Xs); w, V = np.linalg.eigh(C); U = V[:, ::-1][:, :R]
    print(f"Shin pooled covariance: rank {(w / w.max() > 1e-8).sum()}, kept {R} components ({w[::-1][:R].sum() / w.sum():.4f} of the variance)", flush=True)
    d = dict(key=KEY, set=json.dumps(SET), U=U)
    for nm, z in (("a", A), ("b", B), ("p", P)):
        d[f"V{nm}"] = tangent(z["X"].astype(np.float64), U).astype(np.float32); d[f"y{nm}"] = z["y"]
        d[f"person{nm}"] = z["subject"].astype(str); d[f"session{nm}"] = z["session"].astype(str)
        print(f"features {nm}: {d[f'V{nm}'].shape}", flush=True)
    np.savez(f, **d); return np.load(f, allow_pickle=True)


def centre(V, groups):
    out = V.astype(np.float64).copy()
    for g in np.unique(groups):
        m = groups == g; out[m] -= out[m].mean(0)
    return out


def train_set(F, name):
    """-> (V recentred, y, person) for an expert."""
    if name == "mivma":
        pa = np.char.add(F["persona"], np.char.add("|", np.array([str(int(s[0]) // 2) for s in F["sessiona"]])))
        pb = np.char.add(F["personb"], np.char.add("|", np.array([str(int(s[0]) // 2) for s in F["sessionb"]])))
        V = centre(np.r_[F["Va"], F["Vb"]], np.r_[pa, pb])  # mean of the pseudo-session A_k + B_k (MI l/r, MA, rest)
        y = np.r_[np.zeros(len(F["ya"]), int), np.where(F["yb"] == 1, 1, -1)]  # B classes: 0 rest, 1 subtraction
        keep = y >= 0; return V[keep], y[keep], np.r_[F["persona"], F["personb"]][keep]
    if name == "mavr":
        V = centre(F["Vb"], np.char.add(F["personb"], F["sessionb"])); return V, F["yb"].astype(int), F["personb"]
    raise SystemExit(f"unknown expert {name}")


def fit(V, y, grp):
    sd = V.std(0) + 1e-9; Zs = V / sd; cv = {}
    for C in CS:
        pr = np.empty(len(y))
        for tr, te in GroupKFold(5).split(Zs, y, grp):
            m = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(Zs[tr], y[tr]); pr[te] = m.predict(Zs[te])
        cv[C] = np.mean([np.mean([np.mean(pr[(grp == q) & (y == c)] == c) for c in (0, 1)]) for q in np.unique(grp)])
    Cb = max(cv, key=cv.get); m = LogisticRegression(C=Cb, class_weight="balanced", max_iter=4000).fit(Zs, y)
    return m, sd, cv, Cb


def causal_centre(V, sess, order):
    """rows in replay layout; each row minus the running mean of its session's rows delivered so far (itself included)."""
    out = np.empty_like(V, dtype=np.float64); acc = {}
    for i in order:
        s = sess[i]; n, t = acc.get(s, (0, 0.0)); n, t = n + 1, t + V[i]; acc[s] = (n, t); out[i] = V[i] - t / n
    return out


def proteus_sess(F):
    return np.char.add(F["personp"], np.char.add("|", F["sessionp"]))


def orders(n):
    pz = np.load(REPO + "/runs/shuffle_perms.npz"); o = {"rec": np.arange(n)}
    for k in ORDERS:
        s, d = k.rstrip("0123456789"), k[len(k.rstrip("0123456789")):]; o[k] = pz[f"{s}|{d}|all_loso"].astype(int)
        assert sorted(o[k]) == list(range(n)), k
    return o


def main(names):
    F = features(); sess = proteus_sess(F); n = len(sess); O = orders(n); cols = {}
    rep = open(f"{OUT}/cv_report.txt", "a")
    for nm in names:
        V, y, grp = train_set(F, nm); m, sd, cv, Cb = fit(V, y, grp)
        line = f"{nm}: Shin n {len(y)} ({np.bincount(y).tolist()}), {len(np.unique(grp))} people; grouped 5-fold CV BA " + \
               " ".join(f"C{C:g} {100 * a:.1f}" for C, a in cv.items()) + f" -> C {Cb:g}"
        print(line, flush=True); rep.write(line + "\n")
        out = {}
        for k, o in O.items():
            Zp = causal_centre(F["Vp"], sess, o) / sd; out[k] = np.log(np.clip(m.predict_proba(Zp), 1e-6, 1)).astype(np.float32)
        np.savez(f"{OUT}/{nm}.npz", **out, classes=np.array({"mivma": ["MI", "MA"], "mavr": ["rest", "MA"]}[nm]))
        cols[nm] = out; print(f"written {OUT}/{nm}.npz keys {sorted(out)}", flush=True)
    if len(cols) > 1:
        np.savez(f"{OUT}/shin2.npz", **{k: np.concatenate([cols[nm][k] for nm in names], 1) for k in O})
        print(f"written {OUT}/shin2.npz ({'+'.join(names)})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["mivma", "mavr"])
