"""RUNDETECT common (ledger s217): the nuisance view, the leave-one-person-out run-discriminant projection, delivery
orders per session and the label metrics. Label-free at use: the projection and the length prior are fitted on the
OTHER people's true runs; the held-out person's runs are only read to score purity / ARI.
Nuisance view: log-variance per channel in 6 bands (120 Hz windows), cached in runs/rundetect/blv.npz."""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../.."); sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
A = env.A; p0 = "all_loso"; Z = A.L.Z; IDX = A.pl.B[f"idx|{p0}"].astype(int)
SES, PER, Y = Z[f"sess|{p0}"], Z[f"person|{p0}"], Z[f"y|{p0}"].astype(int)
PEOPLE = list(dict.fromkeys(PER)); SESSIONS = list(dict.fromkeys(SES)); N = len(SES)
BANDS = [(4, 8), (8, 13), (13, 20), (20, 30), (30, 45), (45, 58)]; FS = 120.0; NDIM, REG = int(os.environ.get("NDIM", 6)), float(os.environ.get("REG", 1.0))  # chosen on the held-out same/different-run distance ratio over 8 settings (s217)
PERMS = np.load(REPO + "/runs/shuffle_perms.npz")
with np.load(REPO + "/data/proteus_windows.npz") as _w:
    RUN = np.char.add(SES.astype(str), np.char.add("|", np.char.add(_w["task"].astype(str), np.char.add("|", _w["run"].astype(str)))[IDX]))
RUNI = np.unique(RUN, return_inverse=True)[1]  # true run as an int, for scoring only
for s in SESSIONS:
    assert (np.diff(np.flatnonzero(SES == s)) == 1).all(), "a session's rows are not contiguous"


def blv():
    f = REPO + "/runs/rundetect/blv.npz"
    if os.path.exists(f):
        v = np.load(f)["F"]; assert v.shape == (N, 41 * len(BANDS)); return v
    from scipy.signal import butter, sosfiltfilt
    with np.load(REPO + "/data/proteus_windows.npz") as w:
        X = w["X"][IDX].astype(np.float64)
    F = np.concatenate([np.log(sosfiltfilt(butter(4, b, "bandpass", fs=FS, output="sos"), X, axis=-1).var(-1) + 1e-12) for b in BANDS], 1)
    assert np.isfinite(F).all(); np.savez(f, F=F.astype(np.float32)); return F


def sess_centre(F):
    out = F.copy()
    for s in SESSIONS:
        m = SES == s; out[m] -= F[m].mean(0)
    return out


def projection(F, held, ndim=NDIM, reg=REG, exclude=None):
    """Run-discriminant directions from people != held: generalized eigvecs of between-run vs within-run scatter of
    session-centred features. Returns W (D, ndim), normalised so the within-run covariance is I, and eigenvalues lam
    (between-run variance per direction in those units)."""
    tr = (PER != held) & (PER != exclude); Fc = sess_centre(F)[tr]; r = RUN[tr]
    mu = {u: Fc[r == u].mean(0) for u in np.unique(r)}; M = np.stack([mu[u] for u in r])
    Sw = np.cov((Fc - M).T); Sb = np.cov(M.T); Sw += reg * np.trace(Sw) / len(Sw) * np.eye(len(Sw))
    from scipy.linalg import eigh
    lam, V = eigh(Sb, Sw); o = np.argsort(lam)[::-1][:ndim]; return V[:, o], lam[o]


def projected():
    """-> X (N, NDIM) uncentred, each person in its own leave-one-person-out projection, scaled so the within-run SD
    is 1 per direction; LAM {person: (NDIM,) between-run variance in those units}. The scales are cross-fitted (nested:
    projection fit without the held person and q, measured on q), since the fitting people's own eigenvalues overfit."""
    f = REPO + f"/runs/rundetect/proj_d{NDIM}_r{REG:g}.npz"
    if os.path.exists(f):
        z = np.load(f); return z["X"], {p: z["lam_" + p] for p in PEOPLE}
    F = blv(); Fc = sess_centre(F); X = np.zeros((N, NDIM)); lams = {}

    def stats(W, q):  # within / between-run variance per direction on person q
        m = PER == q; V = Fc[m] @ W; r = RUN[m]; M = np.stack([V[r == u].mean(0) for u in r]); return ((V - M) ** 2).mean(0), (M ** 2).mean(0)
    for h in PEOPLE:
        W, _ = projection(F, h); ws, bs = [], []
        for q in PEOPLE:
            if q == h:
                continue
            Wq, _ = projection(F, h, exclude=q); a, b = stats(Wq, q); ws.append(a); bs.append(b)
        sw, sb = np.sqrt(np.mean(ws, 0)), np.mean(bs, 0)
        m = PER == h; X[m] = (F[m] @ W) / sw; lams[h] = sb / sw ** 2
    np.savez(f, X=X, **{"lam_" + p: v for p, v in lams.items()}); return X, lams


def length_prior(held, lo=20, hi=44, floor=0.02):
    L = np.array([len(np.flatnonzero(RUN == u)) for u in np.unique(RUN[PER != held])])
    p = np.bincount(np.clip(L, lo, hi) - lo, minlength=hi - lo + 1).astype(float); p = p / p.sum() + floor / (hi - lo + 1)
    full = np.zeros(hi + 1); full[lo:] = p / p.sum(); return full  # p(L = l), l = 0..hi


def order_rows(order, s):
    """Rows (replay layout) of session s in delivery order for order 'recording' or '<scheme><seed>'."""
    rows = np.flatnonzero(SES == s)
    if order == "recording":
        return rows
    sc, sd = order.rstrip("0123456789"), int(order[len(order.rstrip("0123456789")):])
    g = PERMS[f"{sc}|{sd}|{p0}"]; rk = np.empty(N, int); rk[g] = np.arange(N); return rows[np.argsort(rk[rows])]


def purity_ari(lab, true):
    from sklearn.metrics import adjusted_rand_score
    pur = sum(np.bincount(true[lab == u]).max() for u in np.unique(lab)) / len(lab)
    return pur, adjusted_rand_score(true, lab)
