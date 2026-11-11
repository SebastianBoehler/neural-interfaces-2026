"""Retune grid (declared in grid.txt): mix(wslot, wt) -> rebal(alpha) -> SessionEM(lam_em) -> TSEM(lam_t).
TSEM is factored: the per-window class likelihoods depend only on its input (seeds), so the 3 lam_t values share one
pass. Usage: python grid.py OUT.npz BASE[,BASE..] [rho=0.7] [configs=all|ship|a,e,s,t;...]  (NJ workers)
Saves per base/config/proto per-person absolute BA (new: 10 people, seen: P04/P10) after TSEM and before TSEM."""
import itertools, os, sys, time
import paths
W = paths.DATA
import numpy as np
import session_em as SE
import tsem as T
from multiprocessing import Pool

Z = dict(np.load(W + "/inputs.npz", allow_pickle=True))
PROTOS = ("all_loso", "seen_session"); PEOPLE = [f"P{i:02d}" for i in range(1, 11)]; SEEN = ["P04", "P10"]
ALPHAS, EMLAMS, WSLOTS, WTS, TLAMS = (1.0, 1.15, 1.3), (0.35, 0.5, 0.7), (1.5, 2.0, 2.5), (3.0, 5.0, 7.0), (0.12, 0.15, 0.2)
SHIP = (1.15, 0.5, 2.0, 5.0)


def rebal(P, sess, alpha, k=5.0, W=30):  # = diag_common.rebal (checked: replay == stored final)
    out = np.empty_like(P)
    for s in np.unique(sess):
        r = np.flatnonzero(sess == s); X = P[r]
        cs = np.vstack([np.zeros((1, 3)), np.cumsum(X, 0)])
        n = np.arange(len(X)); lo = np.maximum(0, n - W)
        m = (k / 3 + cs[n] - cs[lo]) / (k + (n - lo))[:, None]
        q = X / m ** alpha
        out[r] = q / q.sum(1, keepdims=True)
    return out


def tsem_lik(P, sess, X, rho=0.7, shrink=0.7, m0=20, nmin=15):
    """T.tsem with the likelihood kept: L (n,3) (NaN = row left unchanged), tt (n,) = t of the row."""
    L = np.full_like(P, np.nan); tt = np.zeros(len(P))
    for s in np.unique(sess):
        r = np.flatnonzero(sess == s); lab = P[r].argmax(1); conf = P[r].max(1)
        for t in range(nmin, len(r)):
            A_, f = X[r[:t]], X[r[t]]
            m, sd = A_.mean(0), np.maximum(A_.std(0), 1e-3)
            Zm, z = (A_ - m) / sd, (f - m) / sd
            q = max(2, int(rho * t / 3)); sel = []
            for k in range(3):
                ik = np.flatnonzero(lab[:t] == k); sel.append(ik[np.argsort(-conf[ik])[:q]])
            if min(len(x) for x in sel) < 2:
                continue
            mu = np.stack([Zm[x].sum(0) / (len(x) + m0) for x in sel])
            R = np.concatenate([Zm[x] - mu[k] for k, x in enumerate(sel)])
            Cv = np.cov(R.T); Cv = (1 - shrink) * Cv + shrink * np.eye(len(Cv)) * np.trace(Cv) / len(Cv)
            L[r[t]] = T._lda(z[None], mu, np.linalg.inv(Cv))[0]; tt[r[t]] = t
    return L, tt


def apply_lam(P, L, tt, lam, n0=60):
    out = P.copy(); m = ~np.isnan(L[:, 0])
    e = lam * tt[m] / (tt[m] + n0)
    q = P[m] * L[m] ** e[:, None]; out[m] = q / q.sum(1, keepdims=True)
    return out


def scores(P, proto):
    y, cell, per = Z[f"y|{proto}"], Z[f"cell|{proto}"], Z[f"person|{proto}"]; pred = P.argmax(1); acc = {}
    for c in np.unique(cell):
        m = cell == c; yy, pp = y[m], pred[m]
        acc.setdefault(per[m][0], []).append(100 * np.mean([np.mean(pp[yy == k] == k) for k in np.unique(yy)]))
    return np.array([np.mean(acc[p]) for p in (PEOPLE if proto == "all_loso" else SEEN)])


def pre_tsem(base, proto, a, le, wsl, wt):
    Ps, ws = Z[f"{base}|Ps|{proto}"], Z[f"{base}|ws|{proto}"].copy(); ws[0], ws[-1] = wsl, wt
    mix = sum(w / ws.sum() * P for w, P in zip(ws, Ps))
    b = rebal(mix, Z[f"sess|{proto}"], a)
    return SE.apply(Z[f"GE|{proto}"], b, Z[f"sess|{proto}"], lam=le, n0=60, m0=20, shrink=0.7)


def task(args):
    base, cfg, rho, tlams = args; out = {}
    for proto in PROTOS:
        e = pre_tsem(base, proto, *cfg)
        L, tt = tsem_lik(e, Z[f"sess|{proto}"], Z[f"pc128|{proto}"], rho=rho)
        out[proto] = (scores(e, proto), [scores(apply_lam(e, L, tt, lt), proto) for lt in tlams])
    return base, cfg, out


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    outf, bases = sys.argv[1], sys.argv[2].split(",")
    kw = dict(a.split("=") for a in sys.argv[3:]); rho = float(kw.get("rho", 0.7)); cf = kw.get("configs", "all")
    TL = tuple(float(v) for v in kw["tlams"].split(",")) if "tlams" in kw else TLAMS
    combos = (list(itertools.product(ALPHAS, EMLAMS, WSLOTS, WTS)) if cf == "all" else [SHIP] if cf == "ship"
              else [tuple(float(v) for v in c.split(",")) for c in cf.split(";")])
    # checks on the shipped config: replay == stored final; factored TSEM == T.tsem
    for base in bases:
        for proto in PROTOS:
            e = pre_tsem(base, proto, *SHIP)
            ref = T.tsem(Z[f"{base}|final|{proto}"], Z[f"sess|{proto}"], T.static(Z[f"pc128|{proto}"]), lam=0.15,
                         rho=rho, shrink=0.7, m0=20, n0=60, nmin=15)
            L, tt = tsem_lik(e, Z[f"sess|{proto}"], Z[f"pc128|{proto}"], rho=rho)
            print(base, proto, "replay vs stored final max|diff|", np.abs(e - Z[f"{base}|final|{proto}"]).max(),
                  "| factored TSEM vs T.tsem max|diff|", np.abs(apply_lam(e, L, tt, 0.15) - ref).max(), flush=True)
    jobs = [(b, c, rho, TL) for b in bases for c in combos]
    t0 = time.time(); res = {}
    with Pool(int(os.environ.get("NJ", "8"))) as pool:
        for i, (b, c, o) in enumerate(pool.imap_unordered(task, jobs)):
            for proto, (pre, post) in o.items():
                res[f"{b}|{','.join(map(str, c))}|pre|{proto}"] = pre
                for lt, s in zip(TL, post):
                    res[f"{b}|{','.join(map(str, c))}|{lt}|{proto}"] = s
            if i % 20 == 0:
                print(f"{i + 1}/{len(jobs)} done {time.time() - t0:.0f}s", flush=True)
    np.savez(outf, **res)
    print("saved", outf, len(res), f"{time.time() - t0:.0f}s", flush=True)
