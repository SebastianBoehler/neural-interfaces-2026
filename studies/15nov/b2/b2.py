"""s284 B2: per-class Gaussian sampling in core1's tangent-space features, added to the LR training set at weight w.
Core1 features/heads come from s263 variant Z (cache reused by path); only Core.fit changes. w = 0 runs the unmodified Core.fit.
Synthetic rows (n per class = real rows of that class in the fold's training people): pick a training person p (uniform), class c;
mean = population class mean + between-person deviation (low-rank Gaussian over the P people's class means);
within-person noise = low-rank Gaussian over p's centred class rows; both shrunk toward isotropic diagonal noise (alpha 0.3).
Usage: b2.py prep | fold W I (W in 0,0.1,0.3,1). Outputs runs/b2/w<W>/fold<I>.json."""
import json, os, sys, time
import numpy as np
REPO = os.environ.get("NI26_ROOT") or os.path.abspath(__file__ + "/../../../..")
os.environ["NI26_ROOT"] = REPO
sys.path[:0] = [REPO + "/experiments", REPO + "/studies/14nov/s263"]
import protocol  # noqa: E402
import ts_core2 as T  # noqa: E402
import s263  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

HEADS = s263.HEADS
ALPHA = 0.3
STATS = {}


def synth(F, y, subj, rng):
    """Returns (Xs, ys) sampled from the training rows only (F, y, subj are already the fold's training rows)."""
    people = sorted(set(subj))
    Xs, ys = [], []
    for c in np.unique(y):
        n = int((y == c).sum())
        mu = {p: F[(subj == p) & (y == c)].mean(0) for p in people}
        M = np.stack([mu[p] for p in people])
        pm = M.mean(0)
        B = M - pm  # between-person deviations (P, D)
        sd = np.sqrt(F[y == c].var(0) + 1e-6)
        cen = {p: F[(subj == p) & (y == c)] - mu[p] for p in people}
        if True:
            pick = rng.integers(len(people), size=n)
            zb = rng.standard_normal((n, len(people))) / np.sqrt(max(len(people) - 1, 1))
            X = pm + np.sqrt(1 - ALPHA) * zb @ B + np.sqrt(ALPHA) * rng.standard_normal((n, F.shape[1])) * sd * 0.5
            for k, p in enumerate(people):
                r = np.flatnonzero(pick == k)
                if len(r):
                    A = cen[p]
                    X[r] += np.sqrt(1 - ALPHA) * (rng.standard_normal((len(r), len(A))) / np.sqrt(max(len(A) - 1, 1))) @ A
            Xs.append(X.astype(F.dtype)), ys.append(np.full(n, c))
    return np.concatenate(Xs), np.concatenate(ys)


class B2Core(T.Core):
    def __init__(self, fe, heads, store, w, subj, seed=0):
        super().__init__(fe, heads, store)
        self.w, self.subj, self.seed = w, subj, seed

    def fit(self, idx):
        if self.w == 0:
            return super().fit(idx)
        fe = self.fe
        assert not (fe.cfg["ct"] or fe.cfg["cb"]), "B2 assumes core1 trains on fe.F rows"
        self.priors = [fe.prior(b, idx) for b in range(len(fe.C))]
        self.d2p = [d2[idx].mean() for d2 in fe.d2]
        self.szp = fe.sz_prior(idx)
        X, y = fe.F[idx], fe.y[idx]
        Xs, ys = synth(X, y, self.subj[idx], np.random.default_rng(self.seed))
        assert not any(str(s).endswith(self.test_person) for s in set(self.subj[idx])), "synthetic source leaks the held-out person"
        STATS.update(n_real=len(y), n_syn=len(ys), syn_dist=float(np.abs(Xs.mean(0) - X.mean(0)).mean()))
        Xa, ya = np.concatenate([X, Xs]), np.concatenate([y, ys])
        sw = np.concatenate([np.ones(len(y)), np.full(len(ys), self.w)])
        self.models = Parallel(n_jobs=len(self.heads), prefer="threads")(delayed(fit_w)(h, Xa, ya, sw) for h in self.heads)


def fit_w(h, X, y, sw):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return make_pipeline(StandardScaler(), LogisticRegression(C=float(h[2:]), max_iter=2000)).fit(X, y, logisticregression__sample_weight=sw)


def main():
    cmd = sys.argv[1]
    d = protocol.load()
    if cmd == "prep":
        sys.argv = [sys.argv[0], "prep", "Z"]
        return s263.main()
    w = float(sys.argv[2]); i = int(sys.argv[3])
    fe = s263.load_feats("Z")
    name, tr, te = list(protocol.folds(d, "all_loso"))[i]
    store = {h: np.full((len(d["y"]), 3), np.nan) for h in HEADS}
    t0 = time.time()
    m = B2Core(fe, HEADS, store, w, np.asarray(d["subject"]))
    m.test_person = name
    m.fit(np.flatnonzero(tr))
    fit = time.time() - t0
    protocol.predict_streams(m, d, te, proba=True)
    res = {h: protocol.score(d["y"][te], store[h][te].argmax(1), d, te)["balanced_accuracy_cell_mean"] for h in HEADS}
    out = f"{REPO}/runs/b2/w{sys.argv[2]}"
    os.makedirs(out, exist_ok=True)
    json.dump(dict(w=w, person=name, ba=res, fit_s=fit, stats=STATS), open(f"{out}/fold{i}.json", "w"))
    print("w", w, name, {h: round(100 * b, 2) for h, b in res.items()}, STATS, f"fit {fit:.0f}s", flush=True)


if __name__ == "__main__":
    main()
