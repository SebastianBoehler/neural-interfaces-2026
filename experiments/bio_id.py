"""BIO1: label-free open-set person identification from the first n windows of a session.

Session signature = mean window feature (gain-free log-PSD shape and/or band correlation, CAR). A two-covariance
Gaussian model (between-person B, within-person/session W, window noise S/n) is fitted in PCA space on TRAIN people of
external multi-session corpora; the score is the same-person log-likelihood ratio of (gallery mean, probe).
External test: held-out people, 10-person galleries with one session each, probes = first n windows of another
session (genuine) or of a non-gallery person (impostor). PROTEUS: leave-one-session-out, as BIO0.
Usage: python experiments/bio_id.py OUT.json [DATA_ROOT]
"""
import glob, json, os, sys
import numpy as np

NS = (5, 10, 20, 40, 60)
KEEP = int(os.environ.get("BIO_KEEP", 60))  # windows kept per session for short-n probes and the window-noise estimate
CORPORA = tuple(os.environ.get("BIO_CORPORA", "lee2019_mi,stieger2021,yang2025,kumar2024,guttmannflury2025_mi,zhou2020").split(","))
VIEWS = {"psd": (0, 1599), "corr": (1599, 3239), "both": (0, 3239)}
IU = np.triu_indices(41, 1)


def feats(X):
    """(n, 41, 480) microvolt windows at 120 Hz -> (n, 3239): log-PSD shape 2-40 Hz, alpha and beta correlation (Fisher z)."""
    X = X.astype(np.float64); X = X - X.mean(1, keepdims=True); X = X - X.mean(2, keepdims=True)
    seg = np.stack([X[:, :, s:s + 120] for s in range(0, 361, 60)], 2) * np.hanning(120)  # (n, 41, 7, 120), 1 Hz bins
    P = (np.abs(np.fft.rfft(seg, axis=-1)) ** 2).mean(2)[:, :, 2:41]
    L = np.log(P + 1e-12); out = [(L - L.mean(2, keepdims=True)).reshape(len(X), -1)]
    F = np.fft.rfft(X, axis=-1); f = np.fft.rfftfreq(480, 1 / 120)
    for lo, hi in ((8, 13), (13, 30)):
        B = F[:, :, (f >= lo) & (f < hi)]; C = np.real(B @ np.conj(B.transpose(0, 2, 1)))
        d = np.sqrt(np.einsum("nii->ni", C)) + 1e-12; R = np.clip(C / d[:, :, None] / d[:, None, :], -0.999, 0.999)
        out.append(np.arctanh(R[:, IU[0], IU[1]]))
    return np.concatenate(out, 1).astype(np.float32)


def load_sessions(files, tag):
    """-> list of dict(corpus, person, session, full (D,), head (<=KEEP, D))."""
    out = []
    for fn in files:
        z = np.load(fn, allow_pickle=True); sub, ses = z["subject"].astype(str), z["session"].astype(str)
        for key in dict.fromkeys(zip(sub, ses)):
            r = np.flatnonzero((sub == key[0]) & (ses == key[1]))
            if len(r) < 30:
                continue
            F = np.concatenate([feats(z["X"][r[i:i + 256]]) for i in range(0, len(r), 256)])
            out.append(dict(corpus=tag, person=f"{tag}/{key[0]}", session=key[1], full=F.mean(0), head=F[:KEEP]))
        print(tag, fn.split("/")[-1], len(out), flush=True)
    return out


class TwoCov:
    def fit(self, S, view, dim=64, shrink=0.2):
        a, b = VIEWS[view]; self.sl = slice(a, b); self.mu = {}
        for c in {s["corpus"] for s in S}:
            self.mu[c] = np.mean([s["full"][self.sl] for s in S if s["corpus"] == c], 0)
        Xc = np.stack([s["full"][self.sl] - self.mu[s["corpus"]] for s in S]); self.sd = Xc.std(0) + 1e-6
        _, _, Vt = np.linalg.svd(Xc / self.sd, full_matrices=False); dim = min(dim, len(Vt) - 2); self.V = Vt[:dim].T
        Z = (Xc / self.sd) @ self.V; per = np.array([s["person"] for s in S])
        means, Wd = [], []
        for p in np.unique(per):
            m = per == p; means.append(Z[m].mean(0))
            if m.sum() > 1:
                Wd.append(Z[m] - Z[m].mean(0))
        Wd = np.concatenate(Wd); W = Wd.T @ Wd / max(1, len(Wd) - len(means))
        Hd = [(s["head"][:, self.sl] / self.sd) @ self.V for s in S]
        Nd = np.concatenate([h - h.mean(0) for h in Hd])
        Sw = Nd.T @ Nd / len(Nd); Bc = np.cov(np.stack(means).T) - W / 2
        sh = lambda M: (1 - shrink) * M + shrink * np.eye(dim) * np.trace(M) / dim  # noqa: E731
        ev, U = np.linalg.eigh(sh(Bc)); self.B = (U * np.maximum(ev, 1e-3 * ev.max())) @ U.T; self.W, self.S = sh(W), sh(Sw)
        return self

    def proj(self, x, mu):
        return ((x[..., self.sl] - mu) / self.sd) @ self.V

    def llr(self, g, m, p, n):
        """gallery mean g of m full sessions vs probe p from n windows -> same-person log-likelihood ratio."""
        Wg, Wp = self.W / m, self.W + (self.S / n if n else 0); d = len(g)
        A, C = self.B + Wg, self.B + Wp; same = np.block([[A, self.B], [self.B, C]]); x = np.concatenate([g, p])
        ls = -0.5 * (x @ np.linalg.solve(same, x) + np.linalg.slogdet(same)[1])
        ld = -0.5 * (g @ np.linalg.solve(A, g) + p @ np.linalg.solve(C, p) + np.linalg.slogdet(A)[1] + np.linalg.slogdet(C)[1])
        return (ls - ld) / d


def probe(s, n):
    return s["full"] if n is None else s["head"][:n].mean(0)


def external_trials(M, S, rng, reps=30, G=10):
    """-> {n: (genuine best-score, genuine correct, impostor best-score)} pooled over corpora."""
    res = {n: ([], [], []) for n in NS}
    for c in sorted({s["corpus"] for s in S}):
        by = {}
        for s in S:
            if s["corpus"] == c:
                by.setdefault(s["person"], []).append(s)
        multi = [p for p in by if len(by[p]) > 1]
        if len(multi) < 4:
            continue
        mu = np.mean([s["full"][M.sl] for s in S if s["corpus"] == c], 0)
        for _ in range(reps):
            gal = list(rng.choice(multi, min(G, len(multi) // 2), replace=False)); gs, pr = {}, []
            for p in gal:
                i, j = rng.choice(len(by[p]), 2, replace=False); gs[p] = M.proj(by[p][i]["full"], mu); pr.append((p, by[p][j]))
            for p in by:
                if p not in gal:
                    pr.append((None, by[p][rng.integers(len(by[p]))]))
            for truth, s in pr:
                for n in NS:
                    x = M.proj(probe(s, n), mu); sc = np.array([M.llr(gs[p], 1, x, n) for p in gal])
                    if truth is None:
                        res[n][2].append(sc.max())
                    else:
                        res[n][0].append(sc.max()); res[n][1].append(gal[int(sc.argmax())] == truth)
    return res


def stream_trials(M, S, rng, reps=20, G=10, grid=tuple(range(10, 201, 5))):
    """Session-level trials: the probe is scored at every n of `grid`; -> list of (genuine, best score per n, correct id per n)."""
    out = []
    for c in sorted({s["corpus"] for s in S}):
        by = {}
        for s in S:
            if s["corpus"] == c:
                by.setdefault(s["person"], []).append(s)
        multi = [p for p in by if len(by[p]) > 1]
        if len(multi) < 4:
            continue
        mu = np.mean([s["full"][M.sl] for s in S if s["corpus"] == c], 0)
        for _ in range(reps):
            gal = list(rng.choice(multi, min(G, len(multi) // 2), replace=False)); gs, pr = {}, []
            for p in gal:
                i, j = rng.choice(len(by[p]), 2, replace=False); gs[p] = M.proj(by[p][i]["full"], mu); pr.append((p, by[p][j]))
            for p in by:
                if p not in gal:
                    pr.append((None, by[p][rng.integers(len(by[p]))]))
            for truth, s in pr:
                cum = np.cumsum(s["head"].astype(np.float64), 0); best, ok = np.full(len(grid), np.nan), np.zeros(len(grid), bool)
                for k, n in enumerate(grid):
                    if n > len(cum):
                        break
                    x = M.proj(cum[n - 1] / n, mu); sc = np.array([M.llr(gs[p], 1, x, n) for p in gal])
                    best[k] = sc.max(); ok[k] = truth is not None and gal[int(sc.argmax())] == truth
                out.append((truth is not None, best, ok))
    return out


def stream_trials2(M, S, rng, m=1, step=1, reps=20, G=10, lo=10, hi=200):
    """As the package scores: every `step`-th window from index `lo` (lo + 1 windows), galleries of `m` sessions, and
    the centring vector computed without the probe session. -> list of (genuine, best score per n, correct id per n)."""
    out = []; grid = tuple(range(lo + 1, hi + 1, step)); pj = lambda x: (x[M.sl] / M.sd) @ M.V  # noqa: E731
    for c in sorted({s["corpus"] for s in S}):
        by = {}
        for s in S:
            if s["corpus"] == c:
                by.setdefault(s["person"], []).append(s)
        multi = [p for p in by if len(by[p]) > m]
        if len(multi) < 4:
            continue
        allp = [pj(s["full"]) for s in S if s["corpus"] == c]; tot, nall = np.sum(allp, 0), len(allp)
        for _ in range(reps):
            gal = list(rng.choice(multi, min(G, len(multi) // 2), replace=False)); g0, pr = {}, []
            for p in gal:
                ii = rng.choice(len(by[p]), m + 1, replace=False); g0[p] = np.mean([pj(by[p][i]["full"]) for i in ii[:m]], 0); pr.append((p, by[p][ii[m]]))
            for p in by:
                if p not in gal:
                    pr.append((None, by[p][rng.integers(len(by[p]))]))
            for truth, s in pr:
                mu = (tot - pj(s["full"])) / (nall - 1); cum = np.cumsum(s["head"].astype(np.float64), 0)
                best, ok = np.full(len(grid), np.nan), np.zeros(len(grid), bool)
                for k, n in enumerate(grid):
                    if n > len(cum):
                        break
                    x = (cum[n - 1] / n / M.sd) @ M.V - mu; sc = np.array([M.llr(g0[p] - mu, m, x, n) for p in gal])
                    best[k] = sc.max(); ok[k] = truth is not None and gal[int(sc.argmax())] == truth
                out.append((truth is not None, best, ok))
    return out


def stream_summary(tr, taus=np.arange(0.18, 0.421, 0.01), ks=(1, 2, 3)):
    """Per threshold and persistence k (consecutive grid points above it): session-level false accepts, correct accepts,
    wrong-identity accepts of genuine sessions, and the median grid index of the first correct accept."""
    rows = []
    for k in ks:
        for tau in taus:
            fa, ta, wa, first = [], [], [], []
            for gen, best, ok in tr:
                above = np.nan_to_num(best, nan=-np.inf) >= tau
                run = np.convolve(above, np.ones(k), "valid") >= k if len(above) >= k else np.zeros(0, bool)
                hit = np.flatnonzero(run) + k - 1
                if not gen:
                    fa.append(len(hit) > 0)
                else:
                    good = [h for h in hit if ok[h]]; ta.append(len(good) > 0); wa.append(any(not ok[h] for h in hit))
                    if good:
                        first.append(good[0])
            rows.append(dict(k=int(k), tau=round(float(tau), 2), far=float(np.mean(fa)), tar=float(np.mean(ta)), wrong_id=float(np.mean(wa)),
                             first=float(np.median(first)) if first else None, n_imp=len(fa), n_gen=len(ta)))
    return rows


def summarize(res):
    out = {}
    for n, (g, ok, im) in res.items():
        if not len(im) or not len(g):
            continue
        g, ok, im = np.array(g), np.array(ok), np.array(im); row = dict(rank1=float(ok.mean()), n_gen=len(g), n_imp=len(im))
        for far in (0.01, 0.02, 0.05):
            t = float(np.quantile(im, 1 - far)); row[f"tar@{far}"] = float(np.mean(ok & (g >= t))); row[f"thr@{far}"] = t
        out[n] = row
    return out


def proteus_trials(M, P):
    people = sorted({s["person"] for s in P}); rows = {}
    for n in NS + (None,):
        gen, imp = [], []
        for q in P:
            rest = [s for s in P if s is not q]; mu = np.mean([s["full"][M.sl] for s in rest], 0); sc = {}
            for p in people:
                ss = [s for s in rest if s["person"] == p]
                if ss:
                    sc[p] = M.llr(M.proj(np.mean([s["full"] for s in ss], 0), mu), len(ss), M.proj(probe(q, n), mu), n)
            own = sc.pop(q["person"], None); best = max(sc.values())
            imp.append(best)  # the probe as a new person: best score against everyone else
            if own is not None:
                gen.append((q["person"][-3:] + "s" + q["session"], own, best))
        rows[str(n)] = dict(genuine=[(a, round(b, 3), round(c, 3)) for a, b, c in gen], impostor_max=sorted(round(x, 3) for x in imp))
    return rows


def main():
    out, root = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "data"); rng = np.random.default_rng(0)
    S = [s for c in CORPORA for s in load_sessions(sorted(glob.glob(f"{root}/{c}/windows*.npz") or glob.glob(f"{root}/{c}*.npz")), c)]
    if os.environ.get("BIO_EXPORT"):  # fit on all external people and save the model for the routing gate
        view, dim = os.environ.get("BIO_VIEW", "psd"), int(os.environ.get("BIO_DIM", 64)); M = TwoCov().fit(S, view, dim)
        np.savez(os.environ["BIO_EXPORT"], lo=M.sl.start, hi=M.sl.stop, sd=M.sd, V=M.V, B=M.B, W=M.W, S=M.S,
                 n_sessions=len(S), n_people=len({s["person"] for s in S}))
        return print("exported", view, dim, len(S), flush=True)
    if os.environ.get("BIO_STREAM"):  # session-level calibration of the routing threshold on held-out external people
        for s in S:
            s["head"] = s["head"][:, :1599]
        people = sorted({s["person"] for s in S}); test = set(rng.choice(people, int(0.3 * len(people)), replace=False))
        M = TwoCov().fit([s for s in S if s["person"] not in test], "psd", 64)
        te = [s for s in S if s["person"] in test]
        if os.environ.get("BIO_STREAM2"):  # package-faithful: every window, probe excluded from the centring, m sessions
            res = {}
            for m in (1, 3):
                tr = stream_trials2(M, te, np.random.default_rng(1), m=m, step=int(os.environ.get("BIO_STEP", 1)))
                res[f"m{m}"] = stream_summary(tr, ks=(1,)) if tr else []
                for r in res[f"m{m}"]:
                    if r["tau"] in (0.2, 0.23, 0.25, 0.27, 0.3, 0.33, 0.36):
                        print("m", m, r, flush=True)
                json.dump(res, open(os.environ["BIO_STREAM2"], "w"), indent=1)
            return
        rows = stream_summary(stream_trials(M, te, np.random.default_rng(1)))
        json.dump(rows, open(os.environ["BIO_STREAM"], "w"), indent=1)
        for r in rows:
            if r["tau"] in (0.2, 0.23, 0.25, 0.27, 0.3, 0.33, 0.36, 0.4):
                print(r, flush=True)
        return
    P = load_sessions([f"{root}/proteus_windows.npz"], "proteus")
    people = sorted({s["person"] for s in S}); test = set(rng.choice(people, int(0.3 * len(people)), replace=False))
    tr, te = [s for s in S if s["person"] not in test], [s for s in S if s["person"] in test]
    print(f"external sessions {len(S)} people {len(people)} (train {len(people) - len(test)}, test {len(test)}); proteus sessions {len(P)}", flush=True)
    R = {}
    for view in VIEWS:
        for dim in (32, 64, 128):
            M = TwoCov().fit(tr, view, dim); ext = summarize(external_trials(M, te, np.random.default_rng(1)))
            Mall = TwoCov().fit(S, view, dim); R[f"{view}_d{dim}"] = dict(external=ext, proteus=proteus_trials(Mall, P))
            print(view, dim, {n: (round(v["rank1"], 3), round(v["tar@0.02"], 3), round(v["thr@0.02"], 3)) for n, v in ext.items()}, flush=True)
            for n in ("10", "20", "60", "None"):
                print("   proteus n", n, R[f"{view}_d{dim}"]["proteus"][n], flush=True)
            json.dump(R, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
