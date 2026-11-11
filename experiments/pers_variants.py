"""PERS2 (ledger s17): personal-model recipes compared on ten people (within-session proxy) and on the real
cross-session cases (P04, P10: session 1 -> sessions 2-3), core level.

Recipes: PS1 mix (baseline, head at 3x), person-upweighted core (sample weight w on the person's public windows),
both together, head C and mix weight, and a personal recentring prior (the public session's mean log-covariance as the
causal reference prior of the later session, k pseudo-windows) instead of the population prior.
Usage: python experiments/pers_variants.py OUT.json
"""
import json, os, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT + "/experiments"]
import protocol  # noqa: E402
import ts_core2 as T  # noqa: E402

d = protocol.load(); fe = T.Feats(d, T.parse("car_ov17_fs2_t0.1_le_k2")); y = d["y"]
per = np.array([s[-3:] for s in d["subject"]]); ses = d["session"].astype(str); run = d["run"].astype(int)
ba = lambda yy, p: 100 * np.mean([np.mean(p.argmax(1)[yy == k] == k) for k in range(3)])  # noqa: E731


def causal_std(Z):
    n = np.arange(1, len(Z) + 1)[:, None]; m = np.cumsum(Z, 0) / n; v = np.cumsum(Z ** 2, 0) / n - m ** 2
    return np.where(n >= 4, (Z - m) / np.sqrt(np.maximum(v, 1e-6)), Z - m)


def trial(others, pub, tests):
    idx = np.concatenate([others] + pub); pri = [fe.prior(b, idx) for b in range(len(fe.C))]; d2p = [q[idx].mean() for q in fe.d2]
    pubrows = np.concatenate(pub); ppri = [fe.L[b][pubrows].mean(0) for b in range(len(fe.C))]

    def cz(r, prior=pri, k=2.0):
        k0 = fe.k; fe.k = k
        try:
            return fe.causal(r, prior, d2p, None)
        finally:
            fe.k = k0
    Fpub = [cz(r) for r in pub]; Xtr = np.concatenate([fe.F[others]] + Fpub); ytr = y[idx]; yt = np.concatenate([y[r] for r in tests])
    sc = StandardScaler().fit(Xtr); Ztr = sc.transform(Xtr); npub = len(pubrows)
    feats = {"pop": [cz(r) for r in tests], "pers_k2": [cz(r, ppri, 2.0) for r in tests], "pers_k10": [cz(r, ppri, 10.0) for r in tests]}
    cores = {}
    for w in (1, 5, 20):
        sw = np.r_[np.ones(len(others)), np.full(npub, float(w))]
        cores[w] = LogisticRegression(C=10.0, max_iter=2000).fit(Ztr, ytr, sample_weight=sw)
    pca = PCA(256, random_state=0).fit(sc.transform(fe.F[others])); view = lambda Fx: pca.transform(sc.transform(Fx))  # noqa: E731
    Vp = [view(Fx) for Fx in Fpub]; Zp = np.concatenate([(v - v.mean(0)) / np.maximum(v.std(0), 1e-6) for v in Vp]); yp = y[pubrows]
    heads = {C: LogisticRegression(C=C, max_iter=3000).fit(Zp, yp) for C in (0.03, 0.1, 0.3)}
    pc = lambda w, f="pop": np.concatenate([cores[w].predict_proba(sc.transform(Fx)) for Fx in feats[f]])  # noqa: E731
    ph = lambda C, f="pop": np.concatenate([heads[C].predict_proba(causal_std(view(Fx))) for Fx in feats[f]])  # noqa: E731
    P1, H = pc(1), ph(0.1); out = {"core": ba(yt, P1), "PS1 mix 3x (baseline)": ba(yt, (P1 + 3 * H) / 4), "PS1 mix 1x": ba(yt, (P1 + H) / 2), "PS1 mix 6x": ba(yt, (P1 + 6 * H) / 7),
                                  "PS1 C0.03 3x": ba(yt, (P1 + 3 * ph(0.03)) / 4), "PS1 C0.3 3x": ba(yt, (P1 + 3 * ph(0.3)) / 4),
                                  "core w5": ba(yt, pc(5)), "core w20": ba(yt, pc(20)), "core w5 + PS1 3x": ba(yt, (pc(5) + 3 * H) / 4), "core w20 + PS1 3x": ba(yt, (pc(20) + 3 * H) / 4),
                                  "personal prior k2: core": ba(yt, pc(1, "pers_k2")), "personal prior k10: core": ba(yt, pc(1, "pers_k10")),
                                  "personal prior k10: PS1 mix 3x": ba(yt, (pc(1, "pers_k10") + 3 * ph(0.1, "pers_k10")) / 4),
                                  "personal prior k10: core w5 + PS1 3x": ba(yt, (pc(5, "pers_k10") + 3 * ph(0.1, "pers_k10")) / 4)}
    return out


R = {}
for p in sorted(set(per)):
    others = np.flatnonzero(per != p); s1 = (per == p) & (ses == "1"); runs = sorted(set(run[s1]))
    A, B = np.flatnonzero(s1 & np.isin(run, runs[:3])), np.flatnonzero(s1 & np.isin(run, runs[3:]))
    R[f"{p} within"] = trial(others, [A], [B]); later = sorted(set(ses[per == p]) - {"1"})
    if later:
        R[f"{p} cross"] = trial(others, [np.flatnonzero(s1)], [np.flatnonzero((per == p) & (ses == s)) for s in later])
    print(p, "done", flush=True); json.dump(R, open(sys.argv[1], "w"), indent=1)
names = list(next(iter(R.values()))); within = [k for k in R if k.endswith("within")]
print(f"{'recipe':40s} within-10 mean gain | P04 cross  P10 cross | P04 within P10 within")
for n in names:
    g = lambda k: R[k][n] - R[k]["core"]  # noqa: E731
    print(f"{n:40s} {np.mean([g(k) for k in within]):+10.2f}          | {g('P04 cross'):+8.2f} {g('P10 cross'):+9.2f} | {g('P04 within'):+9.2f} {g('P10 within'):+9.2f}")
