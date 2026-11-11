"""Routing v4 dev replay: the streaming identity gate (experiments/bio_gate.Bio, the class that goes into the package)
merged with the v3 gate (ledger s14).  Usage: python route4.py MODEL.npz [bases] [fresh]"""
import os, sys, time
import numpy as np
REPO = __import__("os").path.abspath(__file__ + "/../../../..")
sys.path[:0] = [REPO + "/replay", REPO + "/experiments"]
import env
from bio_gate import Bio, fold_params
A = env.A; R3, R2, RR, RL, gm, pl, SC = A.R3, A.R2, A.RR, A.RL, A.gm, A.pl, A.SC
z = np.load(REPO + "/data/proteus_windows.npz", allow_pickle=True); X = z["X"]
PER = np.array([s[-3:] for s in z["subject"]]); STR = np.char.add(np.char.add(PER, "|"), z["session"].astype(str))
B0 = Bio.__new__(Bio); B0.win = np.hanning(120); FB = np.stack([B0.feat(x) for x in X])
MODEL = dict(np.load(sys.argv[1]))
print("model", int(MODEL["n_people"]), "people", int(MODEL["n_sessions"]), "sessions; dim", MODEL["V"].shape[1], flush=True)
_BIO = {}
TAU = float(os.environ.get('TAU', 0.26))  # 0.26 is the uploaded v4.1 threshold
PSC = float(os.environ.get('PSC', 0.1))
R3.G3 = float(os.environ.get('G3', 3.0))
print('personal head C', PSC, 'slot gain', R3.G3, 'kappa', os.environ.get('KAPPA', 0), flush=True)
print('tau', TAU, flush=True)


def bio_stream(proto, f, s, gi, names, t0):
    """-> r (n,), j (n,), score of the top person (n,) from the streaming class, one test session."""
    key = (proto, f, s, t0)
    if key not in _BIO:
        itr = np.load(f"{RL.FE}/{proto}-{f}.npz")["idx_tr"]; b = Bio(fold_params(MODEL, FB[itr], STR[itr], list(names), t0=t0, tau=TAU))
        r, j, sc = np.zeros(len(gi)), np.full(len(gi), -1), np.full(len(gi), np.nan)
        for t, g in enumerate(gi):
            j[t], r[t] = b.step(X[g]); sc[t] = b.last[1] if b.last else np.nan
        _BIO[key] = (r, j, sc)
    return _BIO[key]


KAPPA = float(os.environ.get("KAPPA", 0))
G = A.L.G; T_ = G.T


def tsem_lik_prior(PM, KP):
    def f(P, sess, X, rho=0.7, shrink=0.7, m0=20, nmin=15):
        Lk = np.full_like(P, np.nan); tt = np.zeros(len(P))
        for s in np.unique(sess):
            r = np.flatnonzero(sess == s); lab = P[r].argmax(1); conf = P[r].max(1)
            for t in range(nmin, len(r)):
                A_, fx = X[r[:t]], X[r[t]]
                m, sd = A_.mean(0), np.maximum(A_.std(0), 1e-3)
                Zm, z = (A_ - m) / sd, (fx - m) / sd
                q = max(2, int(rho * t / 3)); sel = []
                for k in range(3):
                    ik = np.flatnonzero(lab[:t] == k); sel.append(ik[np.argsort(-conf[ik])[:q]])
                if min(len(x) for x in sel) < 2:
                    continue
                kp = KP[r[t]]
                mu = np.stack([(Zm[x].sum(0) + kp * PM[r[t], k]) / (len(x) + m0) for k, x in enumerate(sel)])
                R = np.concatenate([Zm[x] - mu[k] for k, x in enumerate(sel)])
                Cv = np.cov(R.T); Cv = (1 - shrink) * Cv + shrink * np.eye(len(Cv)) * np.trace(Cv) / len(Cv)
                Lk[r[t]] = T_._lda(z[None], mu, np.linalg.inv(Cv))[0]; tt[r[t]] = t
        return Lk, tt
    return f


def route4(mixes, shadows, a, oof, c1, mode, t0):
    outs, info = {}, []; R3.A = a
    for proto, folds in RR.FOLDS.items():
        idx = pl.B[f"idx|{proto}"]; pos = {int(g): k for k, g in enumerate(idx)}; Tsh = shadows[proto]
        mixr, routed = mixes[proto].copy(), []
        PM, KP = np.zeros((len(idx), 3, 128)), np.zeros(len(idx))
        for f in folds:
            te, names, Gn, Q, per, _ = R3.gate_model(proto, f, oof); hk = f"{proto}|{f}"
            pri = {q: np.mean([np.stack([R3.stdz(Xq[:, :128])[yq == k].mean(0) for k in range(3)]) for Xq, yq in per[q]], 0) for q in names}
            if hk not in A._HEADS:
                A._HEADS[hk] = R3.ps1_heads(per, names, C=PSC)
            for s, (Xs, y, gi) in te.items():
                r = np.array([pos[int(g)] for g in gi])
                if (proto, s) not in A._H:
                    Zc = RR.causal_std(Xs[:, :256]); A._H[(proto, s)] = np.stack([A._HEADS[hk][q].predict_proba(Zc) for q in names])
                r3, j3 = R3.rmap(R2.scores_session(Xs, Tsh[r], Gn, Q), "lin05")
                rb, jb, sb = bio_stream(proto, f, s, np.asarray(gi), names, t0)
                if mode == "v3":
                    rr, j = r3, j3
                elif mode == "bio":
                    rr, j = rb, jb
                else:
                    use = rb > r3; rr, j = np.where(use, rb, r3), np.where(use, jb, j3)
                if a > 0 and (rr > 0).any():
                    mixr[r] = R3.slot_mix(mixes[proto][r], c1[proto][r], A._H[(proto, s)], rr, j); routed.append(r)
                    for t in np.flatnonzero(rr > 0):
                        KP[r[t]] = KAPPA * rr[t]; PM[r[t]] = pri[names[int(j[t])]]
                    t1 = int(np.flatnonzero(rr > 0)[0]); tb = np.flatnonzero(rb > 0)
                    info.append((proto, s.split("/")[-1], names[int(j[t1])], t1, int(tb[0]) if len(tb) else -1, float(np.nanmax(sb)), float((rr > 0).mean())))
        out = Tsh.copy()
        if routed:
            lik0 = G.tsem_lik
            if KAPPA > 0 and mode != "v3":
                G.tsem_lik = tsem_lik_prior(PM, KP)
            try:
                P = gm.pipe_g(mixr, proto)
            finally:
                G.tsem_lik = lik0
            rr_ = np.concatenate(routed); out[rr_] = P[rr_]
        outs[proto] = out
    return outs, info


def system4(bn, mode, t0):
    mixes, sh, a = {}, {}, 0.0
    for p in A.PROTOS:
        mixes[p], a = A.mix(bn, p, A.FAMS); _, sh[p] = A.shadow(mixes[p], p)
    return route4(mixes, sh, a, sh["all_loso"], {p: A.members(bn, p)["c1"] for p in A.PROTOS}, mode, t0)


bases = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] != "-" else list(env.EXPL)
CFG = (("v3", "v3", 10), ("v4 max t10", "max", 10)) if "fresh" in sys.argv else (("v3", "v3", 10), ("v4 max t10", "max", 10), ("v4 max t20", "max", 20), ("bio only t10", "bio", 10))
for bn in bases:
    t0_ = time.time(); res = {}
    for tag, mode, t0 in CFG:
        out, info = system4(bn, mode, t0); dn, ds = SC.person(out["all_loso"], "all_loso"), SC.person(out["seen_session"], "seen_session")
        res[tag] = (dn, ds); d0n, d0s = res["v3"]
        nnew = [i for i in info if i[0] == "all_loso"]; nseen = [i for i in info if i[0] == "seen_session"]
        print(f"{bn} {tag:13s} J {0.5 * dn.mean() + 0.5 * ds.mean():.4f} dJ {0.5 * (dn - d0n).mean() + 0.5 * (ds - d0s).mean():+.2f} new {(dn - d0n).mean():+.2f} "
              f"seen {(ds - d0s).mean():+.2f} (P04 {ds[0] - d0s[0]:+.2f}, P10 {ds[1] - d0s[1]:+.2f}) | routed new {len(nnew)} seen {len(nseen)}", flush=True)
        if bn == bases[0]:
            for i in info:
                print(f"      {i[0]:12s} {i[1]:8s} -> {i[2]} first routed window {i[3]}, first bio window {i[4]}, max bio score {i[5]:+.3f}, routed share {i[6]:.2f}")
    print(f"   {time.time() - t0_:.0f}s", flush=True)
new = sorted((round(float(np.nanmax(v[2])), 3), k[2].split("/")[-1]) for k, v in _BIO.items() if k[0] == "all_loso" and k[3] == 10)
print("new-person sessions, max identity score:", new)
