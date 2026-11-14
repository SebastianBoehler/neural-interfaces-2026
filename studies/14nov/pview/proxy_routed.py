"""s262: the returning-people personal path on the within-session proxy of all ten people (oracle recognition).
Proxy as PV6 (tproxy.py): each public session's acquisition runs play the person's public session, the online runs are the
later session and are scored. Personal parts fitted on the acquisition rows only, then applied to the online rows with the
person known and ramp r = 1 (acquisition rows are never routed: r = 0). Package functions: R3.ps1_heads, R3.slot_mix,
RR.causal_std, ownspace.Personal (own-space stage, beta 0.10), gm.pipe_g, TSEM class-mean prior (kappa 20, chain.py's form).
ER inputs (chain.py system()): rest member at weight 1 in the mix (a = 1/16), eye-net log posteriors as 3 more GE columns.
Arms (all in the same code path `routed`):
  tail0  no ER, no personal parts (zero-change: equals tproxy.npz base)    ERoff  ER inputs, no personal parts
  P / PT heads + own-space stage on the tail mix (a = 1/15) / with the prior
  ER / ERT  ER inputs + heads + own-space stage / with the prior
Usage: [BASES=..] [ORDERS=within0,..] [ARMS=..] python studies/14nov/pview/proxy_routed.py -> runs/pview/routed262_<orders>.npz"""
import os, sys, time
import numpy as np
os.environ.setdefault("SCHEMES", "within,full,runs"); os.environ.setdefault("PERMS", "4")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.abspath(HERE + "/../../13nov"), os.path.abspath(HERE + "/../../12nov")]
import shuffle as S  # noqa: E402
import data as D  # noqa: E402
import ownspace as O  # noqa: E402
A, gm, p0 = S.A, S.gm, S.p0; G, Z, R3, RR = gm.L.G, A.L.Z, A.R3, A.RR; T_ = G.T
BASES = os.environ.get("BASES", "B2_pool,B2_s345,B2_s012,B2_s0,B2_s1,B2_s2").split(",")
ORDERS = os.environ.get("ORDERS", "within0,within1,within2,within3,full0,full1,runs0,runs1").split(",")
ARMS = os.environ.get("ARMS", "tail0,ERoff,P,PT,ER,ERT").split(",")
KAP, BETA, PSC, G3 = 20.0, 0.10, 0.1, 3.0
assert R3.G3 == G3, "R3.G3 is not the package's head gain"
TASK = np.load(D.REPO + "/data/proteus_windows.npz")["task"].astype(str); ONL = np.char.startswith(TASK, "Online")
TP = np.load(D.OUT + "/tproxy.npz"); PF = np.load(D.REPO + "/runs/shuffle_perms.npz")
PC256 = np.full((D.N, 256), np.nan)  # core1 PC256 in the held-out person's basis (the heads' input), recording layout
for q in D.PEOPLE:
    z = np.load(f"{D.REPO}/runs/replay_data/feats_exact/all_loso-{q}.npz", allow_pickle=True)
    for k in range(len(z["sessions"])):
        PC256[z[f"{k}|idx"].astype(int)] = z[f"{k}|pc"]
assert np.isfinite(PC256).all() and np.abs(PC256[:, :128] - D.PC).max() < 1e-9, "feats_exact PCs differ from pc128|all_loso"
PERS, HEAD = {}, {}  # per (person, session): own-space members and head, fitted on the acquisition rows (order-free)


def fitted(s, acq):
    if s not in HEAD:
        q = D.PER[acq[0]]; assert (D.PER[acq] == q).all() and not ONL[acq].any()
        HEAD[s] = R3.ps1_heads({q: [(PC256[acq], D.Y[acq])]}, [q], C=PSC)[q]
        PERS[s] = O.Personal([acq])
        Xs = R3.stdz(PC256[acq][:, :128]); HEAD[s, "pm"] = np.stack([Xs[D.Y[acq] == k].mean(0) for k in range(3)])
    return HEAD[s], PERS[s], HEAD[s, "pm"]


def lik_prior(PM, KP):  # chain.py's tsem_lik_prior (kp pseudo-windows of PM in the class means, denominator included)
    def f(P, sess, X, rho=0.7, shrink=0.7, m0=20, nmin=15):
        Lk = np.full_like(P, np.nan); tt = np.zeros(len(P))
        for s in np.unique(sess):
            r = np.flatnonzero(sess == s); lab = P[r].argmax(1); conf = P[r].max(1)
            for t in range(nmin, len(r)):
                A_, fx = X[r[:t]], X[r[t]]; m, sd = A_.mean(0), np.maximum(A_.std(0), 1e-3); Zm, z = (A_ - m) / sd, (fx - m) / sd
                q = max(2, int(rho * t / 3)); sel = [np.flatnonzero(lab[:t] == k)[np.argsort(-conf[np.flatnonzero(lab[:t] == k)])[:q]] for k in range(3)]
                if min(len(x) for x in sel) < 2:
                    continue
                kp = KP[r[t]]; mu = np.stack([(Zm[x].sum(0) + kp * PM[r[t], k]) / (len(x) + m0 + kp) for k, x in enumerate(sel)])
                Rr = np.concatenate([Zm[x] - mu[k] for k, x in enumerate(sel)]); Cv = np.cov(Rr.T)
                Cv = (1 - shrink) * Cv + shrink * np.eye(len(Cv)) * np.trace(Cv) / len(Cv)
                Lk[r[t]] = T_._lda(z[None], mu, np.linalg.inv(Cv))[0]; tt[r[t]] = t
        return Lk, tt
    return f


def routed(mix, a, c1, idx, personal, kappa=0.0):
    """-> (n, 3) outputs in store layout. personal False: the plain tail on `mix`."""
    n = len(mix); mixr, PM, KP, plan = mix.copy(), np.zeros((n, 3, 128)), np.zeros(n), []
    if personal:
        R3.A = a
        sess = Z[f"sess|{p0}"]
        for s in dict.fromkeys(sess):
            rows = np.flatnonzero(sess == s); gi = idx[rows]; acq = gi[~ONL[gi]]
            head, pers, pm = fitted(s, acq)
            r = ONL[gi].astype(float); j = np.zeros(len(rows), int)
            H = head.predict_proba(RR.causal_std(PC256[gi]))[None]
            mixr[rows] = R3.slot_mix(mix[rows], c1[rows], H, r, j); KP[rows] = kappa * r; PM[rows] = pm; plan.append((rows, gi, r, pers))
    lik0 = G.tsem_lik
    if personal and kappa > 0:
        G.tsem_lik = lik_prior(PM, KP)
    try:
        out = gm.pipe_g(mixr, p0)
    finally:
        G.tsem_lik = lik0
    for rows, gi, r, pers in plan:  # own-space stage, routed rows only
        m = pers.stream(gi)
        for t in np.flatnonzero(r > 0):
            L = np.log(np.clip(out[rows[t]], 1e-9, 1)) + BETA * r[t] * sum(np.log(np.clip(m[k][t], 1e-3, 1)) for k in "TCS") / 3
            e = np.exp(L - L.max()); out[rows[t]] = e / e.sum()
    return out


def ba_online(P, idx):
    on = ONL[idx]; per = D.PER[idx]
    return np.array([D.ba(idx[on & (per == q)], P[on & (per == q)].argmax(1)) for q in D.PEOPLE])


def ld(path):
    z = np.load(path); assert (z["index"].astype(int) == np.arange(D.N)).all(), path; return z["proba"]


R, t0 = {}, time.time()
for order in ORDERS:
    sc, sd = order.rstrip("0123456789"), order[len(order.rstrip("0123456789")):]
    P = S.perm_for(sc, int(sd)); S.apply(P); idx = Z[f"idx|{p0}"].astype(int)
    assert (idx == PF[f"{sc}|{sd}|{p0}"]).all() and (idx == np.arange(D.N)[P[p0]]).all(), "the order is not the pinned order file's"
    eye = np.log(np.clip(np.mean([ld(f"{D.REPO}/runs/ocushuf/{order}/oc_nomix-{p0}-s{k}.npz") for k in range(3)], 0), 1e-3, 1))[P[p0]]
    rest = np.mean([ld(f"{D.REPO}/runs/restshuf/{order}/rest4_sh_r1-{p0}-s{k}.npz") for k in range(3)], 0)[P[p0]]
    ge0 = Z[f"GE|{p0}"]
    for bn in BASES:
        ref = TP[f"{order}|{bn}|base"]; mix0, a0 = A.mix(bn, p0, A.FAMS); assert abs(a0 - 1 / 15) < 1e-12; c1 = A.members(bn, p0)["c1"]
        for arm in ARMS:
            er = arm.startswith("ER"); pers = arm in ("P", "PT", "ER", "ERT"); kap = KAP if arm.endswith("T") and pers else 0.0
            try:
                if er:
                    A.EXTRA["rt_rest"] = {p0: rest}; Z[f"GE|{p0}"] = np.c_[ge0, eye]; mix, a = A.mix(bn, p0, A.FAMS, [("rt_rest", 1.0)]); assert abs(a - 1 / 16) < 1e-12
                    c1 = A.members(bn, p0)["c1"]
                else:
                    mix, a = mix0, a0
                out = routed(mix, a, c1, idx, pers, kap)
            finally:
                Z[f"GE|{p0}"] = ge0; A.EXTRA.pop("rt_rest", None); R3.A = 1 / 15
            R[order, bn, arm] = ba_online(out, idx)
            if arm == "tail0":
                assert np.allclose(R[order, bn, arm], ref, atol=1e-9), f"{order} {bn}: tail0 {R[order, bn, arm]} != tproxy base {ref} (zero-change failed)"
        b = R[order, bn, "tail0"] if "tail0" in ARMS else ref
        print(f"{order} {bn}: tail {b.mean():.2f} | " + " ".join(f"{k} {(R[order, bn, k] - b).mean():+.2f}" for k in ARMS if k != "tail0")
              + f" | P01 " + " ".join(f"{k} {R[order, bn, k][0] - b[0]:+.2f}" for k in ARMS if k != "tail0") + f" [{time.time() - t0:.0f}s]", flush=True)
S.apply({q: np.arange(S.N[q]) for q in S.PR})
os.makedirs(D.OUT, exist_ok=True)
np.savez(f"{D.OUT}/routed262_{os.environ.get('TAG', '')}{'-'.join(ORDERS)}{'' if len(BASES) == 6 else '_smoke'}.npz", **{f"{o}|{b}|{k}": v for (o, b, k), v in R.items()})
print("written", flush=True)
