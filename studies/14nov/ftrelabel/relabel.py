"""RELABEL (ledger s221): per-refit FT labels for new people. The label files of s178/s187 hold one output per window, as
emitted when it arrived; re-applying the chain's state at a later refit cannot be read from them. This script reruns the
label chain (shufcombo LABEL_OUT, LABEL_ARM=V6r: the v6 chain = v6.1's labels K for new people) with hooks on the two
adaptive stages (gm.em_g, gm.tsem_g). At each refit time t of the grid (delivery position within the session) it applies
the stages' state at t (fitted on windows 0..t-1 exactly as the stock stage fits it for window t) to every earlier window:
  EM:   q_j(t) = b_j * lda_t(z_j)^e(t)     TSEM: out_j(t) = q_j(t) * L_t(x_j)^e'(t)
b_j is the stage input as emitted (rebalancing stays as emitted); e, e' are the stock exponents at t with the gate value of
the window at t (computed from windows < t). Causal: nothing at or after t enters. Checks (fail loudly): the state applied to
window t reproduces the stock stage's output for window t; the file's as-emitted labels agree with chain_v6r_<order>.npz
(argmax agreement > 0.98; today's chain carries the s214 fix).
Output runs/labels/relab_v6r_<order>.npz: the keys of a label file plus relab (N, K, 3) (NaN where j >= t_k), tgrid (K,),
pos (N,) delivery position within the session. Usage: [KIND=v51b] SCHEME=within PSEED=0 OCU_DIR=runs/ocushuf/within0 python relabel.py
KIND=v51b (s229): LABEL_ARM=P0r, v5.1b's own stages -> relab_v51b_<order>.npz, checked against chain_v51b_<order>.npz."""
import os, runpy, sys
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ORD = f"{os.environ['SCHEME']}{os.environ['PSEED']}"; T0, EVERY = 20, 10  # s15's refit grid
KIND = os.environ.get("KIND", "v6r"); ARM = {"v6r": "V6r", "v51b": "P0r"}[KIND]; NAME = f"relab_{KIND}_{ORD}"  # s229: v51b = v5.1b's own chain
# FT_TAG / T17_DIR may name another tag's files: the label arms read only their base (frozen, re-streamed) outputs
for k in ("TSEM_TGT", "NUIS", "RUNOFF", "LABEL_ARM", "LABEL_OUT"):
    assert not os.environ.get(k), f"{k} would change the label chain (the chain_* files were written without it)"
os.environ.update(LABEL_OUT=NAME, LABEL_ARM=ARM, GATE_P0="1")  # GATE_P0 changes only the V6r arm's gate
sys.path[:0] = [REPO + "/studies/13nov", REPO + "/replay"]
import shuffle as S  # noqa: E402  (shufcombo imports the same module object)
gm = S.gm; L, M, SE, G = gm.L, gm.M, gm.SE, gm.L.G
EM0, TS0, CALLS = gm.em_g, gm.tsem_g, []


def grid(n):
    return list(range(T0, n, EVERY))


def em_hook(b, proto, g, bem=gm.BEM, nmin=gm.NMIN):
    out = EM0(b, proto, g, bem, nmin)
    if proto != "all_loso":
        return out
    F, sess = L.Z[f"GE|{proto}"], L.Z[f"sess|{proto}"]; K = max(len(grid(int((sess == s).sum()))) for s in set(sess))
    R = np.full((len(b), K, 3), np.nan); pos = np.zeros(len(b), int); lam, n0, m0, shrink, iters = 0.5, 60, 20, 0.7, 2
    for s in dict.fromkeys(sess):
        r = np.flatnonzero(sess == s); pos[r] = np.arange(len(r))
        for k, t in enumerate(grid(len(r))):
            q = b[r[:t + 1]].copy()
            if t >= nmin:
                A, B = F[r[:t]], b[r[:t]]; m, sd = A.mean(0), np.maximum(A.std(0), SE.SD_FLOOR); Zm = (A - m) / sd; Q = B
                for _ in range(iters):
                    mu = (Q / (Q.sum(0, keepdims=True) + m0)).T @ Zm; C = np.cov((Zm - Q @ mu).T)
                    C = (1 - shrink) * C + shrink * np.eye(len(C)) * np.trace(C) / len(C); Ci = np.linalg.inv(C)
                    Q = B * SE._softmax_lda(Zm, mu, Ci) ** lam; Q = Q / Q.sum(1, keepdims=True)
                e = lam * t / (t + n0) * (1 + bem * (M.h(t, 30) * g[r[t]]))
                q = q * SE._softmax_lda((F[r[:t + 1]] - m) / sd, mu, Ci) ** e; q = q / q.sum(1, keepdims=True)
            assert np.allclose(q[t], out[r[t]], atol=1e-9), f"EM state at t={t} does not reproduce the stock output"
            R[r[:t], k] = q[:t]
    CALLS.append(("em", b, g, out, R, pos, L.Z[f"y|{proto}"].astype(int).copy())); return out


def ts_hook(P, proto, g, bts=gm.BTS):
    out = TS0(P, proto, g, bts)
    if proto != "all_loso":
        return out
    assert CALLS and CALLS[-1][0] == "em" and CALLS[-1][3] is P, "TSEM's input is not the EM output just recorded"
    Rem = CALLS[-1][4]; R = Rem.copy(); X, sess = L.Z[f"pc128|{proto}"], L.Z[f"sess|{proto}"]; rho, shrink, m0, nmin = 0.7, 0.7, 20, 15
    for s in dict.fromkeys(sess):
        r = np.flatnonzero(sess == s); lab, conf = P[r].argmax(1), P[r].max(1)
        for k, t in enumerate(grid(len(r))):
            ref = P[r[t]]
            if t >= nmin:
                A = X[r[:t]]; m, sd = A.mean(0), np.maximum(A.std(0), 1e-3); Zm = (A - m) / sd; q_ = max(2, int(rho * t / 3)); sel = []
                for c in range(3):
                    ik = np.flatnonzero(lab[:t] == c); sel.append(ik[np.argsort(-conf[ik])[:q_]])
                if min(len(x) for x in sel) >= 2:
                    mu = np.stack([Zm[x].sum(0) / (len(x) + m0) for x in sel]); Rr = np.concatenate([Zm[x] - mu[c] for c, x in enumerate(sel)])
                    Cv = np.cov(Rr.T); Cv = (1 - shrink) * Cv + shrink * np.eye(len(Cv)) * np.trace(Cv) / len(Cv)
                    Lk = G.T._lda((X[r[:t + 1]] - m) / sd, mu, np.linalg.inv(Cv)); e = 0.15 * t / (t + 60) * (1 + bts * (M.h(t, 30) * g[r[t]]))
                    ref = P[r[t]] * Lk[t] ** e; ref = ref / ref.sum()
                    q = Rem[r[:t], k] * Lk[:t] ** e; R[r[:t], k] = q / q.sum(1, keepdims=True)
            assert np.allclose(ref, out[r[t]], atol=1e-9), f"TSEM state at t={t} does not reproduce the stock output"
    CALLS.append(("ts", P, g, out, R, *CALLS[-1][5:])); return out


gm.em_g, gm.tsem_g = em_hook, ts_hook
try:
    runpy.run_path(REPO + "/studies/13nov/shufcombo.py", run_name="shufcombo")
except SystemExit:
    pass
f = f"{REPO}/runs/labels/{NAME}.npz"; z = dict(np.load(f)); ref = np.load(f"{REPO}/runs/labels/chain_{KIND}_{ORD}.npz")
_, _, _, out, R, pos, y = CALLS[-1]; P = S.perm_for(os.environ["SCHEME"], int(os.environ["PSEED"]))["all_loso"]; o = np.argsort(P)
assert np.array_equal(out[o], z["all_loso"]), "the last TSEM call is not the label arm"
d = np.abs(z["all_loso"] - ref["all_loso"]).max(); agr = np.mean(z["all_loso"].argmax(1) == ref["all_loso"].argmax(1))
# 14 Nov: today's chain differs from the 13 Nov files (s214 fixed the eye outputs' order per protocol): within 1 changes 42 of
# 2655 argmax labels, the others 0-6. ftrel.py takes the as-emitted labels from this file, so all its arms share one chain.
print(f"as-emitted labels against chain_{KIND}_{ORD}: max diff {d:.1e}, argmax agreement {agr:.4f}", flush=True); assert agr > float(os.environ.get("MIN_AGREE", 0.98)), agr
# MIN_AGREE (s229): chain_v51b of within 0-3, full 0/1, runs 0/1 were written 13 Nov 13:16-14:49, before the s184 GE, Hilbert
# and teacher re-streams (15:21-15:43) that shufcombo now adds: today's chain agrees 0.948-0.976 there, 1.0000 on the later files
R, pos, y = R[o], pos[o], y[o]
np.savez(f, **z, relab=R, tgrid=np.array(grid(10**6)[:R.shape[1]]), pos=pos)
for k in range(0, R.shape[1], 4):
    m = ~np.isnan(R[:, k, 0]); print(f"refit {k} (t={T0 + EVERY * k}): n={m.sum()} acc relabelled {np.mean(R[m, k].argmax(1) == y[m]):.4f} "
                                     f"as emitted {np.mean(z['all_loso'][m].argmax(1) == y[m]):.4f}", flush=True)
print("written", f, flush=True)
