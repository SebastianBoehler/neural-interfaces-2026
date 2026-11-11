"""FORGET (ledger s20): core1 with an exponentially forgetting causal recentring reference at test time.

Uses the outer heads of the weight-space cache (runs/wsp/cache). Writes posteriors per half-life to OUT.npz.
Usage: python experiments/forget_ref.py CACHE_DIR OUT.npz [half-lives, default inf,80,30]
"""
import glob, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT + "/experiments"]
import protocol  # noqa: E402
import ts_core2 as T  # noqa: E402

CACHE, OUT = sys.argv[1], sys.argv[2]
HL = [float(h) for h in (sys.argv[3] if len(sys.argv) > 3 else "inf,80,30").split(",")]
d = protocol.load(); fe = T.Feats(d, T.parse("car_ov17_fs2_t0.1_le_k2")); st = d["stream"]


def causal_forget(r, priors, rho):
    n = len(r); w = np.zeros(n); acc = 0.0
    for t in range(n):
        acc = rho * acc + 1.0; w[t] = acc
    out = []
    for b in range(len(fe.C)):
        Lb = fe.L[b][r]; S = np.empty_like(Lb); a = np.zeros_like(Lb[0])
        for t in range(n):
            a = rho * a + Lb[t]; S[t] = a
        out.append(T.whiten_tangent(fe.C[b][r], T.expm((fe.k * priors[b] + S) / (fe.k + w)[:, None, None])))
    M = fe.M[r]; Sm = np.empty_like(M); a = np.zeros_like(M[0])
    for t in range(n):
        a = rho * a + M[t]; Sm[t] = a
    out.append(M - Sm / (fe.k + w)[:, None])
    return np.concatenate(out, 1)


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


res = {}
for fn in sorted(glob.glob(CACHE + "/*-*.npz")):
    z = np.load(fn, allow_pickle=True); name = os.path.basename(fn)[:-4]; rows, idx = z["rows"], z["train"]
    pri = [fe.prior(b, idx) for b in range(len(fe.C))]
    for h in HL:
        rho = 1.0 if np.isinf(h) else 0.5 ** (1.0 / h); F = np.zeros((len(rows), fe.F.shape[1]))
        for s in dict.fromkeys(st[rows]):
            m = st[rows] == s; F[m] = causal_forget(rows[m], pri, rho)
        P = softmax(((F - z["o_mean"]) / z["o_scale"]) @ z["o_coef"].astype(np.float64).T + z["o_b"])
        res[f"h{h:g}|{name}|proba"] = P.astype(np.float32); res[f"h{h:g}|{name}|rows"] = rows
        if np.isinf(h):
            print(name, "h=inf max |diff| vs cache", float(np.abs(P - z["proba"]).max()), flush=True)
    np.savez(OUT, **res)
