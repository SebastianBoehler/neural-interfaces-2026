"""SOFT OFFSETS (ledger s251 (2)): the s210 run offsets with soft membership. For new people (all_loso), each row of
SessionEM's view (GE, incl. the eye columns) and TSEM's view (pc128), per session in delivery order, becomes
    v_t - n_t / (n_t + KAPPA) * (m_w - m_s),  m_w = sum_{j<t} w'_tj v_j / n_t,  n_t = sum_{j<t} w'_tj,
m_s the session's prefix mean (current row excluded), w' from fpw.py (forms: zero = w' 0, the plumbing check; kern = w;
top10 = w on the 10 earlier windows with the smallest relative distance). Causal; nothing crosses sessions.
Installed by wrapping gm.em_g / gm.tsem_g (the views are swapped in for the call and restored after); shufcombo.py is run
unmodified through runpy, once per form in one process, env as the s242 r15 arm. The current delivery order is read from
the permuted replay store (A.pl.B['idx|all_loso'], as designdec/stage.py). SAVE_OUT -> runs/softfp/so_<order>_<form>.npz;
shufcombo's summary npz is redirected into runs/softfp/.
(3) LOCAL TEMPLATES IN TSEM (s251 (3), forms tzero / tloc): grid.tsem_lik replaced for all_loso; at window t the earlier
windows get relative weights a_j = w_tj / mean_{j'<t} w_tj' (fpw kernel); the centre is the a-weighted mean of the
earlier rows and each class template is sum_j a_j z_j / (sum_j a_j + m0) over the same confident seed windows as stock
(covariance, seeds, shrinkage unchanged). tzero runs the same function with a = None: the stock code path, the plumbing
check (must equal s242r15 exactly).
Usage: [BASES=...] [input dirs as wrap.py] python softoff.py <order> zero,kern,top10,tzero,tloc"""
import os, runpy, sys, time
order, forms = sys.argv[1], sys.argv[2].split(","); sc = order.rstrip("0123456789"); REPO = os.path.abspath(__file__ + "/../../../..")
env = dict(SCHEME=sc, PSEED=order[len(sc):], STRICT="1", GATE_P0="1", OCU_DIR=f"runs/ocushuf/{order}", FT_V="r15", REC_PFX="ttt7",
           T6_DIR="runs/ttt/probs7", T17_DIR="runs/ttt/probs18", FT_TAG=f"ttt18{order}LP", PYTHONWARNINGS="ignore")
os.environ.update({k: os.environ.get(k, v) for k, v in env.items()}); os.environ.setdefault("BASES", "B2_pool"); os.chdir(REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import fpw  # noqa: E402
C = fpw.C; gm = C.A.gm; KAPPA, TOPK = 5.0, 10
CFG = {"form": "off", "calls": 0}; _W = {}


def wform(rows, form=None):
    form = form or CFG["form"]; key = (form, rows.tobytes())
    if key not in _W:
        W = fpw.wmat(rows)
        if form == "zero":
            W = np.zeros_like(W)
        elif form == "top10":
            out = np.zeros_like(W)
            for t in range(1, len(W)):
                j = np.argsort(-W[t, :t], kind="stable")[:TOPK]; out[t, j] = W[t, j]
            W = out
        else:
            assert form == "kern", form
        _W[key] = W
    return _W[key]


def recentre(V):
    P = C.A.pl.B[f"idx|{C.p0}"].astype(int); ss = gm.L.Z[f"sess|{C.p0}"]; assert len(P) == len(V) == len(ss) == C.N
    assert (C.SES[P] == ss).all(), "the replay store's order does not match its session column"
    out = V.astype(np.float64).copy()
    for s in dict.fromkeys(ss):
        pos = np.flatnonzero(ss == s); Vs = out[pos].copy(); W = wform(P[pos]); n = W.sum(1)
        cs = np.vstack([np.zeros((1, Vs.shape[1])), np.cumsum(Vs, 0)])[:-1]; t = np.arange(len(pos)); ms = cs / np.maximum(t, 1)[:, None]
        mw = (W @ Vs) / np.maximum(n, 1e-12)[:, None]; f = np.where(t > 0, n / (n + KAPPA), 0.0)
        out[pos] = Vs - f[:, None] * (mw - ms)
    return out.astype(V.dtype)


def tsem_local(stock, T):
    def tsem_lik(P, sess, X, rho=0.7, shrink=0.7, m0=20, nmin=15):
        if CFG["form"] not in ("tzero", "tloc") or len(P) != C.N:
            return stock(P, sess, X, rho, shrink, m0, nmin)
        CFG["calls"] += 1; L = np.full_like(P, np.nan); tt = np.zeros(len(P)); Pi = C.A.pl.B[f"idx|{C.p0}"].astype(int)
        assert (C.SES[Pi] == sess).all(), "the replay store's order does not match TSEM's session column"
        for s in np.unique(sess):
            r = np.flatnonzero(sess == s); lab = P[r].argmax(1); conf = P[r].max(1); W = wform(Pi[r], "kern") if CFG["form"] == "tloc" else None
            for t in range(nmin, len(r)):
                A_, f = X[r[:t]], X[r[t]]; a = None
                if W is not None and W[t, :t].sum() > 0:
                    a = W[t, :t] / W[t, :t].mean()
                m, sd = (A_.mean(0) if a is None else (a @ A_) / a.sum()), np.maximum(A_.std(0), 1e-3)
                Zm, z = (A_ - m) / sd, (f - m) / sd
                q = max(2, int(rho * t / 3)); sel = []
                for k in range(3):
                    ik = np.flatnonzero(lab[:t] == k); sel.append(ik[np.argsort(-conf[ik])[:q]])
                if min(len(x) for x in sel) < 2:
                    continue
                if a is None:
                    mu = np.stack([Zm[x].sum(0) / (len(x) + m0) for x in sel])
                else:
                    mu = np.stack([(a[x, None] * Zm[x]).sum(0) / (a[x].sum() + m0) for x in sel])
                R = np.concatenate([Zm[x] - mu[k] for k, x in enumerate(sel)])
                Cv = np.cov(R.T); Cv = (1 - shrink) * Cv + shrink * np.eye(len(Cv)) * np.trace(Cv) / len(Cv)
                L[r[t]] = T._lda(z[None], mu, np.linalg.inv(Cv))[0]; tt[r[t]] = t
        return L, tt
    return tsem_lik


def install():
    em0, ts0 = gm.em_g, gm.tsem_g; G = gm.L.G; G.tsem_lik = tsem_local(G.tsem_lik, G.T)

    def wrapped(stock, key):
        def f(b, proto, g, *a, **k):
            if proto != C.p0 or CFG["form"] in ("off", "tzero", "tloc"):
                return stock(b, proto, g, *a, **k)
            z = f"{key}|{proto}"; v0 = gm.L.Z[z]; gm.L.Z[z] = recentre(v0); CFG["calls"] += 1
            try:
                return stock(b, proto, g, *a, **k)
            finally:
                gm.L.Z[z] = v0
        return f
    gm.em_g, gm.tsem_g = wrapped(em0, "GE"), wrapped(ts0, "pc128")


OUT = f"{REPO}/runs/softfp"; os.makedirs(OUT, exist_ok=True); _savez = np.savez


def savez(path, *a, **k):
    p = str(path)
    if os.path.basename(p).startswith("shufcombo_"):
        p = f"{OUT}/{TAG}_{os.path.basename(p)}"
    assert p.startswith(OUT), f"softoff.py would write outside runs/softfp: {p}"
    return _savez(p, *a, **k)


if __name__ == "__main__":
    np.savez = savez; install()
    for fm in forms:
        CFG.update(form=fm, calls=0); TAG = f"so_{order}_{fm}"; os.environ["SAVE_OUT"] = f"runs/softfp/{TAG}.npz"; t0 = time.time()
        print(f"=== SOFTOFF {order} {fm} (BASES {os.environ['BASES']})", flush=True)
        runpy.run_path(f"{REPO}/studies/13nov/shufcombo.py", run_name="__main__")
        assert CFG["calls"] > 0, "the hook never ran: silent no-op"
        print(f"=== {fm}: {time.time() - t0:.0f}s, hooked stage calls {CFG['calls']}", flush=True)
