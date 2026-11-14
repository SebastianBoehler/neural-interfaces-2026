"""DESIGNDEC stage (ledger s239): gated SessionEM + gated TSEM (replay/gm.py em_g, tsem_g; grid.tsem_lik) run window by
window together with the run smoother (smoother.py), so that the smoothed run posterior of windows < t can stand in for
the stages' soft labels at window t (one-way: windows < t are never re-emitted).
  DD_FB  off | rep | ext | fwd   (rep: smoothed posterior replaces SessionEM's past inputs B and TSEM's seed labels;
         ext: the urn's extrinsic message multiplied into B_s and into TSEM's e_s; fwd: the filter's emitted posterior)
  DD_B   J (J-wprior, urn weights, s226/s228c) | true (true runs and caps; contiguous orders only)
  DD_ON  latch (from the s228c detector latch, default) | all (from window 0)
install(gm) replaces gm.em_g / gm.tsem_g: em_g runs the joint pass for new people in a non-recording order and keeps
the final output; tsem_g returns it for that exact array (anything else: the stock stages). The current window order is
read from the permuted replay store (A.pl.B['idx|all_loso'] = the delivery order after shuffle.apply)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, HERE + "/../rundetect")
import smoother as SM  # noqa: E402
import common as C  # noqa: E402
import detect as D  # noqa: E402
import structured as ST  # noqa: E402
import package as PK  # noqa: E402  (detector thresholds, X, LAM, CHECKS)
gm, SE = C.A.gm, C.A.gm.SE
CFG = {"fb": os.environ.get("DD_FB", "off"), "bnd": os.environ.get("DD_B", "J"), "on": os.environ.get("DD_ON", "latch"),
       "stg": os.environ.get("DD_STG", "both")}  # stg: which stage takes the soft labels (both | em | ts)


def check():
    assert CFG["fb"] in ("off", "rep", "ext", "fwd") and CFG["bnd"] in ("J", "true") and CFG["on"] in ("latch", "all") and CFG["stg"] in ("both", "em", "ts"), CFG


check()
T_LAT = int(os.environ.get("DD_TLAT", 0))  # > 0: soft labels only once t >= this (diagnostic)


def latch(rows):
    p = C.PER[rows[0]]; ws = [w for w in PK.CHECKS if w <= len(rows)]; v = D.stat(PK.X, rows, "lag", ws)
    hit = [w for w, x in zip(ws, v) if x < PK.THR[f"lag|{p}"][PK.WS.index(w)]]; return hit[0] if hit else None


def hyps(rows):
    """rows: recording rows of one session in delivery order -> (lp, reset (T, H), cap (T, H)), as jointcount.hyp_cache."""
    T = len(rows)
    if CFG["bnd"] == "true":
        r = C.RUNI[rows]; reset = np.r_[True, r[1:] != r[:-1]][:, None]
        assert reset.sum() == len(np.unique(r)), "true runs are not contiguous in this order"
        return np.zeros(1), reset, np.array([SM.J.TRUE_CAP[C.RUN[x]] for x in rows])[:, None]
    p = C.PER[rows[0]]; Ls, lp, cap = ST.hyps("wprior", p); _, bounds = ST.posterior_path(PK.X[rows], PK.LAM[p], Ls, lp, use_x=False)
    blk = np.stack([ST.block_index(bounds, t) for t in range(T)]); reset = np.vstack([np.ones(len(Ls), bool), blk[1:] != blk[:-1]])
    capx = np.concatenate([cap, np.full((len(Ls), 2), 12)], 1); return lp, reset, capx[np.arange(len(Ls))[None], blk]


def em_adapt(A, B, f, p, t, g, bem, lam=0.5, n0=60, m0=20, shrink=0.7, iters=2):  # = gm.EMg._adapt
    m, sd = A.mean(0), np.maximum(A.std(0), SE.SD_FLOOR); Z, z = (A - m) / sd, (f - m) / sd; Q = B
    for _ in range(iters):
        mu = (Q / (Q.sum(0, keepdims=True) + m0)).T @ Z; Cm = np.cov((Z - Q @ mu).T)
        Cm = (1 - shrink) * Cm + shrink * np.eye(len(Cm)) * np.trace(Cm) / len(Cm); Ci = np.linalg.inv(Cm)
        Q = B * SE._softmax_lda(Z, mu, Ci) ** lam; Q = Q / Q.sum(1, keepdims=True)
    e = lam * t / (t + n0) * (1 + bem * (gm.M.h(t, 30) * g)); q = p * SE._softmax_lda(z[None], mu, Ci)[0] ** e
    return q / q.sum()


def tsem_at(Xs, f, P, rho=0.7, shrink=0.7, m0=20):  # = grid.tsem_lik for one t (P: label posteriors of windows < t)
    t = len(Xs); m, sd = Xs.mean(0), np.maximum(Xs.std(0), 1e-3); Zm, z = (Xs - m) / sd, (f - m) / sd
    lab, conf = P.argmax(1), P.max(1); q = max(2, int(rho * t / 3)); sel = []
    for k in range(3):
        ik = np.flatnonzero(lab == k); sel.append(ik[np.argsort(-conf[ik])[:q]])
    if min(len(x) for x in sel) < 2:
        return None
    mu = np.stack([Zm[x].sum(0) / (len(x) + m0) for x in sel]); R = np.concatenate([Zm[x] - mu[k] for k, x in enumerate(sel)])
    Cv = np.cov(R.T); Cv = (1 - shrink) * Cv + shrink * np.eye(len(Cv)) * np.trace(Cv) / len(Cv)
    return gm.L.G.T._lda(z[None], mu, np.linalg.inv(Cv))[0]


def norm(x):
    return x / x.sum(1, keepdims=True)


def joint(b, F, X, g, rows, bem, nmin, bts, stats=None):
    """one session in delivery order -> (SessionEM output, final output). fb 'off' = the stock stages exactly."""
    check(); fb = CFG["fb"]; T = len(b); e = b.copy(); fin = b.copy(); w0 = 0 if CFG["on"] == "all" else latch(rows)
    u = SM.PrefixUrn(*hyps(rows)) if fb != "off" else None; w0 = max(w0, T_LAT) if w0 is not None else None
    for t in range(T):
        soft = u.soft(t, fb) if (fb != "off" and w0 is not None and t >= w0 and t > 0) else None
        if t >= nmin:
            B = b[:t] if soft is None or CFG["stg"] == "ts" else norm(soft if fb != "ext" else b[:t] * soft)
            try:
                o = em_adapt(F[:t], B, F[t], b[t], t, g[t], bem)
            except np.linalg.LinAlgError:
                o = b[t]
            e[t] = o if np.isfinite(o).all() else b[t]
        if t >= 15:
            Lk = tsem_at(X[:t], X[t], e[:t] if soft is None or CFG["stg"] == "em" else norm(soft if fb != "ext" else e[:t] * soft))
            if Lk is not None:
                ee = 0.15 * t / (t + 60) * (1 + bts * (gm.M.h(t, 30) * g[t])); q = e[t] * Lk ** ee; fin[t] = q / q.sum()
            else:
                fin[t] = e[t]
        else:
            fin[t] = e[t]
        if u is not None:
            u.step(t, fin[t])
    if stats is not None:
        stats.append(w0)
    return e, fin


STATE = {"e": None, "fin": None, "calls": 0, "latch": []}


def install(gm_=gm, p0="all_loso"):
    stock_em, stock_ts = gm_.em_g, gm_.tsem_g

    def em_g(b, proto, g, bem=None, nmin=None):
        bem = gm_.BEM if bem is None else bem; nmin = gm_.NMIN if nmin is None else nmin
        Pd = C.A.pl.B[f"idx|{p0}"].astype(int)
        if proto != p0 or (Pd == np.arange(len(Pd))).all():
            return stock_em(b, proto, g, bem, nmin)
        sess, F, X = gm_.L.Z[f"sess|{proto}"], gm_.L.Z[f"GE|{proto}"], gm_.L.Z[f"pc128|{proto}"]
        e, fin = b.copy(), b.copy()
        for s in dict.fromkeys(sess):
            r = np.flatnonzero(sess == s)
            e[r], fin[r] = joint(b[r], F[r], X[r], g[r], Pd[r], bem, nmin, gm_.BTS, stats=STATE["latch"])
        STATE.update(e=e, fin=fin); STATE["calls"] += 1; return e

    def tsem_g(P, proto, g, bts=None):
        if STATE["e"] is not None and P is STATE["e"]:
            out = STATE["fin"]; STATE.update(e=None, fin=None); return out
        return stock_ts(P, proto, g, gm_.BTS if bts is None else bts)
    gm_.em_g, gm_.tsem_g = em_g, tsem_g
