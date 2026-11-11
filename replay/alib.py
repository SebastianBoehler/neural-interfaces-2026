"""Lean study (Part A): any member subset / extra member through the packaged chain
mix -> rebal (a 1.15, k 5, W 30) -> GEB SessionEM -> GEB TSEM (eb2/gm.py) -> routing v3 (route2/rlib3, lin05),
routing's personal head share = core1's share of the lean mix (package: Route3.bind(a = w[0])), gallery debias from
the lean system's own all_loso output. Members (dev stack, retune/inputs.npz) split into families:
  core1, hilb (slot = mean of the two at weight 2 -> each 1), tf (FBCNet TF ea/raw, 0.5 each), shallow (ea/raw, 1 each),
  sbnfb (SBN FBCNet raw k20, ea k2, ea k5: 1 each), sbnsinc (1), sbnfbl (1), teacher (5)."""
import paths  # noqa: F401
import numpy as np
import gm  # noqa: E402  (early/lib as L, early/mech as M, GEB stages)
L, M = gm.L, gm.M
import rlib3 as R3  # noqa: E402
pl, R2, RL, RR = R3.pl, R3.R2, R3.L, R3.R
import score as SC  # noqa: E402  (tsem_confirm/score.py)

PROTOS = ("all_loso", "seen_session")
FAM = {"core1": [("c1", 1.0)], "hilb": [("hb", 1.0)], "tf": [(1, 0.5), (2, 0.5)], "shallow": [(3, 1.0), (4, 1.0)],
       "sbnfb": [(5, 1.0), (6, 1.0), (9, 1.0)], "sbnsinc": [(7, 1.0)], "sbnfbl": [(8, 1.0)], "teacher": [(10, 5.0)]}
FAMS = list(FAM)
C1 = {p: pl.get("B2_pool", "core1", p) for p in PROTOS}
EXTRA = {}  # name -> {proto: (n, 3)} extra members aligned to the base rows (Part B / B3)
for p in PROTOS:
    assert (L.Z[f"idx|{p}"] == pl.B[f"idx|{p}"]).all()
assert (L.Z["idx|all_loso"] == np.arange(len(L.Z["idx|all_loso"]))).all()


def members(bn, p):
    Ps = L.Z[f"{bn}|Ps|{p}"]; out = {i: Ps[i] for i in range(1, 11)}
    out["c1"] = C1[p]; out["hb"] = 2 * Ps[0] - C1[p]
    out.update({k: v[p] for k, v in EXTRA.items()})
    return out


def mix(bn, p, fams, extra=(), wscale=None, slot="c1"):
    """fams: family names; extra: [(name, weight)]; wscale: {family: factor}. -> (mix (n,3), core1 share)."""
    mem = members(bn, p); acc, wsum, wc1 = 0.0, 0.0, 0.0
    items = [(k, w * (wscale or {}).get(f, 1.0)) for f in fams for k, w in FAM[f]] + list(extra)
    for k, w in items:
        acc = acc + w * mem[k]; wsum += w; wc1 += w if k == slot else 0.0
    return acc / wsum, wc1 / wsum


LAM = 0.15  # TSEM lambda (packaged 0.15); both tails (gm.pipe_g looks up gm.tsem_g at call time)


def _tsem_g(P, proto, g, bts=gm.BTS):  # = gm.tsem_g with lambda LAM
    Lk, tt = L.G.tsem_lik(P, L.Z[f"sess|{proto}"], L.Z[f"pc128|{proto}"], rho=0.7)
    out = P.copy(); m = ~np.isnan(Lk[:, 0])
    e = LAM * tt[m] / (tt[m] + 60) * (1 + bts * (M.h(tt[m], 30) * g[m]))
    q = P[m] * Lk[m] ** e[:, None]; out[m] = q / q.sum(1, keepdims=True); return out


gm.tsem_g = _tsem_g


def shadow(m, p):
    b = M.rebal(m, p, None); g = gm.gate(b, p); e = gm.em_g(b, p, g); return e, gm.tsem_g(e, p, g)


_HEADS, _H = {}, {}


def route(mixes, shadows, a, oof, c1=None):
    c1 = C1 if c1 is None else c1  # member whose share the personal head takes (package: the w[0] member)
    """routing v3 (lin05) on the shadows; returns ({proto: out}, n routed new sessions, n routed seen sessions)."""
    outs, nr = {}, {"all_loso": 0, "seen_session": 0}; R3.A = a
    for proto, folds in RR.FOLDS.items():
        idx = pl.B[f"idx|{proto}"]; pos = {int(g): k for k, g in enumerate(idx)}; Tsh = shadows[proto]
        mixr, routed = mixes[proto].copy(), []
        for f in folds:
            te, names, Gn, Q, per, _ = R3.gate_model(proto, f, oof); hk = f"{proto}|{f}"
            if hk not in _HEADS:
                _HEADS[hk] = R3.ps1_heads(per, names)
            for s, (X, y, gi) in te.items():
                r = np.array([pos[int(g)] for g in gi])
                if (proto, s) not in _H:
                    Zc = RR.causal_std(X[:, :256]); _H[(proto, s)] = np.stack([_HEADS[hk][q].predict_proba(Zc) for q in names])
                Sc = R2.scores_session(X, Tsh[r], Gn, Q); rr, j = R3.rmap(Sc, "lin05")
                if a > 0 and (rr > 0).any():
                    mixr[r] = R3.slot_mix(mixes[proto][r], c1[proto][r], _H[(proto, s)], rr, j); routed.append(r); nr[proto] += 1
        out = Tsh.copy()
        if routed:
            P = gm.pipe_g(mixr, proto); rr_ = np.concatenate(routed); out[rr_] = P[rr_]
        outs[proto] = out
    return outs, nr["all_loso"], nr["seen_session"]


def system(bn, fams, extra=(), wscale=None, routing=True, slot="c1"):
    """-> dict(out={proto: final}, pre={proto: TSEM input}, nnew, nseen)."""
    mixes, sh, pre, a = {}, {}, {}, 0.0
    for p in PROTOS:
        mixes[p], a = mix(bn, p, fams, extra, wscale, slot); pre[p], sh[p] = shadow(mixes[p], p)
    if not routing:
        return dict(out=sh, pre=pre, nnew=0, nseen=0)
    out, nn, ns = route(mixes, sh, a, sh["all_loso"], {p: members(bn, p)[slot] for p in PROTOS})
    return dict(out=out, pre=pre, nnew=nn, nseen=ns)


def seed_acc(S, proto, rho=0.7, nmin=15):  # = core3/screen.seed_acc
    y, ss = L.Z[f"y|{proto}"], L.Z[f"sess|{proto}"]; lab, conf = S.argmax(1), S.max(1); acc = []
    for s in np.unique(ss):
        r = np.flatnonzero(ss == s)
        for t in range(nmin, len(r), 5):
            q = max(2, int(rho * t / 3))
            sel = np.concatenate([ik[np.argsort(-conf[r][ik], kind="stable")[:q]]
                                  for ik in (np.flatnonzero(lab[r][:t] == k) for k in range(3))])
            acc.append(np.mean(lab[r][sel] == y[r][sel]))
    return 100 * np.mean(acc)


def summary(res):
    """per-person absolute BA (new 10, seen 2), seed precision (new/seen)."""
    return dict(dn=SC.person(res["out"]["all_loso"], "all_loso"), ds=SC.person(res["out"]["seen_session"], "seen_session"),
                sp_new=seed_acc(res["pre"]["all_loso"], "all_loso"), sp_seen=seed_acc(res["pre"]["seen_session"], "seen_session"),
                nnew=res["nnew"], nseen=res["nseen"])
