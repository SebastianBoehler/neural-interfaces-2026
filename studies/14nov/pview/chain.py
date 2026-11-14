"""PV2: personal views for recognised returning people in the chain (P04, P10; seen_session: trained on session 1,
tested on sessions 2 and 3), on retv6's machinery (exec'd up to its main loop, not edited) in its s214 form:
GATE_V51B=1 (the gate and the shadow see v5.1b only), members re-streamed (RT_MEMBERS=1), FT_SFX=QF, FT_FORM=s15.
Arms, every addition acting in the routed tail only (rows the gate has not routed yet emit ER's output; checked):
  base  routed v5.1b;  ER  eye view + rest member (s214's v6.2-safe returning rows; reproduced against runs/retv6/p0_*)
  O     ER + the personal ocular model's log posteriors as 3 more SessionEM view columns
  M     ER + the personal ocular model as a member in the routed mix, weight w = 1 against the mix's 16, times ramp r
  S     ER + a SessionEM template prior (personal class means, kappa 20 pseudo-windows, fading as the session fills)
  T     ER + a TSEM template prior (route4's personal `pri`, kappa 20; route4's KAPPA path with the denominator fixed)
  OST   O + S + T;  W  O + M (w = 2 s) + S, T (kappa 20 s), s = the person's strength (personal.strength)
Usage: [BASES=..] [ORDERS=within0,..] [ARMS=..] python studies/14nov/pview/chain.py -> runs/pview/chain_<orders>.npz"""
import os, sys, time
import numpy as np
for k, v in (("GATE_V51B", "1"), ("FT_SFX", "QF"), ("FT_FORM", "s15"), ("RT_MEMBERS", "1")):
    assert os.environ.setdefault(k, v) == v, f"{k} must be {v} here (s214 form)"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import data as D, personal as PS  # noqa: E402
F6 = D.REPO + "/studies/13nov/retv6.py"; SRC = open(F6).read(); HEAD = SRC.split("\nOOF = {bn: oof_rec(bn)")[0]; assert HEAD != SRC, "retv6 layout changed"
R6 = {"__file__": F6, "__name__": "retv6_lib"}; exec(compile(HEAD, F6, "exec"), R6)  # noqa: S102
K, S, A, SC, ns, Z, p, IDX = (R6[k] for k in ("K", "S", "A", "SC", "ns", "Z", "p", "IDX")); gm, R3 = A.gm, A.R3
assert p == "seen_session" and R6["GATE_V51B"] and R6["RT_MEMBERS"]
BASES = os.environ.get("BASES", "B2_pool,B2_s345,B2_s012,B2_s0,B2_s1,B2_s2").split(",")
SAVE_OUT, OUTW = os.environ.get("SAVE_OUT"), {}  # s245: npz path for per-window outputs of every arm
ORDERS = os.environ.get("ORDERS", "within0").split(","); ARMS = os.environ.get("ARMS", "base,ER,O,M,S,T,OST,W,H25,H33,H40,B20,B30").split(","); assert "ER" in ARMS, "ER is the reference arm"
# PV3 (coordinator, s227): common, larger personal strengths. The package's routed head share at r = 1 is
# g a / (1 - a + g a) with g = route_slot_g 3, a = w[0]: 1/15 in v5.1b (0.176), 1/16 with the rest member (0.167).
# Hxx sets g so that the share is xx % at a = 1/16 (g = 15 s / (1 - s)); Bxx sets the own-space stage's beta (0.10 today).
HSH = {"H25": 0.25, "H33": 1 / 3, "H40": 0.40}; BOWN = {"B20": 0.20, "B30": 0.30}
KAP, WM = 20.0, 1.0
SEEN = ("P04", "P10"); OCU = {q: PS.Ocular(PS.public_rows(q)) for q in SEEN}; STR = {q: PS.strength(q) for q in SEEN}
for q in SEEN:  # the personal model's rows are exactly the fold's training rows of that person (no test session inside)
    it = np.load(f"{D.REPO}/runs/replay_data/feats_exact/seen_session-{q}_s23.npz")["idx_tr"].astype(int)
    assert set(OCU[q].rows) == set(it[D.PER[it] == q]), q
EMP = {(q, o): PS.em_prior(q, o) for q in SEEN for o in (False, True)}
print("PV2 strengths (s, LORO personal BA, eye BA): " + ", ".join(f"{q} {v[0]:.2f} ({v[1]:.1f}, {v[2]:.1f})" for q, v in STR.items()), flush=True)
HOOK = dict(M=0.0, S=0.0, T=0.0, scale=False, rowsP=None, POC=None)  # set per arm


class EMp(gm.EMg):  # gm.EMg with personal class means as kp pseudo-windows in mu (kp = 0: identical to EMg)
    kp, pm, mk = 0.0, None, None  # mk (d,): columns the prior pulls (None: all)

    def _adapt(self, A_, B, f, p_, t):
        if self.kp <= 0:
            return super()._adapt(A_, B, f, p_, t)
        m, sd = A_.mean(0), np.maximum(A_.std(0), D_SD)
        Zs, z = (A_ - m) / sd, (f - m) / sd; Q = B
        for _ in range(self.iters):
            mk = np.ones(Zs.shape[1]) if self.mk is None else self.mk
            mu = (Q.T @ Zs + self.kp * mk * self.pm) / (Q.sum(0)[:, None] + self.m0 + self.kp * mk)  # = (Q/den).T @ Zs + kp pm / den when mk = 1
            C = np.cov((Zs - Q @ mu).T); C = (1 - self.shrink) * C + self.shrink * np.eye(len(C)) * np.trace(C) / len(C)
            Ci = np.linalg.inv(C); Q = B * gm.SE._softmax_lda(Zs, mu, Ci) ** self.lam; Q = Q / Q.sum(1, keepdims=True)
        e = self.lam * t / (t + self.n0) * (1 + self.bem * (gm.M.h(t, 30) * self.g))
        q = p_ * gm.SE._softmax_lda(z[None], mu, Ci)[0] ** e; return q / q.sum()


D_SD = gm.SE.SD_FLOOR; _em0, _pipe0, _slot0 = gm.em_g, gm.pipe_g, R3.slot_mix


def em_hook(b, proto, g, bem=gm.BEM, nmin=gm.NMIN):  # = gm.em_g with per-row prior (routed rows only)
    F, sess = Z[f"GE|{proto}"], Z[f"sess|{proto}"]; out = np.empty_like(b); kp, pm, mk = HOOK["emkp"], HOOK["empm"], HOOK.get("emmk")
    for s in dict.fromkeys(sess):
        e = EMp(bem, lam=0.5, n0=60, m0=20, shrink=0.7, nmin=nmin)
        for r in np.flatnonzero(sess == s):
            e.g, e.kp, e.pm, e.mk = g[r], kp[r], pm[r], None if mk is None else mk[r]; out[r] = e.step(F[r], b[r])
    return out


def tsem_lik_prior(PM, KP):  # route4's form with kp also in the denominator (ledger: the s19/s57b over-scaling)
    T_ = ns["T_"]

    def f(P, sess, X, rho=0.7, shrink=0.7, m0=20, nmin=15):
        Lk = np.full_like(P, np.nan); tt = np.zeros(len(P)); KPs = KP * HOOK["tsscale"]
        for s in np.unique(sess):
            r = np.flatnonzero(sess == s); lab = P[r].argmax(1); conf = P[r].max(1)
            for t in range(nmin, len(r)):
                A_, fx = X[r[:t]], X[r[t]]; m, sd = A_.mean(0), np.maximum(A_.std(0), 1e-3); Zm, z = (A_ - m) / sd, (fx - m) / sd
                q = max(2, int(rho * t / 3)); sel = [np.flatnonzero(lab[:t] == k)[np.argsort(-conf[np.flatnonzero(lab[:t] == k)])[:q]] for k in range(3)]
                if min(len(x) for x in sel) < 2:
                    continue
                kp = KPs[r[t]]; mu = np.stack([(Zm[x].sum(0) + kp * PM[r[t], k]) / (len(x) + m0 + kp) for k, x in enumerate(sel)])
                R = np.concatenate([Zm[x] - mu[k] for k, x in enumerate(sel)]); Cv = np.cov(R.T)
                Cv = (1 - shrink) * Cv + shrink * np.eye(len(Cv)) * np.trace(Cv) / len(Cv)
                Lk[r[t]] = T_._lda(z[None], mu, np.linalg.inv(Cv))[0]; tt[r[t]] = t
        return Lk, tt
    return f


ns["tsem_lik_prior"] = tsem_lik_prior


def session_of(H):
    return next(k for k, v in A._H.items() if v is H)


def slot_hook(mix, c1, H, r, j):  # after route4's personal-head slot: the personal ocular member at r * w
    out = _slot0(mix, c1, H, r, j); key = session_of(H); names, gi, _ = K.NM[key]
    for t in np.flatnonzero(r > 0):
        q = names[int(j[t])][-3:]; assert q == D.PER[gi[t]], f"{key}: routed to {q}, the session is {D.PER[gi[t]]}"
        w = HOOK["M"] * (2 * STR[q][0] if HOOK["scale"] else 1.0) * r[t]
        if w > 0:
            out[t] = (16 * out[t] + w * HOOK["POC"][gi[t]]) / (16 + w)
    return out


def pipe_hook(mixr, proto):
    if HOOK["S"] > 0:
        gm.em_g = em_hook
    try:
        return _pipe0(mixr, proto)
    finally:
        gm.em_g = _em0


gm.pipe_g, R3.slot_mix = pipe_hook, slot_hook


def personal_streams(Pp):
    """personal ocular posteriors per global window index, each session streamed in its delivery order (rows of the
    permuted store), with the person's own model."""
    POC = np.full((D.N, 3), np.nan); GEp, ss, gidx = Z[f"GE|{p}"][:, :48], Z[f"sess|{p}"], IDX[Pp]
    for s in dict.fromkeys(ss):
        r = np.flatnonzero(ss == s); q = D.PER[gidx[r[0]]]; assert (D.PER[gidx[r]] == q).all()
        POC[gidx[r]] = OCU[q].stream(GEp[r])
    return POC


def routed_rows(proto):
    """-> {(proto, s): (rows in the permuted store, r, person)} from route4's record of this call."""
    pos = {int(g): k for k, g in enumerate(A.pl.B[f"idx|{proto}"])}; out = {}
    for key, (r, j) in K.REC.items():
        names, gi, _ = K.NM[key]; q = names[int(j[np.flatnonzero(r > 0)[0]])][-3:]
        assert all(names[int(x)][-3:] == q for x in j[r > 0]), f"{key}: the routed person changes inside the session"
        out[key] = (np.array([pos[int(g)] for g in gi]), r, q)
    return out


def set_em_prior(proto, with_o):
    n = len(Z[f"sess|{proto}"]); d = Z[f"GE|{proto}"].shape[1]; kp, pm, mk = np.zeros(n), np.zeros((n, 3, d)), np.ones((n, d))
    for rows, r, q in routed_rows(proto).values():
        if with_o in PS.COLS:  # PV5: matched-coordinate prior on the gaze (+ event-rate) columns only
            pm_, mk_ = EMPM[q, with_o]; assert pm_.shape[1] == d, (pm_.shape, d); mk[rows] = mk_
        else:
            pm_ = EMP[q, with_o]; assert pm_.shape[1] == d, (pm_.shape, d)
        kp[rows] = HOOK["S"] * r * (STR[q][0] if HOOK["scale"] else 1.0); pm[rows] = pm_
    HOOK["emkp"], HOOK["empm"], HOOK["emmk"] = kp, pm, mk


def system(bn, arm, Pp, oof, inp, POC):
    ft, ocu, rest = inp; fams, ge0, folds0 = list(A.FAMS), Z[f"GE|{p}"], A.RR.FOLDS
    assert R3.G3 == 3.0 and K.BETA == 0.10, "the head gain or the own-space beta was left changed"
    if arm in HSH or arm == "TH25":
        R3.G3 = 15 * HSH[arm[-3:]] / (1 - HSH[arm[-3:]])
    K.BETA = BOWN.get(arm, 0.10)
    o, w = arm in ("O", "OST", "W") or arm.startswith("OG"), arm == "W"
    if arm.startswith(("OG", "SG")):  # PV5: matched coordinates; POC is the dict of matched personal streams then
        POC = POC[arm[1:]] if arm.startswith("OG") else None; o = arm[1:] if arm.startswith("SG") else True
    elif isinstance(POC, dict):
        POC = POC["orig"]
    HOOK.update(M={"M": WM, "W": 1.0}.get(arm, 0.0), S=KAP if arm in ("S", "OST", "W") or arm.startswith("SG") else 0.0, scale=w, POC=POC)
    ns["KAPPA"] = KAP if arm in ("T", "OST", "W", "TH25") else float(arm[1:]) if arm in ("T10", "T40") else 0.0  # PV4: T10, T40, TH25
    extra = [] if arm == "base" else [("rt_rest", 1.0)]
    if arm != "base":
        A.EXTRA["rt_rest"] = {p: rest[Pp]}; Z[f"GE|{p}"] = np.c_[(ge0, ocu[Pp]) + ((np.log(np.clip(POC[IDX[Pp]], 1e-3, 1)),) if o is True else ())]
    try:
        A.RR.FOLDS = {p: folds0[p]}; K.REC.clear()
        mix, a = A.mix(bn, p, fams, extra)
        ge_add = Z[f"GE|{p}"]; Z[f"GE|{p}"] = ge0; _, sh = A.shadow(A.mix(bn, p, list(A.FAMS))[0], p); Z[f"GE|{p}"] = ge_add
        assert abs(a - (1 / 15 if arm == "base" else 1 / 16)) < 1e-12, (arm, a)
        orig_rr = ns["route4"]
        if HOOK["S"] > 0 or ns["KAPPA"] > 0:  # the priors need the routing of this call: wrap the routed tail
            _pf = gm.pipe_g

            def pg(mixr, proto):
                if HOOK["S"] > 0:
                    set_em_prior(proto, o)
                HOOK["tsscale"] = np.ones(len(mixr))
                if w:
                    for rows, r, q in routed_rows(proto).values():
                        HOOK["tsscale"][rows] = STR[q][0]
                return _pf(mixr, proto)
            gm.pipe_g = pg
        try:
            out0 = orig_rr({p: mix}, {p: sh}, a, oof, {p: A.members(bn, p)["c1"]}, "max", 10)[0]
        finally:
            gm.pipe_g = pipe_hook
        rr = routed_rows(p); out = K.staged(out0)[0][p]
    finally:
        A.RR.FOLDS = folds0; Z[f"GE|{p}"] = ge0; ns["KAPPA"] = 0.0; R3.G3, K.BETA = 3.0, 0.10; HOOK.update(M=0.0, S=0.0, scale=False)
        for k in [k for k in A.EXTRA if k.startswith("rt_")]:
            del A.EXTRA[k]
    return out, rr


EMPM = {}


def matched(order, Pp, POC):
    """PV5: personal ocular models and SessionEM priors on columns computed by one code path for both sides."""
    Fm = PS.matched_feats(order); gi = IDX[Pp]; GEt = Z[f"GE|{p}"]
    assert GEt.shape[1] == 48 and np.array_equal(GEt[:, :32], Fm[gi, :32]), "gaze columns differ between training and test"
    ts = np.load(D.REPO + "/runs/geshuf.npz")[f"{order}|seen_session"]
    assert np.allclose(GEt[:, 32:48], ts[gi], rtol=0, atol=1e-9), "test event rates are not geshuf's delivery-order stream (same code path as training)"
    for c, cols in PS.COLS.items():
        P_ = np.full((D.N, 3), np.nan); ss = Z[f"sess|{p}"]
        for q in SEEN:
            m = PS.Ocular(PS.public_rows(q), Fm[:, cols]); EMPM[q, c] = PS.em_prior_masked(q, Fm, cols, 51)
            for s_ in dict.fromkeys(ss[D.PER[gi] == q]):
                r = np.flatnonzero(ss == s_); P_[gi[r]] = m.stream(GEt[r][:, cols])
        POC["G" + c[1:]] = P_
        print(f"  {order} matched {c}: personal ocular alone P04 {D.ba(gi[D.PER[gi] == 'P04'], P_[gi[D.PER[gi] == 'P04']].argmax(1)):.1f} P10 {D.ba(gi[D.PER[gi] == 'P10'], P_[gi[D.PER[gi] == 'P10']].argmax(1)):.1f}", flush=True)


def main():
    t0 = time.time(); OOF = {bn: R6["oof_rec"](bn) for bn in BASES}; RES = {}
    for order in ORDERS:
        P = R6["perm"](order); Pp = P[p]; inp = R6["inputs"](order)
        for c in (ns["_BIO"], A._H, K.STREAMS, K.REC):
            c.clear()
        S.apply(P); K.RANK["cur"] = {q: K._rank(K.IDX[q][P[q]]) for q in K.PR}
        R6["apply_members"](R6["member_deltas"](order), Pp); POC = {"orig": personal_streams(Pp)}
        if any(a.startswith(("OG", "SG")) for a in ARMS):
            matched(order, Pp, POC)
        y = Z[f"y|{p}"].astype(int); gi = IDX[Pp]
        print(f"{order}: personal ocular alone (delivery order, causal) P04 {D.ba(gi[D.PER[gi] == 'P04'], POC['orig'][gi[D.PER[gi] == 'P04']].argmax(1)):.1f} "
              f"P10 {D.ba(gi[D.PER[gi] == 'P10'], POC['orig'][gi[D.PER[gi] == 'P10']].argmax(1)):.1f}; acc {np.mean(POC['orig'][gi].argmax(1) == y):.3f}", flush=True)
        REF = np.load(f"{D.REPO}/runs/retv6/p0_g1_s15_QF_seen_session_{order}.npz") if os.path.exists(f"{D.REPO}/runs/retv6/p0_g1_s15_QF_seen_session_{order}.npz") else None
        for bn in BASES:
            outs, rrs = {}, {}
            for arm in ARMS:
                outs[arm], rrs[arm] = system(bn, arm, Pp, OOF[bn], inp, POC)
            for arm in ARMS:  # rows before the session's first routed window emit ER's output (base: its own); the priors and the member must not reach them
                if arm in ("base", "ER"):
                    continue
                for (rows, r, q), (rows_e, r_e, _) in zip(rrs[arm].values(), rrs["ER"].values()):
                    assert np.array_equal(rows, rows_e) and np.array_equal(r, r_e), f"{arm}: routing differs from ER's"
                    pre = rows[:int(np.flatnonzero(r > 0)[0])]  # before the first routed window (gi is in delivery order)
                    if arm not in ("O", "OST", "W") and not arm.startswith("OG"):
                        assert np.allclose(outs[arm][pre], outs["ER"][pre], atol=1e-12), f"{arm} leaks into rows the gate has not routed"
                    outs[arm][pre] = outs["ER"][pre]
            for arm in ARMS:
                RES[order, bn, arm] = SC.person(outs[arm], p)
                if SAVE_OUT:  # s245: per-window outputs in retv6's SAVE_OUT layout (recording order), ER checked against it
                    OUTW[f"{order}|{bn}|{arm}"] = np.asarray(outs[arm])[np.argsort(Pp)].astype(np.float32)
            if SAVE_OUT and os.path.exists(f"{D.REPO}/runs/chainout/ret_{order}.npz"):
                d_ = np.abs(OUTW[f"{order}|{bn}|ER"] - np.load(f"{D.REPO}/runs/chainout/ret_{order}.npz")[f"{order}|{bn}|ER"]).max()
                assert d_ < 1e-5, f"{order} {bn}: saved ER differs from runs/chainout/ret_{order}.npz by {d_}"
            if REF is not None:
                for arm in [a for a in ("base", "ER") if a in ARMS]:  # both checked against s214 when run
                    ref = REF[f"{order}|{bn}|{arm}"]; assert np.allclose(RES[order, bn, arm], ref, atol=1e-6), f"{order} {bn} {arm}: {RES[order, bn, arm]} != s214 {ref}"
            b = RES[order, bn, "ER"]
            print(f"  {order} {bn}: ER P04 {b[0]:.2f} P10 {b[1]:.2f}{' (= s214)' if REF is not None else ' (no s214 reference)'} | "
                  + " ".join(f"{k} {(RES[order, bn, k] - b).mean():+.2f} ({RES[order, bn, k][0] - b[0]:+.2f}, {RES[order, bn, k][1] - b[1]:+.2f})" for k in ARMS if k != "ER") + f" [{time.time() - t0:.0f}s]", flush=True)
    S.apply({q: np.arange(S.N[q]) for q in K.PR}); K.RANK["cur"] = None
    os.makedirs(D.OUT, exist_ok=True); tag = os.environ.get("TAG", "") + "-".join(ORDERS) + ("" if len(BASES) == 6 else "_smoke")  # TAG: a run with other arms keeps its own file
    np.savez(f"{D.OUT}/chain_{tag}.npz", **{f"{o}|{bn}|{k}": v for (o, bn, k), v in RES.items()}); print(f"written runs/pview/chain_{tag}.npz", flush=True)
    if SAVE_OUT:
        np.savez_compressed(f"{D.REPO}/{SAVE_OUT}", **OUTW); print("per-window outputs ->", SAVE_OUT, flush=True)


if __name__ == "__main__":
    main()
