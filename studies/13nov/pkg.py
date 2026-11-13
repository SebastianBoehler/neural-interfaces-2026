"""PKG (ledger s138, package form after the evaluation-path review of 13 Nov): the design inputs in the uploaded chain,
new and seen people measured together. Chain: routing v4.1 (route4 "max", t10) + the own-space stage on routed windows
(v5_replay.staged), six bases. As in a package:
  - the gate's calibration (route4's oof) is fixed to the shipped members' population shadow (P0);
  - routing comes from the population shadow; personal nets enter only the routed tail, from the session's first routed
    window (the dev form of s78/s119 gave them to whole sessions in both tails from window 0: arm T1w reproduces it);
  - fine-tuning (cum10) runs only on sessions the gate never routes (decided from T0's routing);
  - no SessionBN shallow member (the package has none).
Arms (stage 0.15 except P0):
  P0  shipped members, stage 0.10 (v5.1b)
  T0  slots 5-8 from ttt6's frozen nets (the twins of FT) in both tails
  T1b T0 + the personal nets' frozen twins (pft base) in the routed tail from the first routed window (control of T1)
  T1  T0 + personal nets (pft pfh60) in the routed tail from the first routed window (v5.2's form)
  T1w T0 + personal nets in both tails from window 0 (the dev form of s119; not shippable)
  F   T1 + FT on never-routed sessions      FO  F + OCU      O  T1 + OCU      FOR  FO + REST      R  T1 + REST
Usage: [BASES=B2_pool] [FT_DIR=.. FT_PFX=.. FT_SEEDS=a-b OCU_SEEDS=..] python studies/13nov/pkg.py"""
import glob, os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
F_ = REPO + "/studies/12nov/v5_replay.py"; exec(compile(open(F_).read().split("\nres = []")[0], F_, "exec"))  # noqa: S102  (A, SC, ns, REC, NM, staged, BETA)
import grid as G_  # noqa: E402  (second TSEM for the PLS lever; replay/ is on the path after v5_replay)
Z = A.L.Z; PR = A.PROTOS; IDX = {p: A.pl.B[f"idx|{p}"].astype(int) for p in PR}; SLOT = {"fb": 6, "fbraw": 5, "sinc": 7, "fbl": 8}
BASES = os.environ.get("BASES", "B2_pool,B2_s345,B2_s012,B2_s0,B2_s1,B2_s2").split(",")


def load(pat, p, n=1):
    zs = [np.load(f) for f in sorted(glob.glob(pat))]; assert len(zs) == n and all((z["index"] == IDX[p]).all() for z in zs), (pat, len(zs)); return zs[0]["proba"] if n == 1 else np.mean([z["proba"] for z in zs], 0)


PF = REPO + "/runs/ttt/probs_pf"; T6 = REPO + "/" + os.environ.get("FT_DIR", "runs/ttt/probs6"); FPX = os.environ.get("FT_PFX", "ttt6")
_a, _b = (int(x) for x in os.environ.get("FT_SEEDS", "0-5").split("-")); FTS = range(_a, _b + 1); OCS = os.environ.get("OCU_SEEDS", "0,1,2").split(",")
FTV = os.environ.get("FT_VAR", "cum10")  # the fine-tuning arm read from FT_DIR, e.g. gC5 in runs/ttt/probs7 (FT_PFX=ttt7)
print(f"pkg: FT {T6} {FPX} {FTV} seeds {list(FTS)}; OCU seeds {OCS}; REST seeds {os.environ.get('REST_SEEDS', '0,1,2')}; chain route4 max t10 + own-space stage (v4.1/v5), gate fixed from P0", flush=True)
PN = {(a, v): load(f"{PF}/pft_{a}_{v}-seen_session-s*.npz", "seen_session", 3) for a in SLOT for v in ("base", "pfh60")}
FRS = {(a, s): {p: load(f"{T6}/{FPX}_{a}_base-{p}-s{s}.npz", p) for p in PR} for a in SLOT for s in FTS}
CUS = {(a, s): {p: load(f"{T6}/{FPX}_{a}_{FTV}-{p}-s{s}.npz", p) for p in PR} for a in SLOT for s in FTS}
FR = {a: {p: np.mean([FRS[a, s][p] for s in FTS], 0) for p in PR} for a in SLOT}
CU = {a: {p: np.mean([CUS[a, s][p] for s in FTS], 0) for p in PR} for a in SLOT}
# s152: the study nets of seed K reproduce base B2_sK's own SessionBN slots, so each base gets its own seeds' FT change
BASESEEDS = {"B2_s0": [0], "B2_s1": [1], "B2_s2": [2], "B2_s012": [0, 1, 2], "B2_s345": [3, 4, 5], "B2_pool": [0, 1, 2, 3, 4, 5]}
OC = {p: np.mean([load(f"{REPO}/runs/ocunet/probs/oc_nomix-{p}-s{s}.npz", p) for s in OCS], 0) for p in PR}
RSS = os.environ.get("REST_SEEDS", "0,1,2").split(",")  # explicit seeds, as OCU_SEEDS
A.EXTRA["rest"] = {p: np.mean([load(f"{REPO}/runs/rest4/probs/rest4_sh_r1-{p}-s{s}.npz", p) for s in RSS], 0) for p in PR}
A.EXTRA["restfb"] = {p: np.mean([load(f"{REPO}/runs/rest4/probs/rest4_fb_r1-{p}-s{s}.npz", p) for s in RSS], 0) for p in PR}  # s144: the FBCNet rest-class net
REF = {p: Z[f"GE|{p}"].copy() for p in PR}; GEO = {p: np.c_[REF[p], np.log(np.clip(OC[p], 1e-3, 1))] for p in PR}
# s155 levers inside the combination (all on the shipped path): view scale or oracle, cognitive teacher, mix exponent, PLS view, weak gate
ORC = {p: np.log(0.05 + 0.85 * np.eye(3)[Z[f"y|{p}"].astype(int)]) for p in PR}  # ORACLE eye-net view: true labels (ceiling only)
_cog = f"{REPO}/runs/teach/teacher_cog1_all_loso.npy"
if os.path.exists(_cog):
    A.EXTRA["cog1"] = {"all_loso": np.load(_cog), "seen_session": Z["B2_pool|Ps|seen_session"][10]}
_mix0 = A.mix


MXS = os.environ.get("OCU_MIX_SEEDS", "0,1,2").split(",")  # s156: the eye net's Mixup arm as a second view family
OCM = {p: np.mean([load(f"{REPO}/runs/ocunet/probs/oc_mix-{p}-s{s}.npz", p) for s in MXS], 0) for p in PR}


def ge_view(sp, p):
    if not sp.get("ocu"):
        return REF[p]
    L = ORC[p] if sp.get("ocuorc") else np.log(np.clip(OC[p], 1e-3, 1)); V = np.c_[REF[p], sp.get("ocuw", 1.0) * L]
    return np.c_[V, np.log(np.clip(OCM[p], 1e-3, 1))] if sp.get("ocmix") else V


def weak_rows(bn, p, RTS0, q=0.5, T0=30):  # weak-session gate (s133 form): out-of-person quantile of the shipped mix's early confidence
    mix = _mix0(bn, p, A.FAMS)[0]; sess = Z[f"sess|{p}"]; per = Z[f"person|{p}"]; ss = np.unique(sess)
    conf = {u: mix[np.flatnonzero(sess == u)[:T0]].max(1).mean() for u in ss}; sp_ = {u: per[np.flatnonzero(sess == u)[0]] for u in ss}
    thr = {u: np.quantile([conf[v] for v in ss if sp_[v] != sp_[u]], q) for u in ss}; return np.array([conf[u] < thr[u] for u in sess])


_PLS = {}


def pls_view():  # s74's PLS-DA view of core1's 256 components, per fold on training people only
    if not _PLS:
        from sklearn.cross_decomposition import PLSRegression
        p = "all_loso"; sess = Z[f"sess|{p}"]; V = np.zeros((len(sess), 16))
        for f in sorted(glob.glob(f"{REPO}/runs/replay_data/feats_exact/all_loso-P*.npz")):
            z = np.load(f, allow_pickle=True); Xc = z["pc_tr"].copy(); itr = z["idx_tr"].astype(int)
            for u in np.unique(sess[itr]):
                m = sess[itr] == u; Xc[m] -= Xc[m].mean(0)
            sd = Xc.std(0); W = PLSRegression(16, scale=False).fit(Xc / sd, np.eye(3)[z["y_tr"].astype(int)]).x_rotations_
            for k in [k for k in z.files if k.endswith("|pc")]:
                V[z[k.replace("|pc", "|idx")].astype(int)] = (z[k] / sd) @ W
        assert np.abs(V).sum(1).min() > 0; _PLS["v"] = V
    return _PLS["v"]
SPEC = {"T0": dict(), "T1b": dict(pn="base"), "T1": dict(pn="pfh60"), "T1w": dict(pn="pfh60", w0=True), "F": dict(pn="pfh60", ft=True),
        "FO": dict(pn="pfh60", ft=True, ocu=True), "O": dict(pn="pfh60", ocu=True), "FOR": dict(pn="pfh60", ft=True, ocu=True, rest=True),
        "R": dict(pn="pfh60", rest=True),
        "OR": dict(pn="pfh60", ocu=True, rest=True),  # s150: the stack without fine-tuning (no run-time cost)
        "V6": dict(ship=True, ocu=True, rest=True, beta=0.10), "V6b": dict(ship=True, ocu=True, rest=True, beta=0.15),  # s151: v5.1b + OCU + REST on the shipped members
        "SF": dict(ship=True, ft=True, beta=0.10), "V6F": dict(ship=True, ft=True, ocu=True, rest=True, beta=0.10)}  # s152: FT as a change to the shipped members
_V = dict(ship=True, ft=True, ocu=True, rest=True, beta=0.10)  # s155: refinements and retests inside V6F
SPEC.update({"V6F_ft05": {**_V, "ftw": 0.5}, "V6F_ft15": {**_V, "ftw": 1.5}, "V6F_r05": {**_V, "rest": 0.5}, "V6F_r2": {**_V, "rest": 2.0},
             "V6F_o05": {**_V, "ocuw": 0.5}, "V6F_o2": {**_V, "ocuw": 2.0}, "V6F_orc": {**_V, "ocuorc": True}, "V6F_cog": {**_V, "cog": 1.0},
             "V6F_pls": {**_V, "pls": 0.15}, "V6F_mix": {**_V, "mixe": 0.9}, "V6F_weak": {**_V, "ftweak": True},
             "V6F_ocm": {**_V, "ocmix": True}})  # s156: a second eye-net family in the view
SPEC.update({
        "FOR2": dict(pn="pfh60", ft=True, ocu=True, rest=True, restfb=True), "FORw2": dict(pn="pfh60", ft=True, ocu=True, rest=2.0)})  # s144


def routing():  # -> rows from each routed session's first routed window on (RTW), rows of routed sessions (RTS), from the last route4 call
    W, S = {p: np.zeros(len(IDX[p]), bool) for p in PR}, {p: np.zeros(len(IDX[p]), bool) for p in PR}
    for (proto, s), (r, j) in REC.items():
        names, gi, f = NM[(proto, s)]; pos = {int(g): k for k, g in enumerate(IDX[proto])}; rows = np.array([pos[int(g)] for g in gi])
        S[proto][rows] = True; W[proto][rows[int(np.flatnonzero(np.asarray(r) > 0)[0]):]] = True
    return W, S


def famswap(arrs, tag, rest, restfb=False):
    fams, extra = list(A.FAMS), []
    for a, slot in SLOT.items():
        A.EXTRA[f"pk_{a}_{tag}"] = arrs[a]; fam = next(f for f, items in A.FAM.items() if any(k == slot for k, _ in items))
        if fam in fams:
            fams.remove(fam); extra += list(A.FAM[fam])
        extra = [(k, w) for k, w in extra if k != slot] + [(f"pk_{a}_{tag}", 1.0)]
    return fams, extra + ([("rest", float(rest))] if rest else []) + ([("restfb", 1.0)] if restfb else [])


def system(bn, arm, oof=None, RTS0=None):
    global BETA
    sp = SPEC.get(arm, {}); BETA = sp.get("beta", 0.10 if arm == "P0" else 0.15)
    for p in PR:
        Z[f"GE|{p}"] = ge_view(sp, p)
    if sp.get("mixe"):
        A.mix = lambda *a_, **k_: (lambda m, s_: ((m ** sp["mixe"]) / (m ** sp["mixe"]).sum(1, keepdims=True), s_))(*_mix0(*a_, **k_))
    try:
        c1 = {p: A.members(bn, p)["c1"] for p in PR}
        if arm == "P0" or sp.get("ship"):  # the shipped members everywhere; FT enters only as each base's own seeds' change on never-routed rows
            assert not sp.get("pn"), f"{arm}: personal nets need study nets, not the shipped members"
            pf, pe = list(A.FAMS), ([("rest", float(sp["rest"]))] if sp.get("rest") else [])
            cogx = [("cog1", float(sp["cog"]))] if sp.get("cog") else []
            if sp.get("ft"):
                assert bn in BASESEEDS and all(k in FTS for k in BASESEEDS[bn]), (bn, list(FTS))
                arrs = {}
                for a, slot in SLOT.items():
                    arrs[a] = {}
                    for p in PR:
                        base = A.members(bn, p)[slot].copy(); dlt = sp.get("ftw", 1.0) * np.mean([CUS[a, k][p] - FRS[a, k][p] for k in BASESEEDS[bn]], 0)
                        off = RTS0[p] | (~weak_rows(bn, p, RTS0) if sp.get("ftweak") else False)
                        x = np.where(off[:, None], base, np.clip(base + dlt, 1e-6, None)); arrs[a][p] = x / x.sum(1, keepdims=True)
                    assert np.abs(arrs[a]["all_loso"] - A.members(bn, "all_loso")[slot]).max() > 1e-3, f"{arm}: the FT change is a no-op"
                pf, pe = famswap(arrs, arm + "_ship", sp.get("rest"))
            pe = pe + cogx
        else:
            pop = {a: {p: np.where(RTS0[p][:, None], FR[a][p], CU[a][p]) if sp.get("ft") else FR[a][p].copy() for p in PR} for a in SLOT}
            pf, pe = famswap(pop, arm + "_pop", sp.get("rest"), sp.get("restfb"))
        mp, sh, a = {}, {}, 0.0
        for p in PR:
            mp[p], a = A.mix(bn, p, pf, pe); _, sh[p] = A.shadow(mp[p], p)
        REC.clear(); out0 = ns["route4"](mp, sh, a, sh["all_loso"] if oof is None else oof, c1, "max", 10)[0]  # pass 1: routing
        if arm not in ("P0", "T0") and not sp.get("ship"):
            RTW, RTS = routing(); rt = {a: {p: x.copy() for p, x in pop[a].items()} for a in SLOT}
            for k in SLOT:  # not "a": that is the core1 share route4 needs
                m = RTS["seen_session"] if sp.get("w0") else RTW["seen_session"]; rt[k]["seen_session"][m] = PN[(k, sp["pn"])][m]
                if sp.get("w0"):
                    pop[k]["seen_session"][m] = PN[(k, sp["pn"])][m]
            if RTW["all_loso"].any():
                print(f"   {arm}: {int(RTW['all_loso'].sum())} new rows routed (false accepts); no personal nets exist for them here", flush=True)
            rf, re_ = famswap(rt, arm + "_rt", sp.get("rest"), sp.get("restfb"))
            if sp.get("w0"):  # dev form: the shadow (and so the gate's input) also sees the personal nets
                pf, pe = famswap(pop, arm + "_pop", sp.get("rest"), sp.get("restfb"))
                for p in PR:
                    mp[p], a = A.mix(bn, p, pf, pe); _, sh[p] = A.shadow(mp[p], p)
            mr = {p: A.mix(bn, p, rf, re_)[0] for p in PR}
            REC.clear(); out0 = ns["route4"](mr, sh, a, sh["all_loso"] if oof is None else oof, c1, "max", 10)[0]  # pass 2: routed tail with personal nets
        out = staged(out0)[0]
        if sp.get("pls"):  # s74 form: a second TSEM pass through the PLS view after the chain, new people's never-routed rows only
            P = out["all_loso"]; L, tt = G_.tsem_lik(P, Z["sess|all_loso"], pls_view())
            Q = G_.apply_lam(P, L, tt, sp["pls"]); out["all_loso"] = np.where(RTS0["all_loso"][:, None], P, Q)
        return out, sh["all_loso"]
    finally:
        A.mix = _mix0
        for p in PR:
            Z[f"GE|{p}"] = REF[p]


ARMS = ["P0"] + [k for k in SPEC if k in ("T0", "T1") or not os.environ.get("ARMS") or k in os.environ["ARMS"].split(",")]; R = {}  # ARMS=FOR,FOR2 runs a subset (P0, T0, T1 always)
for bn in BASES:
    out, oof = system(bn, "P0"); R[bn, "P0"] = out; _, RP0 = routing()  # ship arms take never-routed rows from v5.1b's own routing
    R[bn, "T0"] = system(bn, "T0", oof, {p: np.zeros(len(IDX[p]), bool) for p in PR})[0]; _, RTS0 = routing()  # T0 uses no FT, so its mask is unused
    print(f"{bn}: routed sessions' rows new {RTS0['all_loso'].sum()} of {len(IDX['all_loso'])}, seen {RTS0['seen_session'].sum()} of {len(IDX['seen_session'])}; FT on the rest", flush=True)
    for arm in ARMS[2:]:
        R[bn, arm] = system(bn, arm, oof, RP0 if SPEC.get(arm, {}).get("ship") else RTS0)[0]
    s = {k: (SC.person(R[bn, k]["all_loso"], "all_loso"), SC.person(R[bn, k]["seen_session"], "seen_session")) for k in ARMS}
    print(f"   T0 new {s['T0'][0].mean():.2f} seen {s['T0'][1].mean():.2f} | " + " | ".join(f"{k} {s[k][0].mean() - s['T0'][0].mean():+.2f}/{s[k][1].mean() - s['T0'][1].mean():+.2f}" for k in ARMS if k != "T0"), flush=True)
    for k in ARMS:
        R[bn, k] = s[k]
rng = np.random.default_rng(0)
if os.environ.get("PKG_SAVE"):  # per-person results of every arm and base, for paired differences between arms
    np.savez(os.environ["PKG_SAVE"], **{f"{b}|{k}|{j}": R[b, k][j] for (b, k) in R for j in (0, 1)})
for ref, arms in (("P0", [k for k in ARMS if k != "P0"]), ("T0", ARMS), ("T1b", [k for k in ("T1", "T1w") if k in ARMS]), ("T1", [k for k in ARMS if k not in ("P0", "T0", "T1", "T1b", "T1w") and not SPEC.get(k, {}).get("ship")])):
    if ref not in ARMS or not arms:
        continue
    print(f"\n{'=== THE UPLOADED SYSTEM' if ref == 'P0' else '--- a STUDY BASE, not the uploaded system'} ({ref})", flush=True)
    print(f"against {ref}, six-base mean: new [person 80 %] people>=0, without the best | seen (P04, P10) [per-base range] | J")
    for k in arms:
        if k == ref:
            continue
        dn = np.array([R[b, k][0] - R[b, ref][0] for b in BASES]); ds = np.array([R[b, k][1] - R[b, ref][1] for b in BASES]); d = dn.mean(0); q = int(np.argmax(d))
        bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])
        print(f"  {k:4s} new {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] {(d >= 0).sum()}/10, w/o P{q + 1:02d} {np.delete(d, q).mean():+.2f} | seen {ds.mean():+.2f} "
              f"(P04 {ds[:, 0].mean():+.2f}, P10 {ds[:, 1].mean():+.2f}) [{ds.mean(1).min():+.2f}, {ds.mean(1).max():+.2f}] | J {0.5 * d.mean() + 0.5 * ds.mean():+.2f}", flush=True)
