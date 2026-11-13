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
Z = A.L.Z; PR = A.PROTOS; IDX = {p: A.pl.B[f"idx|{p}"].astype(int) for p in PR}; SLOT = {"fb": 6, "fbraw": 5, "sinc": 7, "fbl": 8}
BASES = os.environ.get("BASES", "B2_pool,B2_s345,B2_s012,B2_s0,B2_s1,B2_s2").split(",")


def load(pat, p, n=1):
    zs = [np.load(f) for f in sorted(glob.glob(pat))]; assert len(zs) == n and all((z["index"] == IDX[p]).all() for z in zs), (pat, len(zs)); return zs[0]["proba"] if n == 1 else np.mean([z["proba"] for z in zs], 0)


PF = REPO + "/runs/ttt/probs_pf"; T6 = REPO + "/" + os.environ.get("FT_DIR", "runs/ttt/probs6"); FPX = os.environ.get("FT_PFX", "ttt6")
_a, _b = (int(x) for x in os.environ.get("FT_SEEDS", "0-5").split("-")); FTS = range(_a, _b + 1); OCS = os.environ.get("OCU_SEEDS", "0,1,2").split(",")
print(f"pkg: FT {T6} {FPX} seeds {list(FTS)}; OCU seeds {OCS}; REST seeds {os.environ.get('REST_SEEDS', '0,1,2')}; chain route4 max t10 + own-space stage (v4.1/v5), gate fixed from P0", flush=True)
PN = {(a, v): load(f"{PF}/pft_{a}_{v}-seen_session-s*.npz", "seen_session", 3) for a in SLOT for v in ("base", "pfh60")}
FR = {a: {p: np.mean([load(f"{T6}/{FPX}_{a}_base-{p}-s{s}.npz", p) for s in FTS], 0) for p in PR} for a in SLOT}
CU = {a: {p: np.mean([load(f"{T6}/{FPX}_{a}_cum10-{p}-s{s}.npz", p) for s in FTS], 0) for p in PR} for a in SLOT}
OC = {p: np.mean([load(f"{REPO}/runs/ocunet/probs/oc_nomix-{p}-s{s}.npz", p) for s in OCS], 0) for p in PR}
RSS = os.environ.get("REST_SEEDS", "0,1,2").split(",")  # explicit seeds, as OCU_SEEDS
A.EXTRA["rest"] = {p: np.mean([load(f"{REPO}/runs/rest4/probs/rest4_sh_r1-{p}-s{s}.npz", p) for s in RSS], 0) for p in PR}
REF = {p: Z[f"GE|{p}"].copy() for p in PR}; GEO = {p: np.c_[REF[p], np.log(np.clip(OC[p], 1e-3, 1))] for p in PR}
SPEC = {"T0": dict(), "T1b": dict(pn="base"), "T1": dict(pn="pfh60"), "T1w": dict(pn="pfh60", w0=True), "F": dict(pn="pfh60", ft=True),
        "FO": dict(pn="pfh60", ft=True, ocu=True), "O": dict(pn="pfh60", ocu=True), "FOR": dict(pn="pfh60", ft=True, ocu=True, rest=True),
        "R": dict(pn="pfh60", rest=True)}


def routing():  # -> rows from each routed session's first routed window on (RTW), rows of routed sessions (RTS), from the last route4 call
    W, S = {p: np.zeros(len(IDX[p]), bool) for p in PR}, {p: np.zeros(len(IDX[p]), bool) for p in PR}
    for (proto, s), (r, j) in REC.items():
        names, gi, f = NM[(proto, s)]; pos = {int(g): k for k, g in enumerate(IDX[proto])}; rows = np.array([pos[int(g)] for g in gi])
        S[proto][rows] = True; W[proto][rows[int(np.flatnonzero(np.asarray(r) > 0)[0]):]] = True
    return W, S


def famswap(arrs, tag, rest):
    fams, extra = list(A.FAMS), []
    for a, slot in SLOT.items():
        A.EXTRA[f"pk_{a}_{tag}"] = arrs[a]; fam = next(f for f, items in A.FAM.items() if any(k == slot for k, _ in items))
        if fam in fams:
            fams.remove(fam); extra += list(A.FAM[fam])
        extra = [(k, w) for k, w in extra if k != slot] + [(f"pk_{a}_{tag}", 1.0)]
    return fams, extra + ([("rest", 1.0)] if rest else [])


def system(bn, arm, oof=None, RTS0=None):
    global BETA
    sp = SPEC.get(arm, {}); BETA = 0.10 if arm == "P0" else 0.15
    for p in PR:
        Z[f"GE|{p}"] = GEO[p] if sp.get("ocu") else REF[p]
    try:
        c1 = {p: A.members(bn, p)["c1"] for p in PR}
        if arm == "P0":
            pf, pe = list(A.FAMS), []
        else:
            pop = {a: {p: np.where(RTS0[p][:, None], FR[a][p], CU[a][p]) if sp.get("ft") else FR[a][p].copy() for p in PR} for a in SLOT}
            pf, pe = famswap(pop, arm + "_pop", sp.get("rest"))
        mp, sh, a = {}, {}, 0.0
        for p in PR:
            mp[p], a = A.mix(bn, p, pf, pe); _, sh[p] = A.shadow(mp[p], p)
        REC.clear(); out0 = ns["route4"](mp, sh, a, sh["all_loso"] if oof is None else oof, c1, "max", 10)[0]  # pass 1: routing
        if arm not in ("P0", "T0"):
            RTW, RTS = routing(); rt = {a: {p: x.copy() for p, x in pop[a].items()} for a in SLOT}
            for k in SLOT:  # not "a": that is the core1 share route4 needs
                m = RTS["seen_session"] if sp.get("w0") else RTW["seen_session"]; rt[k]["seen_session"][m] = PN[(k, sp["pn"])][m]
                if sp.get("w0"):
                    pop[k]["seen_session"][m] = PN[(k, sp["pn"])][m]
            if RTW["all_loso"].any():
                print(f"   {arm}: {int(RTW['all_loso'].sum())} new rows routed (false accepts); no personal nets exist for them here", flush=True)
            rf, re_ = famswap(rt, arm + "_rt", sp.get("rest"))
            if sp.get("w0"):  # dev form: the shadow (and so the gate's input) also sees the personal nets
                pf, pe = famswap(pop, arm + "_pop", sp.get("rest"))
                for p in PR:
                    mp[p], a = A.mix(bn, p, pf, pe); _, sh[p] = A.shadow(mp[p], p)
            mr = {p: A.mix(bn, p, rf, re_)[0] for p in PR}
            REC.clear(); out0 = ns["route4"](mr, sh, a, sh["all_loso"] if oof is None else oof, c1, "max", 10)[0]  # pass 2: routed tail with personal nets
        return staged(out0)[0], sh["all_loso"]
    finally:
        for p in PR:
            Z[f"GE|{p}"] = REF[p]


ARMS = ["P0"] + list(SPEC); R = {}
for bn in BASES:
    out, oof = system(bn, "P0"); R[bn, "P0"] = out
    R[bn, "T0"] = system(bn, "T0", oof, {p: np.zeros(len(IDX[p]), bool) for p in PR})[0]; _, RTS0 = routing()  # T0 uses no FT, so its mask is unused
    print(f"{bn}: routed sessions' rows new {RTS0['all_loso'].sum()} of {len(IDX['all_loso'])}, seen {RTS0['seen_session'].sum()} of {len(IDX['seen_session'])}; FT on the rest", flush=True)
    for arm in ARMS[2:]:
        R[bn, arm] = system(bn, arm, oof, RTS0)[0]
    s = {k: (SC.person(R[bn, k]["all_loso"], "all_loso"), SC.person(R[bn, k]["seen_session"], "seen_session")) for k in ARMS}
    print(f"   T0 new {s['T0'][0].mean():.2f} seen {s['T0'][1].mean():.2f} | " + " | ".join(f"{k} {s[k][0].mean() - s['T0'][0].mean():+.2f}/{s[k][1].mean() - s['T0'][1].mean():+.2f}" for k in ARMS if k != "T0"), flush=True)
    for k in ARMS:
        R[bn, k] = s[k]
rng = np.random.default_rng(0)
for ref, arms in (("T0", ARMS), ("T1b", ["T1", "T1w"]), ("T1", ["F", "O", "FO", "FOR", "R"])):
    print(f"\nagainst {ref}, six-base mean: new [person 80 %] people>=0, without the best | seen (P04, P10) [per-base range] | J")
    for k in arms:
        if k == ref:
            continue
        dn = np.array([R[b, k][0] - R[b, ref][0] for b in BASES]); ds = np.array([R[b, k][1] - R[b, ref][1] for b in BASES]); d = dn.mean(0); q = int(np.argmax(d))
        bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])
        print(f"  {k:4s} new {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] {(d >= 0).sum()}/10, w/o P{q + 1:02d} {np.delete(d, q).mean():+.2f} | seen {ds.mean():+.2f} "
              f"(P04 {ds[:, 0].mean():+.2f}, P10 {ds[:, 1].mean():+.2f}) [{ds.mean(1).min():+.2f}, {ds.mean(1).max():+.2f}] | J {0.5 * d.mean() + 0.5 * ds.mean():+.2f}", flush=True)
