"""PKG (ledger s138): the package-faithful replay of the design inputs, new and seen people measured together.
Chain: routing v4.1 (route4 "max", t10) + the own-space stage on routed windows (v5_replay.staged), six bases.
The gate's calibration (route4's oof) is fixed to the shipped members' outputs, as in a package; fine-tuning (cum10,
ttt6 seeds 0-5) is applied only to sessions the gate never routes; returning people get the personal nets (pfh60) on
routed sessions and the stage at BETA; OCU appends the eye-net log posteriors to SessionEM's view everywhere.
Arms:  P0  shipped members, stage 0.10 (v5.1b)       P1  personal nets + stage 0.15 (§119's next candidate)
       P1t P1 with the four SessionBN slots of unrouted sessions from ttt6's frozen nets (the twins of FT)
       P2  P1t + FT on never-routed sessions          P3  P2 + OCU          P4  P1t + OCU
Usage: [BASES=B2_pool] python studies/13nov/pkg.py"""
import glob, os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
F = REPO + "/studies/12nov/v5_replay.py"; exec(compile(open(F).read().split("\nres = []")[0], F, "exec"))  # noqa: S102  (A, SC, ns, REC, NM, staged, BETA)
Z = A.L.Z; PR = A.PROTOS; IDX = {p: A.pl.B[f"idx|{p}"].astype(int) for p in PR}; SLOT = {"fb": 6, "fbraw": 5, "sinc": 7, "fbl": 8}
BASES = os.environ.get("BASES", "B2_pool,B2_s345,B2_s012,B2_s0,B2_s1,B2_s2").split(",")


def load(pat, p, n):
    zs = [np.load(f) for f in sorted(glob.glob(pat))]; assert (len(zs) == n if n else len(zs) >= 1) and all((z["index"] == IDX[p]).all() for z in zs), (pat, len(zs)); return np.mean([z["proba"] for z in zs], 0)


PF, T2, T6 = REPO + "/runs/ttt/probs_pf", REPO + "/runs/ttt/probs2", REPO + "/runs/ttt/probs6"
PN = {(a, v): {"all_loso": load(f"{T2}/ttt2_{a}_base-all_loso-s*.npz", "all_loso", 0)  # as pnet.py: every stored seed (sinc has one)
                 , "seen_session": load(f"{PF}/pft_{a}_{v}-seen_session-s*.npz", "seen_session", 3)}
      for a in list(SLOT) + ["sh"] for v in ("base", "pfh60")}
T6 = REPO + "/" + os.environ.get("FT_DIR", "runs/ttt/probs6"); FPX = os.environ.get("FT_PFX", "ttt6")  # fresh seeds: FT_DIR=runs/diag/ttt/probs8 FT_PFX=ttt8 FT_SEEDS=6-11
_a, _b = (int(x) for x in os.environ.get("FT_SEEDS", "0-5").split("-")); FTS = range(_a, _b + 1); OCS = os.environ.get("OCU_SEEDS", "0,1,2").split(",")
print(f"pkg: FT {T6} {FPX} seeds {list(FTS)}; OCU seeds {OCS}; chain route4 max t10 + own-space stage (v4.1/v5)", flush=True)
FR = {a: {p: np.mean([load(f"{T6}/{FPX}_{a}_base-{p}-s{s}.npz", p, 1) for s in FTS], 0) for p in PR} for a in SLOT}
CU = {a: {p: np.mean([load(f"{T6}/{FPX}_{a}_cum10-{p}-s{s}.npz", p, 1) for s in FTS], 0) for p in PR} for a in SLOT}
OC = {p: np.mean([load(f"{REPO}/runs/ocunet/probs/oc_nomix-{p}-s{s}.npz", p, 1) for s in OCS], 0) for p in PR}
REF = {p: Z[f"GE|{p}"].copy() for p in PR}; GEO = {p: np.c_[REF[p], np.log(np.clip(OC[p], 1e-3, 1))] for p in PR}


def routed_rows():  # rows of every session with at least one routed window, from REC/NM of the last route4 call
    rows = {p: np.zeros(len(IDX[p]), bool) for p in PR}
    for (proto, s) in REC:
        names, gi, f = NM[(proto, s)]; pos = {int(g): k for k, g in enumerate(IDX[proto])}; rows[proto][[pos[int(g)] for g in gi]] = True
    return rows


def arm_members(arm, RT):  # not "members": v5_replay.staged calls its own members()
    """EXTRA entries and the family swap for an arm; RT = routed rows (fixed from P1t)."""
    if arm == "P0":
        return list(A.FAMS), []
    fams, extra = list(A.FAMS), []
    for a in list(SLOT) + ["sh"]:
        P = {p: PN[(a, "pfh60")][p].copy() for p in PR}
        if a in SLOT and arm != "P1":
            for p in PR:
                src = CU[a][p] if arm in ("P2", "P3") else FR[a][p]; P[p][~RT[p]] = src[~RT[p]]
        A.EXTRA[f"pk_{a}_{arm}"] = P
        if a == "sh":
            extra.append((f"pk_{a}_{arm}", 1.0)); continue
        fam = next(f for f, items in A.FAM.items() if any(k == SLOT[a] for k, _ in items))
        if fam in fams:
            fams.remove(fam); extra += [(k, w) for k, w in A.FAM[fam]]
        extra = [(k, w) for k, w in extra if k != SLOT[a]] + [(f"pk_{a}_{arm}", 1.0)]
    return fams, extra


def system(bn, arm, oof, RT):
    global BETA
    fams, extra = arm_members(arm, RT); BETA = 0.10 if arm == "P0" else 0.15; mixes, sh, a = {}, {}, 0.0; REC.clear()
    for p in PR:
        Z[f"GE|{p}"] = GEO[p] if arm in ("P3", "P4") else REF[p]
    try:
        for p in PR:
            mixes[p], a = A.mix(bn, p, fams, extra); _, sh[p] = A.shadow(mixes[p], p)
        out0 = ns["route4"](mixes, sh, a, sh["all_loso"] if oof is None else oof, {p: A.members(bn, p)["c1"] for p in PR}, "max", 10)[0]
        return staged(out0)[0], sh["all_loso"]
    finally:
        for p in PR:
            Z[f"GE|{p}"] = REF[p]


ARMS = ("P0", "P1", "P1t", "P2", "P3", "P4"); R = {}
for bn in BASES:
    out, oof = system(bn, "P0", None, None); R[bn, "P0"] = out          # the shipped members fix the gate's calibration
    out, _ = system(bn, "P1t", oof, {p: np.zeros(len(IDX[p]), bool) for p in PR}); RT = routed_rows()
    print(f"{bn}: never-routed sessions get FT; routed rows new {RT['all_loso'].sum()} of {len(RT['all_loso'])}, seen {RT['seen_session'].sum()} of {len(RT['seen_session'])}", flush=True)
    for arm in ARMS[1:]:
        R[bn, arm], _ = system(bn, arm, oof, RT)
        mv = {p: int((routed_rows()[p] != RT[p]).sum()) for p in PR}
        if arm != "P1" and any(mv.values()):
            print(f"   WARNING {arm}: routing moved on {mv} rows; the FT rows stay fixed from P1t", flush=True)
    s = {k: (SC.person(R[bn, k]["all_loso"], "all_loso"), SC.person(R[bn, k]["seen_session"], "seen_session")) for k in ARMS}
    print(f"   P0 J {0.5 * s['P0'][0].mean() + 0.5 * s['P0'][1].mean():.2f} | " + " | ".join(f"{k} new {s[k][0].mean() - s['P0'][0].mean():+.2f} seen {s[k][1].mean() - s['P0'][1].mean():+.2f}" for k in ARMS[1:]), flush=True)
    for k in ARMS:
        R[bn, k] = s[k]
rng = np.random.default_rng(0)
for ref in ("P0", "P1", "P1t"):
    print(f"\nagainst {ref}, six-base mean (new [person bootstrap 10-90] bases ahead | seen P04 P10 | J)")
    for k in ARMS:
        if k == ref:
            continue
        dn = np.array([R[b, k][0] - R[b, ref][0] for b in BASES]); ds = np.array([R[b, k][1] - R[b, ref][1] for b in BASES]); d = dn.mean(0)
        bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])
        print(f"  {k:4s} new {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] {(dn.mean(1) > 0).sum()}/{len(BASES)} | seen {ds.mean():+.2f} (P04 {ds[:, 0].mean():+.2f}, P10 {ds[:, 1].mean():+.2f}, bases {(ds.mean(1) > 0).sum()}/{len(BASES)}) | J {0.5 * d.mean() + 0.5 * ds.mean():+.2f}", flush=True)
