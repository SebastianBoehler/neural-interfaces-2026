"""SHUFCOMBO (ledger s165/s167): v5.1b and the combination under non-chronological delivery, members re-streamed,
new people (all_loso; their windows take the unrouted tail, gm.pipe_g). For a delivery order (SCHEME, PSEED):
  P0rec  v5.1b in recording order (reference)        P0s  v5.1b, stages in the delivery order, members as recorded
  P0r    v5.1b with the SessionBN slots re-streamed in the delivery order (each base's own seeds' change, ttt17 base -
         ttt6 base, added to the shipped slot) and the stages in that order
  V6Fr   P0r + fine-tuning re-streamed (delta1: the first base seed's cum10 - base change of ttt17, applied to the slot)
         + eye-net view + rest member (both as computed in recording order unless OCU_DIR points at ocushuf outputs)
  V6Frec the same combination in recording order (ttt6 cum10 delta1), for the order loss of the combination itself
Usage: SCHEME=full PSEED=0 [OCU_DIR=runs/ocushuf/full0] python studies/13nov/shufcombo.py"""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SCHEME, PSEED = os.environ["SCHEME"], int(os.environ["PSEED"]); os.environ["SCHEMES"] = SCHEME; os.environ["PERMS"] = str(PSEED + 1)
import shuffle as S  # noqa: E402
A, SC, gm, p0 = S.A, S.SC, S.gm, S.p0; REPO = S.REPO; Z = A.L.Z; N = S.N[p0]
SLOT = {"fbraw": 5, "fb": 6, "sinc": 7, "fbl": 8}; BS = {"B2_s0": [0], "B2_s1": [1], "B2_s2": [2], "B2_s012": [0, 1, 2], "B2_s345": [3, 4, 5], "B2_pool": [0, 1, 2, 3, 4, 5]}
T6D, T17D = os.environ.get("T6_DIR", REPO + "/runs/ttt/probs6"), os.environ.get("T17_DIR", REPO + "/runs/ttt/probs17")
IDX = S.ORIG[[i for i, d in enumerate(S.DICTS) if d is A.pl.B][0]][f"idx|{p0}"].astype(int); assert (IDX == np.arange(N)).all()


def ld(path):
    z = np.load(path); assert (z["index"] == IDX).all(), path; return z["proba"]


FV = os.environ.get("FT_V", "cum10"); RPX = os.environ.get("REC_PFX", "ttt6")  # s171: FT_V=gC5 with REC_PFX=ttt7, T6_DIR=runs/ttt/probs7, T17_DIR=.../probs18, FT_TAG=ttt18...
FR6 = {(a, s): ld(f"{T6D}/{RPX}_{a}_base-{p0}-s{s}.npz") for a in SLOT for s in range(6)}
CU6 = {(a, s): ld(f"{T6D}/{RPX}_{a}_{FV}-{p0}-s{s}.npz") for a in SLOT for s in range(6)}
TG = os.environ.get("FT_TAG", f"ttt17{SCHEME}{PSEED}"); FR17 = {(a, s): ld(f"{T17D}/{TG}_{a}_base-{p0}-s{s}.npz") for a in SLOT for s in range(6)}
CU17 = {(a, s): ld(f"{T17D}/{TG}_{a}_{FV}-{p0}-s{s}.npz") for a in SLOT for s in range(6)}
OCD = os.environ.get("OCU_DIR"); OCS = sorted(glob.glob(f"{REPO}/{OCD}/oc_nomix-{p0}-s*.npz")) if OCD else [f"{REPO}/runs/ocunet/probs/oc_nomix-{p0}-s{s}.npz" for s in range(3)]
assert OCS, OCD; OCU = np.log(np.clip(np.mean([ld(f) for f in OCS], 0), 1e-3, 1))
REST = np.mean([ld(f"{REPO}/runs/rest4/probs/rest4_sh_r1-{p0}-s{s}.npz") for s in range(3)], 0)
print(f"SHUFCOMBO {SCHEME} {PSEED}: FT {FV} re-streamed {TG}, recording-order {RPX}; eye net {'re-streamed ' + OCD if OCD else 'recording order (3 seeds)'}; rest member recording order; new people", flush=True)


def norm(x):
    x = np.clip(x, 1e-6, None); return x / x.sum(1, keepdims=True)


def chain(bn, arm, P):
    """members for the arm in recording index order, then everything permuted by P, stages run in that order."""
    S.apply({p: (P if p == p0 else np.arange(S.N[p])) for p in S.PR}); base = {s_: A.L.Z[f"{bn}|Ps|{p0}"][s_][np.argsort(P)] for s_ in SLOT.values()}  # undo P: recording order
    ks = BS[bn]; arrs = {}
    for a, sl in SLOT.items():
        x = base[sl].copy()
        if arm in ("P0r", "V6Fr"):
            x = x + np.mean([FR17[a, k] - FR6[a, k] for k in ks], 0)
        if arm == "V6Fr":
            x = x + (CU17[a, ks[0]] - FR17[a, ks[0]])
        if arm == "V6Frec":
            x = x + (CU6[a, ks[0]] - FR6[a, ks[0]])
        arrs[a] = norm(x)
    fams, extra = list(A.FAMS), []
    for a, sl in SLOT.items():
        A.EXTRA[f"sc_{a}"] = {p0: arrs[a][P], "seen_session": A.L.Z[f"{bn}|Ps|seen_session"][sl]}
        fam = next(f for f, it in A.FAM.items() if any(k == sl for k, _ in it))
        if fam in fams:
            fams.remove(fam); extra += list(A.FAM[fam])
        extra = [(k, w) for k, w in extra if k != sl] + [(f"sc_{a}", 1.0)]
    if arm in ("V6Fr", "V6Frec"):
        A.EXTRA["sc_rest"] = {p0: REST[P], "seen_session": A.L.Z[f"{bn}|Ps|seen_session"][3]}; extra = extra + [("sc_rest", 1.0)]
    ge0 = A.L.Z[f"GE|{p0}"].copy()
    if arm in ("V6Fr", "V6Frec"):
        A.L.Z[f"GE|{p0}"] = np.c_[ge0, OCU[P]]
    try:
        out = gm.pipe_g(A.mix(bn, p0, fams, extra)[0], p0)
    finally:
        A.L.Z[f"GE|{p0}"] = ge0
    return SC.person(out, p0)


Pperm = S.perm_for(SCHEME, PSEED)[p0]; ident = np.arange(N); R = {}
for bn in S.BASES:
    R[bn] = {"P0rec": chain(bn, "P0", ident), "P0s": chain(bn, "P0", Pperm), "P0r": chain(bn, "P0r", Pperm), "V6Fr": chain(bn, "V6Fr", Pperm), "V6Frec": chain(bn, "V6Frec", ident)}
    print(f"{bn}: P0rec {R[bn]['P0rec'].mean():.2f} | P0s {R[bn]['P0s'].mean() - R[bn]['P0rec'].mean():+.2f} P0r {R[bn]['P0r'].mean() - R[bn]['P0rec'].mean():+.2f} | "
          f"V6Fr - P0r {R[bn]['V6Fr'].mean() - R[bn]['P0r'].mean():+.2f} | V6Frec - P0rec {R[bn]['V6Frec'].mean() - R[bn]['P0rec'].mean():+.2f}", flush=True)
S.apply({p: np.arange(S.N[p]) for p in S.PR}); rng = np.random.default_rng(0)
for lab, a, b in (("v5.1b, stages shuffled", "P0s", "P0rec"), ("v5.1b, stages + SessionBN members re-streamed", "P0r", "P0rec"),
                  ("combination over v5.1b, both shuffled", "V6Fr", "P0r"), ("combination over v5.1b, recording order", "V6Frec", "P0rec")):
    d = np.mean([R[bn][a] - R[bn][b] for bn in S.BASES], 0); bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])
    print(f"{lab:48s} new {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] without P10 {np.delete(d, 9).mean():+.2f}", flush=True)
np.savez(f"{REPO}/runs/shufcombo_{SCHEME}{PSEED}{'_ocu' if OCD else ''}{'' if FV == 'cum10' else '_' + FV}.npz", **{f"{bn}|{k}": v for bn, r in R.items() for k, v in r.items()})
