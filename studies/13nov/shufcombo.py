"""SHUFCOMBO (ledger s165/s167): v5.1b and the combination under non-chronological delivery, members re-streamed,
new people (all_loso; their windows take the unrouted tail, gm.pipe_g). For a delivery order (SCHEME, PSEED):
  P0rec  v5.1b in recording order (reference)        P0s  v5.1b, stages in the delivery order, members as recorded
  P0r    v5.1b with the SessionBN slots re-streamed in the delivery order (each base's own seeds' change, ttt17 base -
         ttt6 base, added to the shipped slot) and the stages in that order
  V6Fr   P0r + fine-tuning re-streamed (delta1: the first base seed's cum10 - base change of ttt17, applied to the slot)
         + eye-net view + rest member (both as computed in recording order unless OCU_DIR points at ocushuf outputs)
  V6Frec the same combination in recording order (ttt6 cum10 delta1), for the order loss of the combination itself
s176, every re-streamed arm (P0r, V6Fr, V6r) also takes, when present for this order: core1 as member streamed in the
order (runs/coremember.npz, its change against its own recording-order rebuild added to the stored member, hb kept),
TSEM's pc128 view rebuilt in the order (runs/coreshuf_p<PSEED>.npz, sign-aligned change added), the rest net
re-streamed (runs/restshuf/<order>/). STRICT=1 fails when any of them is missing. The header prints what stays recorded.
GATE_P0=1: the combination's gate from v5.1b's chain under the same order (the package's gate sees its inner v5.1b).
LABEL_OUT=name: write FT pseudo-labels (v5.1b B2_pool emitted in this order) to studies/13nov/<name>.npz and exit.
Usage: SCHEME=full PSEED=0 [OCU_DIR=runs/ocushuf/full0] [STRICT=1] python studies/13nov/shufcombo.py"""
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
REST = np.mean([ld(f"{REPO}/runs/rest4/probs/rest4_sh_r1-{p0}-s{s}.npz") for s in range(3)], 0); ORD = f"{SCHEME}{PSEED}"
RSD, CMP = os.environ.get("RS_DIR", f"{REPO}/runs/restshuf"), os.environ.get("CM_NPZ", f"{REPO}/runs/coremember.npz")  # overrides: smoke tests only
RSS = sorted(glob.glob(f"{RSD}/{ORD}/rest4_sh_r1-{p0}-s*.npz")); RESTR = np.mean([ld(f) for f in RSS], 0) if RSS else REST
CMF = [CMP] if "CM_NPZ" in os.environ else sorted(glob.glob(f"{REPO}/runs/coremember*.npz"))  # coremember.py writes one file per order set
CM = next((z for z in map(np.load, filter(os.path.exists, CMF)) if f"{SCHEME}{PSEED}|{p0}" in z), {})
DC1 = CM[f"{ORD}|{p0}"] - CM[f"rec|{p0}"] if f"{ORD}|{p0}" in CM else None  # core1 member: delivery order - its own recording-order rebuild
CSF = [os.environ["CS_NPZ"]] if "CS_NPZ" in os.environ else sorted(glob.glob(f"{REPO}/runs/coreshuf_p{PSEED}*.npz"))  # coreshuf.py writes one file per scheme set
CS = next((z for z in map(np.load, filter(os.path.exists, CSF)) if f"{SCHEME}|{p0}" in z), {}); DPC = None  # CS_NPZ=none: pc128 as recorded
if f"{SCHEME}|{p0}" in CS:
    ref, rec = S.ORIG[[i for i, d in enumerate(S.DICTS) if d is Z][0]][f"pc128|{p0}"], CS[f"rec|{p0}"]
    sg = np.sign([np.corrcoef(ref[:, j], rec[:, j])[0, 1] for j in range(ref.shape[1])]); DPC = sg * (CS[f"{SCHEME}|{p0}"] - rec)
REC_LEFT = [n for n, ok in (("core1 member", DC1 is not None), ("pc128 view", DPC is not None), ("rest net", bool(RSS)), ("eye net", bool(OCD))) if not ok] + [
    "GE view (SessionEM's)", "Hilbert member", "transformer and shallow members (causal statistics)", "teacher member"]
assert not (os.environ.get("STRICT") and len(REC_LEFT) > 4), f"STRICT: still in recording order: {REC_LEFT[:-4]}"
print(f"SHUFCOMBO {SCHEME} {PSEED}: FT {FV} re-streamed {TG}, recording-order {RPX}; eye net {'re-streamed ' + OCD if OCD else 'recording order (3 seeds)'}; "
      f"rest net {'re-streamed ' + str(len(RSS)) + ' seeds' if RSS else 'recording order'}; core1 member {'re-streamed' if DC1 is not None else 'recording order'}; "
      f"pc128 {'rebuilt in the order' if DPC is not None else 'recording order'}; new people\nSTILL IN RECORDING ORDER: {', '.join(REC_LEFT)}", flush=True)


GATE_P0, LABEL_OUT, GB, LAB = bool(os.environ.get("GATE_P0")), os.environ.get("LABEL_OUT"), {}, {}
if GATE_P0:
    print("GATE_P0: the combination's gate is computed on v5.1b's chain under the same order (package form)", flush=True)


def norm(x):
    x = np.clip(x, 1e-6, None); return x / x.sum(1, keepdims=True)


def chain(bn, arm, P):
    """members for the arm in recording index order, then everything permuted by P, stages run in that order."""
    S.apply({p: (P if p == p0 else np.arange(S.N[p])) for p in S.PR})
    if arm in ("P0r", "V6Fr", "V6r"):  # s176: core1 (member, keeping hb = 2 Ps[0] - c1 fixed) and pc128 in the delivery order
        if DC1 is not None:
            A.C1[p0] = A.C1[p0] + DC1[P]; Z[f"{bn}|Ps|{p0}"][0] = Z[f"{bn}|Ps|{p0}"][0] + DC1[P] / 2
        if DPC is not None:
            Z[f"pc128|{p0}"] = Z[f"pc128|{p0}"] + DPC[P].astype(Z[f"pc128|{p0}"].dtype)
    base = {s_: A.L.Z[f"{bn}|Ps|{p0}"][s_][np.argsort(P)] for s_ in SLOT.values()}  # undo P: recording order
    ks = BS[bn]; arrs = {}
    for a, sl in SLOT.items():
        x = base[sl].copy()
        if arm in ("P0r", "V6Fr", "V6r"):
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
    if arm in ("V6Fr", "V6Frec", "V6r"):
        A.EXTRA["sc_rest"] = {p0: (RESTR if arm != "V6Frec" else REST)[P], "seen_session": A.L.Z[f"{bn}|Ps|seen_session"][3]}; extra = extra + [("sc_rest", 1.0)]
    ge0 = A.L.Z[f"GE|{p0}"].copy()
    if arm in ("V6Fr", "V6Frec", "V6r"):
        A.L.Z[f"GE|{p0}"] = np.c_[ge0, OCU[P]]
    try:
        b = gm.M.rebal(A.mix(bn, p0, fams, extra)[0], p0, None); rec = (P == np.arange(len(P))).all()
        if arm in ("P0", "P0r"):
            GB[bn, "P0rec" if rec and arm == "P0" else arm] = b
        ref = {"V6Fr": "P0r", "V6r": "P0r", "V6Frec": "P0rec"}.get(arm)  # GATE_P0: the package's gate sees its inner v5.1b, not the v6 chain
        g = gm.gate(GB[bn, ref] if GATE_P0 and ref else b, p0); out = gm.tsem_g(gm.em_g(b, p0, g), p0, g)
    finally:
        A.L.Z[f"GE|{p0}"] = ge0
    if LABEL_OUT and arm == "P0r":
        LAB[bn] = out[np.argsort(P)]  # v5.1b's outputs as emitted in the delivery order, back in recording index layout
    return SC.person(out, p0)


Pperm = S.perm_for(SCHEME, PSEED)[p0]; ident = np.arange(N); R = {}
assert (IDX[Pperm] == np.load(f"{REPO}/runs/shuffle_perms.npz")[f"{SCHEME}|{PSEED}|{p0}"]).all(), "the stages' order is not the order the members were streamed in"
if LABEL_OUT:  # FT pseudo-labels as the package sees them: v5.1b (B2_pool) emitted in this delivery order; seen rows as chain_final
    CH = np.load(f"{REPO}/studies/12nov/chain_final.npz"); y = Z[f"y|{p0}"].astype(int); chain("B2_pool", "P0r", Pperm); lab = LAB["B2_pool"]
    print(f"labels: delivery-order v5.1b acc {np.mean(lab.argmax(1) == y):.4f} | chain_final (recording order) acc {np.mean(CH[p0].argmax(1) == y):.4f} | "
          f"agreement {np.mean(lab.argmax(1) == CH[p0].argmax(1)):.4f}", flush=True)
    np.savez(f"{REPO}/studies/13nov/{LABEL_OUT}.npz", all_loso=lab, seen_session=CH["seen_session"], idx_seen=CH["idx_seen"]); sys.exit(print("written", LABEL_OUT))
for bn in S.BASES:
    R[bn] = {"P0rec": chain(bn, "P0", ident), "P0s": chain(bn, "P0", Pperm), "P0r": chain(bn, "P0r", Pperm), "V6Fr": chain(bn, "V6Fr", Pperm), "V6Frec": chain(bn, "V6Frec", ident),
              "V6r": chain(bn, "V6r", Pperm)}  # s173: the fallback without FT (eye net + rest member only)
    print(f"{bn}: P0rec {R[bn]['P0rec'].mean():.2f} | P0s {R[bn]['P0s'].mean() - R[bn]['P0rec'].mean():+.2f} P0r {R[bn]['P0r'].mean() - R[bn]['P0rec'].mean():+.2f} | "
          f"V6Fr - P0r {R[bn]['V6Fr'].mean() - R[bn]['P0r'].mean():+.2f} | V6Frec - P0rec {R[bn]['V6Frec'].mean() - R[bn]['P0rec'].mean():+.2f}", flush=True)
S.apply({p: np.arange(S.N[p]) for p in S.PR}); rng = np.random.default_rng(0)
for lab, a, b in (("v5.1b, stages shuffled", "P0s", "P0rec"), ("v5.1b, stages + members re-streamed (as header)", "P0r", "P0rec"),
                  ("combination over v5.1b, both shuffled", "V6Fr", "P0r"), ("combination over v5.1b, recording order", "V6Frec", "P0rec"),
                  ("no-FT fallback (eye net + rest) over v5.1b, shuffled", "V6r", "P0r")):
    d = np.mean([R[bn][a] - R[bn][b] for bn in S.BASES], 0); bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])
    print(f"{lab:48s} new {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] without P10 {np.delete(d, 9).mean():+.2f}", flush=True)
# the FT tag is in the name, so draws and label forms do not overwrite each other
np.savez(f"{REPO}/runs/shufcombo_{SCHEME}{PSEED}{'_ocu' if OCD else ''}{'' if FV == 'cum10' else '_' + FV}{'_all' if len(REC_LEFT) == 4 else ''}{'_gp0' if GATE_P0 else ''}{'_' + TG if TG != f'ttt17{SCHEME}{PSEED}' else ''}.npz", **{f"{bn}|{k}": v for bn, r in R.items() for k, v in r.items()})
