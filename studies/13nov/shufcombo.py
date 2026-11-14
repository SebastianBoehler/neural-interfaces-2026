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
LABEL_OUT=name: write FT pseudo-labels (v5.1b B2_pool emitted in this order) to runs/labels/<name>.npz and exit.
Usage: SCHEME=full PSEED=0 [OCU_DIR=runs/ocushuf/full0] [STRICT=1] python studies/13nov/shufcombo.py"""
import glob, hashlib, os, sys


def one_with(files, key, what):
    """the single file among files holding key (s194: fail loudly on ambiguity instead of taking the first)."""
    hit = [f for f in files if os.path.exists(f) and key in np.load(f)]; assert len(hit) <= 1, f"several {what} files hold {key}: {hit}"
    return np.load(hit[0]) if hit else {}
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SCHEME, PSEED = os.environ["SCHEME"], int(os.environ["PSEED"]); os.environ["SCHEMES"] = SCHEME; os.environ["PERMS"] = str(PSEED + 1)
import shuffle as S  # noqa: E402
A, SC, gm = S.A, S.SC, S.gm; p0 = os.environ.get("PROTO", S.p0); OTHER = next(p for p in S.PR if p != p0); REPO = S.REPO; Z = A.L.Z; N = S.N[p0]
# PROTO=seen_session (s183): returning people with recognition forced off, i.e. through the new-person path (gm.pipe_g)
SLOT = {"fbraw": 5, "fb": 6, "sinc": 7, "fbl": 8}; BS = {"B2_s0": [0], "B2_s1": [1], "B2_s2": [2], "B2_s012": [0, 1, 2], "B2_s345": [3, 4, 5], "B2_pool": [0, 1, 2, 3, 4, 5]}
T6D, T17D = os.environ.get("T6_DIR", REPO + "/runs/ttt/probs6"), os.environ.get("T17_DIR", REPO + "/runs/ttt/probs17")
IDX = S.ORIG[[i for i, d in enumerate(S.DICTS) if d is A.pl.B][0]][f"idx|{p0}"].astype(int); assert p0 != "all_loso" or (IDX == np.arange(N)).all()


def ld(path):
    z = np.load(path); assert (z["index"] == IDX).all(), path; return z["proba"]


FV = os.environ.get("FT_V", "cum10"); RPX = os.environ.get("REC_PFX", "ttt6")  # s171: FT_V=gC5 with REC_PFX=ttt7, T6_DIR=runs/ttt/probs7, T17_DIR=.../probs18, FT_TAG=ttt18...
FR6 = {(a, s): ld(f"{T6D}/{RPX}_{a}_base-{p0}-s{s}.npz") for a in SLOT for s in range(6)}
HAS_REC = all(os.path.exists(f"{T6D}/{RPX}_{a}_{FV}-{p0}-s{s}.npz") for a in SLOT for s in range(6))  # s194: sweep forms have no recording-order run
CU6 = {(a, s): ld(f"{T6D}/{RPX}_{a}_{FV}-{p0}-s{s}.npz") for a in SLOT for s in range(6)} if HAS_REC else None
if not HAS_REC:
    print(f"no recording-order {RPX} {FV} files: arm V6Frec is skipped (its line is left out)", flush=True)
TG = os.environ.get("FT_TAG", f"ttt17{SCHEME}{PSEED}"); FR17 = {(a, s): ld(f"{T17D}/{TG}_{a}_base-{p0}-s{s}.npz") for a in SLOT for s in range(6)}
CU17 = {(a, s): ld(f"{T17D}/{TG}_{a}_{FV}-{p0}-s{s}.npz") for a in SLOT for s in range(6)}
OCD = os.environ.get("OCU_DIR"); OCS = sorted(glob.glob(f"{REPO}/{OCD}/oc_nomix-{p0}-s*.npz")) if OCD else [f"{REPO}/runs/ocunet/probs/oc_nomix-{p0}-s{s}.npz" for s in range(3)]
assert OCS, OCD; OCU = np.log(np.clip(np.mean([ld(f) for f in OCS], 0), 1e-3, 1))
_ow = hashlib.sha256(np.ascontiguousarray(np.load(f"{REPO}/runs/shuffle_perms.npz")[f"{SCHEME}|{PSEED}|{p0}"].astype(np.int64)).tobytes()).hexdigest() if SCHEME != "rec" else None
for _f in (OCS if OCD and _ow else []):  # s214: where an eye output records its delivery order, it must be this protocol's
    _z = np.load(_f); assert "order_sha" not in _z.files or str(_z["order_sha"]) == _ow, f"{_f}: eye net not in {p0}'s delivery order"
if os.environ.get("OCU_SYN") is not None:  # s188 accuracy curve (ceiling test only): the eye net's posteriors shuffled within person x true class
    # (same accuracy and confusion, no correlation with the EEG members' errors), then +d points by swapping the top class with the true one
    pr, yy, pp = np.exp(OCU), Z[f"y|{p0}"].astype(int), Z[f"person|{p0}"]; rg = np.random.default_rng(0); a0 = np.mean(pr.argmax(1) == yy); sp = pr.copy()
    for q in np.unique(pp):
        for c in range(3):
            r = np.flatnonzero((pp == q) & (yy == c)); sp[r] = pr[rg.permutation(r)]
    wrong = np.flatnonzero(sp.argmax(1) != yy); k = int(round(float(os.environ["OCU_SYN"]) / 100 * len(yy)))
    for r in rg.permutation(wrong)[:k]:
        t, m = yy[r], sp[r].argmax(); sp[r, [t, m]] = sp[r, [m, t]]
    OCU = np.log(np.clip(sp, 1e-3, 1)); print(f"OCU_SYN +{os.environ['OCU_SYN']}: eye view accuracy {a0:.4f} -> {np.mean(sp.argmax(1) == yy):.4f} (shuffled within person x class)", flush=True)
OCU_TX = os.environ.get("OCU_TX", "")  # s196 (D1/D2): transforms of the eye log-posterior columns
if OCU_TX in ("shufrun", "demeanrun", "demeansess"):  # diagnostics (non-causal): which part of the eye view's error correlation hurts
    yy, pp, rr, ss = Z[f"y|{p0}"].astype(int), Z[f"person|{p0}"], S.RUN[IDX], Z[f"sess|{p0}"]; rg = np.random.default_rng(0); o2 = OCU.copy()
    if OCU_TX == "shufrun":  # keeps each run's bias, removes the time structure inside a run
        for key in set(zip(pp, yy, rr)):
            r = np.flatnonzero((pp == key[0]) & (yy == key[1]) & (rr == key[2])); o2[r] = OCU[rg.permutation(r)]
    else:  # remove the per-run (or per-session) mean; demeanrun adds the session mean back
        grp = np.char.add(ss.astype(str), "|" + rr.astype(str)) if OCU_TX == "demeanrun" else ss
        for g in np.unique(grp):
            r = grp == g; o2[r] = OCU[r] - OCU[r].mean(0) + (OCU[ss == ss[r][0]].mean(0) if OCU_TX == "demeanrun" else 0)
    OCU = o2; print(f"OCU_TX {OCU_TX}: diagnostic transform of the eye view (non-causal)", flush=True)
elif OCU_TX:
    assert OCU_TX.startswith("causal"), OCU_TX; print(f"OCU_TX {OCU_TX}: causal centring of the eye view in the delivery order (package form)", flush=True)


def ocu_view(P):
    """the eye view rows in delivery order; causal[W] subtracts the running mean of the session's earlier windows (window W)."""
    v = OCU[P]
    if not OCU_TX.startswith("causal"):
        return v
    W = int(OCU_TX[6:] or 10 ** 6); ss = A.L.Z[f"sess|{p0}"]; out = v.copy()
    for g in dict.fromkeys(ss):
        r = np.flatnonzero(ss == g); cs = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v[r], 0)]); n = np.arange(len(r)); lo = np.maximum(0, n - W)
        m = np.where((n - lo)[:, None] > 0, (cs[n] - cs[lo]) / np.maximum(n - lo, 1)[:, None], 0.0); out[r] = v[r] - m  # prefix mean, current window excluded
    return out


if os.environ.get("OCU_ORACLE"):  # s188 ceiling test only, never shippable: the eye view replaced by a smoothed one-hot of the true labels
    OCU = np.log(0.6 * np.eye(3)[Z[f"y|{p0}"].astype(int)] + 0.4 / 3); print("OCU_ORACLE: the eye view is the true label (ceiling test)", flush=True)
REST = np.mean([ld(f"{REPO}/runs/rest4/probs/rest4_sh_r1-{p0}-s{s}.npz") for s in range(3)], 0); ORD = f"{SCHEME}{PSEED}"
RSD, CMP = os.environ.get("RS_DIR", f"{REPO}/runs/restshuf"), os.environ.get("CM_NPZ", f"{REPO}/runs/coremember.npz")  # overrides: smoke tests only
RSS = sorted(glob.glob(f"{RSD}/{ORD}/rest4_sh_r1-{p0}-s*.npz")); RESTR = np.mean([ld(f) for f in RSS], 0) if RSS else REST
CMF = [CMP] if "CM_NPZ" in os.environ else sorted(glob.glob(f"{REPO}/runs/coremember*.npz"))  # coremember.py writes one file per order set
CM = one_with(CMF, f"{SCHEME}{PSEED}|{p0}", "core1 member")
DC1 = CM[f"{ORD}|{p0}"] - CM[f"rec|{p0}"] if f"{ORD}|{p0}" in CM else None  # core1 member: delivery order - its own recording-order rebuild
CSF = [os.environ["CS_NPZ"]] if "CS_NPZ" in os.environ else sorted(glob.glob(f"{REPO}/runs/coreshuf_p{PSEED}.npz") + glob.glob(f"{REPO}/runs/coreshuf_p{PSEED}_*.npz"))
# coreshuf.py writes coreshuf_p<seed>[_<schemes>].npz. Exact patterns (s194 bug): "p1*" also matched p10/p11, and within 1 read within 10's pc128
CSH = [f for f in filter(os.path.exists, CSF) if f"{SCHEME}|{p0}" in np.load(f)]; assert len(CSH) <= 1, f"several pc128 files for {SCHEME} {PSEED}: {CSH}"
CS = np.load(CSH[0]) if CSH else {}; DPC = None  # CS_NPZ=none: pc128 as recorded
if f"{SCHEME}|{p0}" in CS:
    ref, rec = S.ORIG[[i for i, d in enumerate(S.DICTS) if d is Z][0]][f"pc128|{p0}"], CS[f"rec|{p0}"]
    sg = np.sign([np.corrcoef(ref[:, j], rec[:, j])[0, 1] for j in range(ref.shape[1])]); DPC = sg * (CS[f"{SCHEME}|{p0}"] - rec)
# s184: GE's event rates (geshuf.py) and the Hilbert member (coremember.py MEMBER=hilbert), each as a change against its own rebuild
GS = np.load(f"{REPO}/runs/geshuf.npz") if os.path.exists(f"{REPO}/runs/geshuf.npz") and os.environ.get("GE_NPZ") != "none" else {}
DGE = GS[f"{ORD}|{p0}"][IDX] - GS["rec"][IDX] if f"{ORD}|{p0}" in GS else None
if DGE is not None:  # GE = [gaze 32 | event rates 16]; the stored event rates must be the rebuild's
    assert np.allclose(S.ORIG[[i for i, d in enumerate(S.DICTS) if d is Z][0]][f"GE|{p0}"][:, 32:], GS["rec"][IDX]), "GE columns 32: are not the event rates"
HBF = sorted(glob.glob(f"{REPO}/runs/hilbertmember*.npz")) if os.environ.get("HB_NPZ") != "none" else []
HB = one_with(HBF, f"{ORD}|{p0}", "Hilbert"); DHB = HB[f"{ORD}|{p0}"] - HB[f"rec|{p0}"] if f"{ORD}|{p0}" in HB else None
TSF = sorted(glob.glob(f"{REPO}/runs/teachshuf*.npz")) if os.environ.get("TS_NPZ") != "none" else []  # s184: teacher, slot 10
TSF = [os.environ["TS_FILE"]] if os.environ.get("TS_FILE") else TSF  # s207: an edited teacher file (studies/14nov/teachgain.py)
TS = one_with(TSF, f"{ORD}|{p0}", "teacher"); DTS = TS[f"{ORD}|{p0}"] - TS[f"rec|{p0}"] if f"{ORD}|{p0}" in TS else None
CORE = [n for n, ok in (("core1 member", DC1 is not None), ("pc128 view", DPC is not None), ("rest net", bool(RSS)), ("eye net", bool(OCD))) if not ok]
REC_LEFT = CORE + [n for n, ok in (("GE event rates", DGE is not None), ("Hilbert member", DHB is not None), ("teacher member", DTS is not None)) if not ok] + [
    "transformer and shallow ea members (causal EA)"]
# s203: STRICT also requires GE, Hilbert and teacher; it used to check only CORE, so a missing one dropped out silently
assert not (os.environ.get("STRICT") and REC_LEFT[:-1]), f"STRICT: still in recording order: {REC_LEFT[:-1]}"
SYN_T, SYN_D = os.environ.get("SYN_T"), float(os.environ.get("SYN_D", 0))  # s207 leverage (oracle diagnostic, never shippable)
if SYN_T:  # +SYN_D accuracy points on one component (s<k> member slot, c1 core1, rest, eye), in recording layout before any
    ZO = next(o for d, o in zip(S.DICTS, S.ORIG) if d is Z); yy = ZO[f"y|{p0}"].astype(int)  # permutation: top and true swapped
    def syn(pr):  # on randomly chosen wrong windows (the swap set is the same for every base and arm, seed 0)
        pr = pr.copy(); w = np.flatnonzero(pr.argmax(1) != yy); a0 = np.mean(pr.argmax(1) == yy)
        for r in np.random.default_rng(0).permutation(w)[:int(round(SYN_D / 100 * len(yy)))]:
            t, m = yy[r], pr[r].argmax(); pr[r, [t, m]] = pr[r, [m, t]]
        return pr, a0, np.mean(pr.argmax(1) == yy)
    if SYN_T[0] == "s":
        k = int(SYN_T[1:])
        for key in [q for q in ZO if q.endswith(f"|Ps|{p0}")]:
            ZO[key] = ZO[key].copy(); ZO[key][k], a0, a1 = syn(ZO[key][k])
    elif SYN_T == "c1":  # core1 also sits in slot 0 as hb = 2 Ps[0] - c1, so Ps[0] moves by half its change
        c1n, a0, a1 = syn(S.C1[p0]); dc = c1n - S.C1[p0]; S.C1[p0] = c1n
        for key in [q for q in ZO if q.endswith(f"|Ps|{p0}")]:
            ZO[key] = ZO[key].copy(); ZO[key][0] = ZO[key][0] + dc / 2
    elif SYN_T == "rest":
        RESTR, a0, a1 = syn(RESTR)
    elif SYN_T == "eye":
        e, a0, a1 = syn(np.exp(OCU)); OCU = np.log(np.clip(e, 1e-3, 1))
    else:
        raise SystemExit(f"SYN_T {SYN_T}: unknown component")
    print(f"SYN {SYN_T} +{SYN_D:g}: component accuracy {a0:.4f} -> {a1:.4f} (last base for slots)", flush=True)
OUTX = os.environ.get("OUTX")  # s207 stage 2: a final-layer edit (studies/14nov/wsp_final.py), per held-out person
OX_T = os.path.basename(OUTX).split("_")[1] if OUTX else None
if OUTX:
    OXZ = np.load(OUTX); ZO = next(o for d, o in zip(S.DICTS, S.ORIG) if d is Z); OXP = ZO[f"person|{p0}"]
    def oedit(pr):  # z' = t (I + B) log p + c, row by row with the row's person's edit; a person absent from the file fails
        out = np.empty_like(pr); lz = np.log(np.clip(pr, 1e-6, 1))
        for i, q in enumerate(OXZ["people"]):
            r = OXP == q; o = OXZ["t"][i] * lz[r] @ (np.eye(3) + OXZ["B"][i]).T + OXZ["c"][i]; o = np.exp(o - o.max(1, keepdims=True)); out[r] = o / o.sum(1, keepdims=True)
        assert np.isin(OXP, OXZ["people"]).all(), "OUTX: people without an edit"; return out
    if OX_T[0] == "s" and int(OX_T[1:]) not in SLOT.values():
        for key in [q for q in ZO if q.endswith(f"|Ps|{p0}")]:
            ZO[key] = ZO[key].copy(); ZO[key][int(OX_T[1:])] = oedit(ZO[key][int(OX_T[1:])])
    elif OX_T == "c1":
        c1n = oedit(S.C1[p0]); dc = c1n - S.C1[p0]; S.C1[p0] = c1n
        for key in [q for q in ZO if q.endswith(f"|Ps|{p0}")]:
            ZO[key] = ZO[key].copy(); ZO[key][0] = ZO[key][0] + dc / 2
    elif OX_T == "rest":
        RESTR = oedit(RESTR)
    elif OX_T == "eye":
        OCU = np.log(np.clip(oedit(np.exp(OCU)), 1e-3, 1))
    print(f"OUTX {OUTX}: final-layer edit of {OX_T}" + (" (fine-tuned slot: applied after the FT delta, in chain)" if OX_T[0] == "s" and int(OX_T[1:]) in SLOT.values() else ""), flush=True)
print(f"SHUFCOMBO {SCHEME} {PSEED}: FT {FV} re-streamed {TG}, recording-order {RPX}; eye net {'re-streamed ' + OCD if OCD else 'recording order (3 seeds)'}; "
      f"rest net {'re-streamed ' + str(len(RSS)) + ' seeds' if RSS else 'recording order'}; core1 member {'re-streamed' if DC1 is not None else 'recording order'}; "
      f"pc128 {'rebuilt in the order' if DPC is not None else 'recording order'}; new people\nSTILL IN RECORDING ORDER: {', '.join(REC_LEFT)}", flush=True)


GATE_P0, LABEL_OUT, GB, LAB = bool(os.environ.get("GATE_P0")), os.environ.get("LABEL_OUT"), {}, {}
SAVE_OUT, OUTS = os.environ.get("SAVE_OUT"), {}  # s193: path for per-window outputs of P0r / V6Fr / V6r (recording layout)
OCU_TS = int(os.environ.get("OCU_TS", 0))  # s188: k copies of the eye view appended to TSEM's pc128 view (0 = off)
LABEL_ARM = os.environ.get("LABEL_ARM", "P0r")  # s187: V6r = labels from the v6 chain itself (eye view + rest member, no FT)
assert LABEL_ARM in ("P0r", "V6r", "BLEND"), LABEL_ARM  # BLEND (s200): the mean of the v5.1b and v6-chain outputs as emitted
if GATE_P0:
    print("GATE_P0: the combination's gate is computed on v5.1b's chain under the same order (package form)", flush=True)


def norm(x):
    x = np.clip(x, 1e-6, None); return x / x.sum(1, keepdims=True)


RUNOFF, RUNOFF_V = float(os.environ.get("RUNOFF", 0)), os.environ.get("RUNOFF_V", "both")  # s210 (panel BCI 2.1)


def runoff(V, ss, rr):
    """rows in delivery order: v - j/(j+k) (mean of the run's j earlier rows - mean of the session's earlier rows)."""
    out = V.copy()
    for g in dict.fromkeys(ss):
        rs = np.flatnonzero(ss == g); cs = np.vstack([np.zeros((1, V.shape[1])), np.cumsum(V[rs], 0)]); n = np.arange(len(rs))
        ms = np.where(n[:, None] > 0, cs[n] / np.maximum(n, 1)[:, None], 0.0)  # session prefix mean, current row excluded
        for u in dict.fromkeys(rr[rs]):
            q = np.flatnonzero(rr[rs] == u); cq = np.vstack([np.zeros((1, V.shape[1])), np.cumsum(V[rs][q], 0)]); j = np.arange(len(q))
            mr = np.where(j[:, None] > 0, cq[j] / np.maximum(j, 1)[:, None], 0.0)
            out[rs[q]] -= (j / (j + RUNOFF))[:, None] * np.where(j[:, None] > 0, mr - ms[q], 0.0)
    return out


NUIS, NUIS_V = int(os.environ.get("NUIS", 0)), os.environ.get("NUIS_V", "pc")  # s218: project out the between-run subspace


def nuis_projectors(key):
    """per held-out person: the top-NUIS directions of between-run offsets (run mean minus session mean, each class
    centred within its session) of view `key`, estimated on the OTHER people's recording-order rows (labels used offline)."""
    ZO = next(o for d, o in zip(S.DICTS, S.ORIG) if d is Z); V = ZO[f"{key}|{p0}"].astype(np.float64); yy = ZO[f"y|{p0}"].astype(int)
    ss, pp = ZO[f"sess|{p0}"], ZO[f"person|{p0}"]; rr = np.char.add(ss.astype(str), np.char.add("|", S.RUN[IDX])); C = V.copy()
    for g in np.unique(ss):
        for c in range(3):
            m = (ss == g) & (yy == c); C[m] -= C[m].mean(0)
    off = {u: (pp[rr == u][0], C[rr == u].mean(0) - C[ss == ss[rr == u][0]].mean(0)) for u in np.unique(rr)}
    U = {}
    for q in np.unique(pp):
        M = np.array([o for (pq, o) in off.values() if pq != q]); U[q] = np.linalg.svd(M - M.mean(0), full_matrices=False)[2][:NUIS].T
    return U


NUIS_U = {k: nuis_projectors(k) for k in (["pc128"] if NUIS_V == "pc" else ["pc128", "GE"])} if NUIS else {}


def nuis_apply(V, pp):  # rows of any order: each row loses its person's projector's span
    out = V.copy()
    for q, U in NUIS_U_CUR.items():
        m = pp == q; d = U.shape[0]; out[np.ix_(m, np.arange(d))] = V[m][:, :d] - (V[m][:, :d] @ U) @ U.T  # GE: its own columns only (eye view untouched)
    return out


def chain(bn, arm, P):
    """members for the arm in recording index order, then everything permuted by P, stages run in that order."""
    S.apply({p: (P if p == p0 else np.arange(S.N[p])) for p in S.PR})
    if arm in ("P0r", "V6Fr", "V6r"):  # s176: core1 (member, keeping hb = 2 Ps[0] - c1 fixed) and pc128 in the delivery order
        if DC1 is not None:
            A.C1[p0] = A.C1[p0] + DC1[P]; Z[f"{bn}|Ps|{p0}"][0] = Z[f"{bn}|Ps|{p0}"][0] + DC1[P] / 2
        if DPC is not None:
            Z[f"pc128|{p0}"] = Z[f"pc128|{p0}"] + DPC[P].astype(Z[f"pc128|{p0}"].dtype)
        if DHB is not None:  # hb = 2 Ps[0] - c1: Ps[0] moves by half the Hilbert change
            Z[f"{bn}|Ps|{p0}"][0] = Z[f"{bn}|Ps|{p0}"][0] + DHB[P] / 2
        if DTS is not None:
            Z[f"{bn}|Ps|{p0}"][10] = Z[f"{bn}|Ps|{p0}"][10] + DTS[P]
        if DGE is not None:
            Z[f"GE|{p0}"] = Z[f"GE|{p0}"].copy(); Z[f"GE|{p0}"][:, 32:] = Z[f"GE|{p0}"][:, 32:] + DGE[P]
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
        if OUTX and OX_T == f"s{sl}":  # s207: the fine-tuned slot's final layer edited after its FT delta
            arrs[a] = oedit(arrs[a])
    fams, extra = list(A.FAMS), []
    for a, sl in SLOT.items():
        A.EXTRA[f"sc_{a}"] = {p0: arrs[a][P], OTHER: A.L.Z[f"{bn}|Ps|{OTHER}"][sl]}
        fam = next(f for f, it in A.FAM.items() if any(k == sl for k, _ in it))
        if fam in fams:
            fams.remove(fam); extra += list(A.FAM[fam])
        extra = [(k, w) for k, w in extra if k != sl] + [(f"sc_{a}", 1.0)]
    if arm in ("V6Fr", "V6Frec", "V6r"):
        A.EXTRA["sc_rest"] = {p0: (RESTR if arm != "V6Frec" else REST)[P], OTHER: A.L.Z[f"{bn}|Ps|{OTHER}"][3]}; extra = extra + [("sc_rest", 1.0)]
    ge0, pc0 = A.L.Z[f"GE|{p0}"].copy(), A.L.Z[f"pc128|{p0}"].copy()
    if arm in ("V6Fr", "V6Frec", "V6r"):
        ov = ocu_view(P); A.L.Z[f"GE|{p0}"] = np.c_[ge0, ov]
        if OCU_TS:  # s188 form A: the eye view in TSEM too. TSEM standardises columns, so weight = k copies (scaling would be a no-op)
            A.L.Z[f"pc128|{p0}"] = np.c_[pc0, np.repeat(ov, OCU_TS, axis=1)].astype(pc0.dtype)
    if NUIS:  # s218: no run id, no order: a fixed projector per held-out person (in the package: one matrix from all ten)
        global NUIS_U_CUR
        for key, Uk in NUIS_U.items():
            NUIS_U_CUR = Uk; A.L.Z[f"{key}|{p0}"] = nuis_apply(A.L.Z[f"{key}|{p0}"].astype(np.float64), A.L.Z[f"person|{p0}"]).astype(A.L.Z[f"{key}|{p0}"].dtype)
    if RUNOFF:  # s210 oracle: true run ids; each view row minus its run's shrunk causal offset from the session (delivery order)
        ss = A.L.Z[f"sess|{p0}"]  # RUN_IDS=<npz with key <scheme><seed>: run label per replay row> replaces the true run ids
        rr = np.char.add(ss.astype(str), np.char.add("|", (np.load(os.environ["RUN_IDS"])[f"{SCHEME}{PSEED}"].astype(str) if os.environ.get("RUN_IDS") else S.RUN[IDX])[P]))
        for key in (["pc128"] if RUNOFF_V == "pc" else ["GE"] if RUNOFF_V == "ge" else ["pc128", "GE"]):
            A.L.Z[f"{key}|{p0}"] = runoff(A.L.Z[f"{key}|{p0}"].astype(np.float64), ss, rr).astype(A.L.Z[f"{key}|{p0}"].dtype)
    try:
        b = gm.M.rebal(A.mix(bn, p0, fams, extra)[0], p0, None); rec = (P == np.arange(len(P))).all()
        if arm in ("P0", "P0r"):
            GB[bn, "P0rec" if rec and arm == "P0" else arm] = b
        ref = {"V6Fr": "P0r", "V6r": "P0r", "V6Frec": "P0rec"}.get(arm)  # GATE_P0: the package's gate sees its inner v5.1b, not the v6 chain
        g = gm.gate(GB[bn, ref] if GATE_P0 and ref else b, p0); out = gm.tsem_g(gm.em_g(b, p0, g), p0, g)
    finally:
        A.L.Z[f"GE|{p0}"], A.L.Z[f"pc128|{p0}"] = ge0, pc0
    if SAVE_OUT and arm in ("P0r", "V6Fr", "V6r"):  # s193: per-window outputs (recording layout) for the session selector
        OUTS[f"{bn}|{arm}"] = out[np.argsort(P)].astype(np.float32)
    if LABEL_OUT and (arm == LABEL_ARM or (LABEL_ARM == "BLEND" and arm in ("P0r", "V6r"))):
        LAB[bn if LABEL_ARM != "BLEND" else (bn, arm)] = out[np.argsort(P)]  # outputs as emitted in the delivery order, back in recording index layout
    return SC.person(out, p0)


ident = np.arange(N); R = {}; Pperm = ident if SCHEME == "rec" else S.perm_for(SCHEME, PSEED)[p0]  # s191: SCHEME=rec = recording order
PERMS_SHA = "6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f"  # s180: every re-streamed input (ttt17/18, ocushuf, restshuf, coremember, coreshuf) read this file
assert hashlib.sha256(open(f"{REPO}/runs/shuffle_perms.npz", "rb").read()).hexdigest() == PERMS_SHA, "runs/shuffle_perms.npz is not the pinned order file"
assert SCHEME == "rec" or (IDX[Pperm] == np.load(f"{REPO}/runs/shuffle_perms.npz")[f"{SCHEME}|{PSEED}|{p0}"]).all(), "the stages' order is not the order the members were streamed in"
assert not (SCHEME == "rec" and os.environ.get("STRICT")), "STRICT has no meaning in recording order (nothing is re-streamed)"
assert not (LABEL_OUT and p0 != "all_loso"), "LABEL_OUT is for new people"
if LABEL_OUT:  # FT pseudo-labels as the package sees them: v5.1b (B2_pool) emitted in this delivery order; seen rows as chain_final
    CH = np.load(f"{REPO}/studies/12nov/chain_final.npz"); y = Z[f"y|{p0}"].astype(int)
    if LABEL_ARM != "P0r":
        chain("B2_pool", "P0r", Pperm)  # the package gate (GATE_P0) of the v6 chain is v5.1b's
    if LABEL_ARM == "BLEND":
        chain("B2_pool", "V6r", Pperm); lab = 0.5 * (LAB["B2_pool", "P0r"] + LAB["B2_pool", "V6r"])
    else:
        chain("B2_pool", LABEL_ARM, Pperm); lab = LAB["B2_pool"]
    print(f"labels ({LABEL_ARM}): delivery-order chain acc {np.mean(lab.argmax(1) == y):.4f} | chain_final (recording order) acc {np.mean(CH[p0].argmax(1) == y):.4f} | "
          f"agreement {np.mean(lab.argmax(1) == CH[p0].argmax(1)):.4f}", flush=True)
    np.savez(f"{REPO}/runs/labels/{LABEL_OUT}.npz", all_loso=lab, seen_session=CH["seen_session"], idx_seen=CH["idx_seen"]); sys.exit(print("written", LABEL_OUT))
for bn in S.BASES:
    R[bn] = {"P0rec": chain(bn, "P0", ident), "P0s": chain(bn, "P0", Pperm), "P0r": chain(bn, "P0r", Pperm), "V6Fr": chain(bn, "V6Fr", Pperm),
              "V6r": chain(bn, "V6r", Pperm)}  # s173: the fallback without FT (eye net + rest member only)
    if HAS_REC:
        R[bn]["V6Frec"] = chain(bn, "V6Frec", ident)
    print(f"{bn}: P0rec {R[bn]['P0rec'].mean():.2f} | P0s {R[bn]['P0s'].mean() - R[bn]['P0rec'].mean():+.2f} P0r {R[bn]['P0r'].mean() - R[bn]['P0rec'].mean():+.2f} | "
          f"V6Fr - P0r {R[bn]['V6Fr'].mean() - R[bn]['P0r'].mean():+.2f}" + (f" | V6Frec - P0rec {R[bn]['V6Frec'].mean() - R[bn]['P0rec'].mean():+.2f}" if HAS_REC else ""), flush=True)
S.apply({p: np.arange(S.N[p]) for p in S.PR}); rng = np.random.default_rng(0)
for lab, a, b in (("v5.1b, stages shuffled", "P0s", "P0rec"), ("v5.1b, stages + members re-streamed (as header)", "P0r", "P0rec"),
                  ("combination over v5.1b, both shuffled", "V6Fr", "P0r"), ("combination over v5.1b, recording order", "V6Frec", "P0rec"),
                  ("no-FT fallback (eye net + rest) over v5.1b, shuffled", "V6r", "P0r")):
    if a not in R[S.BASES[0]]:
        continue
    d = np.mean([R[bn][a] - R[bn][b] for bn in S.BASES], 0)
    if len(d) != 10:  # returning people (PROTO=seen_session): too few people for a bootstrap; per person instead
        print(f"{lab:48s} {p0} mean {d.mean():+.2f} | per person " + " ".join(f"{q} {v:+.2f}" for q, v in zip(np.unique(Z[f'person|{p0}']), d)), flush=True); continue
    bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])
    print(f"{lab:48s} {'new' if p0 == 'all_loso' else p0} {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] without P10 {np.delete(d, 9).mean():+.2f}", flush=True)
# the FT tag is in the name, so draws and label forms do not overwrite each other; s208: so is a non-default
# OCU_DIR (form B runs had overwritten the regular files of within 1-3, full 0/1, runs 0/1)
np.savez(f"{REPO}/runs/shufcombo_{'' if p0 == 'all_loso' else p0 + '_'}{SCHEME}{PSEED}{'_ocu' if OCD else ''}{'' if not OCD or OCD.rstrip('/') == f'runs/ocushuf/{SCHEME}{PSEED}' else '_ocd' + OCD.rstrip('/').replace('runs/', '').replace('/', '-')}{'' if FV == 'cum10' else '_' + FV}{'_all' if not CORE else ''}{'_ge' if DGE is not None else ''}{'_hb' if DHB is not None else ''}{'_ts' if DTS is not None else ''}{f'_ocuts{OCU_TS}' if OCU_TS else ''}{'_' + OCU_TX if OCU_TX else ''}{'_gp0' if GATE_P0 else ''}{'_' + TG if TG != f'ttt17{SCHEME}{PSEED}' else ''}{f'_syn{SYN_T}{SYN_D:g}' if SYN_T else ''}{'_ox' + os.path.basename(OUTX)[6:-4] if OUTX else ''}{'_cm' + os.path.basename(CMP)[:-4] if 'CM_NPZ' in os.environ else ''}{'_rs' + os.path.basename(RSD.rstrip('/')) if 'RS_DIR' in os.environ and RSD.rstrip('/') not in (f'{REPO}/runs/restshuf', 'runs/restshuf', 'runs/diag/restshuf') else ''}{'_tf' + os.path.basename(os.environ['TS_FILE'])[:-4] if os.environ.get('TS_FILE') else ''}{f'_runoff{RUNOFF_V}{RUNOFF:g}' if RUNOFF else ''}{'_rid' + os.path.basename(os.environ['RUN_IDS'])[:-4] if os.environ.get('RUN_IDS') else ''}{f'_nuis{NUIS_V}{NUIS}' if NUIS else ''}.npz", **{f"{bn}|{k}": v for bn, r in R.items() for k, v in r.items()})
if SAVE_OUT:
    np.savez_compressed(f"{REPO}/{SAVE_OUT}", **OUTS); print("per-window outputs ->", SAVE_OUT, flush=True)
