"""DOSE (exploratory, ledger s43): own-space final stage on the real v4.1 chain, dose-response, with and without PFT.
Usage: [BASES=a,b] python studies/12nov/dose.py"""
import glob, os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/studies/12nov"]
import ownspace as O  # noqa: E402
rows = lambda p, s: np.flatnonzero((O.PER == p) & (O.SES == s))  # noqa: E731
Q = {k: np.full((len(O.y), 3), np.nan) for k in "TCS"}
for p in ("P04", "P10"):
    m = O.Personal([rows(p, "1")])
    for b in "23":
        o = m.stream(rows(p, b))
        for k in "TCS":
            Q[k][rows(p, b)] = o[k]
R4 = REPO + "/studies/11nov/rt4/route4.py"; ns = {"__file__": R4, "__name__": "route4"}
sys.argv = ["route4.py", REPO + "/runs/replay_data/bio_model_psd_d64.npz"]
exec(compile(open(R4).read().split("\nbases = sys.argv")[0], R4, "exec"), ns)
A, R3, SC = ns["A"], ns["R3"], ns["SC"]; NM = {}; _gm = R3.gate_model
D, D2 = REPO + "/runs/ttt/probs_pf", REPO + "/runs/ttt/probs2"
IDX = {p: A.pl.B[f"idx|{p}"].astype(int) for p in A.PROTOS}; SLOT = {"fb": 6, "fbraw": 5, "sinc": 7, "fbl": 8, "sh": None}
ARCHS = sorted({os.path.basename(f).split("_")[1] for f in glob.glob(f"{D}/pft_*_base-seen_session-s*.npz")})


def gate_model(proto, f, oof):
    out = _gm(proto, f, oof)
    for s, (Xs, ys, gi) in out[0].items():
        NM[(proto, s)] = (list(out[1]), np.asarray(gi))
    return out


R3.gate_model = gate_model
for arch in ARCHS:
    for v in ("base", "pfh60"):
        zs = [np.load(f) for f in sorted(glob.glob(f"{D}/pft_{arch}_{v}-seen_session-s*.npz"))]; assert zs and all((z["index"] == IDX["seen_session"]).all() for z in zs)
        z2 = [np.load(f) for f in sorted(glob.glob(f"{D2}/ttt2_{arch}_base-all_loso-s*.npz"))]; assert z2
        A.EXTRA[f"{arch}_{v}"] = {"all_loso": np.mean([z["proba"] for z in z2], 0), "seen_session": np.mean([z["proba"] for z in zs], 0)}


def swap(v):
    fams, extra = list(A.FAMS), []
    for arch in ARCHS:
        slot, name = SLOT[arch], f"{arch}_{v}"
        if slot is None:
            extra.append((name, 1.0)); continue
        fam = next(f for f, items in A.FAM.items() if any(k == slot for k, _ in items))
        if fam in fams:
            fams.remove(fam); extra += [(k, w) for k, w in A.FAM[fam]]
        extra = [(k, w) for k, w in extra if k != slot] + [(name, 1.0)]
    return fams, extra


def sys4(bn, v):
    fams, extra = swap(v); mixes, sh, a = {}, {}, 0.0
    for p in A.PROTOS:
        mixes[p], a = A.mix(bn, p, fams, extra); _, sh[p] = A.shadow(mixes[p], p)
    return ns["route4"](mixes, sh, a, sh["all_loso"], {p: A.members(bn, p)["c1"] for p in A.PROTOS}, "max", 10)[0]


def own(on, H0):
    for (proto, s), (names, gi) in NM.items():
        if proto == "seen_session":
            H = H0[(proto, s)].copy()
            if on:
                j = [i for i, q in enumerate(names) if q.endswith(O.PER[gi[0]])][0]
                L = np.log(np.clip(H[j], 1e-9, 1)) + 0.5 / 3 * sum(np.log(np.clip(Q[k][gi], 1e-3, 1)) for k in "TCS")
                e = np.exp(L - L.max(1, keepdims=True)); H[j] = e / e.sum(1, keepdims=True)
            A._H[(proto, s)] = H



def fstage(out, keys, beta, tfun=None):
    """final stage on the seen sessions: chain x (geometric mean of the own-space members)^beta."""
    o = {p: v.copy() for p, v in out.items()}; idx = A.pl.B["idx|seen_session"]; pos = {int(g): k for k, g in enumerate(idx)}
    for (proto, s), (names, gi) in NM.items():
        if proto == "seen_session":
            r = np.array([pos[int(g)] for g in gi]); b = beta if tfun is None else beta * tfun(np.arange(len(r)))[:, None]
            L = np.log(np.clip(out[proto][r], 1e-9, 1)) + b * sum(np.log(np.clip(Q[k][gi], 1e-3, 1)) for k in keys) / len(keys)
            e = np.exp(L - L.max(1, keepdims=True)); o[proto][r] = e / e.sum(1, keepdims=True)
    return o


BASES = os.environ.get("BASES", "B2_pool,B2_s345").split(",")
for bn in BASES:
    ref = ns["system4"](bn, "max", 10)[0]; H0 = dict(A._H); print(SC.absline(bn + " v4.1 (shipped)", ref), flush=True)
    twin = sys4(bn, "base"); pft = sys4(bn, "pfh60")
    for tag, out in (("shipped", ref), ("PFT (vs twins)", pft)):
        b0 = ref if tag == "shipped" else twin
        for keys in ("TCS", "TC", "C", "T"):
            for beta in (0.05, 0.1, 0.15, 0.25, 0.4):
                r = SC.compare(fstage(out, keys, beta), b0)
                print(f"   {tag:14s} F {keys:3s} beta {beta:4.2f}: seen {r['seen']:+.2f} (P04 {r['ds'][0]:+.2f}, P10 {r['ds'][1]:+.2f})", flush=True)
        if tag != "shipped":
            r = SC.compare(out, b0); print(f"   {tag:14s} alone: seen {r['seen']:+.2f} (P04 {r['ds'][0]:+.2f}, P10 {r['ds'][1]:+.2f})")
