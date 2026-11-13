"""Levers of STACK (ledger s130) and the chain run that applies them; imported by stack.py and teach2.py."""
import glob, os, sys, time
import numpy as np
from sklearn.cross_decomposition import PLSRegression
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
import grid as G  # noqa: E402
A = env.A; SC = A.SC; Z = A.L.Z; rng = np.random.default_rng(0); PR = A.PROTOS
BASES = tuple(os.environ.get("BASES", ",".join(env.BASES)).split(",")); SMOKE = os.environ.get("SMOKE") == "1"
LEV = ("FT", "REST", "OCU", "MIX", "PLS"); SLOT = {"fbraw": 5, "fb": 6, "sinc": 7, "fbl": 8}; LAM = 0.15
IDX = {p: A.pl.B[f"idx|{p}"].astype(int) for p in PR}


def load(pat, p, nmin):
    zs = [np.load(f) for f in sorted(glob.glob(pat.replace("PROTO", p)))]
    assert len(zs) >= nmin and all((z["index"] == IDX[p]).all() for z in zs), (pat, p, len(zs))
    return np.mean([z["proba"] for z in zs], 0)


# FT: cum10 fine-tuned SessionBN families (new people only) against the same nets frozen (ttt6 base)
D6 = REPO + "/runs/ttt/probs6"
for a in SLOT:
    fr = {p: load(f"{D6}/ttt6_{a}_base-PROTO-s*.npz", p, 6) for p in PR}
    cu = load(f"{D6}/ttt6_{a}_cum10-all_loso-s*.npz", "all_loso", 6)
    assert np.abs(cu - fr["all_loso"]).max() > 1e-3, f"{a}: cum10 equals frozen (no-op)"
    A.EXTRA[f"{a}_fr"] = fr; A.EXTRA[f"{a}_ft"] = {"all_loso": cu, "seen_session": fr["seen_session"]}
# REST: rest-as-fourth-class ShallowConvNet as an extra member
A.EXTRA["rest"] = {p: load(REPO + "/runs/rest4/probs/rest4_sh_r1-PROTO-s*.npz", p, 3) for p in PR}
# OCU: eye-net log posteriors appended to SessionEM's GE view
OC = {p: load(REPO + "/runs/ocunet/probs/oc_nomix-PROTO-s*.npz", p, 3) for p in PR}
REF = {p: Z[f"GE|{p}"].copy() for p in PR}; GEO = {p: np.c_[REF[p], np.log(np.clip(OC[p], 1e-3, 1))] for p in PR}
# MIX: member mix to the power 0.9, renormalised
_mix = A.mix


def mix09(*a, **k):
    m, s = _mix(*a, **k); q = m ** 0.9; return q / q.sum(1, keepdims=True), s


# PLS: 16 PLS-DA directions of core1's components (per fold, training people only), second TSEM after the chain (s74)
p0 = "all_loso"; sess = Z[f"sess|{p0}"]; per = Z[f"person|{p0}"]; N = len(sess); VIEW = np.zeros((N, 16))
assert (Z[f"idx|{p0}"] == np.arange(N)).all()
for f in sorted(glob.glob(f"{REPO}/runs/replay_data/feats_exact/all_loso-P*.npz")):
    z = np.load(f, allow_pickle=True); Xtr, ytr, itr = z["pc_tr"], z["y_tr"].astype(int), z["idx_tr"].astype(int); Xc = Xtr.copy()
    for s in np.unique(sess[itr]):
        m = sess[itr] == s; Xc[m] -= Xc[m].mean(0)
    sd = Xc.std(0); W = PLSRegression(16, scale=False).fit(Xc / sd, np.eye(3)[ytr]).x_rotations_
    for k in [x for x in z.files if x.endswith("|pc")]:
        VIEW[z[k.replace("|pc", "|idx")].astype(int)] = (z[k] / sd) @ W
assert np.abs(VIEW).sum(1).min() > 0, "a test row never got a PLS view"


def run(S, bn, shipped=False, teach=None, add=(), ft="ft"):
    """S: levers; teach: (EXTRA name, weight) replacing the teacher family; add: extra [(EXTRA name, weight)] members."""
    fams, extra = list(A.FAMS), list(add)
    if teach:
        fams.remove("teacher"); extra.append(teach)
    if not shipped:
        for a, slot in SLOT.items():
            fam = next(f for f, items in A.FAM.items() if any(k == slot for k, _ in items))
            if fam in fams:
                fams.remove(fam); extra += list(A.FAM[fam])
            extra = [(k, w) for k, w in extra if k != slot] + [(f"{a}_{ft if 'FT' in S else 'fr'}", 1.0)]
    assert set(S) <= set(LEV), f"unknown lever in {S}"
    assert ft == "ft" or "FT" in S, f"ft={ft} has no effect without the FT lever"
    assert not (shipped and "FT" in S), "FT swaps the SessionBN families; shipped=True would drop it silently"
    if "REST" in S:
        extra.append(("rest", 1.0))
    for p in PR:
        Z[f"GE|{p}"] = GEO[p] if "OCU" in S else REF[p]
    A.mix = mix09 if "MIX" in S else _mix
    try:
        out = dict(A.system(bn, fams, extra=extra)["out"])
    finally:
        A.mix = _mix
        for p in PR:
            Z[f"GE|{p}"] = REF[p]
    if "PLS" in S:
        L, tt = G.tsem_lik(out[p0], sess, VIEW); out[p0] = G.apply_lam(out[p0], L, tt, LAM)
    return SC.person(out["all_loso"], "all_loso"), SC.person(out["seen_session"], "seen_session")
