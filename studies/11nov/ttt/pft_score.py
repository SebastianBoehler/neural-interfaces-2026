"""Score PFT (ledger s29): personally fine-tuned nets on seen sessions, standalone, as chain members (single family and
all families jointly) and as a final multiplicative stage; always against the same nets frozen ("base").
New-person rows use the frozen wave-2 outputs (fine-tuning needs the person's own labelled session).
fb -> alib member 6; fbraw -> 5; sinc -> 7; fbl -> 8; sh -> extra member at weight 1.
Usage: python pft_score.py PROBS_PF PROBS2"""
import glob, os, sys
import numpy as np
REPO = __import__("os").path.abspath(__file__ + "/../../../..")
sys.path.insert(0, REPO + "/replay")
import env
A = env.A; SC = A.SC; D, D2 = sys.argv[1], sys.argv[2]
IDX = {p: A.pl.B[f"idx|{p}"].astype(int) for p in A.PROTOS}; SLOT = {"fb": 6, "fbraw": 5, "sinc": 7, "fbl": 8, "sh": None}
files = glob.glob(f"{D}/pft_*_base-seen_session-s*.npz"); archs = sorted({os.path.basename(f).split("_")[1] for f in files})
VARS = ["base", "pf20", "pf60", "pf200", "pfh60"]


def member(arch, v):
    fs = sorted(glob.glob(f"{D}/pft_{arch}_{v}-seen_session-s*.npz")); zs = [np.load(f) for f in fs]
    assert zs and all((z["index"] == IDX["seen_session"]).all() for z in zs)
    f2 = sorted(glob.glob(f"{D2}/ttt2_{arch}_base-all_loso-s*.npz")); z2 = [np.load(f) for f in f2]; assert z2, arch
    return {"all_loso": np.mean([z["proba"] for z in z2], 0), "seen_session": np.mean([z["proba"] for z in zs], 0)}, len(zs)


def swap(names):
    """names: {arch: member name} -> system kwargs with those families' slots replaced."""
    fams, extra = list(A.FAMS), []
    for arch, name in names.items():
        slot = SLOT[arch]
        if slot is None:
            extra.append((name, 1.0)); continue
        fam = next(f for f, items in A.FAM.items() if any(k == slot for k, _ in items))
        if fam in fams:
            fams.remove(fam); extra += [(k, w) for k, w in A.FAM[fam]]
        extra = [(k, w) for k, w in extra if k != slot] + [(name, 1.0)]
    return dict(fams=fams, extra=extra)


def seen(kw):
    return np.mean([SC.person(A.system(bn, kw["fams"], extra=kw["extra"])["out"]["seen_session"], "seen_session") for bn in env.EXPL], 0)


M = {}
print("arch variant seeds | standalone seen: P04 P10 (vs frozen) | chain seen vs frozen twin: mean (P04, P10)")
for arch in archs:
    for v in VARS:
        if not glob.glob(f"{D}/pft_{arch}_{v}-seen_session-s*.npz"):
            continue
        P, n = member(arch, v); A.EXTRA[f"{arch}_{v}"] = P; M[(arch, v)] = P
        ss = SC.person(P["seen_session"], "seen_session"); ds = seen(swap({arch: f"{arch}_{v}"}))
        if v == "base":
            ref = (ss, ds); print(f"{arch:5s} base  {n} | {ss[0]:6.2f} {ss[1]:6.2f} | chain seen {ds.mean():.3f} ({ds[0]:.2f}, {ds[1]:.2f})", flush=True); continue
        print(f"{arch:5s} {v:5s} {n} | {ss[0] - ref[0][0]:+6.2f} {ss[1] - ref[0][1]:+6.2f} | {(ds - ref[1]).mean():+.2f} ({ds[0] - ref[1][0]:+.2f}, {ds[1] - ref[1][1]:+.2f})", flush=True)
print("\nall families jointly (chain seen vs all frozen twins): mean (P04, P10)")
ref = seen(swap({a: f"{a}_base" for a in archs}))
print(f"frozen twins: {ref.mean():.3f} ({ref[0]:.2f}, {ref[1]:.2f}); shipped chain: {seen(dict(fams=A.FAMS, extra=[])).mean():.3f}")
for v in VARS[1:]:
    if all((a, v) in M for a in archs):
        ds = seen(swap({a: f"{a}_{v}" for a in archs})); print(f"{v:5s}: {(ds - ref).mean():+.2f} ({ds[0] - ref[0]:+.2f}, {ds[1] - ref[1]:+.2f})", flush=True)
print("\nfinal stage on seen sessions: shipped chain x (geometric mean of the families)^lambda, seen vs shipped: mean (P04, P10)")
base = {bn: A.system(bn, A.FAMS)["out"]["seen_session"] for bn in env.EXPL}; r0 = np.mean([SC.person(base[bn], "seen_session") for bn in env.EXPL], 0)
for v in VARS:
    if not all((a, v) in M for a in archs):
        continue
    E = np.exp(np.mean([np.log(np.clip(M[(a, v)]["seen_session"], 1e-6, 1)) for a in archs], 0)); E /= E.sum(1, keepdims=True)
    for lam in (0.25, 0.5, 1.0):
        ds = np.mean([SC.person((lambda q: q / q.sum(1, keepdims=True))(base[bn] * E ** lam), "seen_session") for bn in env.EXPL], 0) - r0
        print(f"{v:5s} lambda {lam:4.2f}: {ds.mean():+.2f} ({ds[0]:+.2f}, {ds[1]:+.2f})", flush=True)
