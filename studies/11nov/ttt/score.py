"""Score TTT / TTT2: fine-tuned members standalone and in the chain, against their own frozen ("base") outputs.
fb -> replaces alib member 6; fbraw -> 5; sinc -> 7; fbl -> 8; sh -> extra member at weight 1.
Usage: python score.py PROBS_DIR PREFIX(ttt|ttt2)"""
import glob, os, sys
import numpy as np
REPO = __import__("os").path.abspath(__file__ + "/../../../..")
sys.path.insert(0, REPO + "/replay")
import env
A = env.A; SC = A.SC; D, PRE = sys.argv[1], sys.argv[2]
IDX = {p: A.pl.B[f"idx|{p}"].astype(int) for p in A.PROTOS}; SLOT = {"fb": 6, "fbraw": 5, "sinc": 7, "fbl": 8, "sh": None}
G1 = [SC.PEOPLE.index(p) for p in SC.G1]; G2 = [SC.PEOPLE.index(p) for p in SC.G2]
files = glob.glob(f"{D}/{PRE}_*-all_loso-s*.npz"); archs = sorted({os.path.basename(f)[len(PRE) + 1:].split("_")[0] for f in files})


def member(arch, v):
    out, seeds = {}, None
    for p in A.PROTOS:
        fs = sorted(glob.glob(f"{D}/{PRE}_{arch}_{v}-{p}-s*.npz")); sd = [f.rsplit("-s", 1)[1][:-4] for f in fs]
        seeds = sd if seeds is None else [s for s in seeds if s in sd]
    for p in A.PROTOS:
        zs = [np.load(f"{D}/{PRE}_{arch}_{v}-{p}-s{s}.npz") for s in seeds]
        assert zs and all((z["index"] == IDX[p]).all() for z in zs), (arch, v, p)
        out[p] = np.mean([z["proba"] for z in zs], 0)
    return out, seeds


def chain(arch, name):
    slot = SLOT[arch]
    if slot is None:
        return A.system, dict(fams=A.FAMS, extra=[(name, 1.0)])
    fam = next(f for f, items in A.FAM.items() if any(k == slot for k, _ in items))
    keep = [(k, w) for k, w in A.FAM[fam] if k != slot]
    return A.system, dict(fams=[f for f in A.FAMS if f != fam], extra=keep + [(name, 1.0)])


print("arch variant seeds | standalone new seen (vs frozen) | chain dJ vs frozen: J new (G1, G2) seen")
for arch in archs:
    vs = sorted({os.path.basename(f)[len(PRE) + 1 + len(arch) + 1:].split("-")[0] for f in files if os.path.basename(f).startswith(f"{PRE}_{arch}_")}, key=lambda v: (v != "base", v))
    ref = None
    for v in vs:
        P, seeds = member(arch, v); sn, ss = SC.person(P["all_loso"], "all_loso"), SC.person(P["seen_session"], "seen_session"); A.EXTRA[f"{arch}_{v}"] = P
        fn, kw = chain(arch, f"{arch}_{v}"); dn, ds = [], []
        for bn in env.EXPL:
            s = A.summary(fn(bn, kw["fams"], extra=kw["extra"])); dn.append(s["dn"]); ds.append(s["ds"])
        dn, ds = np.mean(dn, 0), np.mean(ds, 0)
        if v == "base":
            ref = (sn, ss, dn, ds); print(f"{arch:5s} base     {len(seeds)} | {sn.mean():6.2f} {ss.mean():6.2f} | chain J {0.5 * dn.mean() + 0.5 * ds.mean():.3f}", flush=True); continue
        d = dn - ref[2]
        print(f"{arch:5s} {v:8s} {len(seeds)} | {sn.mean() - ref[0].mean():+6.2f} {ss.mean() - ref[1].mean():+6.2f} | {0.5 * d.mean() + 0.5 * (ds - ref[3]).mean():+.2f} {d.mean():+.2f} ({d[G1].mean():+.2f}, {d[G2].mean():+.2f}) {(ds - ref[3]).mean():+.2f}", flush=True)
