"""Build the chain with fine-tuned members swapped in and export its final posteriors as the next round's labels.
Usage: python round.py PROBS_DIR PREFIX VARIANT OUT.npz [archs, default all found]
fb -> member 6, fbraw -> 5, sinc -> 7, fbl -> 8, sh -> extra member (weight 1)."""
import glob, os, sys
import numpy as np
REPO = __import__("os").path.abspath(__file__ + "/../../../..")
sys.path.insert(0, REPO + "/replay")
import env
A = env.A; SC = A.SC; D, PRE, VAR, OUT = sys.argv[1:5]
IDX = {p: A.pl.B[f"idx|{p}"].astype(int) for p in A.PROTOS}; SLOT = {"fb": 6, "fbraw": 5, "sinc": 7, "fbl": 8, "sh": None}
archs = sys.argv[5].split(",") if len(sys.argv) > 5 else sorted({os.path.basename(f)[len(PRE) + 1:].split("_")[0] for f in glob.glob(f"{D}/{PRE}_*_{VAR}-all_loso-s*.npz")})


def member(arch, v):
    out = {}
    for p in A.PROTOS:
        zs = [np.load(f) for f in sorted(glob.glob(f"{D}/{PRE}_{arch}_{v}-{p}-s*.npz"))]
        assert zs and all((z["index"] == IDX[p]).all() for z in zs), (arch, v, p); out[p] = np.mean([z["proba"] for z in zs], 0)
    return out, len(zs)


def build(v):
    drop, extra = set(), []
    for a in archs:
        P, n = member(a, v); A.EXTRA[f"{a}_{v}"] = P; extra.append((f"{a}_{v}", 1.0))
        if SLOT[a] is not None:
            drop.add(SLOT[a])
    fams, keep = [], []
    for f in A.FAMS:
        items = A.FAM[f]
        if any(k in drop for k, _ in items):
            keep += [(k, w) for k, w in items if k not in drop]
        else:
            fams.append(f)
    return fams, keep + extra


res = {}
for v in ("base", VAR):
    fams, extra = build(v); J = []
    for bn in env.EXPL:
        r = A.system(bn, fams, extra=extra); s = A.summary(r); J.append((s["dn"], s["ds"]))
        if bn == "B2_pool":
            res[v] = r["out"]
    dn, ds = np.mean([j[0] for j in J], 0), np.mean([j[1] for j in J], 0); res[v + "_s"] = (dn, ds)
    print(f"{v:8s} archs {archs}: chain J {0.5 * dn.mean() + 0.5 * ds.mean():.3f} new {dn.mean():.2f} seen {ds.mean():.2f}", flush=True)
dn, ds = res[VAR + "_s"][0] - res["base_s"][0], res[VAR + "_s"][1] - res["base_s"][1]
G1 = [SC.PEOPLE.index(p) for p in SC.G1]; G2 = [SC.PEOPLE.index(p) for p in SC.G2]
print(f"{VAR} - base: dJ {0.5 * dn.mean() + 0.5 * ds.mean():+.2f} new {dn.mean():+.2f} (G1 {dn[G1].mean():+.2f}, G2 {dn[G2].mean():+.2f}; {int((dn >= 0).sum())}/10) seen {ds.mean():+.2f}")
y = SC.B["y|all_loso"]; print("label accuracy on new people: frozen chain %.4f, this chain %.4f" % (np.mean(res["base"]["all_loso"].argmax(1) == y), np.mean(res[VAR]["all_loso"].argmax(1) == y)))
np.savez(OUT, all_loso=res[VAR]["all_loso"], seen_session=res[VAR]["seen_session"], idx_seen=IDX["seen_session"])
