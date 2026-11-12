"""FOCAL with the v5.1 stage (ledger s87): focal-loss SessionBN families against their twins, seen per person, six bases.
Usage: python studies/12nov/focal_v51.py"""
import glob, os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
F = REPO + "/studies/12nov/v5_replay.py"; exec(compile(open(F).read().split("\nres = []")[0], F, "exec"))  # noqa: S102
D = REPO + "/runs/netaug/probs"; SLOT = {"fbraw": 5, "fb": 6, "sinc": 7, "fbl": 8}; IDX = {p: A.pl.B[f"idx|{p}"].astype(int) for p in A.PROTOS}
for a in SLOT:
    for v in ("twin", "focal1", "focal2"):
        A.EXTRA[f"{a}_{v}"] = {p: np.mean([np.load(f)["proba"] for f in sorted(glob.glob(f"{D}/na_{a}_{v}-{p}-s*.npz"))], 0) for p in A.PROTOS}


def swap(v):
    fams, extra = list(A.FAMS), []
    for a, slot in SLOT.items():
        fam = next(f for f, items in A.FAM.items() if any(k == slot for k, _ in items))
        if fam in fams:
            fams.remove(fam); extra += [(k, w) for k, w in A.FAM[fam]]
        extra = [(k, w) for k, w in extra if k != slot] + [(f"{a}_{v}", 1.0)]
    return fams, extra


def run(bn, v):
    fams, extra = swap(v); mixes, sh, a = {}, {}, 0.0; REC.clear()
    for p in A.PROTOS:
        mixes[p], a = A.mix(bn, p, fams, extra); _, sh[p] = A.shadow(mixes[p], p)
    out0 = ns["route4"](mixes, sh, a, sh["all_loso"], {p: A.members(bn, p)["c1"] for p in A.PROTOS}, "max", 10)[0]
    return out0, staged(out0)[0]


R = []
for bn in ("B2_pool", "B2_s345", "B2_s012", "B2_s0", "B2_s1", "B2_s2"):
    t0, ts = run(bn, "twin"); line = []
    for v in ("focal1", "focal2"):
        f0, fs = run(bn, v); a = SC.compare(f0, t0); b = SC.compare(fs, ts); R.append((v, a, b))
        line.append(f"{v}: without stage seen {a['seen']:+.2f} (P04 {a['ds'][0]:+.2f}, P10 {a['ds'][1]:+.2f}) | with stage seen {b['seen']:+.2f} (P04 {b['ds'][0]:+.2f}, P10 {b['ds'][1]:+.2f}) new {b['new']:+.2f}")
    print(f"{bn}: " + " || ".join(line), flush=True)
for v in ("focal1", "focal2"):
    rs = [b for x, _, b in R if x == v]; print(f"MEAN {v} with the stage: seen {np.mean([r['seen'] for r in rs]):+.2f} (P04 {np.mean([r['ds'][0] for r in rs]):+.2f}, P10 {np.mean([r['ds'][1] for r in rs]):+.2f}) new {np.mean([r['new'] for r in rs]):+.2f}")
