"""PNET (ledger s78): personal (fine-tuned) nets on seen sessions + the v5.1 own-space stage, six bases.
Usage: python studies/12nov/pnet.py"""
import glob, os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
F = REPO + "/studies/12nov/v5_replay.py"; exec(compile(open(F).read().split("\nres = []")[0], F, "exec"))  # noqa: S102  (defines A, R3, SC, ns, REC, staged)
D, D2 = REPO + "/runs/ttt/probs_pf", REPO + "/runs/ttt/probs2"
IDX = {p: A.pl.B[f"idx|{p}"].astype(int) for p in A.PROTOS}; SLOT = {"fb": 6, "fbraw": 5, "sinc": 7, "fbl": 8, "sh": None}
ARCHS = sorted({os.path.basename(f).split("_")[1] for f in glob.glob(f"{D}/pft_*_base-seen_session-s*.npz")}); assert len(ARCHS) == 5, ARCHS
for arch in ARCHS:
    for v in ("base", "pfh60", "pf60"):
        zs = [np.load(f) for f in sorted(glob.glob(f"{D}/pft_{arch}_{v}-seen_session-s*.npz"))]; assert len(zs) == 3 and all((z["index"] == IDX["seen_session"]).all() for z in zs)
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


def run(bn, v):
    fams, extra = swap(v); mixes, sh, a = {}, {}, 0.0; REC.clear()
    for p in A.PROTOS:
        mixes[p], a = A.mix(bn, p, fams, extra); _, sh[p] = A.shadow(mixes[p], p)
    out0 = ns["route4"](mixes, sh, a, sh["all_loso"], {p: A.members(bn, p)["c1"] for p in A.PROTOS}, "max", 10)[0]
    return out0, staged(out0)[0]


R = {}
for bn in ("B2_pool", "B2_s345", "B2_s012", "B2_s0", "B2_s1", "B2_s2"):
    t0, ts = run(bn, "base"); ref = SC.person(ts["seen_session"], "seen_session"); line = []
    for v in ("pfh60", "pf60"):
        _, ps = run(bn, v); d = SC.person(ps["seen_session"], "seen_session") - ref; dn = SC.person(ps["all_loso"], "all_loso").mean() - SC.person(ts["all_loso"], "all_loso").mean(); R[bn, v] = d
        line.append(f"{v} + stage: seen {d.mean():+.2f} (P04 {d[0]:+.2f}, P10 {d[1]:+.2f}) new {dn:+.2f}")
    st = SC.person(ts["seen_session"], "seen_session") - SC.person(t0["seen_session"], "seen_session")
    print(f"{bn}: twins + stage seen {SC.person(ts['seen_session'], 'seen_session').mean():.2f} (stage on twins {st.mean():+.2f}) | " + " | ".join(line), flush=True)
for v in ("pfh60", "pf60"):
    D_ = np.array([R[b, v] for b in {k[0] for k in R}]); ok = sum((d.mean() >= 0.3) and (d[1] >= -0.5) for d in D_)
    print(f"MEAN {v}: seen {D_.mean():+.2f} (P04 {D_[:, 0].mean():+.2f}, P10 {D_[:, 1].mean():+.2f}); bases meeting the lever {ok}/6")
