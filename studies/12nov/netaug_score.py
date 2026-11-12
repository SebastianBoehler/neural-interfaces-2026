"""Score NETAUG / cmix / swa (ledger s51, s56, s57a): standalone and in the v3 chain, each arm against the in-harness twin.
Usage: python studies/12nov/netaug_score.py PROBS_DIR"""
import glob, os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
A = env.A; SC = A.SC; D = sys.argv[1]
IDX = {p: A.pl.B[f"idx|{p}"].astype(int) for p in A.PROTOS}; SLOT = {"fb": 6, "fbraw": 5, "sinc": 7, "fbl": 8}


def member(arch, arm):
    out, seeds = {}, None
    for p in A.PROTOS:
        fs = sorted(glob.glob(f"{D}/na_{arch}_{arm}-{p}-s*.npz")); tw = {os.path.basename(f).split("-s")[-1] for f in glob.glob(f"{D}/na_{arch}_twin-{p}-s*.npz")}
        fs = [f for f in fs if os.path.basename(f).split("-s")[-1] in tw]  # only seeds the twin also has
        if not fs:
            return None, 0
        zs = [np.load(f) for f in fs]; assert all((z["index"] == IDX[p]).all() for z in zs); out[p] = np.mean([z["proba"] for z in zs], 0)
        seeds = len(fs) if seeds is None else min(seeds, len(fs))
    return out, seeds


def kw(names):
    fams, extra = list(A.FAMS), []
    for arch, name in names.items():
        slot = SLOT[arch]; fam = next(f for f, items in A.FAM.items() if any(k == slot for k, _ in items))
        if fam in fams:
            fams.remove(fam); extra += [(k, w) for k, w in A.FAM[fam]]
        extra = [(k, w) for k, w in extra if k != slot] + [(name, 1.0)]
    return fams, extra


def chain(names, bn):
    fams, extra = kw(names); return A.system(bn, fams, extra=extra)["out"]


arms = sorted({os.path.basename(f).split("_")[2].split("-")[0] for f in glob.glob(f"{D}/na_*_*-all_loso-s*.npz")} - {"twin"}); M = {}
print("arch arm seeds | standalone new / seen vs twin | chain dJ vs twin swapped (B2_pool, B2_s345)")
for arch in SLOT:
    tw, n = member(arch, "twin")
    if tw is None:
        print(arch, "no twin yet"); continue
    A.EXTRA[f"{arch}_twin"] = tw; M[(arch, "twin")] = tw; ref = {bn: chain({arch: f"{arch}_twin"}, bn) for bn in env.EXPL}
    sn, ss = SC.person(tw["all_loso"], "all_loso").mean(), SC.person(tw["seen_session"], "seen_session").mean(); print(f"{arch:5s} twin   {n} | {sn:.2f} / {ss:.2f}")
    for arm in arms:
        P, n = member(arch, arm)
        if P is None:
            continue
        A.EXTRA[f"{arch}_{arm}"] = P; M[(arch, arm)] = P
        d = [SC.compare(chain({arch: f"{arch}_{arm}"}, bn), ref[bn])["J"] for bn in env.EXPL]
        print(f"{arch:5s} {arm:6s} {n} | {SC.person(P['all_loso'], 'all_loso').mean() - sn:+.2f} / {SC.person(P['seen_session'], 'seen_session').mean() - ss:+.2f} | {d[0]:+.2f} {d[1]:+.2f}", flush=True)
full = [a for a in SLOT if (a, "twin") in M]; ref = {bn: chain({a: f"{a}_twin" for a in full}, bn) for bn in env.EXPL}
print(f"\nall families with a twin ({', '.join(full)}) swapped together, against all twins:")
for arm in arms:
    if all((a, arm) in M for a in full):
        r = [SC.compare(chain({a: f"{a}_{arm}" for a in full}, bn), ref[bn]) for bn in env.EXPL]
        print(f"   {arm:6s} dJ {r[0]['J']:+.2f} / {r[1]['J']:+.2f} | new {r[0]['new']:+.2f} / {r[1]['new']:+.2f} | seen {r[0]['seen']:+.2f} / {r[1]['seen']:+.2f} | new>=0 {r[0]['npos']}/10", flush=True)
