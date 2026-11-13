import os, sys, warnings, time
import numpy as np
warnings.filterwarnings("ignore")
REPO = os.getcwd(); R4 = REPO + "/studies/11nov/rt4/route4.py"; ns = {"__file__": R4, "__name__": "route4"}
sys.argv = ["route4.py", REPO + "/runs/replay_data/bio_model_psd_d64.npz"]
exec(compile(open(R4).read().split("\nbases = sys.argv")[0], R4, "exec"), ns)
sys.path.insert(0, REPO + "/studies/13nov")
import stacklib as S
A, SC, Z, PR = S.A, S.SC, S.Z, S.PR
assert A is ns["A"]
def arm(levers, bn, mode, oof=None):
    fams, extra = list(A.FAMS), []
    for a, slot in S.SLOT.items():
        fam = next(f for f, items in A.FAM.items() if any(k == slot for k, _ in items))
        if fam in fams:
            fams.remove(fam); extra += list(A.FAM[fam])
        extra = [(k, w) for k, w in extra if k != slot] + [(f"{a}_{'ft' if 'FT' in levers else 'fr'}", 1.0)]
    if "REST" in levers:
        extra.append(("rest", 1.0))
    for p in PR:
        Z[f"GE|{p}"] = S.GEO[p] if "OCU" in levers else S.REF[p]
    try:
        mixes, sh, a = {}, {}, 0.0
        for p in PR:
            mixes[p], a = A.mix(bn, p, fams, extra); _, sh[p] = A.shadow(mixes[p], p)
        o = sh["all_loso"] if oof is None else oof; c1 = {p: A.members(bn, p)["c1"] for p in PR}
        out = A.route(mixes, sh, a, o, c1)[0] if mode == "v3" else ns["route4"](mixes, sh, a, o, c1, "max", 10)[0]
    finally:
        for p in PR:
            Z[f"GE|{p}"] = S.REF[p]
    return SC.person(out["all_loso"], "all_loso"), SC.person(out["seen_session"], "seen_session"), sh["all_loso"]
ARMS = {"FT": ("FT",), "OCU": ("OCU",), "FT+OCU": ("FT", "OCU"), "FT+REST+OCU": ("FT", "REST", "OCU")}
R = {}
for bn in S.BASES:
    t0 = time.time(); b3 = arm((), bn, "v3"); b4 = arm((), bn, "v4"); oof0 = b3[2]
    for k, L in ARMS.items():
        x3 = arm(L, bn, "v3"); x3f = arm(L, bn, "v3", oof0); x4f = arm(L, bn, "v4", oof0)
        R[bn, k] = [(x3[0] - b3[0]).mean(), (x3[1] - b3[1]).mean(), (x3f[1] - b3[1]).mean(), (x4f[0] - b4[0]).mean(), (x4f[1] - b4[1]).mean(), (x4f[1] - b4[1])[0], (x4f[1] - b4[1])[1]]
    print(bn, f"({time.time() - t0:.0f}s)", " | ".join(f"{k}: v3 new {r[0]:+.2f} seen {r[1]:+.2f}; v3 fixed gate seen {r[2]:+.2f}; v4.1 fixed gate new {r[3]:+.2f} seen {r[4]:+.2f} (P04 {r[5]:+.2f} P10 {r[6]:+.2f})" for k, r in ((k, R[bn, k]) for k in ARMS)), flush=True)
print("MEAN over bases")
for k in ARMS:
    m = np.mean([R[bn, k] for bn in S.BASES], 0)
    print(f"  {k:12s}: v3 as coded new {m[0]:+.2f} seen {m[1]:+.2f} | v3, gate fixed to the base's outputs: seen {m[2]:+.2f} | v4.1, gate fixed: new {m[3]:+.2f} seen {m[4]:+.2f} (P04 {m[5]:+.2f}, P10 {m[6]:+.2f})")
