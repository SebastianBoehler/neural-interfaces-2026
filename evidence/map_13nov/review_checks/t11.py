import glob, os, sys, warnings, time
import numpy as np
warnings.filterwarnings("ignore")
REPO = os.getcwd(); F = REPO + "/studies/12nov/v5_replay.py"
g = {"__file__": F, "__name__": "v5head"}; exec(compile(open(F).read().split("\nres = []")[0], F, "exec"), g)
A, SC, ns, REC, NM, staged = g["A"], g["SC"], g["ns"], g["REC"], g["NM"], g["staged"]; g["BETA"] = 0.15
D, D2 = REPO + "/runs/ttt/probs_pf", REPO + "/runs/ttt/probs2"
IDX = {p: A.pl.B[f"idx|{p}"].astype(int) for p in A.PROTOS}; SLOT = {"fb": 6, "fbraw": 5, "sinc": 7, "fbl": 8, "sh": None}
ARCHS = sorted({os.path.basename(f).split("_")[1] for f in glob.glob(f"{D}/pft_*_base-seen_session-s*.npz")}); assert len(ARCHS) == 5, ARCHS
for arch in ARCHS:
    for v in ("base", "pfh60"):
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
def mixsh(bn, v):
    fams, extra = swap(v); mixes, sh, a = {}, {}, 0.0
    for p in A.PROTOS:
        mixes[p], a = A.mix(bn, p, fams, extra); _, sh[p] = A.shadow(mixes[p], p)
    return mixes, sh, a
def routed(bn, mixes, sh, a, oof):
    REC.clear(); out0 = ns["route4"](mixes, sh, a, oof, {p: A.members(bn, p)["c1"] for p in A.PROTOS}, "max", 10)[0]; return staged(out0)[0]
pos = {int(gi): k for k, gi in enumerate(IDX["seen_session"])}; rows_all = []
for bn in ("B2_pool", "B2_s345", "B2_s012", "B2_s0", "B2_s1", "B2_s2"):
    t0 = time.time(); mt, st, a = mixsh(bn, "base"); mp, sp, a2 = mixsh(bn, "pfh60"); assert abs(a - a2) < 1e-12
    tw = routed(bn, mt, st, a, st["all_loso"]); first = {s: int(np.flatnonzero(r > 0)[0]) for (proto, s), (r, j) in REC.items() if proto == "seen_session"}
    coded = routed(bn, mp, sp, a, sp["all_loso"])  # pnet.py: personal nets on whole seen sessions, in both tails
    mh = {p: mt[p].copy() for p in A.PROTOS}  # causal form: personal nets from the first routed window, in the routed tail only; gate from the twins' shadow
    for s, t1 in first.items():
        r = np.array([pos[int(x)] for x in NM[("seen_session", s)][1]]); mh["seen_session"][r[t1:]] = mp["seen_session"][r[t1:]]
    causal = routed(bn, mh, st, a, st["all_loso"])
    ref = SC.person(tw["seen_session"], "seen_session"); dc = SC.person(coded["seen_session"], "seen_session") - ref; dh = SC.person(causal["seen_session"], "seen_session") - ref
    rows_all.append((dc, dh)); print(f"{bn} ({time.time() - t0:.0f}s) first routed windows {dict((k.split('/')[-1], v) for k, v in first.items())} | personal nets + stage 0.15 against twins + stage 0.15, seen: as coded {dc.mean():+.2f} (P04 {dc[0]:+.2f}, P10 {dc[1]:+.2f}) | from the first routed window, routed tail only {dh.mean():+.2f} (P04 {dh[0]:+.2f}, P10 {dh[1]:+.2f})", flush=True)
m = np.mean(rows_all, 0); print(f"MEAN as coded {m[0].mean():+.2f} (P04 {m[0][0]:+.2f}, P10 {m[0][1]:+.2f}) | causal form {m[1].mean():+.2f} (P04 {m[1][0]:+.2f}, P10 {m[1][1]:+.2f})")
