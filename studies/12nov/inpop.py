"""INPOP (ledger s82): own-space stage on a within-session proxy where the person's runs 1-3 are in the population.
Usage: python studies/12nov/inpop.py"""
import glob, os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/replay", REPO + "/studies/12nov"]
import env  # noqa: E402
import grid as G  # noqa: E402
import ownspace as O  # noqa: E402
A = env.A; Z = A.L.Z; p = "all_loso"; RUN = np.load(REPO + "/data/proteus_windows.npz", allow_pickle=True)["run"].astype(int)
assert (Z[f"idx|{p}"] == np.arange(len(O.y))).all()
D = REPO + "/runs/ttt/probs_pfx"; SLOTS = {"fbraw": 5, "fb": 6, "sinc": 7, "fbl": 8}; rng = np.random.default_rng(0)
PF = {a: np.mean([np.load(f)["proba"] for f in sorted(glob.glob(f"{D}/pfx_{a}_base-proxy-s*.npz"))], 0) for a in list(SLOTS) + ["sh"]}
IDXP = np.load(f"{D}/pfx_fb_base-proxy-s0.npz")["index"]; assert all((np.load(f)["index"] == IDXP).all() for f in glob.glob(f"{D}/pfx_*_base-proxy-s*.npz"))
B = np.flatnonzero(np.isin(np.arange(len(O.y)), IDXP)); pos = {int(g): k for k, g in enumerate(IDXP)}; RB = np.array([pos[int(g)] for g in B])
PEOPLE = sorted(set(O.PER)); sess = np.array([f"{O.PER[i]}|late" for i in B])


def chain(bn, inpop):
    Ps, ws = Z[f"{bn}|Ps|{p}"][:, B].copy(), Z[f"{bn}|ws|{p}"].copy()
    if inpop:
        for a, k in SLOTS.items():
            Ps[k] = PF[a][RB]  # Ps[k] is member k (checked: Ps[0] = mean of core1 and the Hilbert core; 10 = teacher)
        Ps[3], Ps[4] = PF["sh"][RB], PF["sh"][RB]  # members 3, 4: ShallowConvNet; 1, 2 (TF FBCNet) and 9 stay LOSO
    mix = sum(w / ws.sum() * P for w, P in zip(ws, Ps)); b = G.rebal(mix, sess, 1.15)
    e = G.SE.apply(Z[f"GE|{p}"][B], b, sess, lam=0.5, n0=60, m0=20, shrink=0.7); L, tt = G.tsem_lik(e, sess, Z[f"pc128|{p}"][B]); return G.apply_lam(e, L, tt, 0.15)


MEM = {}
for q in PEOPLE:
    a = np.flatnonzero((O.PER == q) & (O.SES == "1") & (RUN <= 3)); r = B[O.PER[B] == q]; MEM[q] = (r, O.Personal([a]).stream(r))


def ba_people(P):
    return np.array([O.ba(MEM[q][0], P[O.PER[B] == q]) for q in PEOPLE])


for bn in ("B2_pool", "B2_s345", "B2_s012"):
    res = {}
    for tag, inpop in (("LOSO nets", False), ("in-population nets", True)):
        P = chain(bn, inpop); ref = ba_people(P); res[tag] = ref; line = []
        for beta in (0.10, 0.25):
            S = P.copy()
            for q in PEOPLE:
                m = MEM[q][1]; k = O.PER[B] == q; L = np.log(np.clip(P[k], 1e-9, 1)) + beta * sum(np.log(np.clip(m[x], 1e-3, 1)) for x in "TCS") / 3
                e = np.exp(L - L.max(1, keepdims=True)); S[k] = e / e.sum(1, keepdims=True)
            d = ba_people(S) - ref; bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])
            line.append(f"stage {beta}: {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] >=+1: {(d >= 1).sum()}/10 | " + " ".join(f"{x:+.1f}" for x in d))
        print(f"{bn} {tag}: chain {ref.mean():.1f} | " + " ".join(f"{x:.0f}" for x in ref), flush=True)
        for l_ in line:
            print("    " + l_, flush=True)
    print(f"  in-population nets against LOSO nets: {(res['in-population nets'] - res['LOSO nets']).mean():+.2f}", flush=True)
