"""SELECTOR (ledger s193): a per-session choice between v5.1b's tail (P0r), the fallback (V6r) and the gC5 combination (V6Fr)
for new people, from the per-window outputs that shufcombo SAVE_OUT wrote (runs/chainout/<order>.npz, most complete form).
1. Oracle ceiling: per base and session, the arm with the best session balanced accuracy; against always-gC5.
2. Label-free causal criteria, read on the first N windows the session delivered (gC5 is emitted for them), then the
   chosen arm for the rest: agreement between P0r and V6Fr, mean margin difference V6Fr - P0r. The threshold is
   chosen on five people and scored on the other five (all 126 splits averaged), never on the scored people.
3. Within 2: which people and which part (eye + rest = V6r - P0r, FT = V6Fr - V6r) carry the loss.
Usage: python studies/13nov/selector.py"""
import glob, itertools, os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../.."); sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
A = env.A; SC = A.SC; Z = A.L.Z; p = "all_loso"; y = Z[f"y|{p}"].astype(int); sess = Z[f"sess|{p}"]; per = Z[f"person|{p}"]
PEOPLE = list(SC.PEOPLE); PERMS = np.load(REPO + "/runs/shuffle_perms.npz"); ARMS = ("P0r", "V6r", "V6Fr")
FILES = sorted(glob.glob(f"{REPO}/runs/chainout/*.npz")); assert FILES, "run shufcombo with SAVE_OUT first"


def sba(P, r):  # balanced accuracy of rows r
    return np.mean([np.mean(P[r][y[r] == c].argmax(1) == c) for c in np.unique(y[r])])


def delivery_rank(order):
    sc, sd = order.rstrip("0123456789"), int(order[len(order.rstrip("0123456789")):]); g = PERMS[f"{sc}|{sd}|{p}"]
    rk = np.empty(len(y), int); rk[g] = np.arange(len(g)); return rk


R = {}
for f in FILES:
    o = os.path.basename(f)[:-4]; z = np.load(f); bases = sorted({k.split("|")[0] for k in z.files}); rk = delivery_rank(o)
    for bn in bases:
        P = {a: z[f"{bn}|{a}"] for a in ARMS}; sessions = list(dict.fromkeys(sess))
        orc = P["V6Fr"].copy()
        for s in sessions:
            r = np.flatnonzero(sess == s); best = max(ARMS, key=lambda a: sba(P[a], r)); orc[r] = P[best][r]
        crit = {}
        for s in sessions:
            r = np.flatnonzero(sess == s); r = r[np.argsort(rk[r])]
            for N in (30, 60):
                h = r[:N]; agree = np.mean(P["P0r"][h].argmax(1) == P["V6Fr"][h].argmax(1))
                dm = np.mean(np.sort(P["V6Fr"][h], 1)[:, -1] - np.sort(P["P0r"][h], 1)[:, -1]); crit[s, N] = (agree, dm, r[N:])
        R[o, bn] = (P, orc, crit, sessions)
print(f"SELECTOR: {len({o for o, _ in R})} orders, {len({b for _, b in R})} bases; arms {ARMS}", flush=True)

# 1. oracle ceiling, per scheme
for sch in ("within", "full", "runs"):
    ks = [k for k in R if k[0].startswith(sch)]
    if not ks:
        continue
    d_or = np.mean([SC.person(R[k][1], p) - SC.person(R[k][0]["V6Fr"], p) for k in ks], 0)
    d_fb = np.mean([SC.person(R[k][0]["V6r"], p) - SC.person(R[k][0]["V6Fr"], p) for k in ks], 0)
    print(f"{sch:6s}: oracle per session over always-gC5 {d_or.mean():+.2f} (without P10 {np.delete(d_or, 9).mean():+.2f}); always-fallback {d_fb.mean():+.2f}", flush=True)
ceil = np.mean([np.mean(SC.person(R[k][1], p) - SC.person(R[k][0]["V6Fr"], p)) for k in R if k[0].startswith("within")])
print(f"CEILING (within) {ceil:+.2f}: {'below +0.3, stop (pre-registered)' if ceil < 0.3 else 'large enough: label-free criteria follow'}", flush=True)

# 2. label-free criteria, threshold on five people, scored on the other five
if ceil >= 0.3:
    for N, ci, name in ((30, 0, "agreement"), (60, 0, "agreement"), (30, 1, "margin diff"), (60, 1, "margin diff")):
        def apply(k, thr, alt):
            P, _, crit, sessions = R[k]; out = P["V6Fr"].copy()
            for s in sessions:
                c = crit[s, N]; flip = (c[ci] < thr) if ci == 0 else (c[ci] > thr)
                if flip:
                    out[c[2]] = P[alt][c[2]]
            return SC.person(out, p) - SC.person(P["V6Fr"], p)
        ks = [k for k in R if k[0].startswith("within")]; gains = []
        q = np.linspace(0.05, 0.5, 10) if ci == 0 else np.linspace(0.5, 0.95, 10)  # flip the least agreeing / most over-confident sessions
        grid = [(thr, alt) for alt in ("P0r", "V6r") for thr in np.quantile([R[k][2][s, N][ci] for k in ks for s in R[k][3]], q)]
        G = [np.mean([apply(k, *g) for k in ks], 0) for g in grid]  # per-person gain of each rule (labels never enter the rule)
        for half in itertools.combinations(range(10), 5):
            tr, te = list(half), [i for i in range(10) if i not in half]
            best = max(range(len(grid)), key=lambda i: G[i][tr].mean()); gains.append(G[best][te].mean())
        print(f"  criterion {name} on the first {N} windows: held-out gain over always-gC5 {np.mean(gains):+.2f} [{np.percentile(gains, 10):+.2f}, {np.percentile(gains, 90):+.2f}] over 126 splits", flush=True)

# 3. within 2: who and what
for o in ("within2",):
    ks = [k for k in R if k[0] == o]
    if ks:
        v = {a: np.mean([SC.person(R[k][0][a], p) for k in ks], 0) for a in ARMS}
        print(f"{o} per person (P01..P10): eye + rest (V6r - P0r) " + " ".join(f"{x:+.1f}" for x in v["V6r"] - v["P0r"]) + " | FT (V6Fr - V6r) " + " ".join(f"{x:+.1f}" for x in v["V6Fr"] - v["V6r"]), flush=True)
