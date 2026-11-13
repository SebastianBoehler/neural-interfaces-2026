import sys, glob, os, time
import numpy as np
REPO = os.getcwd()
sys.path.insert(0, REPO + "/replay")
import env
import grid as G
A = env.A; SC = A.SC; Z = A.L.Z; PR = A.PROTOS
IDX = {p: A.pl.B[f"idx|{p}"].astype(int) for p in PR}; SLOT = {"fbraw": 5, "fb": 6, "sinc": 7, "fbl": 8}
def load(pat, p, nmin):
    zs = [np.load(f) for f in sorted(glob.glob(pat.replace("PROTO", p)))]
    assert len(zs) >= nmin and all((z["index"] == IDX[p]).all() for z in zs), (pat, p, len(zs))
    return np.mean([z["proba"] for z in zs], 0)
D6 = REPO + "/runs/ttt/probs6"
for a in SLOT:
    fr = {p: load(f"{D6}/ttt6_{a}_base-PROTO-s*.npz", p, 6) for p in PR}
    cu = load(f"{D6}/ttt6_{a}_cum10-all_loso-s*.npz", "all_loso", 6)
    A.EXTRA[f"{a}_fr"] = fr; A.EXTRA[f"{a}_ft"] = {"all_loso": cu, "seen_session": fr["seen_session"]}
def kw(tag):
    fams, extra = list(A.FAMS), []
    for a, slot in SLOT.items():
        fam = next(f for f, items in A.FAM.items() if any(k == slot for k, _ in items))
        if fam in fams:
            fams.remove(fam); extra += list(A.FAM[fam])
        extra = [(k, w) for k, w in extra if k != slot] + [(f"{a}_{tag}", 1.0)]
    return fams, extra
# state check: shipped chain before
ship0 = {bn: A.system(bn, A.FAMS)["out"] for bn in env.BASES}
R = {}
mixes = {}
for bn in env.BASES:
    for tag in ("fr", "ft"):
        fams, extra = kw(tag); r = A.system(bn, fams, extra=extra)["out"]; R[bn, tag] = r
    mixes[bn] = A.mix(bn, "all_loso", *kw("fr"))[0]
print("== swapped-family configuration (stack.py base / ttt_reduce_score base): how different are the six bases?")
print("fams/extra:", kw("fr"))
for bn in env.BASES[1:]:
    print(f"  mix max|diff| vs B2_pool: swapped {np.abs(mixes[bn]-mixes['B2_pool']).max():.4f} | shipped {np.abs(A.mix(bn,'all_loso',A.FAMS)[0]-A.mix('B2_pool','all_loso',A.FAMS)[0]).max():.4f}",
          f"| label agreement of final outputs with B2_pool: swapped {np.mean(R[bn,'fr']['all_loso'].argmax(1)==R['B2_pool','fr']['all_loso'].argmax(1)):.4f} shipped {np.mean(ship0[bn]['all_loso'].argmax(1)==ship0['B2_pool']['all_loso'].argmax(1)):.4f}")
dn = {bn: SC.person(R[bn, "ft"]["all_loso"], "all_loso") - SC.person(R[bn, "fr"]["all_loso"], "all_loso") for bn in env.BASES}
print("FT(cum10) - frozen, new, per base:", {bn: round(float(d.mean()), 3) for bn, d in dn.items()})
print("base 'new' (frozen swapped) per base:", {bn: round(float(SC.person(R[bn,'fr']['all_loso'],'all_loso').mean()), 2) for bn in env.BASES}, "| shipped:", {bn: round(float(SC.person(ship0[bn]['all_loso'],'all_loso').mean()), 2) for bn in env.BASES})
M = np.array([dn[bn] for bn in env.BASES]); print("corr of per-person FT deltas between bases:\n", np.round(np.corrcoef(M), 2))
# state check: shipped chain after the extra runs
for bn in env.BASES:
    o = A.system(bn, A.FAMS)["out"]; assert all(np.abs(o[p] - ship0[bn][p]).max() == 0 for p in PR), bn
print("state check: shipped chain identical before/after the swapped runs")
# == null arms: second TSEM in random 64-dim views of core1's 256 PCs after the chain (as viewnull.py)
p = "all_loso"; sess = Z[f"sess|{p}"]; N = len(sess); rng = np.random.default_rng(123); PC = np.zeros((N, 256))
for f in sorted(glob.glob(f"{REPO}/runs/replay_data/feats_exact/all_loso-P*.npz")):
    z = np.load(f, allow_pickle=True); sd = z["pc_tr"].std(0)
    for k in [x for x in z.files if x.endswith("|pc")]:
        PC[z[k.replace("|pc", "|idx")].astype(int)] = z[k] / sd
ref = {bn: G.scores(ship0[bn][p], p) for bn in env.BASES}; D = []
t0 = time.time()
for draw in range(16):
    W = np.linalg.qr(rng.standard_normal((256, 64)))[0]; row = []
    for bn in env.BASES:
        L, tt = G.tsem_lik(ship0[bn][p], sess, PC @ W); row.append(G.scores(G.apply_lam(ship0[bn][p], L, tt, 0.15), p) - ref[bn])
    D.append(row)
D = np.array(D)  # (draws, bases, people)
m = D.mean(2)  # (draws, bases)
print("null arms (random views), time", round(time.time() - t0))
print("per-draw 'new' delta per base:\n", np.round(m, 2))
ex, fr = m[:, :2].mean(1), m[:, 2:].mean(1)
print("corr(EXPL mean, FRESH mean) over draws:", round(float(np.corrcoef(ex, fr)[0, 1]), 3), "| corr(B2_s345, mean of s0,s1,s2):", round(float(np.corrcoef(m[:, 1], m[:, 3:].mean(1))[0, 1]), 3))
print("SD over draws: single base", np.round(m.std(0), 3), "| six-base mean", round(float(m.mean(1).std()), 3), "| if independent would be ~", round(float(m.std(0).mean() / np.sqrt(6)), 3))
print("draws with all six bases the same sign:", int(((m > 0).all(1) | (m < 0).all(1)).sum()), "of", len(m), "| draws with >= 5/6 same sign:", int(((m > 0).sum(1) >= 5).sum() + ((m < 0).sum(1) >= 5).sum()))
best = int(np.argmax(ex)); print("selecting the best of 16 null draws on EXPL: EXPL", round(float(ex[best]), 2), "-> FRESH", round(float(fr[best]), 2), "| mean of all draws", round(float(m.mean()), 3))
