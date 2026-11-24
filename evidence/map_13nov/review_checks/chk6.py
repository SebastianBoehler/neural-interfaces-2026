import sys, os, glob, numpy as np
sys.dont_write_bytecode = True
REPO = "<repo>"
sys.path.insert(0, REPO + "/replay")
import env
A = env.A; SC = A.SC
D = REPO + "/runs/ttt/probs6"; PFX = "ttt6"; SLOT = {"fb": 6, "fbraw": 5, "sinc": 7, "fbl": 8}; ARCHS = sorted(SLOT)
def load(a, v):
    return {p: np.stack([np.load(f)["proba"] for f in sorted(glob.glob(f"{D}/{PFX}_{a}_{v}-{p}-s*.npz"))]) for p in A.PROTOS}
def kw(names):
    fams, extra = list(A.FAMS), []
    for arch, name in names.items():
        fam = next(f for f, items in A.FAM.items() if any(k == SLOT[arch] for k, _ in items))
        if fam in fams:
            fams.remove(fam); extra += [(k, w) for k, w in A.FAM[fam]]
        extra = [(k, w) for k, w in extra if k != SLOT[arch]] + [(name, 1.0)]
    return fams, extra
def chain(tag, bn):
    fams, extra = kw({a: f"{a}_{tag}" for a in ARCHS}); return A.system(bn, fams, extra=extra)["out"]
VARS = ["fixA", "cum10", "head"]
P = {(a, v): load(a, v) for a in ARCHS for v in ["base"] + VARS}
for a in ARCHS:
    for v in ["base"] + VARS:
        A.EXTRA[f"{a}_{v}"] = {p: P[(a, v)][p].mean(0) for p in A.PROTOS}
refs, real, dn = {}, {}, {v: [] for v in VARS}
y = SC.B["y|all_loso"]
for bn in env.BASES:
    refs[bn] = chain("base", bn); real[bn] = A.system(bn, A.FAMS)["out"]
    rn, tn = SC.person(real[bn]["all_loso"], "all_loso"), SC.person(refs[bn]["all_loso"], "all_loso")
    print(f"{bn}: packaged-chain replay new {rn.mean():.2f} | TTT reference (four slots = the TTT run's frozen nets) new {tn.mean():.2f} | label acc of ref {np.mean(refs[bn]['all_loso'].argmax(1) == y):.4f}")
    for v in VARS:
        r = SC.compare(chain(v, bn), refs[bn]); dn[v].append(r["dn"])
b0 = env.BASES[0]
print("reference chains across the six bases: max |P - P(B2_pool)| and share of windows with another argmax (new people)")
for bn in env.BASES[1:]:
    print(f"   TTT reference {bn}: {np.abs(refs[bn]['all_loso'] - refs[b0]['all_loso']).max():.3f}, {np.mean(refs[bn]['all_loso'].argmax(1) != refs[b0]['all_loso'].argmax(1)):.4f}   | packaged chain: {np.abs(real[bn]['all_loso'] - real[b0]['all_loso']).max():.3f}, {np.mean(real[bn]['all_loso'].argmax(1) != real[b0]['all_loso'].argmax(1)):.4f}")
rng = np.random.default_rng(0)
for v in VARS:
    M = np.array(dn[v]); print(v, "new per base", np.round(M.mean(1), 2).tolist(), "| per person (mean of bases)", np.round(M.mean(0), 2).tolist())
    print("    correlation of the per-person deltas between bases (min off-diagonal): %.2f" % np.corrcoef(M)[np.triu_indices(6, 1)].min())
    d = M.mean(0); bt = [d[rng.integers(0, 10, 10)].mean() for _ in range(4000)]
    print("    person bootstrap of the six-base mean: 80 %% [%.2f, %.2f], 95 %% [%.2f, %.2f]; people >= 0: %d/10; without the best person %.2f" % (*np.percentile(bt, [10, 90]), *np.percentile(bt, [2.5, 97.5]), (d >= 0).sum(), np.sort(d)[:-1].mean()))
print("--- seen sessions (returning people; the plan gates fine-tuning off when recognised)")
for v in ("fixA", "cum10"):
    ds = np.array([SC.compare(chain(v, bn), refs[bn])["ds"] for bn in env.BASES]); print(v, "seen per base", np.round(ds.mean(1), 2).tolist(), "P04 %.2f P10 %.2f" % tuple(ds.mean(0)))
    ds = np.array([SC.person(A.system(bn, *kw({a: f"{a}_{v}" for a in ARCHS}), routing=False)["out"]["seen_session"], "seen_session") - SC.person(A.system(bn, *kw({a: f"{a}_base" for a in ARCHS}), routing=False)["out"]["seen_session"], "seen_session") for bn in env.BASES])
    print(v, "seen, routing off (a returning session the gate misses)", np.round(ds.mean(1), 2).tolist(), "P04 %.2f P10 %.2f" % tuple(ds.mean(0)))
