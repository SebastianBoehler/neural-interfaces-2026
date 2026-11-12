"""Score TTT3 (ledger s60): package-form in-session fine-tuning, alone and in the v3 chain, against the same nets frozen.
Usage: [PFX=ttt4 VARS=ple20,fix20,ent20,cons20] [ALLBASES=1] python studies/12nov/ttt3_score.py PROBS_DIR"""
import glob, os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
A = env.A; SC = A.SC; D = sys.argv[1]; rng = np.random.default_rng(0)
IDX = {p: A.pl.B[f"idx|{p}"].astype(int) for p in A.PROTOS}; SLOT = {"fb": 6, "fbraw": 5, "sinc": 7, "fbl": 8, "sh": None}; PFX = os.environ.get("PFX", "ttt3"); VARS = os.environ.get("VARS", "pl20,ple20,plelr20,plec20,ore20").split(",")
ARCHS = sorted({os.path.basename(f).split("_")[1] for f in glob.glob(f"{D}/{PFX}_*_base-all_loso-s*.npz")})


def load(arch, v):
    out = {}
    for p in A.PROTOS:
        fs = sorted(glob.glob(f"{D}/{PFX}_{arch}_{v}-{p}-s*.npz")); zs = [np.load(f) for f in fs]; assert zs and all((z["index"] == IDX[p]).all() for z in zs), (arch, v, p)
        out[p] = np.stack([z["proba"] for z in zs])
    return out  # {proto: (seeds, n, 3)}


def kw(names):
    fams, extra = list(A.FAMS), []
    for arch, name in names.items():
        slot = SLOT[arch]
        if slot is None:
            extra.append((name, 1.0)); continue
        fam = next(f for f, items in A.FAM.items() if any(k == slot for k, _ in items))
        if fam in fams:
            fams.remove(fam); extra += [(k, w) for k, w in A.FAM[fam]]
        extra = [(k, w) for k, w in extra if k != slot] + [(name, 1.0)]
    return fams, extra


def chain(tag, bn):
    fams, extra = kw({a: f"{a}_{tag}" for a in ARCHS}); return A.system(bn, fams, extra=extra)["out"]


def boot(dn):
    b = [dn[rng.integers(0, len(dn), len(dn))].mean() for _ in range(2000)]; return np.percentile(b, [10, 90])


P = {(a, v): load(a, v) for a in ARCHS for v in ["base"] + VARS}; print("families", ARCHS, "seeds", {a: len(P[(a, "base")]["all_loso"]) for a in ARCHS})
print("standalone new-person BA, mean of seeds: frozen, then the gain of each variant")
for a in ARCHS:
    b = np.mean([SC.person(x, "all_loso").mean() for x in P[(a, "base")]["all_loso"]])
    print(f"   {a:5s} frozen {b:.2f} | " + " ".join(f"{v} {np.mean([SC.person(x, 'all_loso').mean() for x in P[(a, v)]['all_loso']]) - b:+.2f}" for v in VARS))
for a in ARCHS:
    for v in ["base"] + VARS:
        A.EXTRA[f"{a}_{v}"] = {p: P[(a, v)][p].mean(0) for p in A.PROTOS}
print("\nchain, all families swapped (three seeds each), against the same nets frozen: new [80 % person bootstrap], people >= 0, P04, seen")
for bn in (env.BASES if os.environ.get("ALLBASES") else env.EXPL):
    ref = chain("base", bn)
    for v in VARS:
        r = SC.compare(chain(v, bn), ref); lo, hi = boot(r["dn"]); print(f"   {bn} {v:8s}: new {r['new']:+.2f} [{lo:+.2f}, {hi:+.2f}] {r['npos']}/10 P04 {r['dn'][3]:+.1f} | seen {r['seen']:+.2f} | dJ {r['J']:+.2f}", flush=True)
print("\nruntime-feasible form: one fine-tuned seed next to two frozen seeds per family (three choices of the seed)")
for v in ("pl20", "ple20", "plelr20", "plec20"):
    for bn in (env.BASES if os.environ.get("ALLBASES") else env.EXPL):
        ref = chain("base", bn); res = []
        for k in range(3):
            for a in ARCHS:
                n = len(P[(a, "base")]["all_loso"]); A.EXTRA[f"{a}_one"] = {p: (P[(a, v)][p][k % n] + sum(P[(a, "base")][p][j] for j in range(n) if j != k % n)) / n for p in A.PROTOS}
            res.append(SC.compare(chain("one", bn), ref))
        print(f"   {bn} {v:8s}: new " + " / ".join(f"{r['new']:+.2f}" for r in res) + " | seen " + " / ".join(f"{r['seen']:+.2f}" for r in res), flush=True)
