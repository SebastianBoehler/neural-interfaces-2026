"""v5.3 (ledger s106): what runtime-feasible subsets of the s101 fine-tuning are worth in the chain, new people, six bases.
Stored per-seed outputs of ttt5.py (runs/ttt/probs5; six seeds per family; frozen 'base' and fine-tuned variants).
Forms per family (k = the fine-tuned seed(s); the others stay frozen):
  all6   mean of the six fine-tuned seeds (s101's arm)
  delta1 frozen six-seed mean + (fine-tuned_k - frozen_k), one seed k: one fine-tuning per family, its change
         added to every member of the family
  delta2 the same with the mean change of two seeds
  one6   (fine-tuned_k + the five other frozen seeds) / 6
Usage: [PFX=ttt5 VARS=fixA,fixS,fixLR FORMS=all6,delta1,delta2,one6] python studies/13nov/ttt_reduce_score.py PROBS_DIR"""
import glob, os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay"); sys.path.insert(0, REPO + "/studies/12nov")
import env  # noqa: E402
A = env.A; SC = A.SC; D = sys.argv[1]; rng = np.random.default_rng(0)
IDX = {p: A.pl.B[f"idx|{p}"].astype(int) for p in A.PROTOS}; SLOT = {"fb": 6, "fbraw": 5, "sinc": 7, "fbl": 8}
PFX = os.environ.get("PFX", "ttt5"); VARS = os.environ.get("VARS", "fixA,fixS,fixLR").split(",")
FORMS = os.environ.get("FORMS", "all6,delta1,delta2,one6").split(",")
_a, _b = (int(x) for x in os.environ.get("SEEDS", "0-5").split("-")); SEEDS = list(range(_a, _b + 1))  # six seeds per run, e.g. SEEDS=6-11
assert len(SEEDS) == 6, f"SEEDS must name six seeds (the forms assume six), got {SEEDS}"
print(f"scoring {D} prefix {PFX} seeds {SEEDS} in A.system (routing v3 dev chain, not the shipped v4.1/v5 chain)", flush=True)
ARCHS = sorted({os.path.basename(f).split("_")[1] for f in glob.glob(f"{D}/{PFX}_*_base-all_loso-s*.npz")})
assert sorted(ARCHS) == sorted(SLOT), f"families found in {D}: {ARCHS} (need all four; nothing scored)"


LABELS = set()


def load(arch, v):
    out = {}
    for p in A.PROTOS:
        fs = [f"{D}/{PFX}_{arch}_{v}-{p}-s{s}.npz" for s in SEEDS]; zs = [np.load(f) for f in fs if os.path.exists(f)]
        assert len(zs) == 6 and all((z["index"] == IDX[p]).all() for z in zs), (arch, v, p, len(zs))
        tags = {str(z["labels"]) if "labels" in z.files else "untagged (before 13 Nov 00:50 UTC)" for z in zs}
        assert len({t for t in tags if not t.startswith("untagged")}) <= 1, f"{arch} {v} {p}: mixed label sources {tags}"  # untagged = written before the hash existed
        assert "true" in PFX or not any("chain_true" in t for t in tags), f"{arch} {v} {p}: oracle labels under a packageable name {tags}"
        LABELS.update(tags)
        out[p] = np.stack([z["proba"] for z in zs])
    return out  # {proto: (6, n, 3)}


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


def form(base, ft, f, ks):
    """family posterior (n, 3) per proto for form f with fine-tuned seeds ks"""
    if f == "all6":
        return ft.mean(0)
    if f.startswith("delta"):
        P = base.mean(0) + np.mean([ft[k] - base[k] for k in ks], 0)
        return np.clip(P, 1e-6, None) / np.clip(P, 1e-6, None).sum(-1, keepdims=True)
    if f == "one6":
        return (ft[ks[0]] + sum(base[j] for j in range(6) if j != ks[0])) / 6
    raise ValueError(f)


CHOICES = {"all6": [None], "delta1": [(k,) for k in range(6)], "delta2": [(0, 1), (2, 3), (4, 5)], "one6": [(k,) for k in range(6)]}
P = {(a, v): load(a, v) for a in ARCHS for v in ["base"] + VARS}
print("label sources of the scored files:", sorted(LABELS), flush=True)
for a in ARCHS:
    A.EXTRA[f"{a}_base"] = {p: P[(a, "base")][p].mean(0) for p in A.PROTOS}
print("chain, four families, new people: mean over the choices of the fine-tuned seed(s) [range over choices]; 80 % person "
      "bootstrap of the first choice; people >= 0; seen (gated off in a package)", flush=True)
summary = {}
for v in VARS:
    for f in FORMS:
        rows, per = [], []
        for bn in env.BASES:
            ref = chain("base", bn); res = []
            for ks in CHOICES[f]:
                for a in ARCHS:
                    A.EXTRA[f"{a}_x"] = {p: form(P[(a, "base")][p], P[(a, v)][p], f, ks) for p in A.PROTOS}
                res.append(SC.compare(chain("x", bn), ref))
            dn = res[0]["dn"]; b = [dn[rng.integers(0, 10, 10)].mean() for _ in range(2000)]; lo, hi = np.percentile(b, [10, 90])
            new = [r["new"] for r in res]; rows.append(np.mean(new)); per.append(np.mean([r["dn"] for r in res], 0))
            print(f"   {v:6s} {f:6s} {bn:8s} new {np.mean(new):+.2f} [{min(new):+.2f}, {max(new):+.2f}] boot [{lo:+.2f}, {hi:+.2f}] "
                  f"{res[0]['npos']}/10 | seen {np.mean([r['seen'] for r in res]):+.2f}", flush=True)
        summary[v, f] = np.mean(rows)
        d = np.mean(per, 0); bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [2.5, 10, 90, 97.5]); k = int(np.argmax(d))
        print(f"MEAN {v} {f}: new {np.mean(rows):+.2f} (bases {min(rows):+.2f} to {max(rows):+.2f}; bases are nearly one system, s139) | person "
              f"80% [{bt[1]:+.2f}, {bt[2]:+.2f}] 95% [{bt[0]:+.2f}, {bt[3]:+.2f}] people>=0 {(d >= 0).sum()}/10 | without best (P{k + 1:02d} {d[k]:+.1f}) {np.delete(d, k).mean():+.2f}", flush=True)
print("SUMMARY " + " | ".join(f"{v} {f} {s:+.2f}" for (v, f), s in summary.items()))
