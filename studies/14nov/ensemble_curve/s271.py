"""s271 step 1 scoring (ledger s271 + amendment): marginals of new families over the shipped base, from stored probs18 files.
Base sets: {fb,fbl,sinc}, {fb,fbl,sinc,fbraw}, and the same with sinc -> sincs; old families at 3 seeds (s0-2) and at 6 seeds; new family always 3 seeds (s0-2), added.
Also equal-n swap (base 4 seeds each, 12 nets vs base 3 seeds + new 3 seeds), each new family alone, sincs minus sinc, corr of posteriors with fb.
Person-equal BA of the mean posterior; 80 % interval resamples people and (order, draw) cells (1000 resamples).
Usage: python studies/14nov/ensemble_curve/s271.py"""
import itertools, os, sys
import numpy as np
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import curve as C  # noqa: E402

NEW = ("conf", "eegn", "atc", "deep4", "sincs"); OLD = ("fb", "fbl", "fbraw", "sinc"); cells = list(itertools.product(C.ORD, C.DRAWS)); nc = len(cells)
N = {}
for o, d in cells:
    for f in OLD + NEW:
        for s in range(6 if f in OLD else 3):
            fn = f"{C.D}/ttt18{o}{d}_{f}_base-all_loso-s{s}.npz"; assert os.path.exists(fn), fn
            z = np.load(fn); assert z["proba"].shape == (len(C.y), 3), fn
            N[o, d, f, s] = z["proba"].astype(np.float64)
ref = np.load(f"{C.D}/ttt18within0d1_fb_base-all_loso-s0.npz")["index"]
for f in NEW: assert (np.load(f"{C.D}/ttt18within0d1_{f}_base-all_loso-s0.npz")["index"] == ref).all(), f  # same rows as shipped files
for k in list(N)[::7]:  # guard: 1-net subset equals own file BA
    assert np.allclose(C.subset_ba({kk: v for kk, v in N.items()}, k[0], k[1], [(k[2], k[3])]), C.person_ba(N[k]))
ba = lambda nets: np.stack([C.subset_ba(N, o, d, nets) for o, d in cells])  # (cells, people)
br = np.random.default_rng(1); idx = [(br.integers(0, len(C.PEOPLE), len(C.PEOPLE)), br.integers(0, nc, nc)) for _ in range(1000)]
nets = lambda fams, S: [(f, s) for f in fams for s in range(S)]


def rep(name, v):
    b = [v[np.ix_(oo, pp)].mean() for pp, oo in idx]; lo, hi = np.percentile(b, [10, 90])
    return f"  {name:58s} {v.mean():+6.2f}  [{lo:+.2f}, {hi:+.2f}]  people up {int((v.mean(0) > 0).sum())}/10  cells up {int((v.mean(1) > 0).sum())}/{nc}"


L = [f"s271 step 1 scoring; cells {cells}; {len(C.PEOPLE)} people; BA in %, person-equal, mean posterior; marginal = BA(base + new family 3 seeds, added) - BA(base)",
     "new families: " + ", ".join(NEW) + " (from scratch, SessionBN, 3 seeds); old families 3 seeds = s0-2, 6 seeds = s0-5", ""]
alone = {f: ba(nets([f], 3)) for f in NEW}
L.append("each new family alone (3 nets) vs references (3 nets, s0-2):")
for f in NEW + ("fb", "fbl", "fbraw", "sinc"):
    L.append(f"  {f:6s} alone {ba(nets([f], 3)).mean():.2f}")
L.append("")
for bname, fams in (("{fb,fbl,sinc}", ("fb", "fbl", "sinc")), ("{fb,fbl,sinc,fbraw}", ("fb", "fbl", "sinc", "fbraw")),
                    ("{fb,fbl,sincs}", ("fb", "fbl", "sincs")), ("{fb,fbl,sincs,fbraw}", ("fb", "fbl", "sincs", "fbraw"))):
    for S in (3, 6):
        if "sincs" in fams and S == 6: continue  # sincs has 3 seeds only
        base = ba([(f, s) for f in fams for s in range(3 if f == "sincs" else S)])
        L.append(f"base {bname} old seeds={S}: BA {base.mean():.2f} ({len(fams)} families)")
        for f in NEW:
            if f == "sincs" and "sincs" in fams: continue
            if f == "sincs" and "sinc" in fams: continue  # sincs vs sinc is reported separately
            v = ba([(g, s) for g in fams for s in range(3 if g == "sincs" else S)] + nets([f], 3)) - base
            L.append(rep(f"+{f}", v))
L.append("")
L.append("sincs minus sinc (value of the m7r trunk; 3 seeds each, then within the base):")
L.append(rep("sinc(3 nets) - sincs(3 nets), alone", ba(nets(["sinc"], 3)) - ba(nets(["sincs"], 3))))
for S in (3, 6):
    for bname, rest in (("{fb,fbl}+x", ("fb", "fbl")), ("{fb,fbl,fbraw}+x", ("fb", "fbl", "fbraw"))):
        r = [(f, s) for f in rest for s in range(S)]
        L.append(rep(f"{bname} (old seeds={S}): sinc(3) - sincs(3)", ba(r + nets(["sinc"], 3)) - ba(r + nets(["sincs"], 3))))
L.append(""); L.append("equal-n swap: base {fb,fbl,sinc} at 4 seeds each (12 nets) vs base at 3 seeds + new family (12 nets); value = new - swap:")
b4 = ba(nets(("fb", "fbl", "sinc"), 4)); L.append(f"  base 12 nets (4 seeds each) BA {b4.mean():.2f}")
for f in NEW: L.append(rep(f"{f}: (base3 + {f}) - base4", ba(nets(("fb", "fbl", "sinc"), 3) + nets([f], 3)) - b4))
L.append(""); L.append("mean Pearson correlation of posteriors (flattened, all rows x 3 classes, mean over cells, seed-matched s0-2) with fb:")
cor = lambda f, g, sh: np.mean([np.corrcoef(N[o, d, f, s].ravel(), N[o, d, g, (s + sh) % 3].ravel())[0, 1] for o, d in cells for s in range(3)])
L.append(f"  fb with fb (other seed, reference) {cor('fb', 'fb', 1):.3f}")
for f in NEW + ("fbl", "fbraw", "sinc"): L.append(f"  {f:6s} with fb {cor(f, 'fb', 0):.3f}")
txt = "\n".join(L); print(txt)
os.makedirs(C.REPO + "/evidence/ensemble_curve_15nov", exist_ok=True); open(C.REPO + "/evidence/ensemble_curve_15nov/s271.txt", "w").write(txt + "\n")
