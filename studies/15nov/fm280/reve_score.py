"""s280 REVE scoring (s271.py protocol): each REVE form alone (3 seeds), marginal over the shipped {fb,fbl,sinc} (3 seeds each, new form 3 seeds added),
equal-n swap (base at 4 seeds each vs base 3 + form), posterior correlation with fb. Person-equal BA of the mean posterior; 80 % interval resamples
people and (order, draw) cells (1000). Reads runs/diag/reve280_probs18 (collect.py) and runs/diag/ttt/probs18 (shipped families).
Usage: python studies/15nov/fm280/reve_score.py"""
import itertools, os, sys
import numpy as np
sys.argv = sys.argv[:1]
REPO = os.path.abspath(__file__ + "/../../../..")
sys.path.insert(0, REPO + "/studies/14nov/ensemble_curve")
import curve as C  # noqa: E402

NEW = ("rv1", "rv2", "rv3", "rv0", "rv4"); OLD = ("fb", "fbl", "sinc"); cells = list(itertools.product(C.ORD, C.DRAWS)); nc = len(cells)
DN = REPO + "/runs/diag/reve280_probs18"; N = {}
ref = np.load(f"{C.D}/ttt18within0d1_fb_base-all_loso-s0.npz")["index"]
for o, d in cells:
    for f in OLD + NEW:
        for s in range(3):
            fn = f"{C.D if f in OLD else DN}/ttt18{o}{d}_{f}_base-all_loso-s{s}.npz"; assert os.path.exists(fn), fn
            z = np.load(fn); assert z["proba"].shape == (len(C.y), 3) and (z["index"] == ref).all(), fn; N[o, d, f, s] = z["proba"].astype(np.float64)
for k in list(N)[::5]:
    assert np.allclose(C.subset_ba(N, k[0], k[1], [(k[2], k[3])]), C.person_ba(N[k]))
ba = lambda nets: np.stack([C.subset_ba(N, o, d, nets) for o, d in cells])
br = np.random.default_rng(1); idx = [(br.integers(0, len(C.PEOPLE), len(C.PEOPLE)), br.integers(0, nc, nc)) for _ in range(1000)]
nets = lambda fams, S: [(f, s) for f in fams for s in range(S)]


def rep(name, v):
    b = [v[np.ix_(oo, pp)].mean() for pp, oo in idx]; lo, hi = np.percentile(b, [10, 90])
    return f"  {name:50s} {v.mean():+6.2f}  [{lo:+.2f}, {hi:+.2f}]  people up {int((v.mean(0) > 0).sum())}/10  cells up {int((v.mean(1) > 0).sum())}/{nc}"


base = ba(nets(OLD, 3)); b4 = ba(nets(OLD, 4)) if False else None
L = [f"s280 REVE forms; cells {cells}; BA % person-equal, mean posterior, 3 seeds per form; marginal = BA(shipped three + form) - BA(shipped three)", ""]
L.append(f"shipped three (fb,fbl,sinc, 3 seeds each) BA {base.mean():.2f}; singles (3 nets): " + ", ".join(f"{f} {ba(nets([f], 3)).mean():.2f}" for f in OLD))
L.append("form alone (3 nets) BA, and per-person range: ")
for f in NEW:
    a = ba(nets([f], 3)); L.append(f"  {f:5s} {a.mean():.2f}  (people {a.mean(0).min():.1f}-{a.mean(0).max():.1f})")
L.append(""); L.append("marginal over the shipped three:")
for f in NEW:
    L.append(rep(f"+{f}", ba(nets(OLD, 3) + nets([f], 3)) - base))
L.append(""); L.append("pretraining value like-for-like (pretrained R2 - random-init R0, same recipe): alone and marginal")
L.append(rep("rv2-rv0 alone", ba(nets(["rv2"], 3)) - ba(nets(["rv0"], 3)))); L.append(rep("rv2-rv0 marginal", ba(nets(OLD, 3) + nets(["rv2"], 3)) - ba(nets(OLD, 3) + nets(["rv0"], 3))))
L.append(""); L.append("equal-n swap: base at 4 seeds each is not available (3 stored seeds + s3 may be partial); use the stored 6-seed old families: base4 = fb,fbl,sinc at seeds 0-3")
fam6 = lambda S: [(f, s) for f in OLD for s in range(S)]
for f in OLD:
    for s in range(6):
        for o, d in cells:
            if (o, d, f, s) not in N:
                N[o, d, f, s] = np.load(f"{C.D}/ttt18{o}{d}_{f}_base-all_loso-s{s}.npz")["proba"].astype(np.float64)
b4 = ba(fam6(4)); L.append(f"  base 12 nets (4 seeds each) BA {b4.mean():.2f}")
for f in NEW:
    L.append(rep(f"{f}: (base3 + {f}) - base4", ba(fam6(3) + nets([f], 3)) - b4))
L.append(""); L.append("mean Pearson correlation of posteriors with fb (seed matched, mean over cells); fb-fb other seed as reference:")
cor = lambda f, g, sh: np.mean([np.corrcoef(N[o, d, f, s].ravel(), N[o, d, g, (s + sh) % 3].ravel())[0, 1] for o, d in cells for s in range(3)])
L.append(f"  fb with fb {cor('fb', 'fb', 1):.3f}")
for f in ("fbl", "sinc") + NEW: L.append(f"  {f:5s} with fb {cor(f, 'fb', 0):.3f}")
txt = "\n".join(L); print(txt)
os.makedirs(REPO + "/evidence/reve280_15nov", exist_ok=True); open(REPO + "/evidence/reve280_15nov/score.txt", "w").write(txt + "\n")
